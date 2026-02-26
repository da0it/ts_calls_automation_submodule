from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import os
import random
import re
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .models import AIAnalysis, CallInput, Evidence, IntentResult, Priority
from .nlp_preprocess import PreprocessConfig, build_canonical

logger = logging.getLogger(__name__)


class AIAnalyzer:
    def analyze(self, call: CallInput, allowed_intents: Dict[str, Dict], groups: Optional[Dict[str, Dict]] = None) -> AIAnalysis:
        raise NotImplementedError


class RubertEmbeddingAnalyzer(AIAnalyzer):
    """
    Router analyzer in a simplified "fine-tuned model only" mode.
    If no fine-tuned model is available or confidence is too low, routes to misc.triage.
    """

    def __init__(
        self,
        model_name: str = "ai-forever/ruBert-base",
        device: Optional[str] = None,
        min_confidence: float = 0.55,
        max_text_chars: int = 4000,
        preprocess_cfg: Optional[PreprocessConfig] = None,
        tuned_model_path: Optional[str] = None,
        tuned_blend_alpha: float = 0.0,  # kept for backward-compatible config, unused
        finetuned_enabled: bool = False,
        finetuned_model_path: Optional[str] = None,
        finetuned_blend_alpha: float = 1.0,  # kept for backward-compatible config, unused
        finetuned_learning_rate: float = 2e-5,
        finetuned_epochs: int = 3,
        finetuned_batch_size: int = 16,
        finetuned_max_length: int = 256,
        finetuned_weight_decay: float = 0.01,
        **_: Any,
    ):
        self.model_name = str(model_name).strip() or "ai-forever/ruBert-base"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_confidence = float(max(0.0, min(1.0, min_confidence)))
        self.max_text_chars = int(max(200, min(20000, max_text_chars)))

        self.preprocess_cfg = preprocess_cfg or PreprocessConfig(
            drop_fillers=True,
            dedupe=True,
            keep_timestamps=True,
            do_lemmatize=True,
            drop_stopwords=False,
            max_chars=self.max_text_chars,
        )

        self.finetuned_enabled = bool(finetuned_enabled)
        self.finetuned_model_path = str(finetuned_model_path or "").strip()
        self.finetuned_learning_rate = float(max(1e-6, min(1e-3, finetuned_learning_rate)))
        self.finetuned_epochs = int(max(1, min(12, finetuned_epochs)))
        self.finetuned_batch_size = int(max(4, min(64, finetuned_batch_size)))
        self.finetuned_max_length = int(max(64, min(512, finetuned_max_length)))
        self.finetuned_weight_decay = float(max(0.0, min(0.2, finetuned_weight_decay)))

        self.tuned_model_path = str(tuned_model_path or "").strip()
        self._state_lock = RLock()
        self._tuned_artifact: Optional[Dict[str, Any]] = None
        self._finetuned_model: Optional[AutoModelForSequenceClassification] = None
        self._finetuned_tokenizer: Optional[Any] = None
        self._active_finetuned_intents: Optional[Tuple[str, ...]] = None
        self._active_finetuned_path: str = ""
        self._last_train_report: Optional[Dict[str, Any]] = None
        self._last_train_error: str = ""

        self._load_tuned_artifact_from_disk()

    def analyze(self, call: CallInput, allowed_intents: Dict[str, Dict], groups: Optional[Dict[str, Dict]] = None) -> AIAnalysis:
        started = time.time()
        prep = build_canonical([(s.start, s.text, s.role) for s in call.segments], self.preprocess_cfg)
        text = self._extract_text_with_context(prep.canonical_text, self.max_text_chars)
        intent_ids = sorted(allowed_intents.keys())

        probs, meta = self._predict_with_finetuned_model(text, intent_ids)
        if probs is None:
            return self._triage_result(
                call_id=call.call_id,
                reason=f"finetuned_unavailable:{meta.get('reason', 'unknown')}",
                processing_time_ms=(time.time() - started) * 1000.0,
                text_len=len(text),
                prep_meta=prep.meta,
                model_meta=meta,
            )

        best_idx = int(torch.argmax(probs).item())
        best_intent_id = intent_ids[best_idx]
        confidence = float(probs[best_idx].item())

        if confidence < self.min_confidence:
            return self._triage_result(
                call_id=call.call_id,
                reason=f"low_confidence:{confidence:.3f}",
                processing_time_ms=(time.time() - started) * 1000.0,
                text_len=len(text),
                prep_meta=prep.meta,
                model_meta=meta,
            )

        meta_intent = allowed_intents.get(best_intent_id, {})
        priority = self._normalize_priority(meta_intent.get("priority", "medium"))
        default_group = str(meta_intent.get("default_group") or "").strip()
        targets = [{"type": "group", "id": default_group, "confidence": confidence}] if default_group else []

        top_k = min(3, len(intent_ids))
        top_indices = torch.topk(probs, k=top_k).indices.tolist()
        top3_intents = [{"intent": intent_ids[int(i)], "score": float(probs[int(i)].item())} for i in top_indices]

        analysis = AIAnalysis(
            intent=IntentResult(
                intent_id=best_intent_id,
                confidence=confidence,
                evidence=[],
                notes=f"finetuned confidence={confidence:.3f}",
            ),
            priority=priority,
            suggested_targets=targets,
            raw={
                "mode": "finetuned_only",
                "model_version": self.model_name,
                "device": self.device,
                "processing_time_ms": round((time.time() - started) * 1000.0, 2),
                "text_length": len(text),
                "prep_meta": prep.meta,
                "top3_intents": top3_intents,
                "finetuned_model": meta,
            },
        )
        logger.info(
            "Intent classified (finetuned-only) call_id=%s intent=%s conf=%.3f",
            call.call_id,
            best_intent_id,
            confidence,
        )
        return analysis

    def get_training_status(self, allowed_intents: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        with self._state_lock:
            artifact = dict(self._tuned_artifact or {})
            report = dict(self._last_train_report or {})
            last_error = str(self._last_train_error or "")

        if not artifact:
            return {
                "active": False,
                "reason": "no_tuned_model",
                "model_path": self.tuned_model_path,
                "finetuned_model": {
                    "enabled": self.finetuned_enabled,
                    "active": False,
                    "model_path": self.finetuned_model_path,
                },
                "last_train_report": report,
                "last_train_error": last_error,
            }

        artifact_intents = list(artifact.get("intent_ids") or [])
        current_intents = sorted(allowed_intents.keys()) if allowed_intents else None
        compatible = current_intents is None or artifact_intents == current_intents

        finetuned_model = artifact.get("finetuned_model") if isinstance(artifact.get("finetuned_model"), dict) else {}
        finetuned_ready = bool(finetuned_model and finetuned_model.get("enabled"))
        active = compatible and finetuned_ready
        reason = "ok" if active else ("intents_mismatch" if not compatible else "no_finetuned_model")

        return {
            "active": active,
            "reason": reason,
            "model_path": self.tuned_model_path,
            "version_id": artifact.get("version_id", ""),
            "trained_at": artifact.get("trained_at", ""),
            "intent_ids": artifact_intents,
            "current_intents": current_intents,
            "metrics": artifact.get("metrics", {}),
            "dataset": artifact.get("dataset", {}),
            "finetuned_model": {
                "enabled": self.finetuned_enabled,
                "active": active,
                "model_path": finetuned_model.get("model_path", self.finetuned_model_path),
                "metrics": finetuned_model.get("metrics", {}),
            },
            "last_train_report": report,
            "last_train_error": last_error,
        }

    def train_tuned_head(
        self,
        allowed_intents: Dict[str, Dict],
        *,
        feedback_path: str,
        output_path: str,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        val_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        started = time.time()
        self._set_last_train_error("")
        if not self.finetuned_enabled:
            raise RuntimeError("fine-tuning is disabled (set ROUTER_FINETUNED_ENABLED=1)")

        samples, dataset_meta = self._collect_training_samples(allowed_intents, feedback_path)
        if not samples:
            raise RuntimeError("no training samples after preprocessing")

        intent_ids = sorted(allowed_intents.keys())
        label_to_idx = {iid: i for i, iid in enumerate(intent_ids)}
        filtered: List[Dict[str, Any]] = []
        for sample in samples:
            intent_id = str(sample.get("intent_id") or "").strip()
            idx = label_to_idx.get(intent_id)
            if idx is None:
                continue
            item = dict(sample)
            item["label_idx"] = int(idx)
            filtered.append(item)

        if len(filtered) < max(30, len(intent_ids) * 3):
            raise RuntimeError(
                f"insufficient labeled data for training: {len(filtered)} samples for {len(intent_ids)} intents"
            )

        texts = [str(row.get("text") or "") for row in filtered]
        labels = [int(row.get("label_idx")) for row in filtered]
        train_idx, val_idx = self._stratified_split(labels, val_ratio=float(val_ratio), random_seed=int(random_seed))
        if not train_idx:
            raise RuntimeError("stratified split produced empty train set")

        report_finetuned, artifact_finetuned = self._train_finetuned_rubert(
            texts=texts,
            labels=labels,
            intent_ids=intent_ids,
            train_idx=train_idx,
            val_idx=val_idx,
            random_seed=int(random_seed),
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
        )

        trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        version_id = f"tuned-{int(time.time())}"
        artifact = {
            "artifact_version": 4,
            "version_id": version_id,
            "trained_at": trained_at,
            "model_name": self.model_name,
            "intent_ids": intent_ids,
            "metrics": report_finetuned.get("metrics", {}),
            "dataset": {
                **dataset_meta,
                "samples_total": len(filtered),
                "samples_train": len(train_idx),
                "samples_val": len(val_idx),
            },
            "finetuned_model": artifact_finetuned,
        }
        self._save_tuned_artifact(output_path, artifact)
        self._activate_tuned_artifact(artifact)

        report = {
            "ok": True,
            "version_id": version_id,
            "trained_at": trained_at,
            "duration_sec": round(time.time() - started, 2),
            "output_path": output_path,
            "metrics": artifact["metrics"],
            "dataset": artifact["dataset"],
            "finetuned_model": report_finetuned,
        }
        self._set_last_train_report(report)
        return report

    def reload_tuned_head_from_disk(self) -> Dict[str, Any]:
        self._load_tuned_artifact_from_disk()
        return self.get_training_status()

    def _triage_result(
        self,
        *,
        call_id: str,
        reason: str,
        processing_time_ms: float,
        text_len: int,
        prep_meta: Dict[str, Any],
        model_meta: Dict[str, Any],
    ) -> AIAnalysis:
        return AIAnalysis(
            intent=IntentResult(
                intent_id="misc.triage",
                confidence=0.0,
                evidence=[],
                notes=reason,
            ),
            priority="medium",
            suggested_targets=[{"type": "group", "id": "technical_support", "confidence": 0.0}],
            raw={
                "mode": "finetuned_only",
                "model_version": self.model_name,
                "device": self.device,
                "processing_time_ms": round(processing_time_ms, 2),
                "text_length": text_len,
                "prep_meta": prep_meta,
                "finetuned_model": model_meta,
            },
        )

    def _predict_with_finetuned_model(
        self,
        text: str,
        intent_ids: List[str],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        if not self.finetuned_enabled:
            return None, {"active": False, "reason": "finetuned_disabled"}

        with self._state_lock:
            artifact = dict(self._tuned_artifact or {})
        if not artifact:
            return None, {"active": False, "reason": "no_tuned_model"}

        finetuned_meta = artifact.get("finetuned_model")
        if not isinstance(finetuned_meta, dict) or not finetuned_meta.get("enabled"):
            return None, {"active": False, "reason": "no_finetuned_model"}

        artifact_intents = list(artifact.get("intent_ids") or [])
        if artifact_intents != intent_ids:
            return None, {
                "active": False,
                "reason": "intents_mismatch",
                "artifact_intents_n": len(artifact_intents),
                "runtime_intents_n": len(intent_ids),
            }

        try:
            model, tokenizer, max_len = self._ensure_finetuned_model_loaded(artifact, intent_ids)
            enc = tokenizer(
                [text],
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=1).squeeze(0)
            return probs, {
                "active": True,
                "trained_at": finetuned_meta.get("trained_at", ""),
                "model_path": finetuned_meta.get("model_path", ""),
            }
        except Exception as exc:
            logger.warning("Failed to run fine-tuned RuBERT head: %s", exc)
            return None, {"active": False, "reason": f"runtime_error:{exc}"}

    def _ensure_finetuned_model_loaded(
        self,
        artifact: Dict[str, Any],
        intent_ids: List[str],
    ) -> Tuple[AutoModelForSequenceClassification, Any, int]:
        intent_key = tuple(intent_ids)
        finetuned_artifact = artifact.get("finetuned_model")
        if not isinstance(finetuned_artifact, dict):
            raise RuntimeError("finetuned model metadata is missing")
        model_path = str(finetuned_artifact.get("model_path") or "").strip()
        if not model_path:
            raise RuntimeError("finetuned model path is empty")

        with self._state_lock:
            if (
                self._finetuned_model is not None
                and self._finetuned_tokenizer is not None
                and self._active_finetuned_intents == intent_key
                and self._active_finetuned_path == model_path
            ):
                max_len = int(finetuned_artifact.get("max_length") or self.finetuned_max_length)
                return self._finetuned_model, self._finetuned_tokenizer, max_len

            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if int(model.config.num_labels) != len(intent_ids):
                raise RuntimeError(
                    f"finetuned model classes mismatch: model={int(model.config.num_labels)} runtime={len(intent_ids)}"
                )

            self._finetuned_model = model
            self._finetuned_tokenizer = tokenizer
            self._active_finetuned_intents = intent_key
            self._active_finetuned_path = model_path
            max_len = int(finetuned_artifact.get("max_length") or self.finetuned_max_length)
            return model, tokenizer, max_len

    def _collect_training_samples(
        self,
        allowed_intents: Dict[str, Dict],
        feedback_path: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        seen = set()
        rows: List[Dict[str, Any]] = []
        source_counts: Dict[str, int] = defaultdict(int)
        class_counts: Dict[str, int] = defaultdict(int)

        for intent_id, meta in allowed_intents.items():
            base_examples = list(meta.get("examples") or [])
            if meta.get("title"):
                base_examples.append(str(meta["title"]))
            for example in base_examples:
                text = self._prepare_training_text(str(example))
                if not text:
                    continue
                key = (intent_id, text.lower())
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"text": text, "intent_id": intent_id, "source": "intent_examples"})
                source_counts["intent_examples"] += 1
                class_counts[intent_id] += 1

        feedback_file = Path(feedback_path)
        if feedback_file.exists() and feedback_file.is_file():
            for raw_line in feedback_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                final = item.get("final") or {}
                intent_id = str(final.get("intent_id") or "").strip()
                if intent_id not in allowed_intents:
                    continue

                text = str(item.get("training_sample") or "").strip()
                if not text:
                    text = str(item.get("transcript_text") or "").strip()
                text = self._prepare_training_text(text)
                if not text:
                    continue

                key = (intent_id, text.lower())
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"text": text, "intent_id": intent_id, "source": "operator_feedback"})
                source_counts["operator_feedback"] += 1
                class_counts[intent_id] += 1

        dataset_meta = {
            "source_counts": dict(source_counts),
            "class_counts": dict(class_counts),
            "feedback_path": str(feedback_file),
        }
        return rows, dataset_meta

    def _prepare_training_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) < 6:
            return ""
        if len(cleaned) > self.max_text_chars:
            cleaned = self._extract_text_with_context(cleaned, self.max_text_chars)
        return cleaned

    def _train_finetuned_rubert(
        self,
        *,
        texts: List[str],
        labels: List[int],
        intent_ids: List[str],
        train_idx: List[int],
        val_idx: List[int],
        random_seed: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        model_path = str(self.finetuned_model_path or "").strip()
        if not model_path:
            raise RuntimeError("ROUTER_FINETUNED_MODEL_PATH is empty")
        if not train_idx:
            raise RuntimeError("empty train set for fine-tuned model")
        if not val_idx:
            val_idx = list(train_idx)

        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_labels = torch.tensor([labels[i] for i in train_idx], dtype=torch.long)
        val_labels = torch.tensor([labels[i] for i in val_idx], dtype=torch.long)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        train_enc = tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=self.finetuned_max_length,
            return_tensors="pt",
        )
        val_enc = tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=self.finetuned_max_length,
            return_tensors="pt",
        )

        train_ds = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], train_labels)
        val_ds = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], val_labels)

        batch_size = int(max(4, min(64, batch_size)))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(intent_ids),
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(max(1e-6, min(1e-3, learning_rate))),
            weight_decay=float(self.finetuned_weight_decay),
        )
        class_weights = self._build_class_weights(train_labels, len(intent_ids)).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        torch.manual_seed(int(random_seed))
        random.seed(int(random_seed))

        best_state = None
        best_val_f1 = -1.0
        best_epoch = 0
        patience = 2
        no_improve = 0
        epochs = int(max(1, min(12, epochs)))

        for epoch in range(1, epochs + 1):
            model.train()
            for input_ids, attention_mask, yb in train_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            val_metrics = self._evaluate_finetuned(model, val_loader, criterion)
            val_f1 = float(val_metrics.get("macro_f1", 0.0))
            if val_f1 > best_val_f1 + 1e-6:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        if best_state is None:
            raise RuntimeError("fine-tuning failed: no best checkpoint")

        model.load_state_dict(best_state)
        train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        train_metrics = self._evaluate_finetuned(model, train_eval_loader, criterion)
        val_metrics = self._evaluate_finetuned(model, val_loader, criterion)

        save_dir = Path(model_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))

        trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta_payload = {
            "model_name": self.model_name,
            "intent_ids": intent_ids,
            "trained_at": trained_at,
            "max_length": self.finetuned_max_length,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        (save_dir / "intent_ids.json").write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report = {
            "enabled": True,
            "best_epoch": best_epoch,
            "model_path": str(save_dir),
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "epochs_requested": epochs,
            },
            "dataset": {
                "samples_total": len(texts),
                "samples_train": len(train_idx),
                "samples_val": len(val_idx),
            },
        }
        artifact = {
            "enabled": True,
            "model_path": str(save_dir),
            "intent_ids": intent_ids,
            "trained_at": trained_at,
            "max_length": self.finetuned_max_length,
            "metrics": report["metrics"],
            "dataset": report["dataset"],
        }
        return report, artifact

    def _evaluate_finetuned(
        self,
        model: AutoModelForSequenceClassification,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        model.eval()
        losses: List[float] = []
        preds: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        with torch.inference_mode():
            for input_ids, attention_mask, yb in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                yb = yb.to(self.device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = criterion(logits, yb)
                losses.append(float(loss.item()))
                preds.append(torch.argmax(logits, dim=1).detach().cpu())
                targets.append(yb.detach().cpu())

        pred = torch.cat(preds, dim=0) if preds else torch.empty(0, dtype=torch.long)
        target = torch.cat(targets, dim=0) if targets else torch.empty(0, dtype=torch.long)
        acc = float((pred == target).float().mean().item()) if target.numel() > 0 else 0.0
        macro_precision, macro_recall, macro_f1 = self._macro_precision_recall_f1(pred, target)
        return {
            "loss": round(sum(losses) / max(1, len(losses)), 6),
            "accuracy": round(acc, 6),
            "macro_precision": round(macro_precision, 6),
            "macro_recall": round(macro_recall, 6),
            "macro_f1": round(macro_f1, 6),
        }

    def _load_tuned_artifact_from_disk(self) -> None:
        if not self.tuned_model_path:
            self._clear_tuned_artifact()
            return
        path = Path(self.tuned_model_path)
        if not path.exists():
            self._clear_tuned_artifact()
            logger.info("No tuned router artifact found at %s", path)
            return
        try:
            payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict):
                raise RuntimeError("invalid tuned artifact payload")
            self._activate_tuned_artifact(payload)
            logger.info("Loaded tuned router artifact from %s", path)
        except Exception as exc:
            self._clear_tuned_artifact()
            logger.warning("Failed to load tuned router artifact from %s: %s", path, exc)

    def _save_tuned_artifact(self, output_path: str, artifact: Dict[str, Any]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(artifact, tmp)
        os.replace(tmp, path)

    def _activate_tuned_artifact(self, artifact: Dict[str, Any]) -> None:
        with self._state_lock:
            self._tuned_artifact = dict(artifact)
            self._finetuned_model = None
            self._finetuned_tokenizer = None
            self._active_finetuned_intents = None
            self._active_finetuned_path = ""

    def _clear_tuned_artifact(self) -> None:
        with self._state_lock:
            self._tuned_artifact = None
            self._finetuned_model = None
            self._finetuned_tokenizer = None
            self._active_finetuned_intents = None
            self._active_finetuned_path = ""

    def _set_last_train_report(self, report: Dict[str, Any]) -> None:
        with self._state_lock:
            self._last_train_report = dict(report)
            self._last_train_error = ""

    def _set_last_train_error(self, error: str) -> None:
        with self._state_lock:
            self._last_train_error = str(error or "")

    def _stratified_split(self, labels: List[int], val_ratio: float, random_seed: int) -> Tuple[List[int], List[int]]:
        by_class: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            by_class[int(label)].append(idx)

        rnd = random.Random(int(random_seed))
        train_idx: List[int] = []
        val_idx: List[int] = []
        val_ratio = max(0.0, min(0.5, float(val_ratio)))

        for _, indices in by_class.items():
            rnd.shuffle(indices)
            if len(indices) <= 1 or val_ratio <= 0.0:
                train_idx.extend(indices)
                continue
            take_val = max(1, int(len(indices) * val_ratio))
            take_val = min(take_val, len(indices) - 1)
            val_idx.extend(indices[:take_val])
            train_idx.extend(indices[take_val:])

        rnd.shuffle(train_idx)
        rnd.shuffle(val_idx)
        return train_idx, val_idx

    def _build_class_weights(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1.0)
        inv = 1.0 / counts
        return inv / inv.mean()

    def _macro_precision_recall_f1(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
        if target.numel() == 0:
            return 0.0, 0.0, 0.0
        labels = sorted({int(x.item()) for x in target})
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        f1_scores: List[float] = []
        for label in labels:
            p = pred == label
            t = target == label
            tp = float((p & t).sum().item())
            fp = float((p & ~t).sum().item())
            fn = float((~p & t).sum().item())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision_scores.append(precision)
            recall_scores.append(recall)
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))

        macro_precision = float(sum(precision_scores) / max(1, len(precision_scores)))
        macro_recall = float(sum(recall_scores) / max(1, len(recall_scores)))
        macro_f1 = float(sum(f1_scores) / max(1, len(f1_scores)))
        return macro_precision, macro_recall, macro_f1

    def _extract_text_with_context(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        start_chars = int(max_chars * 0.6)
        end_chars = int(max_chars * 0.4)
        return text[:start_chars] + "\n[...]\n" + text[-end_chars:]

    def _normalize_priority(self, value: Any) -> Priority:
        raw = str(value or "").strip().lower()
        if raw == "normal":
            raw = "medium"
        if raw not in {"low", "medium", "high", "critical"}:
            raw = "medium"
        return raw  # type: ignore[return-value]
