# routing/ai_analyzer.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from threading import RLock

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .models import AIAnalysis, CallInput, Evidence, IntentResult, Priority
from .nlp_preprocess import PreprocessConfig, build_canonical

logger = logging.getLogger(__name__)


def _fmt_ts(seconds: float) -> str:
    s = int(max(0, seconds))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}" if hh > 0 else f"{mm:02d}:{ss:02d}"


def _l2_normalize(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(p=2, dim=-1, keepdim=True) + 1e-12)


@torch.inference_mode()
def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    s = (last_hidden * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-9)
    return s / d


class AIAnalyzer:
    def analyze(self, call: CallInput, allowed_intents: Dict[str, Dict], groups: Optional[Dict[str, Dict]] = None) -> AIAnalysis:
        raise NotImplementedError


class RubertEmbeddingAnalyzer(AIAnalyzer):
    def __init__(
        self,
        model_name: str = "ai-forever/ruBert-base",
        device: Optional[str] = None,
        min_confidence: float = 0.55,
        max_text_chars: int = 4000,
        preprocess_cfg: Optional[PreprocessConfig] = None,
        tuned_model_path: Optional[str] = None,
        tuned_blend_alpha: float = 0.65,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_confidence = min_confidence
        self.max_text_chars = max_text_chars
        self.tuned_blend_alpha = max(0.0, min(1.0, float(tuned_blend_alpha)))

        self.preprocess_cfg = preprocess_cfg or PreprocessConfig(
            drop_fillers=True,
            dedupe=True,
            keep_timestamps=True,
            do_lemmatize=True,
            drop_stopwords=False,
            max_chars=max_text_chars,
        )

        logger.info("Loading RuBERT model: %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self._intent_cache_key: Optional[Tuple[str, ...]] = None
        self._intent_mat: Optional[torch.Tensor] = None
        self._intent_ids: List[str] = []

        self.tuned_model_path = tuned_model_path
        self._state_lock = RLock()
        self._tuned_artifact: Optional[Dict[str, Any]] = None
        self._tuned_head: Optional[nn.Linear] = None
        self._active_head_intents: Optional[Tuple[str, ...]] = None
        self._last_train_report: Optional[Dict[str, Any]] = None
        self._last_train_error: str = ""

        self._load_tuned_artifact_from_disk()

    def analyze(self, call: CallInput, allowed_intents: Dict[str, Dict], groups: Optional[Dict[str, Dict]] = None) -> AIAnalysis:
        start_time = time.time()

        prep = build_canonical([(s.start, s.text, s.role) for s in call.segments], self.preprocess_cfg)
        text = self._extract_text_with_context(prep.canonical_text, self.max_text_chars)
        lemmas_for_rules = prep.lemmas

        intent_ids, intent_emb = self._build_intent_matrix(allowed_intents)

        q = self._embed([text])
        sims = (q @ intent_emb.T).squeeze(0)

        rule_boosts = self._calculate_rule_boosts(prep, allowed_intents)
        if rule_boosts:
            idx_map = {intent_id: i for i, intent_id in enumerate(intent_ids)}
            for intent_id, delta in rule_boosts.items():
                idx = idx_map.get(intent_id)
                if idx is not None:
                    sims[idx] = sims[idx] + float(delta)

        sim_probs = torch.softmax(sims / 0.15, dim=0)
        tuned_probs, tuned_meta = self._predict_with_tuned_head(q, intent_ids)

        final_probs = sim_probs
        confidence_source = "embed"
        if tuned_probs is not None:
            alpha = self.tuned_blend_alpha
            final_probs = (1.0 - alpha) * sim_probs + alpha * tuned_probs
            confidence_source = "blended"

        top3_indices = torch.topk(final_probs, k=min(3, len(intent_ids)))[1]
        top3_intents = [
            {
                "intent": intent_ids[int(i)],
                "score": float(final_probs[i].item()),
                "sim": float(sims[i].item()),
            }
            for i in top3_indices
        ]

        best_idx = int(torch.argmax(final_probs).item())
        best_intent_id = intent_ids[best_idx]
        best_sim = float(sims[best_idx].item())

        if tuned_probs is not None:
            conf = float(final_probs[best_idx].item())
        else:
            conf = self._calculate_confidence(sims, best_idx)

        if conf < self.min_confidence:
            logger.warning("Low confidence %.3f for call %s, routing to triage", conf, call.call_id)
            best_intent_id = "misc.triage"
            priority: Priority = "normal"
            notes = f"{confidence_source} confidence={conf:.3f} (confidence_too_low -> triage)"
        else:
            meta = allowed_intents.get(best_intent_id, {})
            priority = meta.get("priority", "normal")
            notes = f"{confidence_source} confidence={conf:.3f}; sim={best_sim:.3f}"
            if tuned_probs is not None and tuned_meta.get("version_id"):
                notes += f"; tuned_version={tuned_meta['version_id']}"
            if rule_boosts:
                notes += f"; rule_boosts={{{', '.join(f'{k}:{v:.2f}' for k, v in rule_boosts.items())}}}"

        evidence = self._semantic_evidence(prep, allowed_intents.get(best_intent_id, {}).get("examples", []))

        suggested_targets = []
        meta = allowed_intents.get(best_intent_id, {})
        gid = meta.get("default_group")
        if gid:
            suggested_targets.append({"type": "group", "id": gid, "confidence": conf})

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Intent classified",
            extra={
                "call_id": call.call_id,
                "intent": best_intent_id,
                "confidence": round(conf, 3),
                "similarity": round(best_sim, 3),
                "text_length": len(text),
                "processing_time_ms": round(processing_time_ms, 2),
                "rule_boosts": rule_boosts,
                "confidence_source": confidence_source,
            },
        )

        intent = IntentResult(intent_id=best_intent_id, confidence=conf, evidence=evidence, notes=notes)

        return AIAnalysis(
            intent=intent,
            priority=priority,
            suggested_targets=suggested_targets,
            raw={
                "mode": "rubert_embed",
                "model_version": self.model_name,
                "device": self.device,
                "sim": round(best_sim, 4),
                "processing_time_ms": round(processing_time_ms, 2),
                "top3_intents": top3_intents,
                "prep_meta": prep.meta,
                "lemmas_n": len(lemmas_for_rules),
                "text_length": len(text),
                "rule_boosts": rule_boosts,
                "confidence_source": confidence_source,
                "tuned_model": tuned_meta,
            },
        )

    def get_training_status(self, allowed_intents: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        with self._state_lock:
            artifact = dict(self._tuned_artifact or {})
            report = dict(self._last_train_report or {})
            last_error = self._last_train_error

        if not artifact:
            return {
                "active": False,
                "reason": "no_tuned_model",
                "model_path": self.tuned_model_path,
                "last_train_report": report,
                "last_train_error": last_error,
            }

        artifact_intents = list(artifact.get("intent_ids") or [])
        current_intents = sorted(allowed_intents.keys()) if allowed_intents else None
        compatible = current_intents is None or artifact_intents == current_intents

        reason = "ok" if compatible else "intents_mismatch"
        return {
            "active": compatible,
            "reason": reason,
            "model_path": self.tuned_model_path,
            "version_id": artifact.get("version_id", ""),
            "trained_at": artifact.get("trained_at", ""),
            "blend_alpha": self.tuned_blend_alpha,
            "intent_ids": artifact_intents,
            "current_intents": current_intents,
            "metrics": artifact.get("metrics", {}),
            "dataset": artifact.get("dataset", {}),
            "last_train_report": report,
            "last_train_error": last_error,
        }

    def train_tuned_head(
        self,
        allowed_intents: Dict[str, Dict],
        *,
        feedback_path: str,
        output_path: str,
        epochs: int = 90,
        batch_size: int = 32,
        learning_rate: float = 1e-2,
        val_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        started = time.time()
        self._set_last_train_error("")

        try:
            samples, dataset_meta = self._collect_training_samples(allowed_intents, feedback_path)
            if not samples:
                raise RuntimeError("no training samples after preprocessing")

            intent_ids = sorted(allowed_intents.keys())
            label_to_idx = {iid: i for i, iid in enumerate(intent_ids)}

            filtered_samples: List[Tuple[str, int, str]] = []
            for text, intent_id, source in samples:
                idx = label_to_idx.get(intent_id)
                if idx is None:
                    continue
                filtered_samples.append((text, idx, source))

            if len(filtered_samples) < max(30, len(intent_ids) * 3):
                raise RuntimeError(
                    f"insufficient labeled data for training: {len(filtered_samples)} samples for {len(intent_ids)} intents"
                )

            texts = [row[0] for row in filtered_samples]
            labels = [row[1] for row in filtered_samples]

            features = self._embed_batched(texts, batch_size=64).cpu()
            label_tensor = torch.tensor(labels, dtype=torch.long)

            train_idx, val_idx = self._stratified_split(labels, val_ratio=val_ratio, random_seed=random_seed)
            if not train_idx:
                raise RuntimeError("stratified split produced empty train set")

            train_x = features[train_idx]
            train_y = label_tensor[train_idx]

            if val_idx:
                val_x = features[val_idx]
                val_y = label_tensor[val_idx]
            else:
                val_x = train_x
                val_y = train_y

            in_features = int(features.shape[1])
            num_classes = len(intent_ids)

            model = nn.Linear(in_features, num_classes).to(self.device)
            class_weights = self._build_class_weights(train_y, num_classes).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.01)

            epochs = int(max(10, min(400, epochs)))
            batch_size = int(max(8, min(256, batch_size)))
            best_state = None
            best_val_f1 = -1.0
            best_epoch = 0
            patience = 25
            no_improve = 0

            torch.manual_seed(int(random_seed))
            random.seed(int(random_seed))

            for epoch in range(1, epochs + 1):
                model.train()
                indices = torch.randperm(train_x.size(0))
                for start in range(0, train_x.size(0), batch_size):
                    batch_idx = indices[start : start + batch_size]
                    xb = train_x[batch_idx].to(self.device)
                    yb = train_y[batch_idx].to(self.device)

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                val_metrics = self._evaluate_linear(model, val_x, val_y, batch_size=batch_size)
                val_f1 = float(val_metrics["macro_f1"])

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
                raise RuntimeError("training failed: no best checkpoint")

            model.load_state_dict(best_state)
            train_metrics = self._evaluate_linear(model, train_x, train_y, batch_size=batch_size)
            val_metrics = self._evaluate_linear(model, val_x, val_y, batch_size=batch_size)

            trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            version_id = f"tuned-{int(time.time())}"
            artifact = {
                "artifact_version": 1,
                "version_id": version_id,
                "trained_at": trained_at,
                "model_name": self.model_name,
                "in_features": in_features,
                "intent_ids": intent_ids,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metrics": {
                    "best_epoch": best_epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "epochs_requested": epochs,
                },
                "dataset": {
                    **dataset_meta,
                    "samples_total": len(filtered_samples),
                    "samples_train": len(train_idx),
                    "samples_val": len(val_idx),
                },
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
            }
            self._set_last_train_report(report)
            return report

        except Exception as exc:
            self._set_last_train_error(str(exc))
            raise

    def reload_tuned_head_from_disk(self) -> Dict[str, Any]:
        self._load_tuned_artifact_from_disk()
        return self.get_training_status()

    def _load_tuned_artifact_from_disk(self) -> None:
        if not self.tuned_model_path:
            return
        model_path = Path(self.tuned_model_path)
        if not model_path.exists():
            return

        try:
            payload = torch.load(model_path, map_location="cpu")
            if not isinstance(payload, dict):
                raise RuntimeError("tuned model payload is not a dict")
            self._activate_tuned_artifact(payload)
            logger.info("Loaded tuned routing head: %s", model_path)
        except Exception as exc:
            logger.warning("Failed to load tuned routing head from %s: %s", model_path, exc)

    def _activate_tuned_artifact(self, artifact: Dict[str, Any]) -> None:
        with self._state_lock:
            self._tuned_artifact = dict(artifact)
            self._tuned_head = None
            self._active_head_intents = None

    def _set_last_train_report(self, report: Dict[str, Any]) -> None:
        with self._state_lock:
            self._last_train_report = dict(report)
            self._last_train_error = ""

    def _set_last_train_error(self, error: str) -> None:
        with self._state_lock:
            self._last_train_error = error

    def _predict_with_tuned_head(
        self,
        q: torch.Tensor,
        intent_ids: List[str],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        with self._state_lock:
            artifact = dict(self._tuned_artifact or {})

        if not artifact:
            return None, {"active": False, "reason": "no_tuned_model"}

        artifact_model_name = str(artifact.get("model_name") or "")
        if artifact_model_name != self.model_name:
            return None, {
                "active": False,
                "reason": "model_name_mismatch",
                "artifact_model_name": artifact_model_name,
            }

        artifact_intents = list(artifact.get("intent_ids") or [])
        if artifact_intents != intent_ids:
            return None, {
                "active": False,
                "reason": "intents_mismatch",
                "artifact_intents_n": len(artifact_intents),
                "runtime_intents_n": len(intent_ids),
            }

        try:
            head = self._ensure_tuned_head_loaded(artifact, intent_ids)
            with torch.inference_mode():
                logits = head(q)
                probs = torch.softmax(logits, dim=1).squeeze(0)
            return probs, {
                "active": True,
                "version_id": artifact.get("version_id", ""),
                "trained_at": artifact.get("trained_at", ""),
            }
        except Exception as exc:
            logger.warning("Failed to run tuned routing head: %s", exc)
            return None, {"active": False, "reason": f"runtime_error:{exc}"}

    def _ensure_tuned_head_loaded(self, artifact: Dict[str, Any], intent_ids: List[str]) -> nn.Linear:
        intent_key = tuple(intent_ids)
        with self._state_lock:
            if self._tuned_head is not None and self._active_head_intents == intent_key:
                return self._tuned_head

            state_dict = artifact.get("state_dict")
            if not isinstance(state_dict, dict):
                raise RuntimeError("invalid state_dict in tuned artifact")

            in_features = int(artifact.get("in_features") or 0)
            if in_features <= 0:
                raise RuntimeError("invalid in_features in tuned artifact")

            out_features = len(intent_ids)
            head = nn.Linear(in_features, out_features).to(self.device)
            head.load_state_dict(state_dict)
            head.eval()

            self._tuned_head = head
            self._active_head_intents = intent_key
            return head

    def _save_tuned_artifact(self, output_path: str, artifact: Dict[str, Any]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(artifact, tmp_path)
        os.replace(tmp_path, path)

    def _collect_training_samples(
        self,
        allowed_intents: Dict[str, Dict],
        feedback_path: str,
    ) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
        seen = set()
        rows: List[Tuple[str, str, str]] = []
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
                rows.append((text, intent_id, "intent_examples"))
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

                rows.append((text, intent_id, "operator_feedback"))
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

    def _embed_batched(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            chunks.append(self._embed(batch).detach().cpu())
        return torch.cat(chunks, dim=0)

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

    def _evaluate_linear(self, model: nn.Linear, x: torch.Tensor, y: torch.Tensor, batch_size: int = 64) -> Dict[str, float]:
        model.eval()
        losses: List[float] = []
        preds: List[torch.Tensor] = []
        criterion = nn.CrossEntropyLoss()

        with torch.inference_mode():
            for i in range(0, x.size(0), batch_size):
                xb = x[i : i + batch_size].to(self.device)
                yb = y[i : i + batch_size].to(self.device)
                logits = model(xb)
                loss = criterion(logits, yb)
                losses.append(float(loss.item()))
                preds.append(torch.argmax(logits, dim=1).detach().cpu())

        pred = torch.cat(preds, dim=0) if preds else torch.empty_like(y)
        acc = float((pred == y).float().mean().item()) if y.numel() > 0 else 0.0
        macro_f1 = self._macro_f1(pred, y)

        return {
            "loss": round(sum(losses) / max(1, len(losses)), 6),
            "accuracy": round(acc, 6),
            "macro_f1": round(macro_f1, 6),
        }

    def _macro_f1(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if target.numel() == 0:
            return 0.0

        labels = sorted({int(x.item()) for x in target})
        f1_scores: List[float] = []
        for label in labels:
            p = pred == label
            t = target == label
            tp = float((p & t).sum().item())
            fp = float((p & ~t).sum().item())
            fn = float((~p & t).sum().item())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))

        return float(sum(f1_scores) / max(1, len(f1_scores)))

    def _extract_text_with_context(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text

        start_chars = int(max_chars * 0.6)
        end_chars = int(max_chars * 0.4)

        start_text = text[:start_chars]
        end_text = text[-end_chars:]

        return start_text + "\n[...]\n" + end_text

    def _calculate_rule_boosts(self, prep, allowed_intents: Dict[str, Dict]) -> Dict[str, float]:
        boosts: Dict[str, float] = defaultdict(float)
        text = (prep.canonical_text or "").lower()
        lemmas = [str(x).lower() for x in (prep.lemmas or [])]
        tokens = [str(x).lower() for x in (prep.tokens or [])]
        bag = lemmas + tokens

        for intent_id, meta in allowed_intents.items():
            keywords = meta.get("keywords") or []
            hit_count = 0
            for kw in keywords:
                key = str(kw).strip().lower()
                if not key:
                    continue
                if " " in key:
                    if key in text:
                        hit_count += 1
                    continue
                if any(t.startswith(key) for t in bag):
                    hit_count += 1
                elif re.search(rf"\b{re.escape(key)}\w*", text):
                    hit_count += 1

            if hit_count > 0:
                boosts[intent_id] += min(0.34, 0.11 * hit_count)

        non_triage_score = sum(v for k, v in boosts.items() if k != "misc.triage")
        if non_triage_score >= 0.11 and "misc.triage" in allowed_intents:
            boosts["misc.triage"] -= min(0.22, 0.5 * non_triage_score)

        return dict(boosts)

    def _calculate_confidence(self, sims: torch.Tensor, best_idx: int) -> float:
        sorted_sims = torch.sort(sims, descending=True)[0]
        best = float(sorted_sims[0].item())
        second = float(sorted_sims[1].item()) if len(sorted_sims) > 1 else 0.0

        margin = best - second

        conf_sim = (best - 0.2) / 0.6
        conf_margin = min(1.0, margin / 0.3)

        conf = 0.7 * conf_sim + 0.3 * conf_margin

        return max(0.0, min(1.0, conf))

    def _build_intent_matrix(self, allowed_intents: Dict[str, Dict]) -> Tuple[List[str], torch.Tensor]:
        ids = sorted(allowed_intents.keys())
        key_parts: List[str] = []
        for intent_id in ids:
            meta = allowed_intents[intent_id]
            examples = meta.get("examples") or [meta.get("title", intent_id)]
            normalized_examples = "|".join(str(x).strip().lower() for x in examples[:10])
            key_parts.append(
                f"{intent_id}::{str(meta.get('title', '')).strip().lower()}::{normalized_examples}"
            )
        key = tuple(key_parts)
        if self._intent_cache_key == key and self._intent_mat is not None:
            return self._intent_ids, self._intent_mat

        logger.info("Building intent matrix for %d intents", len(ids))
        emb_list = []
        self._intent_ids = []
        for intent_id in ids:
            meta = allowed_intents[intent_id]
            examples = meta.get("examples") or [meta.get("title", intent_id)]
            ex_text = " ; ".join(str(x) for x in examples[:10])
            v = self._embed([ex_text])
            emb_list.append(v)
            self._intent_ids.append(intent_id)

        mat = torch.cat(emb_list, dim=0)
        self._intent_cache_key = key
        self._intent_mat = mat
        return self._intent_ids, mat

    def _embed(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        return _l2_normalize(pooled)

    def _semantic_evidence(self, prep, examples: List[str], top_k: int = 2) -> List[Evidence]:
        if not examples or not prep.sentences:
            return []

        try:
            ex_emb = self._embed(examples)
            sent_emb = self._embed(prep.sentences)

            sims = (sent_emb @ ex_emb.T).max(dim=1)[0]

            top_indices = torch.topk(sims, k=min(top_k, len(prep.sentences)))[1]

            evidence = []
            for idx in top_indices:
                sent = prep.sentences[int(idx)]
                ts = self._find_timestamp_for_sentence(sent, prep.lines)
                evidence.append(Evidence(text=sent, timestamp=ts))

            return evidence

        except Exception as e:
            logger.warning("Semantic evidence extraction failed: %s, falling back to simple method", e)
            return self._simple_evidence(prep, examples)

    def _find_timestamp_for_sentence(self, sentence: str, lines: List[str]) -> str:
        sent_norm = sentence.lower().strip()

        for line in lines:
            m = re.match(r"^\[(\d{2}:\d{2})\]\s*(.*)$", line)
            if m:
                ts, txt = m.group(1), m.group(2)
                txt_norm = txt.lower().strip()

                if sent_norm in txt_norm or txt_norm in sent_norm:
                    return ts

        return "00:00"

    def _simple_evidence(self, prep, examples: List[str]) -> List[Evidence]:
        if not examples:
            return []

        ex_words = set((" ".join(examples)).lower().split())

        scored = []
        for line in prep.lines:
            txt = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", line).strip()
            w = set(txt.split())
            inter = len(w & ex_words)
            if inter > 0:
                scored.append((inter, line))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Evidence] = []
        for _, line in scored[:2]:
            m = re.match(r"^\[(\d{2}:\d{2})\]\s*(.*)$", line)
            if m:
                ts, txt = m.group(1), m.group(2)
                out.append(Evidence(text=txt, timestamp=ts))
            else:
                out.append(Evidence(text=line, timestamp="00:00"))
        return out
