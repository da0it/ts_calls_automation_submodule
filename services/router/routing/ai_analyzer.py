# routing/ai_analyzer.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
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
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

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


ROLE_UNKNOWN = 0
ROLE_CALLER = 1
ROLE_AGENT = 2
ROLE_SYSTEM = 3
ROLE_VOCAB_SIZE = 4
DEFAULT_URGENCY_PATTERNS = [
    "вся компания",
    "все сотрудники",
    "не можем работать",
    "работа стоит",
    "простой",
    "sla",
    "production down",
    "критическ",
]


@dataclass
class DialogTurn:
    role_id: int
    text: str


class DialogTransformerHead(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_turns: int = 64,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.max_turns = int(max_turns)

        self.input_proj = nn.Linear(self.in_features, self.d_model)
        self.role_emb = nn.Embedding(ROLE_VOCAB_SIZE, self.d_model)
        self.pos_emb = nn.Embedding(self.max_turns, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(
        self,
        turn_embeddings: torch.Tensor,
        role_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # turn_embeddings: [B, T, F]
        # role_ids: [B, T]
        # key_padding_mask: [B, T] (True for padding)
        batch, seq_len, _ = turn_embeddings.shape
        seq_len = min(seq_len, self.max_turns)
        x = turn_embeddings[:, :seq_len, :]
        roles = role_ids[:, :seq_len]
        mask = key_padding_mask[:, :seq_len]

        x = self.input_proj(x)
        pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        x = x + self.role_emb(roles) + self.pos_emb(pos_idx)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x)

        valid = (~mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


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
        finetuned_enabled: bool = False,
        finetuned_model_path: Optional[str] = None,
        finetuned_blend_alpha: float = 0.45,
        finetuned_learning_rate: float = 2e-5,
        finetuned_epochs: int = 3,
        finetuned_batch_size: int = 16,
        finetuned_max_length: int = 256,
        finetuned_weight_decay: float = 0.01,
        dialog_head_enabled: bool = True,
        dialog_blend_alpha: float = 0.55,
        dialog_d_model: int = 256,
        dialog_nhead: int = 4,
        dialog_layers: int = 2,
        dialog_dropout: float = 0.1,
        dialog_max_turns: int = 64,
        dialog_max_turn_chars: int = 280,
        chunk_inference_enabled: bool = False,
        chunk_max_chars: int = 1200,
        chunk_overlap_turns: int = 1,
        chunk_max_count: int = 8,
        chunk_late_weight: float = 0.25,
        chunk_blend_alpha: float = 0.35,
        urgency_escalation_enabled: bool = True,
        urgency_priority_floor: str = "high",
        urgency_patterns: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_confidence = min_confidence
        self.max_text_chars = max_text_chars
        self.tuned_blend_alpha = max(0.0, min(1.0, float(tuned_blend_alpha)))
        self.finetuned_enabled = bool(finetuned_enabled)
        self.finetuned_model_path = str(finetuned_model_path or "").strip()
        self.finetuned_blend_alpha = max(0.0, min(1.0, float(finetuned_blend_alpha)))
        self.finetuned_learning_rate = float(max(1e-6, min(1e-3, finetuned_learning_rate)))
        self.finetuned_epochs = int(max(1, min(12, finetuned_epochs)))
        self.finetuned_batch_size = int(max(4, min(64, finetuned_batch_size)))
        self.finetuned_max_length = int(max(64, min(512, finetuned_max_length)))
        self.finetuned_weight_decay = float(max(0.0, min(0.2, finetuned_weight_decay)))
        self.dialog_head_enabled = bool(dialog_head_enabled)
        self.dialog_blend_alpha = max(0.0, min(1.0, float(dialog_blend_alpha)))
        self.dialog_d_model = int(max(64, min(1024, dialog_d_model)))
        self.dialog_nhead = int(max(1, min(16, dialog_nhead)))
        while self.dialog_nhead > 1 and self.dialog_d_model % self.dialog_nhead != 0:
            self.dialog_nhead -= 1
        self.dialog_layers = int(max(1, min(8, dialog_layers)))
        self.dialog_dropout = float(max(0.0, min(0.5, dialog_dropout)))
        self.dialog_max_turns = int(max(8, min(256, dialog_max_turns)))
        self.dialog_max_turn_chars = int(max(40, min(1200, dialog_max_turn_chars)))
        self.chunk_inference_enabled = bool(chunk_inference_enabled)
        self.chunk_max_chars = int(max(300, min(5000, chunk_max_chars)))
        self.chunk_overlap_turns = int(max(0, min(6, chunk_overlap_turns)))
        self.chunk_max_count = int(max(1, min(24, chunk_max_count)))
        self.chunk_late_weight = float(max(0.0, min(1.5, chunk_late_weight)))
        self.chunk_blend_alpha = float(max(0.0, min(1.0, chunk_blend_alpha)))
        self.urgency_escalation_enabled = bool(urgency_escalation_enabled)
        floor = str(urgency_priority_floor or "high").strip().lower()
        self.urgency_priority_floor = floor if floor in {"low", "medium", "high", "critical"} else "high"
        patterns = urgency_patterns if isinstance(urgency_patterns, list) else None
        self.urgency_patterns = [str(p).strip().lower() for p in (patterns or DEFAULT_URGENCY_PATTERNS) if str(p).strip()]

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
        self._finetuned_model: Optional[AutoModelForSequenceClassification] = None
        self._finetuned_tokenizer: Optional[Any] = None
        self._dialog_head: Optional[DialogTransformerHead] = None
        self._active_head_intents: Optional[Tuple[str, ...]] = None
        self._active_finetuned_intents: Optional[Tuple[str, ...]] = None
        self._active_finetuned_path: str = ""
        self._active_dialog_intents: Optional[Tuple[str, ...]] = None
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
        finetuned_probs, finetuned_meta = self._predict_with_finetuned_model(text, intent_ids)
        dialog_probs, dialog_meta = self._predict_with_dialog_head(call, intent_ids)
        chunk_probs, chunk_meta = self._predict_with_chunked_context(call, intent_ids, intent_emb, allowed_intents)

        final_probs = sim_probs
        confidence_sources = ["embed"]
        if tuned_probs is not None:
            alpha = self.tuned_blend_alpha
            final_probs = (1.0 - alpha) * sim_probs + alpha * tuned_probs
            confidence_sources.append("linear")
        if finetuned_probs is not None:
            alpha = self.finetuned_blend_alpha
            final_probs = (1.0 - alpha) * final_probs + alpha * finetuned_probs
            confidence_sources.append("finetuned")
        if dialog_probs is not None:
            alpha = self.dialog_blend_alpha
            final_probs = (1.0 - alpha) * final_probs + alpha * dialog_probs
            confidence_sources.append("dialog")
        if chunk_probs is not None:
            alpha = self.chunk_blend_alpha
            final_probs = (1.0 - alpha) * final_probs + alpha * chunk_probs
            confidence_sources.append("chunk")
        confidence_source = "+".join(confidence_sources)

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

        if tuned_probs is not None or finetuned_probs is not None or dialog_probs is not None:
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
            if finetuned_probs is not None and finetuned_meta.get("trained_at"):
                notes += f"; finetuned_trained_at={finetuned_meta['trained_at']}"
            if dialog_probs is not None and dialog_meta.get("version_id"):
                notes += f"; dialog_version={dialog_meta['version_id']}"
            if rule_boosts:
                notes += f"; rule_boosts={{{', '.join(f'{k}:{v:.2f}' for k, v in rule_boosts.items())}}}"

        urgency_meta = self._detect_urgency_signals(prep)
        if urgency_meta.get("triggered"):
            old_priority = str(priority)
            priority = self._elevate_priority(old_priority, self.urgency_priority_floor)
            notes += f"; urgency_hits={urgency_meta.get('matched', [])}"
            if str(priority) != old_priority:
                notes += f"; priority_escalated:{old_priority}->{priority}"

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
                "finetuned_model": finetuned_meta,
                "dialog_model": dialog_meta,
                "chunk_model": chunk_meta,
                "urgency": urgency_meta,
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
                "finetuned_model": {
                    "enabled": self.finetuned_enabled,
                    "active": False,
                    "blend_alpha": self.finetuned_blend_alpha,
                    "model_path": self.finetuned_model_path,
                },
                "last_train_report": report,
                "last_train_error": last_error,
            }

        artifact_intents = list(artifact.get("intent_ids") or [])
        current_intents = sorted(allowed_intents.keys()) if allowed_intents else None
        compatible = current_intents is None or artifact_intents == current_intents

        dialog_head = artifact.get("dialog_head") if isinstance(artifact.get("dialog_head"), dict) else {}
        finetuned_model = artifact.get("finetuned_model") if isinstance(artifact.get("finetuned_model"), dict) else {}
        finetuned_ready = bool(finetuned_model and finetuned_model.get("enabled"))
        dialog_ready = bool(dialog_head and dialog_head.get("enabled"))
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
            "linear_head": {
                "active": compatible,
                "blend_alpha": self.tuned_blend_alpha,
                "metrics": artifact.get("metrics", {}),
            },
            "dialog_head": {
                "enabled": self.dialog_head_enabled,
                "active": compatible and dialog_ready,
                "blend_alpha": self.dialog_blend_alpha,
                "max_turns": self.dialog_max_turns,
                "metrics": dialog_head.get("metrics", {}),
            },
            "finetuned_model": {
                "enabled": self.finetuned_enabled,
                "active": compatible and finetuned_ready,
                "blend_alpha": self.finetuned_blend_alpha,
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

            filtered_samples: List[Dict[str, Any]] = []
            for sample in samples:
                intent_id = str(sample.get("intent_id") or "").strip()
                idx = label_to_idx.get(intent_id)
                if idx is None:
                    continue
                item = dict(sample)
                item["label_idx"] = int(idx)
                filtered_samples.append(item)

            if len(filtered_samples) < max(30, len(intent_ids) * 3):
                raise RuntimeError(
                    f"insufficient labeled data for training: {len(filtered_samples)} samples for {len(intent_ids)} intents"
                )

            texts = [str(row.get("text") or "") for row in filtered_samples]
            labels = [int(row.get("label_idx")) for row in filtered_samples]

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

            dialog_artifact: Dict[str, Any] = {"enabled": False}
            dialog_report: Dict[str, Any] = {"enabled": False}
            if self.dialog_head_enabled:
                try:
                    dialog_report, dialog_artifact = self._train_dialog_transformer(
                        samples=filtered_samples,
                        intent_ids=intent_ids,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        val_ratio=val_ratio,
                        random_seed=random_seed,
                    )
                except Exception as dialog_exc:
                    logger.warning("Dialog transformer training failed, keeping linear head only: %s", dialog_exc)
                    dialog_artifact = {
                        "enabled": False,
                        "error": str(dialog_exc),
                    }
                    dialog_report = {
                        "enabled": False,
                        "error": str(dialog_exc),
                    }

            finetuned_artifact: Dict[str, Any] = {"enabled": False}
            finetuned_report: Dict[str, Any] = {"enabled": False}
            if self.finetuned_enabled:
                try:
                    finetuned_report, finetuned_artifact = self._train_finetuned_rubert(
                        texts=texts,
                        labels=labels,
                        intent_ids=intent_ids,
                        train_idx=train_idx,
                        val_idx=val_idx,
                        random_seed=random_seed,
                    )
                except Exception as finetuned_exc:
                    logger.warning("RuBERT fine-tuning failed, keeping linear/dialog heads only: %s", finetuned_exc)
                    finetuned_artifact = {
                        "enabled": False,
                        "error": str(finetuned_exc),
                    }
                    finetuned_report = {
                        "enabled": False,
                        "error": str(finetuned_exc),
                    }

            trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            version_id = f"tuned-{int(time.time())}"
            artifact = {
                "artifact_version": 2,
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
                "dialog_head": dialog_artifact,
                "finetuned_model": finetuned_artifact,
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
                "dialog_head": dialog_report,
                "finetuned_model": finetuned_report,
            }
            self._set_last_train_report(report)
            return report

        except Exception as exc:
            self._set_last_train_error(str(exc))
            raise

    def reload_tuned_head_from_disk(self) -> Dict[str, Any]:
        self._load_tuned_artifact_from_disk()
        return self.get_training_status()

    def _clear_tuned_artifact(self) -> None:
        with self._state_lock:
            self._tuned_artifact = None
            self._tuned_head = None
            self._finetuned_model = None
            self._finetuned_tokenizer = None
            self._dialog_head = None
            self._active_head_intents = None
            self._active_finetuned_intents = None
            self._active_finetuned_path = ""
            self._active_dialog_intents = None

    def _load_tuned_artifact_from_disk(self) -> None:
        if not self.tuned_model_path:
            self._clear_tuned_artifact()
            return
        model_path = Path(self.tuned_model_path)
        if not model_path.exists():
            self._clear_tuned_artifact()
            logger.info("No tuned routing head found at %s. Active tuned state cleared.", model_path)
            return

        try:
            payload = torch.load(model_path, map_location="cpu")
            if not isinstance(payload, dict):
                raise RuntimeError("tuned model payload is not a dict")
            self._activate_tuned_artifact(payload)
            logger.info("Loaded tuned routing head: %s", model_path)
        except Exception as exc:
            self._clear_tuned_artifact()
            logger.warning("Failed to load tuned routing head from %s: %s", model_path, exc)

    def _activate_tuned_artifact(self, artifact: Dict[str, Any]) -> None:
        with self._state_lock:
            self._tuned_artifact = dict(artifact)
            self._tuned_head = None
            self._finetuned_model = None
            self._finetuned_tokenizer = None
            self._dialog_head = None
            self._active_head_intents = None
            self._active_finetuned_intents = None
            self._active_finetuned_path = ""
            self._active_dialog_intents = None

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

    def _predict_with_dialog_head(
        self,
        call: CallInput,
        intent_ids: List[str],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        if not self.dialog_head_enabled:
            return None, {"active": False, "reason": "dialog_head_disabled"}

        with self._state_lock:
            artifact = dict(self._tuned_artifact or {})
        if not artifact:
            return None, {"active": False, "reason": "no_tuned_model"}

        dialog_head_artifact = artifact.get("dialog_head")
        if not isinstance(dialog_head_artifact, dict) or not dialog_head_artifact.get("enabled"):
            return None, {"active": False, "reason": "no_dialog_head"}

        artifact_intents = list(artifact.get("intent_ids") or [])
        if artifact_intents != intent_ids:
            return None, {
                "active": False,
                "reason": "intents_mismatch",
                "artifact_intents_n": len(artifact_intents),
                "runtime_intents_n": len(intent_ids),
            }

        turns = self._segments_to_turns(call.segments)
        if not turns:
            return None, {"active": False, "reason": "no_turns"}

        try:
            model = self._ensure_dialog_head_loaded(artifact, intent_ids)
            turn_texts = [turn.text for turn in turns]
            role_ids = [turn.role_id for turn in turns]
            turn_embeddings = self._embed(turn_texts).detach().cpu()

            seq_x, seq_roles, seq_mask = self._build_dialog_batch(
                seq_embeddings=[turn_embeddings],
                seq_role_ids=[role_ids],
                indices=[0],
            )
            with torch.inference_mode():
                logits = model(
                    seq_x.to(self.device),
                    seq_roles.to(self.device),
                    seq_mask.to(self.device),
                )
                probs = torch.softmax(logits, dim=1).squeeze(0)
            return probs, {
                "active": True,
                "version_id": artifact.get("version_id", ""),
                "trained_at": artifact.get("trained_at", ""),
                "turns_used": len(turns),
            }
        except Exception as exc:
            logger.warning("Failed to run dialog routing head: %s", exc)
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

    def _ensure_dialog_head_loaded(self, artifact: Dict[str, Any], intent_ids: List[str]) -> DialogTransformerHead:
        intent_key = tuple(intent_ids)
        with self._state_lock:
            if self._dialog_head is not None and self._active_dialog_intents == intent_key:
                return self._dialog_head

            dialog_artifact = artifact.get("dialog_head")
            if not isinstance(dialog_artifact, dict) or not dialog_artifact.get("enabled"):
                raise RuntimeError("dialog head artifact is not available")
            state_dict = dialog_artifact.get("state_dict")
            if not isinstance(state_dict, dict):
                raise RuntimeError("invalid dialog state_dict")

            in_features = int(dialog_artifact.get("in_features") or 0)
            if in_features <= 0:
                raise RuntimeError("invalid dialog in_features")
            d_model = int(dialog_artifact.get("d_model") or self.dialog_d_model)
            nhead = int(dialog_artifact.get("nhead") or self.dialog_nhead)
            num_layers = int(dialog_artifact.get("num_layers") or self.dialog_layers)
            dropout = float(dialog_artifact.get("dropout") or self.dialog_dropout)
            max_turns = int(dialog_artifact.get("max_turns") or self.dialog_max_turns)

            model = DialogTransformerHead(
                in_features=in_features,
                num_classes=len(intent_ids),
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                max_turns=max_turns,
            ).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()

            self._dialog_head = model
            self._active_dialog_intents = intent_key
            return model

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
                rows.append(
                    {
                        "text": text,
                        "intent_id": intent_id,
                        "source": "intent_examples",
                        "turns": [DialogTurn(role_id=ROLE_UNKNOWN, text=text)],
                    }
                )
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
                transcript_text = str(item.get("transcript_text") or "").strip()
                if not text:
                    text = transcript_text

                text = self._prepare_training_text(text)
                if not text:
                    continue

                turns = self._feedback_item_to_turns(item, fallback_text=transcript_text or text)
                if not turns:
                    turns = [DialogTurn(role_id=ROLE_UNKNOWN, text=text)]

                key = (intent_id, text.lower())
                if key in seen:
                    continue
                seen.add(key)

                rows.append(
                    {
                        "text": text,
                        "intent_id": intent_id,
                        "source": "operator_feedback",
                        "turns": turns,
                    }
                )
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

    def _prepare_turn_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return ""
        if len(cleaned) > self.dialog_max_turn_chars:
            cleaned = cleaned[: self.dialog_max_turn_chars].strip()
        return cleaned

    def _role_to_id(self, role: Optional[str], speaker: Optional[str] = None) -> int:
        role_l = str(role or "").strip().lower()
        speaker_l = str(speaker or "").strip().lower()
        if not role_l and speaker_l.startswith("system"):
            return ROLE_SYSTEM
        if any(tag in role_l for tag in ("звонящ", "caller", "client", "customer")):
            return ROLE_CALLER
        if any(tag in role_l for tag in ("ответ", "agent", "operator", "support", "менедж")):
            return ROLE_AGENT
        if any(tag in role_l for tag in ("ivr", "bot", "system")):
            return ROLE_SYSTEM
        return ROLE_UNKNOWN

    def _crop_turns(self, turns: List[DialogTurn]) -> List[DialogTurn]:
        if len(turns) <= self.dialog_max_turns:
            return turns
        keep_head = int(self.dialog_max_turns * 0.65)
        keep_tail = self.dialog_max_turns - keep_head
        return turns[:keep_head] + turns[-keep_tail:]

    def _segments_to_turns(self, segments: List[Any]) -> List[DialogTurn]:
        turns: List[DialogTurn] = []
        for seg in segments:
            text = self._prepare_turn_text(getattr(seg, "text", ""))
            if not text:
                continue
            turns.append(
                DialogTurn(
                    role_id=self._role_to_id(getattr(seg, "role", None), getattr(seg, "speaker", None)),
                    text=text,
                )
            )
        return self._crop_turns(turns)

    def _feedback_item_to_turns(self, item: Dict[str, Any], fallback_text: str = "") -> List[DialogTurn]:
        turns: List[DialogTurn] = []
        raw_segments = item.get("transcript_segments")
        if isinstance(raw_segments, list):
            def _safe_float(value: Any) -> float:
                try:
                    return float(value or 0.0)
                except Exception:
                    return 0.0

            ordered = sorted(
                [seg for seg in raw_segments if isinstance(seg, dict)],
                key=lambda seg: _safe_float(seg.get("start", 0.0)),
            )
            for seg in ordered:
                text = self._prepare_turn_text(seg.get("text", ""))
                if not text:
                    continue
                turns.append(
                    DialogTurn(
                        role_id=self._role_to_id(seg.get("role"), seg.get("speaker")),
                        text=text,
                    )
                )

        if turns:
            return self._crop_turns(turns)

        fallback_clean = self._prepare_turn_text(fallback_text)
        if fallback_clean:
            return [DialogTurn(role_id=ROLE_UNKNOWN, text=fallback_clean)]
        return []

    def _build_dialog_batch(
        self,
        *,
        seq_embeddings: List[torch.Tensor],
        seq_role_ids: List[List[int]],
        indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not indices:
            raise RuntimeError("empty dialog batch indices")
        in_features = int(seq_embeddings[indices[0]].shape[1])
        max_len = min(
            self.dialog_max_turns,
            max(int(seq_embeddings[idx].shape[0]) for idx in indices),
        )
        batch = len(indices)

        x = torch.zeros((batch, max_len, in_features), dtype=torch.float32)
        roles = torch.zeros((batch, max_len), dtype=torch.long)
        mask = torch.ones((batch, max_len), dtype=torch.bool)

        for b, idx in enumerate(indices):
            seq = seq_embeddings[idx]
            rids = seq_role_ids[idx]
            take = min(max_len, int(seq.shape[0]), len(rids))
            if take <= 0:
                continue
            x[b, :take, :] = seq[:take, :]
            roles[b, :take] = torch.tensor(rids[:take], dtype=torch.long)
            mask[b, :take] = False
        return x, roles, mask

    def _evaluate_dialog(
        self,
        *,
        model: DialogTransformerHead,
        seq_embeddings: List[torch.Tensor],
        seq_role_ids: List[List[int]],
        labels: torch.Tensor,
        indices: List[int],
        batch_size: int,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        if not indices:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
            }

        model.eval()
        losses: List[float] = []
        preds: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        with torch.inference_mode():
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i : i + batch_size]
                xb, rb, mb = self._build_dialog_batch(
                    seq_embeddings=seq_embeddings,
                    seq_role_ids=seq_role_ids,
                    indices=batch_idx,
                )
                yb = labels[batch_idx].to(self.device)
                logits = model(xb.to(self.device), rb.to(self.device), mb.to(self.device))
                loss = criterion(logits, yb)
                losses.append(float(loss.item()))
                preds.append(torch.argmax(logits, dim=1).detach().cpu())
                targets.append(labels[batch_idx].detach().cpu())

        pred = torch.cat(preds, dim=0)
        target = torch.cat(targets, dim=0)
        acc = float((pred == target).float().mean().item()) if target.numel() > 0 else 0.0
        macro_precision, macro_recall, macro_f1 = self._macro_precision_recall_f1(pred, target)
        return {
            "loss": round(sum(losses) / max(1, len(losses)), 6),
            "accuracy": round(acc, 6),
            "macro_precision": round(macro_precision, 6),
            "macro_recall": round(macro_recall, 6),
            "macro_f1": round(macro_f1, 6),
        }

    def _train_dialog_transformer(
        self,
        *,
        samples: List[Dict[str, Any]],
        intent_ids: List[str],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        val_ratio: float,
        random_seed: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        seq_embeddings: List[torch.Tensor] = []
        seq_role_ids: List[List[int]] = []
        labels: List[int] = []
        turns_count: List[int] = []

        for sample in samples:
            turns = sample.get("turns") or []
            if not isinstance(turns, list):
                turns = []
            dialog_turns: List[DialogTurn] = []
            for turn in turns:
                if isinstance(turn, DialogTurn):
                    dialog_turns.append(turn)
                elif isinstance(turn, dict):
                    text = self._prepare_turn_text(turn.get("text", ""))
                    if not text:
                        continue
                    dialog_turns.append(
                        DialogTurn(
                            role_id=int(turn.get("role_id") or ROLE_UNKNOWN),
                            text=text,
                        )
                    )

            if not dialog_turns:
                fallback_text = self._prepare_turn_text(sample.get("text", ""))
                if not fallback_text:
                    continue
                dialog_turns = [DialogTurn(role_id=ROLE_UNKNOWN, text=fallback_text)]

            dialog_turns = self._crop_turns(dialog_turns)
            turn_texts = [turn.text for turn in dialog_turns]
            role_ids = [int(max(0, min(ROLE_VOCAB_SIZE - 1, turn.role_id))) for turn in dialog_turns]
            turn_emb = self._embed(turn_texts).detach().cpu()
            seq_embeddings.append(turn_emb)
            seq_role_ids.append(role_ids)
            labels.append(int(sample.get("label_idx")))
            turns_count.append(len(dialog_turns))

        if len(seq_embeddings) < max(24, len(intent_ids) * 3):
            raise RuntimeError(
                f"insufficient dialog samples for transformer head: {len(seq_embeddings)}"
            )

        label_tensor = torch.tensor(labels, dtype=torch.long)
        train_idx, val_idx = self._stratified_split(labels, val_ratio=val_ratio, random_seed=random_seed)
        if not train_idx:
            raise RuntimeError("stratified split produced empty train set for dialog head")
        if not val_idx:
            val_idx = list(train_idx)

        in_features = int(seq_embeddings[0].shape[1])
        model = DialogTransformerHead(
            in_features=in_features,
            num_classes=len(intent_ids),
            d_model=self.dialog_d_model,
            nhead=self.dialog_nhead,
            num_layers=self.dialog_layers,
            dropout=self.dialog_dropout,
            max_turns=self.dialog_max_turns,
        ).to(self.device)

        class_weights = self._build_class_weights(label_tensor[train_idx], len(intent_ids)).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.01)

        epochs = int(max(10, min(400, epochs)))
        batch_size = int(max(8, min(128, batch_size)))
        best_state = None
        best_val_f1 = -1.0
        best_epoch = 0
        patience = 20
        no_improve = 0

        torch.manual_seed(int(random_seed))
        random.seed(int(random_seed))

        for epoch in range(1, epochs + 1):
            model.train()
            shuffled = list(train_idx)
            random.shuffle(shuffled)
            for i in range(0, len(shuffled), batch_size):
                batch_idx = shuffled[i : i + batch_size]
                xb, rb, mb = self._build_dialog_batch(
                    seq_embeddings=seq_embeddings,
                    seq_role_ids=seq_role_ids,
                    indices=batch_idx,
                )
                yb = label_tensor[batch_idx].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb.to(self.device), rb.to(self.device), mb.to(self.device))
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            val_metrics = self._evaluate_dialog(
                model=model,
                seq_embeddings=seq_embeddings,
                seq_role_ids=seq_role_ids,
                labels=label_tensor,
                indices=val_idx,
                batch_size=batch_size,
                criterion=criterion,
            )
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
            raise RuntimeError("dialog transformer training failed: no best checkpoint")

        model.load_state_dict(best_state)
        train_metrics = self._evaluate_dialog(
            model=model,
            seq_embeddings=seq_embeddings,
            seq_role_ids=seq_role_ids,
            labels=label_tensor,
            indices=train_idx,
            batch_size=batch_size,
            criterion=criterion,
        )
        val_metrics = self._evaluate_dialog(
            model=model,
            seq_embeddings=seq_embeddings,
            seq_role_ids=seq_role_ids,
            labels=label_tensor,
            indices=val_idx,
            batch_size=batch_size,
            criterion=criterion,
        )

        report = {
            "enabled": True,
            "best_epoch": best_epoch,
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "epochs_requested": epochs,
            },
            "dataset": {
                "samples_total": len(seq_embeddings),
                "samples_train": len(train_idx),
                "samples_val": len(val_idx),
                "turns_avg": round(sum(turns_count) / max(1, len(turns_count)), 2),
                "turns_max": max(turns_count) if turns_count else 0,
            },
        }
        artifact = {
            "enabled": True,
            "in_features": in_features,
            "d_model": self.dialog_d_model,
            "nhead": self.dialog_nhead,
            "num_layers": self.dialog_layers,
            "dropout": self.dialog_dropout,
            "max_turns": self.dialog_max_turns,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "metrics": report["metrics"],
            "dataset": report["dataset"],
        }
        return report, artifact

    def _train_finetuned_rubert(
        self,
        *,
        texts: List[str],
        labels: List[int],
        intent_ids: List[str],
        train_idx: List[int],
        val_idx: List[int],
        random_seed: int,
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

        batch_size = int(max(4, min(64, self.finetuned_batch_size)))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(intent_ids),
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.finetuned_learning_rate),
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

        epochs = int(max(1, min(12, self.finetuned_epochs)))
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
            "learning_rate": self.finetuned_learning_rate,
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
        macro_precision, macro_recall, macro_f1 = self._macro_precision_recall_f1(pred, y)

        return {
            "loss": round(sum(losses) / max(1, len(losses)), 6),
            "accuracy": round(acc, 6),
            "macro_precision": round(macro_precision, 6),
            "macro_recall": round(macro_recall, 6),
            "macro_f1": round(macro_f1, 6),
        }

    def _macro_f1(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return self._macro_precision_recall_f1(pred, target)[2]

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

    def _predict_with_chunked_context(
        self,
        call: CallInput,
        intent_ids: List[str],
        intent_emb: torch.Tensor,
        allowed_intents: Dict[str, Dict],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        if not self.chunk_inference_enabled:
            return None, {"enabled": False, "reason": "disabled"}

        chunks = self._split_call_into_chunks(call)
        if len(chunks) <= 1:
            return None, {"enabled": True, "chunks": len(chunks), "reason": "single_chunk"}

        weighted_probs: List[torch.Tensor] = []
        weights: List[float] = []
        top_intents: List[str] = []
        top_scores: List[float] = []

        idx_map = {intent_id: i for i, intent_id in enumerate(intent_ids)}
        for idx, chunk_text in enumerate(chunks):
            q = self._embed([chunk_text])
            sims = (q @ intent_emb.T).squeeze(0)

            chunk_prep = build_canonical([(float(idx), chunk_text, None)], self.preprocess_cfg)
            chunk_boosts = self._calculate_rule_boosts(chunk_prep, allowed_intents)
            if chunk_boosts:
                for intent_id, delta in chunk_boosts.items():
                    j = idx_map.get(intent_id)
                    if j is not None:
                        sims[j] = sims[j] + float(delta)

            probs = torch.softmax(sims / 0.15, dim=0)
            pos = (idx / max(1, len(chunks) - 1))
            weight = 1.0 + self.chunk_late_weight * pos
            weighted_probs.append(probs * weight)
            weights.append(weight)

            best_idx = int(torch.argmax(probs).item())
            top_intents.append(intent_ids[best_idx])
            top_scores.append(float(probs[best_idx].item()))

        if not weighted_probs or sum(weights) <= 0.0:
            return None, {"enabled": True, "chunks": len(chunks), "reason": "empty_probs"}

        agg_probs = torch.stack(weighted_probs, dim=0).sum(dim=0) / float(sum(weights))
        return agg_probs, {
            "enabled": True,
            "chunks": len(chunks),
            "chunk_max_chars": self.chunk_max_chars,
            "overlap_turns": self.chunk_overlap_turns,
            "blend_alpha": self.chunk_blend_alpha,
            "late_weight": self.chunk_late_weight,
            "top_intents": top_intents,
            "top_scores": [round(x, 4) for x in top_scores],
        }

    def _split_call_into_chunks(self, call: CallInput) -> List[str]:
        entries: List[str] = []

        def _safe_start(seg: Any) -> float:
            try:
                return float(getattr(seg, "start", 0.0) or 0.0)
            except Exception:
                return 0.0

        ordered_segments = sorted(list(call.segments or []), key=_safe_start)
        for seg in ordered_segments:
            raw_text = re.sub(r"\s+", " ", str(getattr(seg, "text", "") or "")).strip()
            if not raw_text:
                continue
            role = str(getattr(seg, "role", "") or "").strip()
            speaker = str(getattr(seg, "speaker", "") or "").strip()
            prefix = role or speaker
            line = f"{prefix}: {raw_text}" if prefix else raw_text
            if len(line) > self.chunk_max_chars:
                line = line[: self.chunk_max_chars].strip()
            if line:
                entries.append(line)

        if not entries:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for line in entries:
            add_len = len(line) + (1 if current else 0)
            if current and current_len + add_len > self.chunk_max_chars:
                chunks.append("\n".join(current))
                overlap = current[-self.chunk_overlap_turns :] if self.chunk_overlap_turns > 0 else []
                current = list(overlap)
                current_len = len("\n".join(current)) if current else 0
                add_len = len(line) + (1 if current else 0)
                if current and current_len + add_len > self.chunk_max_chars:
                    current = []
                    current_len = 0
            if current:
                current_len += 1
            current.append(line)
            current_len += len(line)

        if current:
            chunks.append("\n".join(current))

        if len(chunks) <= self.chunk_max_count:
            return chunks
        if self.chunk_max_count == 1:
            return [chunks[-1]]

        return chunks[: self.chunk_max_count - 1] + [chunks[-1]]

    def _normalize_priority(self, priority: str) -> str:
        value = str(priority or "").strip().lower()
        if value == "normal":
            return "medium"
        if value not in {"low", "medium", "high", "critical"}:
            return "medium"
        return value

    def _elevate_priority(self, priority: str, floor: str) -> Priority:
        rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        current = self._normalize_priority(priority)
        min_floor = self._normalize_priority(floor)
        elevated = current if rank[current] >= rank[min_floor] else min_floor
        return elevated  # type: ignore[return-value]

    def _detect_urgency_signals(self, prep) -> Dict[str, Any]:
        if not self.urgency_escalation_enabled:
            return {"enabled": False, "triggered": False, "matched": []}

        text = str(prep.canonical_text or "").strip().lower()
        if not text:
            return {"enabled": True, "triggered": False, "matched": []}

        matched: List[str] = []
        for pattern in self.urgency_patterns:
            p = str(pattern or "").strip().lower()
            if not p:
                continue
            if " " in p:
                if p in text:
                    matched.append(p)
                continue
            if re.search(rf"\b{re.escape(p)}\w*", text):
                matched.append(p)

        seen = set()
        uniq = []
        for item in matched:
            if item in seen:
                continue
            seen.add(item)
            uniq.append(item)

        return {
            "enabled": True,
            "triggered": bool(uniq),
            "matched": uniq[:10],
            "priority_floor": self.urgency_priority_floor,
        }

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
