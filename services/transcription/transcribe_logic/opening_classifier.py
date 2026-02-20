from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Dict, Iterable, List, Sequence, Tuple


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> List[str]:
    norm = _normalize_text(text)
    if not norm:
        return []
    tokens = re.findall(r"[a-zа-я0-9]+", norm, flags=re.IGNORECASE)
    return [t.lower() for t in tokens if t]


@dataclass
class OpeningSentenceClassifier:
    alpha: float = 1.0
    min_tokens: int = 1

    # persisted state
    trained: bool = False
    log_prior_pos: float = 0.0
    log_prior_neg: float = 0.0
    unk_log_pos: float = 0.0
    unk_log_neg: float = 0.0
    token_log_pos: Dict[str, float] | None = None
    token_log_neg: Dict[str, float] | None = None

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if not texts:
            raise ValueError("empty training set")

        pos_docs = 0
        neg_docs = 0
        pos_counts: Dict[str, int] = {}
        neg_counts: Dict[str, int] = {}
        pos_total = 0
        neg_total = 0

        for text, raw_label in zip(texts, labels):
            label = 1 if int(raw_label) == 1 else 0
            tokens = _tokenize(text)
            if len(tokens) < self.min_tokens:
                continue
            if label == 1:
                pos_docs += 1
                for token in tokens:
                    pos_counts[token] = pos_counts.get(token, 0) + 1
                    pos_total += 1
            else:
                neg_docs += 1
                for token in tokens:
                    neg_counts[token] = neg_counts.get(token, 0) + 1
                    neg_total += 1

        if pos_docs == 0 or neg_docs == 0:
            raise ValueError("both positive and negative samples are required")

        total_docs = pos_docs + neg_docs
        self.log_prior_pos = math.log(pos_docs / total_docs)
        self.log_prior_neg = math.log(neg_docs / total_docs)

        vocab = set(pos_counts.keys()) | set(neg_counts.keys())
        vocab_size = max(1, len(vocab))
        denom_pos = pos_total + self.alpha * vocab_size
        denom_neg = neg_total + self.alpha * vocab_size

        self.token_log_pos = {}
        self.token_log_neg = {}
        for token in vocab:
            c_pos = pos_counts.get(token, 0)
            c_neg = neg_counts.get(token, 0)
            self.token_log_pos[token] = math.log((c_pos + self.alpha) / denom_pos)
            self.token_log_neg[token] = math.log((c_neg + self.alpha) / denom_neg)

        self.unk_log_pos = math.log(self.alpha / denom_pos)
        self.unk_log_neg = math.log(self.alpha / denom_neg)
        self.trained = True

    def predict_proba(self, text: str) -> float:
        if not self.trained or self.token_log_pos is None or self.token_log_neg is None:
            return 0.0
        tokens = _tokenize(text)
        if len(tokens) < self.min_tokens:
            return 0.0

        logp_pos = self.log_prior_pos
        logp_neg = self.log_prior_neg
        for token in tokens:
            logp_pos += self.token_log_pos.get(token, self.unk_log_pos)
            logp_neg += self.token_log_neg.get(token, self.unk_log_neg)

        # stable sigmoid(logp_pos - logp_neg)
        diff = max(-60.0, min(60.0, logp_pos - logp_neg))
        return 1.0 / (1.0 + math.exp(-diff))

    def predict(self, text: str, threshold: float = 0.5) -> int:
        return 1 if self.predict_proba(text) >= threshold else 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "alpha": self.alpha,
            "min_tokens": self.min_tokens,
            "trained": self.trained,
            "log_prior_pos": self.log_prior_pos,
            "log_prior_neg": self.log_prior_neg,
            "unk_log_pos": self.unk_log_pos,
            "unk_log_neg": self.unk_log_neg,
            "token_log_pos": self.token_log_pos or {},
            "token_log_neg": self.token_log_neg or {},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "OpeningSentenceClassifier":
        model = cls(
            alpha=float(payload.get("alpha", 1.0)),
            min_tokens=int(payload.get("min_tokens", 1)),
        )
        model.trained = bool(payload.get("trained", False))
        model.log_prior_pos = float(payload.get("log_prior_pos", 0.0))
        model.log_prior_neg = float(payload.get("log_prior_neg", 0.0))
        model.unk_log_pos = float(payload.get("unk_log_pos", 0.0))
        model.unk_log_neg = float(payload.get("unk_log_neg", 0.0))
        model.token_log_pos = {
            str(k): float(v) for k, v in dict(payload.get("token_log_pos", {})).items()
        }
        model.token_log_neg = {
            str(k): float(v) for k, v in dict(payload.get("token_log_neg", {})).items()
        }
        return model

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "OpeningSentenceClassifier":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)


def evaluate_binary(labels: Sequence[int], probs: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    if len(labels) != len(probs):
        raise ValueError("labels and probs length mismatch")
    if not labels:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = fp = tn = fn = 0
    for y, p in zip(labels, probs):
        pred = 1 if p >= threshold else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 0 and pred == 0:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_test_split(
    rows: Sequence[Tuple[str, int]],
    *,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    import random

    data = list(rows)
    rnd = random.Random(seed)
    rnd.shuffle(data)
    split = int(len(data) * (1.0 - test_ratio))
    split = max(1, min(split, len(data) - 1))
    return data[:split], data[split:]
