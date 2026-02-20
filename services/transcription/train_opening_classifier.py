from __future__ import annotations

import argparse
import csv
import os
from typing import List, Tuple

from transcribe_logic.opening_classifier import (
    OpeningSentenceClassifier,
    evaluate_binary,
    train_test_split,
)


TRUE_LABELS = {"1", "true", "yes", "opening", "open", "agent", "positive", "pos"}
FALSE_LABELS = {"0", "false", "no", "non-opening", "other", "customer", "negative", "neg"}


def _parse_label(raw: str) -> int:
    value = (raw or "").strip().lower()
    if value in TRUE_LABELS:
        return 1
    if value in FALSE_LABELS:
        return 0
    raise ValueError(f"unsupported label value: {raw!r}")


def _load_rows(path: str, text_col: str, label_col: str) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("csv has no header")
        if text_col not in reader.fieldnames:
            raise ValueError(f"text column '{text_col}' not found in csv")
        if label_col not in reader.fieldnames:
            raise ValueError(f"label column '{label_col}' not found in csv")

        for i, row in enumerate(reader, start=2):
            text = (row.get(text_col) or "").strip()
            if not text:
                continue
            raw_label = row.get(label_col)
            try:
                label = _parse_label("" if raw_label is None else str(raw_label))
            except ValueError as e:
                raise ValueError(f"line {i}: {e}") from e
            rows.append((text, label))
    if not rows:
        raise ValueError("no valid training rows found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train opening sentence classifier from CSV.")
    parser.add_argument("--csv", required=True, help="Path to CSV with opening sentence labels.")
    parser.add_argument("--out", required=True, help="Output path for model json.")
    parser.add_argument("--text-col", default="text", help="Name of text column in CSV.")
    parser.add_argument("--label-col", default="label", help="Name of label column in CSV.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing.")
    parser.add_argument("--min-tokens", type=int, default=1, help="Min tokens for sample usage.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Validation ratio (0..0.9).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for reporting binary metrics.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Train on full dataset without holdout metrics.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.csv, args.text_col, args.label_col)
    test_ratio = max(0.0, min(0.9, args.test_ratio))

    classifier = OpeningSentenceClassifier(alpha=args.alpha, min_tokens=max(1, args.min_tokens))

    if args.no_eval or len(rows) < 10 or test_ratio <= 0.0:
        texts = [t for t, _ in rows]
        labels = [y for _, y in rows]
        classifier.fit(texts, labels)
        metrics = None
    else:
        train_rows, test_rows = train_test_split(rows, test_ratio=test_ratio, seed=args.seed)
        train_x = [t for t, _ in train_rows]
        train_y = [y for _, y in train_rows]
        test_x = [t for t, _ in test_rows]
        test_y = [y for _, y in test_rows]

        classifier.fit(train_x, train_y)
        probs = [classifier.predict_proba(t) for t in test_x]
        metrics = evaluate_binary(test_y, probs, threshold=args.threshold)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    classifier.save(args.out)

    print(f"model saved: {args.out}")
    print(f"train_size: {len(rows)}")
    if metrics:
        print(
            "eval: "
            f"accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )
    else:
        print("eval: skipped")


if __name__ == "__main__":
    main()
