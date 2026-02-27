#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


EMPTY_MARK = "__empty__"


def _norm(value: object) -> str:
    raw = str(value or "").strip()
    if raw.lower() in {"", "none", "null", "nan", "-"}:
        return ""
    return raw


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        headers = list(reader.fieldnames or [])
    return rows, headers


def _pick_column(headers: Sequence[str], preferred: str, fallbacks: Iterable[str]) -> str:
    if preferred and preferred in headers:
        return preferred
    for name in fallbacks:
        if name in headers:
            return name
    return ""


def _task_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    n = len(y_true)
    if n == 0:
        return {
            "samples": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "labels": {},
            "confusion": {},
        }

    labels = sorted(set(y_true) | set(y_pred))
    cm: Dict[str, Counter] = {label: Counter() for label in labels}
    for yt, yp in zip(y_true, y_pred):
        cm[yt][yp] += 1

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)

    per_label: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    weighted_f1_sum = 0.0
    total_support = 0

    for label in labels:
        tp = float(cm[label][label])
        fp = float(sum(cm[t][label] for t in labels if t != label))
        fn = float(sum(cm[label][p] for p in labels if p != label))
        support = int(sum(cm[label].values()))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_label[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weighted_f1_sum += f1 * support
        total_support += support

    confusion = {
        true_label: dict(sorted(preds.items(), key=lambda kv: kv[1], reverse=True))
        for true_label, preds in cm.items()
        if sum(preds.values()) > 0
    }

    return {
        "samples": n,
        "accuracy": round(correct / n, 6),
        "macro_precision": round(sum(precisions) / len(precisions), 6),
        "macro_recall": round(sum(recalls) / len(recalls), 6),
        "macro_f1": round(sum(f1s) / len(f1s), 6),
        "weighted_f1": round(weighted_f1_sum / max(1, total_support), 6),
        "labels": per_label,
        "confusion": confusion,
    }


def _build_pairs(rows: Sequence[Dict[str, str]], true_col: str, pred_col: str) -> Tuple[List[str], List[str], int]:
    true_vals: List[str] = []
    pred_vals: List[str] = []
    skipped = 0
    for row in rows:
        yt = _norm(row.get(true_col, ""))
        if not yt:
            skipped += 1
            continue
        yp = _norm(row.get(pred_col, ""))
        if not yp:
            yp = EMPTY_MARK
        true_vals.append(yt)
        pred_vals.append(yp)
    return true_vals, pred_vals, skipped


def _print_task(task_name: str, metrics: Dict[str, object], skipped: int, true_col: str, pred_col: str) -> None:
    print(f"\n== {task_name} ==")
    print(f"columns: true='{true_col}', pred='{pred_col}'")
    print(f"samples={metrics['samples']}, skipped_unlabeled={skipped}")
    print(
        "accuracy={acc:.4f}, macro_p={p:.4f}, macro_r={r:.4f}, macro_f1={f1:.4f}, weighted_f1={wf1:.4f}".format(
            acc=float(metrics["accuracy"]),
            p=float(metrics["macro_precision"]),
            r=float(metrics["macro_recall"]),
            f1=float(metrics["macro_f1"]),
            wf1=float(metrics["weighted_f1"]),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate routing quality from labeled CSV.")
    parser.add_argument("--csv", required=True, help="Path to labeled CSV.")
    parser.add_argument("--out-json", default="", help="Optional path to save JSON report.")

    parser.add_argument("--intent-true-col", default="final_intent_id")
    parser.add_argument("--intent-pred-col", default="ai_intent_id")
    parser.add_argument("--group-true-col", default="final_group_id")
    parser.add_argument("--group-pred-col", default="ai_group_id")
    parser.add_argument("--priority-true-col", default="final_priority")
    parser.add_argument("--priority-pred-col", default="ai_priority")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"[ERROR] csv not found: {csv_path}")
        return 2

    rows, headers = _read_csv(csv_path)
    if not rows:
        print(f"[ERROR] csv is empty: {csv_path}")
        return 2

    intent_true_col = _pick_column(headers, args.intent_true_col, ["final_intent_id", "intent_id"])
    intent_pred_col = _pick_column(headers, args.intent_pred_col, ["ai_intent_id", "pred_intent_id", "intent_id"])
    group_true_col = _pick_column(headers, args.group_true_col, ["final_group_id", "final_group", "group_id"])
    group_pred_col = _pick_column(headers, args.group_pred_col, ["ai_group_id", "pred_group_id", "suggested_group"])
    priority_true_col = _pick_column(headers, args.priority_true_col, ["final_priority", "priority"])
    priority_pred_col = _pick_column(headers, args.priority_pred_col, ["ai_priority", "pred_priority", "priority"])

    missing = [
        name
        for name, val in {
            "intent_true_col": intent_true_col,
            "intent_pred_col": intent_pred_col,
            "group_true_col": group_true_col,
            "group_pred_col": group_pred_col,
            "priority_true_col": priority_true_col,
            "priority_pred_col": priority_pred_col,
        }.items()
        if not val
    ]
    if missing:
        print("[ERROR] missing required columns:", ", ".join(missing))
        print("headers:", ", ".join(headers))
        return 2

    it_y, ip_y, i_skip = _build_pairs(rows, intent_true_col, intent_pred_col)
    gt_y, gp_y, g_skip = _build_pairs(rows, group_true_col, group_pred_col)
    pt_y, pp_y, p_skip = _build_pairs(rows, priority_true_col, priority_pred_col)

    intent_metrics = _task_metrics(it_y, ip_y)
    group_metrics = _task_metrics(gt_y, gp_y)
    priority_metrics = _task_metrics(pt_y, pp_y)

    _print_task("Intent", intent_metrics, i_skip, intent_true_col, intent_pred_col)
    _print_task("Group", group_metrics, g_skip, group_true_col, group_pred_col)
    _print_task("Priority", priority_metrics, p_skip, priority_true_col, priority_pred_col)

    report = {
        "csv": str(csv_path),
        "rows_total": len(rows),
        "intent": {"true_col": intent_true_col, "pred_col": intent_pred_col, "skipped_unlabeled": i_skip, **intent_metrics},
        "group": {"true_col": group_true_col, "pred_col": group_pred_col, "skipped_unlabeled": g_skip, **group_metrics},
        "priority": {"true_col": priority_true_col, "pred_col": priority_pred_col, "skipped_unlabeled": p_skip, **priority_metrics},
    }

    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else (csv_path.parent / "routing_metrics.json")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] report saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

