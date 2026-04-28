#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path


ALLOWED_AUDIO = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}
EMPTY = "__empty__"


def detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        return dialect.delimiter
    except Exception:
        if sample.count(";") >= sample.count(","):
            return ";"
        return ","


def read_rows(path: Path):
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        rows = [dict(row) for row in reader]
        headers = list(reader.fieldnames or [])
    return rows, headers, delimiter


def clean(value) -> str:
    raw = str(value or "").strip()
    if raw.lower() in {"none", "null", "nan"}:
        return ""
    return raw


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def request_json(url, method="GET", payload=None, headers=None, timeout=60):
    body = None
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json; charset=utf-8"

    request = urllib.request.Request(url=url, data=body, method=method.upper(), headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            status = response.getcode()
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc

    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {"raw": raw}
    return status, data


def upload_file(url, file_path: Path, headers=None, timeout=3600):
    boundary = f"----eval-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="audio"; filename="{file_path.name}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )

    request_headers = {"Accept": "application/json", "Content-Type": f"multipart/form-data; boundary={boundary}"}
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(url=url, data=body, method="POST", headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            status = response.getcode()
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except urllib.error.URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc}") from exc

    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {"raw": raw}
    return status, data


def find_audio_path(row, csv_dir: Path, audio_dir: Path | None, indexed_files: dict[str, Path]):
    for key in ["audio_path", "file_path", "path"]:
        raw = clean(row.get(key))
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists():
            return path.resolve()
        if not path.is_absolute():
            local = (csv_dir / raw).resolve()
            if local.exists():
                return local

    filename = clean(row.get("filename"))
    if not filename:
        return None

    candidate = Path(filename).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if not candidate.is_absolute():
        local = (csv_dir / filename).resolve()
        if local.exists():
            return local

    if audio_dir:
        direct = (audio_dir / filename).resolve()
        if direct.exists():
            return direct
        by_name = indexed_files.get(Path(filename).name)
        if by_name is not None:
            return by_name

    return None


def build_report(rows, true_col, pred_col):
    y_true = []
    y_pred = []
    skipped = 0

    for row in rows:
        true_value = clean(row.get(true_col))
        if not true_value:
            skipped += 1
            continue
        pred_value = clean(row.get(pred_col)) or EMPTY
        y_true.append(true_value)
        y_pred.append(pred_value)

    if not y_true:
        return {
            "true_col": true_col,
            "pred_col": pred_col,
            "skipped_unlabeled": skipped,
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
    matrix = {label: Counter() for label in labels}
    for true_value, pred_value in zip(y_true, y_pred):
        matrix[true_value][pred_value] += 1

    correct = sum(1 for true_value, pred_value in zip(y_true, y_pred) if true_value == pred_value)
    precisions = []
    recalls = []
    f1s = []
    weighted_sum = 0.0
    total_support = 0
    labels_report = {}

    for label in labels:
        tp = float(matrix[label][label])
        fp = float(sum(matrix[x][label] for x in labels if x != label))
        fn = float(sum(matrix[label][x] for x in labels if x != label))
        support = int(sum(matrix[label].values()))

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        labels_report[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weighted_sum += f1 * support
        total_support += support

    confusion = {}
    for true_label, row in matrix.items():
        if sum(row.values()) > 0:
            confusion[true_label] = dict(sorted(row.items(), key=lambda item: item[1], reverse=True))

    return {
        "true_col": true_col,
        "pred_col": pred_col,
        "skipped_unlabeled": skipped,
        "samples": len(y_true),
        "accuracy": round(correct / len(y_true), 6),
        "macro_precision": round(sum(precisions) / len(precisions), 6),
        "macro_recall": round(sum(recalls) / len(recalls), 6),
        "macro_f1": round(sum(f1s) / len(f1s), 6),
        "weighted_f1": round(weighted_sum / max(1, total_support), 6),
        "labels": labels_report,
        "confusion": confusion,
    }


def calc_binary_metrics(rows):
    tp = fp = tn = fn = 0
    for row in rows:
        gold = clean(row.get("gold_binary"))
        pred = clean(row.get("pred_binary"))
        if not gold or not pred:
            continue
        if gold == "spam" and pred == "spam":
            tp += 1
        elif gold == "non_spam" and pred == "spam":
            fp += 1
        elif gold == "non_spam" and pred == "non_spam":
            tn += 1
        elif gold == "spam" and pred == "non_spam":
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def print_report(title, report):
    print(f"\n== {title} ==")
    print(f"columns: true='{report['true_col']}', pred='{report['pred_col']}'")
    print(f"samples={report['samples']}, skipped_unlabeled={report['skipped_unlabeled']}")
    print(
        "accuracy={acc:.4f}, macro_p={p:.4f}, macro_r={r:.4f}, macro_f1={f1:.4f}, weighted_f1={wf1:.4f}".format(
            acc=float(report["accuracy"]),
            p=float(report["macro_precision"]),
            r=float(report["macro_recall"]),
            f1=float(report["macro_f1"]),
            wf1=float(report["weighted_f1"]),
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a labeled audio dataset through /api/v1/process-call and compute intent metrics."
    )
    parser.add_argument("--csv", required=True, help="Labeled CSV with at least the intent column and filename/audio_path.")
    parser.add_argument("--audio-dir", default="", help="Root folder with original audio files.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--intent-col", default="call_purpose", help="Gold intent column in the labeled CSV.")
    parser.add_argument("--group-col", default="", help="Optional gold group column.")
    parser.add_argument("--priority-col", default="", help="Optional gold priority column.")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--row-id", type=int, action="append", default=[], help="Only process selected 1-based row(s).")
    parser.add_argument("--out-csv", default="", help="Output CSV with gold labels and system predictions.")
    parser.add_argument("--out-json", default="", help="Output JSON with summary and metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"[ERROR] csv not found: {csv_path}")
        return 2

    audio_dir = Path(args.audio_dir).expanduser().resolve() if args.audio_dir else None
    if audio_dir and not audio_dir.exists():
        print(f"[ERROR] audio dir not found: {audio_dir}")
        return 2

    rows, headers, _ = read_rows(csv_path)
    if not rows:
        print(f"[ERROR] csv is empty: {csv_path}")
        return 2
    if args.intent_col not in headers:
        print(f"[ERROR] column '{args.intent_col}' is required")
        return 2

    selected_rows = {int(x) for x in args.row_id if int(x) > 0}
    csv_dir = csv_path.parent

    audio_index = {}
    if audio_dir:
        for path in audio_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in ALLOWED_AUDIO:
                audio_index[path.name] = path.resolve()

    base_url = args.base_url.rstrip("/")
    status, body = request_json(
        f"{base_url}/api/v1/auth/login",
        "POST",
        {"username": args.username, "password": args.password},
    )
    token = clean(body.get("token"))
    if status != 200 or not token:
        print(f"[ERROR] login failed: status={status}")
        return 2
    headers_auth = {"Authorization": f"Bearer {token}"}

    results = []

    for idx, row in enumerate(rows, start=1):
        if selected_rows and idx not in selected_rows:
            continue

        gold_intent = clean(row.get(args.intent_col))
        gold_group = clean(row.get(args.group_col)) if args.group_col else ""
        gold_priority = clean(row.get(args.priority_col)) if args.priority_col else ""
        gold_binary = "spam" if gold_intent.lower() in {"spam", "spam.call"} else "non_spam"
        audio_path = find_audio_path(row, csv_dir, audio_dir, audio_index)

        item = {
            "row_id": idx,
            "filename": clean(row.get("filename")),
            "audio_path": str(audio_path) if audio_path else "",
            "request_id": "",
            "http_status": "",
            "pipeline_status": "",
            "final_intent_id": gold_intent,
            "ai_intent_id": "",
            "final_group_id": gold_group,
            "ai_group_id": "",
            "final_priority": gold_priority,
            "ai_priority": "",
            "gold_binary": gold_binary,
            "pred_binary": "",
            "ai_confidence": "",
            "call_id": "",
            "ticket_id": "",
            "ticket_url": "",
            "total_time_sec": "",
            "error": "",
            "error_details": "",
        }

        if audio_path is None:
            item["error"] = "audio_not_found"
            results.append(item)
            print(f"[SKIP] row {idx}: audio not found")
            continue

        try:
            request_id = f"eval-{uuid.uuid4().hex}"
            started = time.monotonic()
            item["request_id"] = request_id
            status, payload = upload_file(
                f"{base_url}/api/v1/process-call",
                audio_path,
                headers={**headers_auth, "X-Request-ID": request_id},
                timeout=args.timeout,
            )
            item["http_status"] = status
            item["total_time_sec"] = round(time.monotonic() - started, 6)

            if status != 200:
                item["error"] = f"http_{status}"
                item["error_details"] = clean(payload.get("error") or payload.get("details") or payload.get("raw") or payload)
                results.append(item)
                print(f"[FAIL] row {idx}: http {status}")
                continue

            routing = payload.get("routing") or {}
            ticket = payload.get("ticket") or {}
            item["pipeline_status"] = clean(payload.get("status"))
            item["ai_intent_id"] = clean(routing.get("intent_id"))
            item["ai_group_id"] = clean(routing.get("suggested_group"))
            item["ai_priority"] = clean(routing.get("priority"))
            item["ai_confidence"] = to_float(routing.get("intent_confidence")) or ""
            item["call_id"] = clean(payload.get("call_id"))
            item["ticket_id"] = clean(ticket.get("external_id") or ticket.get("ticket_id"))
            item["ticket_url"] = clean(ticket.get("url"))
            item["pred_binary"] = (
                "spam"
                if item["pipeline_status"] == "spam_blocked" or item["ai_intent_id"] in {"spam", "spam.call"}
                else "non_spam"
            )

            results.append(item)
            print(
                f"[OK] row {idx}: gold={gold_intent} pred={item['ai_intent_id'] or '-'} "
                f"status={item['pipeline_status'] or '-'}"
            )
        except Exception as exc:
            item["error"] = str(exc)
            item["error_details"] = str(exc)
            results.append(item)
            print(f"[FAIL] row {idx}: {exc}")

    ok_rows = [row for row in results if not row["error"]]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv": str(csv_path),
        "audio_dir": str(audio_dir) if audio_dir else "",
        "base_url": base_url,
        "rows_total": len(results),
        "processed_ok": len(ok_rows),
        "processed_with_error": len(results) - len(ok_rows),
        "status_counts": dict(Counter(clean(row.get("pipeline_status")) or "unknown" for row in ok_rows)),
        "intent": build_report(results, "final_intent_id", "ai_intent_id"),
        "binary": calc_binary_metrics(results),
        "avg_total_time_sec": round(
            sum(float(row["total_time_sec"]) for row in ok_rows if row["total_time_sec"] != "") / max(1, len(ok_rows)),
            6,
        )
        if ok_rows
        else None,
    }
    if args.group_col:
        report["group"] = build_report(results, "final_group_id", "ai_group_id")
    if args.priority_col:
        report["priority"] = build_report(results, "final_priority", "ai_priority")

    print_report("Intent", report["intent"])
    if args.group_col:
        print_report("Group", report["group"])
    if args.priority_col:
        print_report("Priority", report["priority"])

    out_csv = (
        Path(args.out_csv).expanduser().resolve()
        if args.out_csv
        else (csv_path.parent / "labeled_audio_eval_results.csv")
    )
    out_json = (
        Path(args.out_json).expanduser().resolve()
        if args.out_json
        else (csv_path.parent / "labeled_audio_eval_metrics.json")
    )

    fieldnames = list(rows[0].keys()) + [
        "row_id",
        "filename",
        "audio_path",
        "request_id",
        "http_status",
        "pipeline_status",
        "final_intent_id",
        "ai_intent_id",
        "final_group_id",
        "ai_group_id",
        "final_priority",
        "ai_priority",
        "gold_binary",
        "pred_binary",
        "ai_confidence",
        "call_id",
        "ticket_id",
        "ticket_url",
        "total_time_sec",
        "error",
        "error_details",
    ]
    fieldnames = list(dict.fromkeys(fieldnames))

    with out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for source, extra in zip(rows, results):
            merged = dict(source)
            merged.update(extra)
            writer.writerow(merged)

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] CSV: {out_csv}")
    print(f"[OK] JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
