#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def clean(value: object) -> str:
    return str(value or "").strip()


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
    return rows, headers


def build_key(row):
    for key in ["ab_id", "filename", "audio_path"]:
        value = clean(row.get(key))
        if value:
            return key, value
    return "", ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge blind manual A/B annotations with system results into one CSV for evaluate_ab_test.py."
    )
    parser.add_argument("--system-csv", required=True)
    parser.add_argument("--manual-csv", required=True)
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    system_csv = Path(args.system_csv).expanduser().resolve()
    manual_csv = Path(args.manual_csv).expanduser().resolve()
    if not system_csv.exists():
        print(f"[ERROR] system csv not found: {system_csv}")
        return 2
    if not manual_csv.exists():
        print(f"[ERROR] manual csv not found: {manual_csv}")
        return 2

    system_rows, _ = read_rows(system_csv)
    manual_rows, _ = read_rows(manual_csv)
    if not system_rows:
        print(f"[ERROR] system csv is empty: {system_csv}")
        return 2
    if not manual_rows:
        print(f"[ERROR] manual csv is empty: {manual_csv}")
        return 2

    manual_index = {}
    duplicate_manual_keys = []
    for row in manual_rows:
        _, value = build_key(row)
        if not value:
            continue
        if value in manual_index:
            duplicate_manual_keys.append(value)
        manual_index[value] = row

    merged_rows = []
    matched = 0
    unmatched_system = []

    for system_row in system_rows:
        _, key = build_key(system_row)
        manual_row = manual_index.get(key)
        if manual_row is None:
            unmatched_system.append(key or clean(system_row.get("filename")) or clean(system_row.get("audio_path")))
            manual_row = {}
        else:
            matched += 1

        merged_rows.append(
            {
                "ab_id": clean(system_row.get("ab_id")) or clean(manual_row.get("ab_id")),
                "filename": clean(system_row.get("filename")) or clean(manual_row.get("filename")),
                "audio_path": clean(system_row.get("audio_path")) or clean(manual_row.get("audio_path")),
                "manual_time_sec": clean(manual_row.get("manual_time_sec")),
                "system_time_sec": clean(system_row.get("system_time_sec")),
                "manual_operator_load_pct": clean(manual_row.get("manual_operator_load_pct")) or "100",
                "system_operator_load_pct": "0",
                "final_intent_id": clean(manual_row.get("final_intent_id")),
                "ai_intent_id": clean(system_row.get("ai_intent_id")),
                "final_group_id": clean(manual_row.get("final_group_id")),
                "ai_group_id": clean(system_row.get("ai_group_id")),
                "final_priority": clean(manual_row.get("final_priority")),
                "ai_priority": clean(system_row.get("ai_priority")),
                "manual_comment": clean(manual_row.get("manual_comment")),
                "manual_order": clean(manual_row.get("manual_order")),
                "pipeline_status": clean(system_row.get("pipeline_status")),
                "ai_confidence": clean(system_row.get("ai_confidence")),
                "transcription_sec": clean(system_row.get("transcription_sec")),
                "routing_sec": clean(system_row.get("routing_sec")),
                "entity_extraction_sec": clean(system_row.get("entity_extraction_sec")),
                "ticket_creation_sec": clean(system_row.get("ticket_creation_sec")),
                "request_id": clean(system_row.get("request_id")),
                "http_status": clean(system_row.get("http_status")),
                "call_id": clean(system_row.get("call_id")),
                "ticket_id": clean(system_row.get("ticket_id")),
                "ticket_url": clean(system_row.get("ticket_url")),
                "error": clean(system_row.get("error")),
                "error_details": clean(system_row.get("error_details")),
            }
        )

    manual_keys = set(manual_index.keys())
    system_keys = {build_key(row)[1] for row in system_rows if build_key(row)[1]}
    unmatched_manual = sorted(manual_keys - system_keys)

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else (system_csv.parent / "ab_eval_merged.csv")
    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else (system_csv.parent / "ab_eval_merge_summary.json")

    with out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(merged_rows[0].keys()))
        writer.writeheader()
        writer.writerows(merged_rows)

    summary = {
        "system_csv": str(system_csv),
        "manual_csv": str(manual_csv),
        "out_csv": str(out_csv),
        "rows_system": len(system_rows),
        "rows_manual": len(manual_rows),
        "rows_merged": len(merged_rows),
        "matched_rows": matched,
        "unmatched_system_rows": len(unmatched_system),
        "unmatched_manual_rows": len(unmatched_manual),
        "duplicate_manual_keys": duplicate_manual_keys,
        "unmatched_system_examples": unmatched_system[:10],
        "unmatched_manual_examples": unmatched_manual[:10],
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] merged csv: {out_csv}")
    print(f"[OK] merge summary: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
