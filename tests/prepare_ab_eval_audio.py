#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import random
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_AUDIO = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}


def clean(value: object) -> str:
    return str(value or "").strip()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_float(value: object):
    try:
        return float(value)
    except Exception:
        return None


def request_json(url: str, method: str = "GET", payload=None, headers=None, timeout: int = 60):
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


def upload_file(url: str, file_path: Path, headers=None, timeout: int = 3600):
    boundary = f"----ab-eval-{uuid.uuid4().hex}"
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


def collect_audio_files(root: Path, recursive: bool):
    finder = root.rglob if recursive else root.glob
    return sorted(path for path in finder("*") if path.is_file() and path.suffix.lower() in ALLOWED_AUDIO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare A/B evaluation artifacts from an audio folder using the deployed /api/v1/process-call."
    )
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--shuffle-manual-order", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Fixed pause between processed calls.")
    parser.add_argument(
        "--jitter-sec",
        type=float,
        default=0.0,
        help="Add random pause in range [0, jitter] after each processed call.",
    )
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    if not audio_dir.exists() or not audio_dir.is_dir():
        print(f"[ERROR] audio dir not found: {audio_dir}")
        return 2

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path.cwd() / "exports" / f"ab_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_audio_files(audio_dir, recursive=not args.no_recursive)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"[ERROR] no audio files found in: {audio_dir}")
        return 2

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
    headers = {"Authorization": f"Bearer {token}"}

    system_rows = []
    manual_rows = []
    rnd = random.Random(args.seed)

    for idx, path in enumerate(files, start=1):
        rel_path = path.relative_to(audio_dir).as_posix()
        ab_id = f"AB{idx:04d}"
        request_id = f"ab-eval-{uuid.uuid4().hex}"
        item = {
            "ab_id": ab_id,
            "filename": rel_path,
            "audio_path": str(path),
            "request_id": request_id,
            "started_at_utc": "",
            "finished_at_utc": "",
            "http_status": "",
            "final_outcome": "",
            "pipeline_status": "",
            "system_time_sec": "",
            "server_total_time_sec": "",
            "transcription_sec": "",
            "routing_sec": "",
            "entity_extraction_sec": "",
            "ticket_creation_sec": "",
            "ai_intent_id": "",
            "ai_group_id": "",
            "ai_priority": "",
            "ai_confidence": "",
            "call_id": "",
            "ticket_id": "",
            "ticket_url": "",
            "error": "",
            "error_details": "",
        }
        try:
            started_at = utc_now_iso()
            started_monotonic = time.monotonic()
            item["started_at_utc"] = started_at
            status, payload = upload_file(
                f"{base_url}/api/v1/process-call",
                path,
                headers={**headers, "X-Request-ID": request_id},
                timeout=args.timeout,
            )
            finished_monotonic = time.monotonic()
            finished_at = utc_now_iso()
            item["http_status"] = status
            item["finished_at_utc"] = finished_at
            item["system_time_sec"] = round(finished_monotonic - started_monotonic, 6)
            if status != 200:
                item["error"] = f"http_{status}"
                item["error_details"] = clean(payload.get("error") or payload.get("details") or payload.get("raw"))
                item["final_outcome"] = "http_error"
            else:
                routing = payload.get("routing") if isinstance(payload.get("routing"), dict) else {}
                processing = payload.get("processing_time") if isinstance(payload.get("processing_time"), dict) else {}
                ticket = payload.get("ticket") if isinstance(payload.get("ticket"), dict) else {}

                item["pipeline_status"] = clean(payload.get("status"))
                item["final_outcome"] = item["pipeline_status"] or "unknown"
                item["server_total_time_sec"] = to_float(payload.get("total_time")) or ""
                item["transcription_sec"] = to_float(processing.get("transcription")) or ""
                item["routing_sec"] = to_float(processing.get("routing")) or ""
                item["entity_extraction_sec"] = to_float(processing.get("entity_extraction")) or ""
                item["ticket_creation_sec"] = to_float(processing.get("ticket_creation")) or ""
                item["ai_intent_id"] = clean(routing.get("intent_id"))
                item["ai_group_id"] = clean(routing.get("suggested_group"))
                item["ai_priority"] = clean(routing.get("priority"))
                item["ai_confidence"] = to_float(routing.get("intent_confidence")) or ""
                item["call_id"] = clean(payload.get("call_id"))
                item["ticket_id"] = clean(ticket.get("external_id") or ticket.get("ticket_id"))
                item["ticket_url"] = clean(ticket.get("url"))
            print(
                f"[{idx}/{len(files)}] {rel_path}: "
                f"status={item['final_outcome'] or item['error'] or item['http_status']} "
                f"e2e={item['system_time_sec'] or '-'}s"
            )
        except Exception as exc:
            item["finished_at_utc"] = utc_now_iso()
            item["error"] = str(exc)
            item["error_details"] = str(exc)
            item["final_outcome"] = "client_error"
            print(f"[{idx}/{len(files)}] {rel_path}: ERROR {exc}")

        system_rows.append(item)
        manual_rows.append(
            {
                "ab_id": ab_id,
                "manual_order": idx,
                "filename": rel_path,
                "audio_path": str(path),
                "manual_time_sec": "",
                "manual_operator_load_pct": "100",
                "final_intent_id": "",
                "final_group_id": "",
                "final_priority": "",
                "manual_comment": "",
            }
        )

        if idx < len(files):
            delay = max(0.0, float(args.sleep_sec))
            if args.jitter_sec > 0:
                delay += rnd.uniform(0.0, float(args.jitter_sec))
            if delay > 0:
                print(f"  sleeping {delay:.2f} sec before next call")
                time.sleep(delay)

    if args.shuffle_manual_order:
        rnd = random.Random(args.seed)
        rnd.shuffle(manual_rows)
        for order, row in enumerate(manual_rows, start=1):
            row["manual_order"] = order

    system_csv = out_dir / "system_results.csv"
    manual_csv = out_dir / "manual_template.csv"
    summary_json = out_dir / "summary.json"

    with system_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(system_rows[0].keys()))
        writer.writeheader()
        writer.writerows(system_rows)

    with manual_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manual_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manual_rows)

    ok_rows = [row for row in system_rows if not row["error"] and str(row["http_status"]) == "200"]
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audio_dir": str(audio_dir),
        "base_url": base_url,
        "rows_total": len(system_rows),
        "rows_ok": len(ok_rows),
        "rows_failed": len(system_rows) - len(ok_rows),
        "sleep_sec": float(args.sleep_sec),
        "jitter_sec": float(args.jitter_sec),
        "avg_system_time_sec": round(
            sum(float(row["system_time_sec"]) for row in ok_rows if row["system_time_sec"] != "") / max(1, len(ok_rows)),
            6,
        ),
        "avg_server_total_time_sec": round(
            sum(float(row["server_total_time_sec"]) for row in ok_rows if row["server_total_time_sec"] != "") / max(
                1,
                len([row for row in ok_rows if row["server_total_time_sec"] != ""]),
            ),
            6,
        )
        if any(row["server_total_time_sec"] != "" for row in ok_rows)
        else None,
        "system_results_csv": str(system_csv),
        "manual_template_csv": str(manual_csv),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[DONE]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nNext steps:")
    print(f"1. Give the operator only: {manual_csv}")
    print("2. Fill manual_time_sec, final_intent_id, final_group_id, final_priority in that file.")
    print("3. Merge results with tests/merge_ab_eval_audio.py.")
    print("4. Run tests/evaluate_ab_test.py on the merged CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
