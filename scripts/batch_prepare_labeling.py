#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Dict, Iterable, List, Tuple


ALLOWED_DEFAULT_EXTS = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
CALLER_ROLE_HINTS = {"звонящий", "caller", "customer", "client"}


def _http_json(
    url: str,
    method: str,
    payload: Dict[str, object] | None = None,
    headers: Dict[str, str] | None = None,
    timeout: int = 60,
) -> Dict[str, object]:
    body = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url=url, data=body, method=method.upper(), headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {e.code}: {raw}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"{method} {url} failed: {e}") from e


def _encode_multipart_form(
    fields: Dict[str, str],
    file_field: str,
    file_path: Path,
) -> Tuple[bytes, str]:
    boundary = f"----batch-label-{uuid.uuid4().hex}"
    chunks: List[bytes] = []

    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")

    file_name = file_path.name
    content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()
    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_name}"\r\n'
        ).encode("utf-8")
    )
    chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    chunks.append(file_bytes)
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))

    return b"".join(chunks), boundary


def _http_multipart_json(
    url: str,
    file_field: str,
    file_path: Path,
    headers: Dict[str, str] | None = None,
    timeout: int = 3600,
    extra_fields: Dict[str, str] | None = None,
) -> Dict[str, object]:
    form_fields = dict(extra_fields or {})
    body, boundary = _encode_multipart_form(form_fields, file_field=file_field, file_path=file_path)
    req_headers = {"Accept": "application/json", "Content-Type": f"multipart/form-data; boundary={boundary}"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url=url, data=body, method="POST", headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed for {file_path.name}: HTTP {e.code}: {raw}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"POST {url} failed for {file_path.name}: {e}") from e


def _collect_audio_files(root: Path, exts: Iterable[str], recursive: bool) -> List[Path]:
    allowed = {e.lower() for e in exts}
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed]
    else:
        files = [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in allowed]
    return sorted(files)


def _join_segments_text(segments: List[Dict[str, object]]) -> str:
    parts: List[str] = []
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _build_training_sample(segments: List[Dict[str, object]], max_chars: int) -> str:
    preferred: List[str] = []
    fallback: List[str] = []
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        role = str(seg.get("role") or "").strip().lower()
        fallback.append(text)
        if any(hint in role for hint in CALLER_ROLE_HINTS):
            preferred.append(text)
    base = " ".join(preferred or fallback)
    base = " ".join(base.split())
    if len(base) <= max_chars:
        return base
    return base[:max_chars].rstrip() + "..."


def _extract_row(source_file: Path, payload: Dict[str, object], max_sample_chars: int) -> Dict[str, str]:
    transcript = payload.get("transcript") if isinstance(payload.get("transcript"), dict) else {}
    routing = payload.get("routing") if isinstance(payload.get("routing"), dict) else {}
    segments = transcript.get("segments") if isinstance(transcript.get("segments"), list) else []
    safe_segments = [s for s in segments if isinstance(s, dict)]

    transcript_text = _join_segments_text(safe_segments)
    training_sample = _build_training_sample(safe_segments, max_chars=max_sample_chars)
    call_id = str(payload.get("call_id") or transcript.get("call_id") or "")

    return {
        "source_file": str(source_file),
        "call_id": call_id,
        "ai_intent_id": str(routing.get("intent_id") or ""),
        "ai_confidence": str(routing.get("intent_confidence") or ""),
        "ai_priority": str(routing.get("priority") or ""),
        "ai_group_id": str(routing.get("suggested_group") or ""),
        "training_sample": training_sample,
        "transcript_text": transcript_text,
        "transcript_segments": json.dumps(safe_segments, ensure_ascii=False),
        "final_intent_id": "",
        "final_group_id": "",
        "label_comment": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch process audio files through orchestrator and prepare CSV for manual intent/group labeling."
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="Orchestrator base URL.")
    parser.add_argument("--username", required=True, help="Username for /api/v1/auth/login.")
    parser.add_argument("--password", required=True, help="Password for /api/v1/auth/login.")
    parser.add_argument("--input-dir", required=True, help="Directory with audio files.")
    parser.add_argument("--output-dir", default="", help="Output directory. Default: ./exports/labeling_<timestamp>.")
    parser.add_argument(
        "--extensions",
        default=",".join(ALLOWED_DEFAULT_EXTS),
        help="Comma-separated allowed file extensions (e.g. .mp3,.wav).",
    )
    parser.add_argument("--no-recursive", action="store_true", help="Do not scan subdirectories.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process (0 = all).")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-file request timeout in seconds.")
    parser.add_argument(
        "--max-training-sample-chars",
        type=int,
        default=320,
        help="Max chars for training_sample field in CSV.",
    )
    parser.add_argument("--save-raw", action="store_true", help="Save per-call raw JSON files.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop batch on first error.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] input dir does not exist: {input_dir}", file=sys.stderr)
        return 2

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path.cwd() / "exports" / f"labeling_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_json"
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    exts = [e.strip().lower() for e in args.extensions.split(",") if e.strip()]
    files = _collect_audio_files(input_dir, exts=exts, recursive=not args.no_recursive)
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        print(f"[WARN] no audio files found in {input_dir} with extensions {exts}")
        return 0

    print(f"[INFO] files to process: {len(files)}")
    print("[INFO] authenticating...")
    login = _http_json(
        f"{base_url}/api/v1/auth/login",
        "POST",
        payload={"username": args.username, "password": args.password},
        timeout=60,
    )
    token = str(login.get("token") or "").strip()
    if not token:
        print("[ERROR] login succeeded but token is missing", file=sys.stderr)
        return 2

    headers = {"Authorization": f"Bearer {token}"}
    results_jsonl = out_dir / "results.jsonl"
    failures_csv = out_dir / "failures.csv"
    dataset_csv = out_dir / "labeling_dataset.csv"

    dataset_rows: List[Dict[str, str]] = []
    failures: List[Dict[str, str]] = []

    for idx, path in enumerate(files, start=1):
        rel = path.relative_to(input_dir) if path.is_relative_to(input_dir) else path
        print(f"[{idx}/{len(files)}] {rel}")
        try:
            payload = _http_multipart_json(
                f"{base_url}/api/v1/process-call",
                file_field="audio",
                file_path=path,
                headers=headers,
                timeout=int(args.timeout),
            )
            enriched = {
                "source_file": str(rel),
                "source_abs_path": str(path),
                "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "response": payload,
            }
            with results_jsonl.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            row = _extract_row(rel, payload, max_sample_chars=int(args.max_training_sample_chars))
            dataset_rows.append(row)

            if args.save_raw:
                raw_name = f"{idx:05d}_{path.stem}.json"
                (raw_dir / raw_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            msg = str(exc)
            print(f"  [ERROR] {msg}", file=sys.stderr)
            failures.append({"source_file": str(rel), "error": msg})
            if args.stop_on_error:
                break

    if dataset_rows:
        fieldnames = [
            "source_file",
            "call_id",
            "ai_intent_id",
            "ai_confidence",
            "ai_priority",
            "ai_group_id",
            "training_sample",
            "transcript_text",
            "transcript_segments",
            "final_intent_id",
            "final_group_id",
            "label_comment",
        ]
        with dataset_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset_rows)

    if failures:
        with failures_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source_file", "error"])
            writer.writeheader()
            writer.writerows(failures)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "total_files": len(files),
        "processed_ok": len(dataset_rows),
        "failed": len(failures),
        "results_jsonl": str(results_jsonl) if results_jsonl.exists() else "",
        "dataset_csv": str(dataset_csv) if dataset_csv.exists() else "",
        "failures_csv": str(failures_csv) if failures_csv.exists() else "",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[DONE]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
