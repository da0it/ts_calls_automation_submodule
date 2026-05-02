#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
ROUTER_DIR = ROOT_DIR / "services" / "router"
if str(ROUTER_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTER_DIR))


def _install_razdel_fallback() -> None:
    try:
        import razdel  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    fallback = types.ModuleType("razdel")

    class _Span:
        def __init__(self, text: str):
            self.text = text

    def sentenize(text: str):
        parts = re.split(r"(?<=[.!?])\s+|\n+", str(text or "").strip(), flags=re.UNICODE)
        return [_Span(part.strip()) for part in parts if part and part.strip()]

    def tokenize(text: str):
        parts = re.findall(r"<[a-z_]+>|[\w]+|[^\w\s]", str(text or ""), flags=re.UNICODE)
        return [_Span(part) for part in parts if part and not part.isspace()]

    fallback.sentenize = sentenize
    fallback.tokenize = tokenize
    sys.modules["razdel"] = fallback


_install_razdel_fallback()

from routing.nlp_preprocess import PreprocessConfig, build_canonical, split_sentences  # noqa: E402


def _build_segments(text: str) -> List[Tuple[float, str, Optional[str]]]:
    sentences = split_sentences(text)
    if not sentences:
        stripped = str(text or "").strip()
        return [(0.0, stripped, None)] if stripped else []
    return [(float(index * 5), sentence, None) for index, sentence in enumerate(sentences)]


def _preprocess_text(text: str, cfg: PreprocessConfig) -> Dict[str, object]:
    result = build_canonical(_build_segments(text), cfg)
    return {
        "preprocessed_text": result.model_text,
        "canonical_text": result.canonical_text,
        "sentences": result.sentences,
        "tokens": result.tokens,
        "meta": result.meta,
    }


def _default_output_csv(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_preprocessed.csv")


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = [dict(row) for row in reader]
        headers = list(reader.fieldnames or [])
    return rows, headers


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _build_config(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        model_text_mode=str(args.model_text_mode or "tokens").strip().lower() or "tokens",
        drop_fillers=not bool(args.keep_fillers),
        drop_stopwords=not bool(args.keep_stopwords),
        dedupe=not bool(args.no_dedupe),
        dedupe_window=int(max(1, args.dedupe_window)),
        max_chars=int(max(200, args.max_chars)),
        keep_timestamps=bool(args.keep_timestamps),
        do_tokenize=True,
        keep_special_tokens=not bool(args.drop_special_tokens),
    )


def _print_single(result: Dict[str, object]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run router text preprocessing for a single text or a CSV dataset."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", default="", help="Single text to preprocess.")
    source.add_argument("--input-csv", default="", help="CSV file with a text column.")

    parser.add_argument("--text-col", default="text", help="Text column in CSV mode.")
    parser.add_argument("--output-csv", default="", help="Output CSV path. Default: <input>_preprocessed.csv")

    parser.add_argument("--model-text-mode", default="tokens", choices=["canonical", "normalized", "plain", "tokens"])
    parser.add_argument("--keep-stopwords", action="store_true", help="Keep stop words in output text.")
    parser.add_argument("--keep-fillers", action="store_true", help="Keep filler phrases like 'спасибо', 'ага', 'алло'.")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable nearby duplicate sentence removal.")
    parser.add_argument("--dedupe-window", type=int, default=2, help="Nearby duplicate window size.")
    parser.add_argument("--keep-timestamps", action="store_true", help="Keep synthetic timestamps in canonical text.")
    parser.add_argument("--drop-special-tokens", action="store_true", help="Drop placeholders like <phone>, <email>.")
    parser.add_argument("--max-chars", type=int, default=4000, help="Maximum output length for canonical text.")

    parser.add_argument("--out-col", default="preprocessed_text", help="Output CSV column for processed model text.")
    parser.add_argument("--canonical-col", default="canonical_text", help="Output CSV column for canonical text.")
    parser.add_argument("--meta-col", default="preprocess_meta", help="Output CSV column for preprocessing metadata JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _build_config(args)

    if args.text:
        result = _preprocess_text(str(args.text), cfg)
        _print_single(result)
        return 0

    input_csv = Path(args.input_csv).expanduser().resolve()
    if not input_csv.exists():
        print(f"[ERROR] csv not found: {input_csv}", file=sys.stderr)
        return 2

    rows, headers = _read_csv(input_csv)
    if not rows:
        print(f"[ERROR] csv is empty: {input_csv}", file=sys.stderr)
        return 2
    if args.text_col not in headers:
        print(f"[ERROR] text column not found: {args.text_col}", file=sys.stderr)
        print("headers:", ", ".join(headers), file=sys.stderr)
        return 2

    output_rows: List[Dict[str, object]] = []
    for row in rows:
        source_text = str(row.get(args.text_col) or "")
        processed = _preprocess_text(source_text, cfg)
        output_row: Dict[str, object] = dict(row)
        output_row[str(args.out_col)] = str(processed["preprocessed_text"])
        output_row[str(args.canonical_col)] = str(processed["canonical_text"])
        output_row[str(args.meta_col)] = json.dumps(processed["meta"], ensure_ascii=False)
        output_rows.append(output_row)

    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else _default_output_csv(input_csv)
    )
    extra_fields = [str(args.out_col), str(args.canonical_col), str(args.meta_col)]
    fieldnames = list(headers)
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    _write_csv(output_csv, output_rows, fieldnames)
    print(f"[OK] saved: {output_csv}")
    print(
        json.dumps(
            {
                "rows_total": len(output_rows),
                "text_col": args.text_col,
                "out_col": args.out_col,
                "canonical_col": args.canonical_col,
                "meta_col": args.meta_col,
                "config": {
                    "model_text_mode": cfg.model_text_mode,
                    "do_tokenize": cfg.do_tokenize,
                    "drop_stopwords": cfg.drop_stopwords,
                    "drop_fillers": cfg.drop_fillers,
                    "dedupe": cfg.dedupe,
                    "keep_timestamps": cfg.keep_timestamps,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
