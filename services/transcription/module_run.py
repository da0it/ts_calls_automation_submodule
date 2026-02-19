from __future__ import annotations
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

import json
import argparse
import os
from transcribe_logic.pipeline import transcribe_with_roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="mp3/ogg/wav")
    ap.add_argument("--out", default="", help="output json (default stdout)")
    ap.add_argument(
        "--whisper-repo-dir",
        default=os.getenv("WHISPER_REPO_DIR", os.path.expanduser("~/whisper-diarization")),
        help="Path to whisper-diarization repo (needed only for WhisperX + NeMo diarization backend)",
    )
    args = ap.parse_args()

    res = transcribe_with_roles(
        args.audio,
        whisper_repo_dir=args.whisper_repo_dir,
    )

    s = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        print(s)


if __name__ == "__main__":
    main()
