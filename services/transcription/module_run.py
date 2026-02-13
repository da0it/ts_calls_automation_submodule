from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import json
import argparse
from transcribe_logic.pipeline import transcribe_with_roles
from transcribe_logic.roles import infer_role_map_from_segments

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="mp3/ogg/wav")
    ap.add_argument("--out", default="", help="output json (default stdout)")
    ap.add_argument("--no-stem", action="store_true", help="Disable source separation in whisper-diarization")
    ap.add_argument(
        "--whisper-repo-dir",
        default="/home/dmitrii/whisper-diarization",
        help="Path to whisper-diarization repo",
    )
    ap.add_argument(
        "--whisper-venv-python",
        default="/home/dmitrii/whisper-diarization/whisper_venv/bin/python",
        help="Path to python inside whisper-diarization venv",
    )
    args = ap.parse_args()

    res = transcribe_with_roles(
        args.audio,
        no_stem=args.no_stem,
        whisper_repo_dir=args.whisper_repo_dir,
        whisper_venv_python=args.whisper_venv_python,
    )

    s = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        print(s)


if __name__ == "__main__":
    main()
