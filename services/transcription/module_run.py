from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import json
import argparse
from transcribe.pipeline import transcribe_with_roles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="mp3/ogg/wav")
    ap.add_argument("--out", default="", help="output json (default stdout)")
    ap.add_argument("--model", default="v3_e2e_rnnt", help="GigaAM model name")
    args = ap.parse_args()

    res = transcribe_with_roles(
        args.audio,
        gigaam_model_name=args.model,
    )

    s = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        print(s)

if __name__ == "__main__":
    main()
