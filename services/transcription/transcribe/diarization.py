from __future__ import annotations
import os
from typing import List, Optional, Tuple
from .config import CFG


def pyannote_turns(wav_path: str, hf_token: Optional[str] = None) -> List[Tuple[float, float, str]]:
    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for pyannote diarization. Put it in .env (HF_TOKEN=...)")

    pipeline = Pipeline.from_pretrained(CFG.pyannote.pipeline_name, token=token)

    try:
        pipeline.instantiate({
            "segmentation": {
                "min_duration_off": CFG.pyannote.min_duration_off,
                "min_duration_on": CFG.pyannote.min_duration_on,
            }
        })
    except Exception:
        pass

    out = pipeline(wav_path, num_speakers=CFG.pyannote.num_speakers)

    ann = None
    if hasattr(out, "exclusive_speaker_diarization"):
        ann = out.exclusive_speaker_diarization
    elif hasattr(out, "speaker_diarization"):
        ann = out.speaker_diarization

    if ann is None:
        attrs = [a for a in dir(out) if not a.startswith("_")]
        raise RuntimeError(f"Unsupported pyannote diarization output type: {type(out)} attrs={attrs[:60]}")

    turns: List[Tuple[float, float, str]] = []
    for turn, _, speaker in ann.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))

    turns.sort(key=lambda x: (x[0], x[1]))
    return turns
