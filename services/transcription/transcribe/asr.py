from __future__ import annotations
import os
import tempfile
from typing import Any, Dict, List, Tuple

from .config import CFG
from .audio_utils import cut_wav_segment, detect_silences


def split_long_turns(
    turns: List[Tuple[float, float, str]],
    max_len: float | None = None,
    overlap: float | None = None,
) -> List[Tuple[float, float, str]]:
    max_len = CFG.turns.long_turn_max_len if max_len is None else max_len
    overlap = CFG.turns.long_turn_overlap if overlap is None else overlap

    out: List[Tuple[float, float, str]] = []
    for s, e, spk in turns:
        if (e - s) <= max_len:
            out.append((s, e, spk))
            continue

        cur = s
        while cur < e:
            nxt = min(e, cur + max_len)
            out.append((cur, nxt, spk))
            if nxt >= e:
                break
            cur = max(cur, nxt - overlap)
    return out


def split_turn_by_silence(
    wav_path: str,
    turn_start: float,
    turn_end: float,
    max_len: float | None = None,
) -> List[Tuple[float, float]]:
    max_len = CFG.asr.piece_max_len if max_len is None else max_len

    sil = detect_silences(
        wav_path,
        turn_start,
        turn_end,
        silence_db=CFG.asr.silence_db,
        silence_min_dur=CFG.asr.silence_min_dur,
    )

    if not sil:
        out: List[Tuple[float, float]] = []
        cur = turn_start
        while cur < turn_end:
            nxt = min(turn_end, cur + max_len)
            out.append((cur, nxt))
            cur = nxt
        return out

    cuts: List[float] = []
    for s0, s1 in sil:
        mid = (s0 + s1) / 2.0
        if (turn_start + CFG.silence.edge_guard_seconds) < mid < (turn_end - CFG.silence.edge_guard_seconds):
            cuts.append(mid)

    cuts = sorted(set(cuts))
    points = [turn_start] + cuts + [turn_end]

    out: List[Tuple[float, float]] = []
    for a, b in zip(points, points[1:]):
        a2 = max(turn_start, a - CFG.silence.split_pad)
        b2 = min(turn_end, b + CFG.silence.split_pad)
        if (b2 - a2) >= CFG.silence.min_piece_seconds:
            out.append((a2, b2))

    final: List[Tuple[float, float]] = []
    for a, b in out:
        if (b - a) <= max_len:
            final.append((a, b))
        else:
            cur = a
            while cur < b:
                nxt = min(b, cur + max_len)
                final.append((cur, nxt))
                cur = nxt
    return final


def transcribe_turns(model, wav_path: str, turns: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        for i, (s, e, spk) in enumerate(turns):
            if (e - s) < CFG.asr.min_dur:
                continue

            pieces = split_turn_by_silence(wav_path, s, e, max_len=CFG.asr.piece_max_len)

            for j, (ps, pe) in enumerate(pieces):
                if (pe - ps) < CFG.asr.min_dur:
                    continue

                seg_wav = os.path.join(td, f"turn_{i:05d}_{j:02d}_{spk}.wav")
                cut_wav_segment(wav_path, seg_wav, ps, pe)

                text = model.transcribe(seg_wav)
                if isinstance(text, dict) and "text" in text:
                    text = text["text"]
                text = str(text).strip()

                if text:
                    out.append({"start": float(ps), "end": float(pe), "speaker": spk, "text": text})
    return out
