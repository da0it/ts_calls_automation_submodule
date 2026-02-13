# transcribe/pipeline.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from transcribe.config import CFG
from transcribe.audio_utils import to_wav_16k_mono_preprocessed
from transcribe.diarization import pyannote_turns
from transcribe.asr import transcribe_turns, split_long_turns
from transcribe.roles import infer_role_map_from_segments


def merge_turns(
    turns: List[Tuple[float, float, str]],
    max_gap: float | None = None,
    min_dur: float | None = None,
) -> List[Tuple[float, float, str]]:
    max_gap = CFG.turns.merge_max_gap if max_gap is None else max_gap
    min_dur = CFG.turns.merge_min_dur if min_dur is None else min_dur

    if not turns:
        return []

    turns = sorted(turns, key=lambda x: (x[0], x[1]))

    merged: List[Tuple[float, float, str]] = []
    cs, ce, cspk = turns[0]

    for s, e, spk in turns[1:]:
        if spk == cspk and (s - ce) <= max_gap:
            ce = max(ce, e)
        else:
            if (ce - cs) >= min_dur:
                merged.append((cs, ce, cspk))
            cs, ce, cspk = s, e, spk

    if (ce - cs) >= min_dur:
        merged.append((cs, ce, cspk))
    return merged


def merge_utterances_same_speaker(
    segs: List[Dict[str, Any]],
    max_gap: float | None = None,
) -> List[Dict[str, Any]]:
    max_gap = CFG.turns.merge_utt_max_gap if max_gap is None else max_gap

    if not segs:
        return []
    segs = sorted(segs, key=lambda x: (x["start"], x["end"]))
    out = [segs[0].copy()]
    for s in segs[1:]:
        prev = out[-1]
        if (
            s.get("speaker") == prev.get("speaker")
            and (s["start"] - prev["end"]) <= max_gap
        ):
            prev["end"] = max(prev["end"], s["end"])
            prev["text"] = (prev.get("text", "").rstrip() + " " + s.get("text", "").lstrip()).strip()
        else:
            out.append(s.copy())
    return out

def fix_short_backchannels(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Исправляет типичную ошибку diarization:
    короткие вставки ("че?", "а?", "угу") прилипают к неправильному speaker.
    Мы пытаемся перевесить speaker на ближайшего "другого" спикера по контексту.
    """
    if not segments:
        return segments

    def norm_text(t: str) -> str:
        t = (t or "").lower().strip()
        t = t.replace("?", "").replace("!", "").replace(".", "").replace(",", "")
        return " ".join(t.split())

    out = [s.copy() for s in segments]

    for i in range(len(out)):
        seg = out[i]
        text = norm_text(seg.get("text", ""))
        if not text:
            continue

        dur = float(seg["end"]) - float(seg["start"])
        words = text.split()

        if dur > CFG.role.short_utt_max_dur:
            continue
        if len(words) > CFG.role.short_utt_max_words:
            continue

        # если текст не в списке коротких "перебивок" — не трогаем
        # (либо можно сделать более мягко: если <=2 слов, то тоже трогать)
        if text not in CFG.role.short_utt_texts:
            continue

        cur_spk = seg.get("speaker")

        # смотрим соседей
        prev_seg = out[i - 1] if i - 1 >= 0 else None
        next_seg = out[i + 1] if i + 1 < len(out) else None

        best_other = None

        # кандидат: предыдущий другой спикер, если рядом
        if prev_seg is not None:
            gap = float(seg["start"]) - float(prev_seg["end"])
            if 0 <= gap <= CFG.role.short_utt_max_gap and prev_seg.get("speaker") != cur_spk:
                best_other = prev_seg

        # кандидат: следующий другой спикер, если рядом (и предыдущего нет)
        if best_other is None and next_seg is not None:
            gap = float(next_seg["start"]) - float(seg["end"])
            if 0 <= gap <= CFG.role.short_utt_max_gap and next_seg.get("speaker") != cur_spk:
                best_other = next_seg

        # если нашли "ближайшего другого" — перекидываем speaker
        if best_other is not None:
            seg["speaker"] = best_other.get("speaker")

    return out



def transcribe_with_roles(
    audio_path: str,
    gigaam_model_name: str = "v3_e2e_rnnt",
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Always pipeline:
      input audio -> wav 16k mono (+preprocess) -> diarization -> merge/split turns -> ASR per turn (+split by silence)
      -> merge adjacent utterances -> infer roles from segments (timing + phrases) -> attach roles.
    """
    import gigaam

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio_mono.wav")
        to_wav_16k_mono_preprocessed(audio_path, wav)

        # diarization -> turns
        turns = pyannote_turns(wav, hf_token=hf_token)
        turns = merge_turns(turns)
        turns = split_long_turns(turns)

        if not turns:
            return {
                "mode": "mono_diarization_turns",
                "input": os.path.basename(audio_path),
                "segments": [],
                "role_mapping": {},
                "note": "Нет turns после diarization.",
            }

        # ASR
        model = gigaam.load_model(gigaam_model_name)
        segs = transcribe_turns(model, wav, turns)

        if not segs:
            return {
                "mode": "mono_diarization_turns",
                "input": os.path.basename(audio_path),
                "segments": [],
                "role_mapping": {},
                "note": "Нет текста после ASR.",
            }

        # Build segments (without roles first)
        segments: List[Dict[str, Any]] = []
        for s in segs:
            segments.append({
                "start": round(float(s["start"]), 2),
                "end": round(float(s["end"]), 2),
                "speaker": str(s["speaker"]),
                "text": str(s.get("text", "")).strip(),
            })

        # merge adjacent utterances for better role inference
        segments = merge_utterances_same_speaker(segments)

        segments = fix_short_backchannels(segments) 

        # infer roles using timing + phrase hits (+ optional IVR detection inside)
        role_map = infer_role_map_from_segments(segments)

        # attach roles
        for seg in segments:
            spk = seg.get("speaker")
            seg["role"] = role_map.get(spk, "спикер")

        return {
            "mode": "mono_diarization_turns",
            "input": os.path.basename(audio_path),
            "segments": segments,
            "role_mapping": role_map,
            "note": (
                "Всегда приводим к mono 16k и делаем diarization turns -> ASR по turn (с доп. делением по паузам). "
                "Роли определяются по таймингу и ключевым фразам (плюс эвристика IVR)."
            ),
        }
