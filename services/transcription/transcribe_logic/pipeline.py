# transcribe/pipeline.py
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional
from transcribe_logic.audio_utils import to_wav_16k_mono_preprocessed
from transcribe_logic.diarization_ext import whisper_diarize_via_cli
from transcribe_logic.roles import infer_role_map_from_segments

def _round_segments(segments: List[Dict[str, Any]], ndigits: int = 2) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in segments:
        ss = s.copy()
        if "start" in ss:
            ss["start"] = round(float(ss["start"]), ndigits)
        if "end" in ss:
            ss["end"] = round(float(ss["end"]), ndigits)
        if "text" in ss and ss["text"] is not None:
            ss["text"] = str(ss["text"]).strip()
        out.append(ss)
    out.sort(key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))
    return out


def _merge_adjacent_same_speaker(
    segments: List[Dict[str, Any]],
    max_gap: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Склеивает соседние сегменты одного speaker, если они почти подряд.
    Полезно, чтобы роль-логика работала стабильнее.
    """
    if not segments:
        return []

    segs = sorted(segments, key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))
    out = [segs[0].copy()]

    for s in segs[1:]:
        prev = out[-1]
        same_spk = s.get("speaker") == prev.get("speaker") and s.get("speaker") is not None
        gap = float(s.get("start", 0.0)) - float(prev.get("end", 0.0))

        if same_spk and 0 <= gap <= max_gap:
            prev["end"] = max(float(prev.get("end", 0.0)), float(s.get("end", 0.0)))
            prev_text = str(prev.get("text", "") or "").rstrip()
            cur_text = str(s.get("text", "") or "").lstrip()
            prev["text"] = (prev_text + " " + cur_text).strip()
        else:
            out.append(s.copy())

    return out


def _ensure_speakers_exist(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Некоторые версии whisper-diarization могут писать SRT без явного speaker тега.
    Тогда speaker будет None. Чтобы пайплайн не падал, ставим дефолт.
    """
    for s in segments:
        if not s.get("speaker"):
            s["speaker"] = "SPEAKER_00"
    return segments


def transcribe_with_roles(
    audio_path: str,
    *,
    hf_token: Optional[str] = None,  # не используется здесь, оставлено для совместимости интерфейса
    no_stem: bool = False,
    whisper_repo_dir: str = "/home/dmitrii/whisper-diarization",
    whisper_venv_python: str = "/home/dmitrii/whisper-diarization/.venv/bin/python",
) -> Dict[str, Any]:
    """
    Новый пайплайн (без pyannote и без GigaAM):
      input audio -> mono wav 16k -> запуск внешнего whisper-diarization (diarize.py)
      -> читаем <wav>.srt -> сегменты (start/end/text/(speaker))
      -> (опционально) склеиваем соседние сегменты одного speaker
      -> infer roles (ответчик/звонящий/ivr/спикер) по таймингу + ключевым фразам
    """
    if hf_token:
        # тут не нужно, но пусть не ломает старый вызов
        os.environ["HF_TOKEN"] = hf_token

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio_mono.wav")

        # 1) приводим вход к mono 16k
        to_wav_16k_mono_preprocessed(audio_path, wav)

        # 2) внешний diarizer (в отдельном venv)
        segments = whisper_diarize_via_cli(
            wav,
            repo_dir=whisper_repo_dir,
            venv_python=whisper_venv_python,
            no_stem=no_stem,
            language="ru",
            whisper_model="medium"
        )

        if not segments:
            return {
                "mode": "whisper_diarization_cli",
                "input": os.path.basename(audio_path),
                "segments": [],
                "role_mapping": {},
                "note": "Внешний diarizer не вернул сегментов (проверь .srt output).",
            }

        # 3) гарантируем наличие speaker поля
        segments = _ensure_speakers_exist(segments)

        # 4) слегка приводим сегменты в порядок
        segments = _round_segments(segments, ndigits=2)
        segments = _merge_adjacent_same_speaker(segments, max_gap=0.7)

        # 5) роли по сегментам (тайминг + фразы; IVR может помечаться как 'ivr')
        role_map = infer_role_map_from_segments(segments)

        for seg in segments:
            spk = seg.get("speaker")
            seg["role"] = role_map.get(spk, "спикер")

        return {
            "mode": "whisper_diarization_cli",
            "input": os.path.basename(audio_path),
            "segments": segments,
            "role_mapping": role_map,
            "note": (
                "Делаем mono 16k -> whisper-diarization (Whisper+NeMo) через отдельный venv -> "
                "парсим SRT и назначаем роли по таймингу/фразам."
            ),
        }
