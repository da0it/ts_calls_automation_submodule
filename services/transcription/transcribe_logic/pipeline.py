# transcribe/pipeline.py
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional
from transcribe_logic.audio_utils import to_wav_16k_mono_preprocessed
from transcribe_logic.roles import infer_role_map_from_segments, assign_roles_to_segments
from transcribe_logic.whisperx_ext import whisperx_diarize_via_cli
from transcribe_logic.whisperx_runtime import whisperx_diarize_inprocess

def _default_whisperx_venv_python() -> str:
    return os.getenv(
        "WHISPERX_VENV_PYTHON",
        os.path.expanduser("~/whisperx_venv/bin/python"),
    )


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


def _smooth_short_speaker_flips(
    segments: List[Dict[str, Any]],
    *,
    max_flip_sec: float = 0.9,
    max_flip_words: int = 3,
) -> List[Dict[str, Any]]:
    """
    Убирает "дребезг" diarization: короткий сегмент другого спикера
    между двумя сегментами одного и того же спикера.
    """
    if len(segments) < 3:
        return segments

    segs = [s.copy() for s in sorted(segments, key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))]
    changed = False

    for i in range(1, len(segs) - 1):
        prev = segs[i - 1]
        cur = segs[i]
        nxt = segs[i + 1]

        prev_spk = prev.get("speaker")
        cur_spk = cur.get("speaker")
        next_spk = nxt.get("speaker")
        if not prev_spk or not cur_spk or not next_spk:
            continue
        if prev_spk != next_spk or cur_spk == prev_spk:
            continue

        s = float(cur.get("start", 0.0))
        e = float(cur.get("end", s))
        dur = max(0.0, e - s)
        words = len(str(cur.get("text", "") or "").split())
        if dur <= max_flip_sec or words <= max_flip_words:
            cur["speaker"] = prev_spk
            changed = True

    return _merge_adjacent_same_speaker(segs, max_gap=0.35) if changed else segs


def _ensure_speakers_exist(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Если diarization выключен или не вернул speaker у части сегментов,
    ставим дефолт, чтобы последующая роль-логика не падала.
    """
    for s in segments:
        if not s.get("speaker"):
            s["speaker"] = "SPEAKER_00"
    return segments


def transcribe_with_roles(
    audio_path: str,
    *,
    hf_token: Optional[str] = None,  # не используется здесь, оставлено для совместимости интерфейса
    whisper_repo_dir: str = os.getenv("WHISPER_REPO_DIR", os.path.expanduser("~/whisper-diarization")),
) -> Dict[str, Any]:
    """
    Пайплайн WhisperX:
      input audio -> mono wav 16k -> whisperx transcribe+align+diarization
      -> infer roles (ответчик/звонящий/ivr/спикер) по таймингу + ключевым фразам
    """
    if hf_token:
        # Поддерживаем прежний интерфейс вызова.
        os.environ["HF_TOKEN"] = hf_token

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio_mono.wav")

        # 1) приводим вход к mono 16k
        to_wav_16k_mono_preprocessed(audio_path, wav)

        diarization_backend = os.getenv("WHISPERX_DIARIZATION_BACKEND", "pyannote").strip().lower()
        common_kwargs = dict(
            model=os.getenv("WHISPERX_MODEL", "large-v3"),
            language=os.getenv("WHISPERX_LANGUAGE", "ru"),
            device=os.getenv("WHISPERX_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPERX_COMPUTE_TYPE", "int8"),
            batch_size=int(os.getenv("WHISPERX_BATCH_SIZE", "4")),
            vad_method=os.getenv("WHISPERX_VAD_METHOD", "silero").strip().lower(),
            diarize=os.getenv("WHISPERX_DIARIZE", "1").strip().lower() in {"1", "true", "yes", "on"},
            diarize_model=os.getenv("WHISPERX_DIARIZE_MODEL", "pyannote/speaker-diarization-3.1"),
            diarization_backend=diarization_backend,
            nemo_repo_dir=os.getenv("WHISPER_REPO_DIR", whisper_repo_dir),
            hf_token=os.getenv("HF_TOKEN"),
        )
        persistent = os.getenv("WHISPERX_PERSISTENT", "1").strip().lower() in {"1", "true", "yes", "on"}
        if persistent:
            segments = whisperx_diarize_inprocess(wav, **common_kwargs)
            mode = "whisperx_persistent"
        else:
            segments = whisperx_diarize_via_cli(
                wav,
                venv_python=_default_whisperx_venv_python(),
                **common_kwargs,
            )
            mode = "whisperx_cli"
        note = (
            f"ASR backend whisperx ({mode}): mono 16k -> whisperx transcribe+align+{diarization_backend} diarization -> role inference."
        )

        if not segments:
            return {
                "mode": mode,
                "input": os.path.basename(audio_path),
                "segments": [],
                "role_mapping": {},
                "note": "Backend returned no segments.",
            }

        # 3) гарантируем наличие speaker поля
        segments = _ensure_speakers_exist(segments)

        # 4) слегка приводим сегменты в порядок
        segments = _round_segments(segments, ndigits=2)
        smooth_segments = os.getenv("WHISPERX_SMOOTH_SEGMENTS", "0").strip().lower() in {"1", "true", "yes", "on"}
        if smooth_segments:
            segments = _smooth_short_speaker_flips(
                segments,
                max_flip_sec=float(os.getenv("WHISPERX_FLIP_MAX_SEC", "0.9")),
                max_flip_words=int(os.getenv("WHISPERX_FLIP_MAX_WORDS", "3")),
            )
            segments = _merge_adjacent_same_speaker(
                segments,
                max_gap=float(os.getenv("WHISPERX_MERGE_GAP_SEC", "0.35")),
            )

        # 5) роли по сегментам (тайминг + фразы; IVR может помечаться как 'ivr')
        role_map = infer_role_map_from_segments(segments)
        role_map = assign_roles_to_segments(segments, role_map)

        return {
            "mode": mode,
            "input": os.path.basename(audio_path),
            "segments": segments,
            "role_mapping": role_map,
            "note": note,
        }
