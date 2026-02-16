# transcribe/pipeline.py
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional
from transcribe_logic.audio_utils import to_wav_16k_mono_preprocessed
from transcribe_logic.diarization_ext import whisper_diarize_via_cli
from transcribe_logic.roles import infer_role_map_from_segments, assign_roles_to_segments
from transcribe_logic.whisperx_ext import whisperx_diarize_via_cli

def _default_whisper_venv_python(repo_dir: str) -> str:
    candidates = [
        os.path.join(repo_dir, ".venv", "bin", "python"),
        os.path.join(repo_dir, "whisper_venv", "bin", "python"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _default_whisperx_venv_python() -> str:
    return os.getenv(
        "WHISPERX_VENV_PYTHON",
        os.getenv("WHISPER_VENV_PYTHON", os.path.expanduser("~/whisper-diarization/whisper_venv/bin/python")),
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
    whisper_repo_dir: str = os.getenv("WHISPER_REPO_DIR", os.path.expanduser("~/whisper-diarization")),
    whisper_venv_python: Optional[str] = None,
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
    if whisper_venv_python is None:
        whisper_venv_python = os.getenv(
            "WHISPER_VENV_PYTHON",
            _default_whisper_venv_python(whisper_repo_dir),
        )

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio_mono.wav")

        # 1) приводим вход к mono 16k
        to_wav_16k_mono_preprocessed(audio_path, wav)

        backend = os.getenv("ASR_BACKEND", "faster").strip().lower()
        if backend == "whisperx":
            diarization_backend = os.getenv("WHISPERX_DIARIZATION_BACKEND", "pyannote").strip().lower()
            segments = whisperx_diarize_via_cli(
                wav,
                venv_python=_default_whisperx_venv_python(),
                model=os.getenv("WHISPERX_MODEL", "large-v3"),
                language=os.getenv("WHISPERX_LANGUAGE", os.getenv("WHISPER_LANGUAGE", "ru")),
                device=os.getenv("WHISPERX_DEVICE", "cpu"),
                compute_type=os.getenv("WHISPERX_COMPUTE_TYPE", "int8"),
                batch_size=int(os.getenv("WHISPERX_BATCH_SIZE", "4")),
                diarize=os.getenv("WHISPERX_DIARIZE", "1").strip().lower() in {"1", "true", "yes", "on"},
                diarize_model=os.getenv("WHISPERX_DIARIZE_MODEL", "pyannote/speaker-diarization-3.1"),
                diarization_backend=diarization_backend,
                nemo_repo_dir=os.getenv("WHISPER_REPO_DIR", whisper_repo_dir),
                hf_token=os.getenv("HF_TOKEN"),
            )
            mode = "whisperx"
            note = (
                f"ASR backend whisperx: mono 16k -> whisperx transcribe+align+{diarization_backend} diarization -> role inference."
            )
        else:
            # 2) внешний diarizer (в отдельном venv)
            segments = whisper_diarize_via_cli(
                wav,
                repo_dir=whisper_repo_dir,
                venv_python=whisper_venv_python,
                no_stem=no_stem,
                language=os.getenv("WHISPER_LANGUAGE", "ru"),
                whisper_model=os.getenv("WHISPER_MODEL", "small"),
                batch_size=int(os.getenv("WHISPER_BATCH_SIZE", "8")),
                device=os.getenv("WHISPER_DEVICE") or None,
                suppress_numerals=os.getenv("WHISPER_SUPPRESS_NUMERALS", "0").strip().lower() in {"1", "true", "yes", "on"},
            )
            mode = "whisper_diarization_cli"
            note = (
                "ASR backend faster: mono 16k -> whisper-diarization (Whisper+NeMo) via separate venv -> "
                "parse SRT and infer roles."
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
        # merge только для faster-whisper бэкенда (whisperx уже нарезает по словам)
        if backend == "whisperx":
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
        else:
            segments = _merge_adjacent_same_speaker(segments, max_gap=0.7)

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
