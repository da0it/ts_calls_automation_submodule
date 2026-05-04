from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from transcribe_logic.audio_utils import to_wav_16k_mono_preprocessed
from transcribe_logic.config import get_whisperx_settings
from transcribe_logic.whisperx_runtime import whisperx_transcribe_inprocess

# Функция приводит список сегментов к аккуратному виду. Принимает список словарей сегментов и количество цифр после запятой, которое нужно
# сохранить при округлении. Возвращает новый список словарей с сегментами.
def _round_segments(segments: List[Dict[str, Any]], ndigits: int = 2) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in segments:

        # Для каждого сегмента создается копия. Исходный словарь s не изменяется напрямую
        ss = s.copy()
        
        # Если у сегмента есть поле start оно приводится к float и округляется до ndigits после запятой
        if "start" in ss:
            ss["start"] = round(float(ss["start"]), ndigits)

        # Если у сегмента есть поле end оно приводится к float и округляется до ndigits после запятой
        if "end" in ss:
            ss["end"] = round(float(ss["end"]), ndigits)

        # Если в сегменте содержится текст, он очищается и приводится к строке
        if "text" in ss and ss["text"] is not None:
            ss["text"] = str(ss["text"]).strip()
        out.append(ss)

    # Сортировка итоговых сегментов по времени начала и по времени конца.
    out.sort(key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))
    return out

# Функция _normalize_speaker_label очищает и нормализует метку говорящего,
# преобразуя технические обозначения вроде SPEAKER_00 в вид Speaker 1.
def _normalize_speaker_label(raw_speaker: Any) -> Optional[str]:
    speaker = str(raw_speaker or "").strip()
    if not speaker or speaker.upper() == "UNKNOWN":
        return None

    match = re.match(r"^speaker[\s_-]*(\d+)$", speaker, flags=re.IGNORECASE)
    if match:
        numeric = int(match.group(1))
        if "_" in speaker or speaker.upper().startswith("SPEAKER_"):
            return f"Speaker {numeric + 1}"
        return f"Speaker {numeric}"

    return speaker

# Функция проходит по сегментам и приводит поле speaker к нормальному виду
def _attach_basic_diarization(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    # Пересобираются все сегменты. Берётся исходная speaker-метка и нормализуется.
    for segment in segments:
        normalized_speaker = _normalize_speaker_label(segment.get("speaker"))
        segment["speaker"] = normalized_speaker or ""
        segment.pop("role", None)
    return segments

# Главная функция модуля. Принимает путь к аудиофайлу, необязательный Hugging Face - токен для диаризации. 
def transcribe_with_roles(
    audio_path: str,
    *,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    with tempfile.TemporaryDirectory() as td:

        # Формируется путь к временному WAV-файлу
        wav = os.path.join(td, "audio_mono.wav")

        # Исходный аудиофайл приводится к стандартному формату:
        to_wav_16k_mono_preprocessed(audio_path, wav)

        # Запускается транскрибация через WhisperX runtime. get_whisperx_settings возвращает словарь настроек, ** распаковывает его
        # в именованные аргументы.
        segments = whisperx_transcribe_inprocess(wav, **get_whisperx_settings())

        # Фиксация режима работы
        mode = "whisperx_runtime"

        note = f"ASR backend whisperx ({mode}): mono 16k -> whisperx transcribe+align."

        # Если WhisperX не вернул сегменты, функция возвращает пустой результат
        if not segments:
            return {
                "mode": mode,
                "input": os.path.basename(audio_path),
                "segments": [],
                "note": "Backend returned no segments.",
            }

        # Нормализация speaker-меток и временных меток
        segments = _attach_basic_diarization(segments)
        segments = _round_segments(segments, ndigits=2)
        has_speaker_labels = any(str(segment.get("speaker") or "").strip() for segment in segments)
        if has_speaker_labels:
            note += " Speaker labels are shown only when diarization data is available."
        else:
            note += " No speaker labels were produced by the backend."

        # Итоговый словарь с результатами транскрибации
        return {
            "mode": mode,
            "input": os.path.basename(audio_path),
            "segments": segments,
            "note": note,
        }
