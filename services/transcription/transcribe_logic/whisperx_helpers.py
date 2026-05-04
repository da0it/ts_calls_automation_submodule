from __future__ import annotations

import os
from typing import Any, Dict, List

# Если WhisperX не вернул говорящего, в сегменте будет:
UNKNOWN_SPEAKER = ""

# Функция читает переменную окружения и преобразует её в bool.
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# Функция служит для добавления информации о говорящих к результату транскрибации. Функция работает только если диаризация включена
# и выполнены все необходимые условия. Возвращает обновленный словарь сегментов.
def maybe_assign_diarization_speakers(
    whisperx: Any,
    result: Dict[str, Any],
    *,
    audio_path: str,
    device: str,
) -> Dict[str, Any]:
    if not _env_bool("WHISPERX_ENABLE_DIARIZATION", False):
        return result

    # Диаризация часто требует доступ к моделям Hugging Face, для чего необходим токен. Код проверяет сразу несколько возможных переменных
    # окружения.
    hf_token = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_TOKEN", "").strip()
        or os.getenv("HF_HUB_TOKEN", "").strip()
    )
    if not hf_token:
        return result

    # Проверка, есть ли в установленной версии библиотеки whisperx нужные объекты
    diarization_pipeline = getattr(whisperx, "DiarizationPipeline", None)
    assign_word_speakers = getattr(whisperx, "assign_word_speakers", None)
    if diarization_pipeline is None or assign_word_speakers is None:
        return result

    min_speakers = max(1, int(os.getenv("WHISPERX_MIN_SPEAKERS", "2")))
    max_speakers = max(min_speakers, int(os.getenv("WHISPERX_MAX_SPEAKERS", str(min_speakers))))

    diarize_model = diarization_pipeline(
        use_auth_token=hf_token,
        device=device,
    )
    diarize_segments = diarize_model(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    return assign_word_speakers(diarize_segments, result)

# Функция приводит результат WhisperX к внутреннему формату сегментов. Принимает словарь result, внутри которого ожидается поле
# "segments". Возвращается список словарей.
def to_segments(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seg in result.get("segments", []):
        text = str(seg.get("text", "") or "").strip()
        if not text:
            continue
        out.append(
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "speaker": str(seg.get("speaker", UNKNOWN_SPEAKER) or "").strip(),
                "text": text,
            }
        )

    return out
