# Модуль реализует вариант транскрибации внутри текущего процесса, без запуска отдельного CLI-worker. То есть Whisper
# импортируется напрямую, модели кэшируются в памяти, а затем используются для распознавания и выравнивания аудио.

from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple

from transcribe_logic.config import resolve_whisperx_device
from transcribe_logic.whisperx_helpers import maybe_assign_diarization_speakers, to_segments

# Создание кэшей моделей
_CACHE_LOCK = threading.RLock()

# Кэш ASR-моделей Whisper. Ключ состоит из: model, device, compute_type, language, vad_method
_ASR_CACHE: Dict[Tuple[str, str, str, str, str], Any] = {}

# Кэш моделей выравнивания. Ключ состоит из: language_code, device
_ALIGN_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}

# Функция импортирует библиотеку whisperx. Если импорт успешен возвращается сам модуль. Если импорт не удался - выбрасывается ошибка
def _load_whisperx():
    try:
        import whisperx
    except Exception as exc:
        raise RuntimeError("Failed to import whisperx in runtime mode.") from exc
    return whisperx

# Функция загружает ASR-модель WhisperX или возвращает её из кэша. Формируется ключ кэша. 
# Например: ("large-v3", "cuda", "int8", "ru", "silero"). Если кэш содержит этот ключ, то модель возвращается из кэша
def _get_asr_model(
    whisperx: Any,
    *,
    model: str,
    device: str,
    compute_type: str,
    language: str,
    vad_method: str,
) -> Any:
    key = (model, device, compute_type, language, vad_method)
    cached = _ASR_CACHE.get(key)
    if cached is not None:
        return cached

    # Параметры для загрузки модели
    kwargs: Dict[str, Any] = {
        "compute_type": compute_type,
        "language": language,
        "vad_method": vad_method,
    }
    try:
        asr_model = whisperx.load_model(model, device, **kwargs)
    except TypeError:
        # Backward compatibility for whisperx versions without vad_method.
        kwargs.pop("vad_method", None)
        asr_model = whisperx.load_model(model, device, **kwargs)

    _ASR_CACHE[key] = asr_model
    return asr_model

# Функция загружает модель выравнивания WhisperX или возвращает её из кэша.
def _get_align_model(whisperx: Any, *, language_code: str, device: str) -> Tuple[Any, Any]:

    # Формирование ключа кэша
    key = (language_code, device)
    cached = _ALIGN_CACHE.get(key)
    if cached is not None:
        return cached

    # Если модель в кэше не найдена, она загружается через WhisperX
    align_model, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=device,
    )

    # После загрузки модель и метаданные сохраняются в кэш и возвращаются
    _ALIGN_CACHE[key] = (align_model, metadata)
    return align_model, metadata

# Функция выполняет предварительную загрузку моделей WhisperX при старте сервиса. Задача функции - заранее загрузить 
# ASR и align модели в кэш, чтобы реальный первый запрос был обработан быстрее.
def warmup_whisperx_runtime(
    *,
    model: str,
    language: str,
    device: str,
    compute_type: str,
    vad_method: str,
) -> None:
    
    # Приведение устройства для работы whisperx к единому виду (ЦПУ или ГПУ)
    device = resolve_whisperx_device(device)

    # Загрузка модели
    whisperx = _load_whisperx()

    # Работа с кэшем выполняется под блокировкой. Это необходимо, чтобы несколько потоков не начали загружать одни и те же модели
    with _CACHE_LOCK:
        _get_asr_model(
            whisperx,
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            vad_method=vad_method,
        )

        # Загрузка модели выравнивания для указанного языка
        _get_align_model(whisperx, language_code=language, device=device)

# Главная функция runtime. Она выполняет транскрибацию аудиофайла через WhisperX внутри текущего процесса.
# Использует кэш ASR- и align-моделей, выполняет распознавание, выравнивание,
# опциональную диаризацию и возвращает сегменты во внутреннем формате.
def whisperx_transcribe_inprocess(
    audio_path: str,
    *,
    model: str = "large-v3",
    language: str = "ru",
    device: str = "auto",
    compute_type: str = "int8",
    batch_size: int = 1,
    vad_method: str = "silero",
) -> List[Dict[str, Any]]:
    device = resolve_whisperx_device(device)
    whisperx = _load_whisperx()

    with _CACHE_LOCK:
        audio = whisperx.load_audio(audio_path)

        asr_model = _get_asr_model(
            whisperx,
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            vad_method=vad_method,
        )
        result = asr_model.transcribe(audio, batch_size=batch_size, language=language)

        align_model, metadata = _get_align_model(
            whisperx,
            language_code=result["language"],
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        result = maybe_assign_diarization_speakers(
            whisperx,
            result,
            audio_path=audio_path,
            device=device,
        )
        return to_segments(result)
