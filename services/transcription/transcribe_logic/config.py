# transcribe/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict

AUTO_DEVICE = "auto"

# Класс хранит параметры аудиоформата
@dataclass
class AudioCfg:

    # Частота дискретизации в Гц
    sample_rate: int = 16000

    # Количество каналов после предобработки
    mono_channels: int = 1

    # Нижняя граница частотного фильтра
    highpass_hz: int = 80

    # Верхняя граница частотног фильтра
    lowpass_hz: int = 7900

    # pcm_s16le — это несжатый PCM 16-bit little-endian. Такой формат хорошо подходит для последующей обработки и распознавания.
    cut_codec: str = "pcm_s16le"

# Класс хранит настройки нарезки аудио
@dataclass
class CutCfg:

    # Дополнительный отступ вокруг вырезаемого фрагмента
    pad_seconds: float = 0.0

# Класс хранит настройки детекции тишины и разбиения речи на части
@dataclass
class SilenceCfg:
    
    # Порог громкости для определения тишины
    silence_db: float = -35.0

    # Минимальная длительность тишины
    silence_min_dur: float = 0.25

    # Максимальная длина речевого фрагмента при разбиении
    split_max_len: float = 4.0

    # Небольшой запас при вырезании сегмента
    split_pad: float = 0.05

    # Защитный отступ от краев сегмента
    edge_guard_seconds: float = 0.3

    # Минимальная длительность получившегося фрагмента
    min_piece_seconds: float = 0.4

# Класс отвечает за параметры обработки речевых реплик
@dataclass
class TurnsCfg:

    # Максимальный разрыв между соседними короткими сегментами, при котором можно их объединить
    merge_max_gap: float = 0.1

    # Минимальная длительность сегмента при объединении
    merge_min_dur: float = 0.25

    # Максимальная длина длинной реплики
    long_turn_max_len: float = 6

    # Перекрытие между частями длинной реплики (помогает не потерять слова на границе разреза)
    long_turn_overlap: float = 0.2

    # Склейка транскрибированной речи при малом времени между сегментами
    merge_utt_max_gap: float = 0.7

# Класс хранит настройки ASR
@dataclass
class ASRCfg:
    # Минимальная длительность аудиофайла (в минутах), для начала распознавания
    min_dur: float = 0.25

# Класс хранит параметры проверки стерео
@dataclass
class StereoCfg:

    # Максимальная разница RMS-громкости между каналами, при которой стерео считается ложным или почти одинаковым
    rms_diff_db: float = 1.0

# Главный класс конфигурации, объединяет все группы настроек в единый объект конфигурации
@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    cut: CutCfg = field(default_factory=CutCfg)
    silence: SilenceCfg = field(default_factory=SilenceCfg)
    turns: TurnsCfg = field(default_factory=TurnsCfg)
    asr: ASRCfg = field(default_factory=ASRCfg)
    stereo: StereoCfg = field(default_factory=StereoCfg)

# Функция нормализует название устройства для применения whisper. Принимает название строкой. Возвращает нормализованное название
def normalize_whisperx_device(device: str | None) -> str:
    value = (device or "").strip().lower()
    if not value:
        return AUTO_DEVICE
    if value == "gpu":
        return "cuda"
    return value

# Функция отвечает за проверку доступности CUDA. @lru_cache(maxsize=1) означает, что результат будет сохранен и повторные вызовы
# не будут заново импортировать torch и проверять GPU.
@lru_cache(maxsize=1)
def _cuda_is_available() -> bool:
    try:
        import torch
    except Exception:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False

# Функция определяет итоговое устройство для применение whisper
def resolve_whisperx_device(device: str | None = None) -> str:

    # Название устройства нормализуется функцией normalize_whisperx_device
    device = normalize_whisperx_device(device)
    if device != AUTO_DEVICE:
        return device
    return "cuda" if _cuda_is_available() else "cpu"

# Функция берет устройство из переменной окружения. Если не задана, используется AUTO_DEVICE
def get_whisperx_device_from_env(name: str = "WHISPERX_DEVICE") -> str:
    return resolve_whisperx_device(os.getenv(name, AUTO_DEVICE))

# Функция собирает настройки Whisper из переменных окружения. Возвращает словарь конфигурации.
def get_whisperx_settings() -> Dict[str, Any]:
    vad_method = os.getenv("WHISPERX_VAD_METHOD", "silero").strip().lower()
    if not vad_method:
        vad_method = "silero"

    return {
        "model": os.getenv("WHISPERX_MODEL", "large-v3"),
        "language": os.getenv("WHISPERX_LANGUAGE", "ru"),
        "device": get_whisperx_device_from_env(),
        "compute_type": os.getenv("WHISPERX_COMPUTE_TYPE", "int8"),
        "batch_size": int(os.getenv("WHISPERX_BATCH_SIZE", "1")),
        "vad_method": vad_method,
    }

# Создается глобальный объект конфигурации. После этого другие модели могут его импортировать, например, так: from .config import CFG
CFG = Config()