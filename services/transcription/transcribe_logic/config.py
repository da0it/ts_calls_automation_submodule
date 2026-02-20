# transcribe/config.py
from __future__ import annotations
from dataclasses import dataclass, field
import os

@dataclass
class AudioCfg:
    sample_rate: int = 16000
    mono_channels: int = 1

    highpass_hz: int = 80
    lowpass_hz: int = 7900

    cut_codec: str = "pcm_s16le"

@dataclass
class CutCfg:
    pad_seconds: float = 0.0

@dataclass
class SilenceCfg:
    # silencedetect params
    silence_db: float = -35.0
    silence_min_dur: float = 0.25

    # splitting by silence
    split_max_len: float = 4.0
    split_pad: float = 0.05
    edge_guard_seconds: float = 0.3
    min_piece_seconds: float = 0.4

@dataclass
class TurnsCfg:
    merge_max_gap: float = 0.1
    merge_min_dur: float = 0.25

    long_turn_max_len: float = 6
    long_turn_overlap: float = 0.2

    # merge already-transcribed utterances in final timeline
    merge_utt_max_gap: float = 0.7

@dataclass
class ASRCfg:
    # minimal duration to process
    min_dur: float = 0.25

    # ASR inner splitting by silences
    silence_db: float = -35.0
    silence_min_dur: float = 0.15
    piece_max_len: float = 4
    piece_pad: float = 0.05

@dataclass
class PyannoteCfg:
    min_duration_off: float = 0.1
    min_duration_on: float = 0.1
    num_speakers: int = 2

    pipeline_name: str = "pyannote/speaker-diarization-3.1"

@dataclass
class RoleCfg:
    min_role_turn: float = 0.8

    # анализ старта звонка
    intro_window_sec: float = 12.0     # первые N секунд
    intro_reliable_until_sec: float = 1.0  # если первый сегмент позже — старт звонка считаем потерянным
    early_weight: float = 1.0          # вес ранней речи
    first_start_weight: float = 0.4    # штраф за "поздно начал"
    min_confidence: float = 0.25       # если меньше — не уверены
    opening_max_start_sec: float = 25.0
    opening_min_score: float = 1.1
    opening_near_best_delta: float = 0.35
    opening_model_enabled: bool = os.getenv("OPENING_CLASSIFIER_ENABLED", "0").strip() in {"1", "true", "yes", "on"}
    opening_model_path: str = os.getenv("OPENING_CLASSIFIER_PATH", "")
    opening_model_min_proba: float = float(os.getenv("OPENING_CLASSIFIER_MIN_PROBA", "0.55"))

    # ключевые слова (регексы/подстроки; делаем простыми)
    answerer_phrases: tuple[str, ...] = (
        "служба поддержки", "техподдерж", "поддержк",
        "меня зовут", "могу помочь", "чем могу помочь",
        "компания", "оператор", "слушаю вас",
        "сервис", "сервисн", "центр",
        "добрый день", "добрый вечер", "доброе утро",
        "контакт-центр", "колл-центр", "горячая линия",
        "вас приветствует", "приветствую",
        "чем я могу", "как вам помочь", "подскажите номер",
        "у нас нет", "у нас заказ", "у нас начинается",
        "секунд", "минут", "ожидайте", "сейчас проверю", "проверим",
        "оформлен", "подтвержд", "перевожу",
    )
    caller_phrases: tuple[str, ...] = (
        "у меня", "не работает", "проблем", "ошибка",
        "подскаж", "хочу", "мне нужно", "почему",
        "купил", "купила", "номер заказа",
        "доставк", "гарант", "возврат",
        "я звоню", "я хотел", "я хотела",
        "мой заказ", "мой номер", "мой договор",
        "скажите пожалуйста", "можно узнать",
    )
    ivr_phrases: tuple[str, ...] = (
        "нажмите", "для связи", "ваш звонок", "оставайтесь на линии",
        "переводим", "ожидайте", "робот", "автоответчик",
    )
    opening_phrases: tuple[str, ...] = (
        "добрый день", "добрый вечер", "доброе утро",
        "меня зовут", "чем могу помочь", "чем я могу помочь",
        "вас приветствует", "служба поддержки", "оператор",
        "компания", "контакт-центр", "колл-центр",
    )

    short_utt_max_dur: float = 0.55
    short_utt_max_words: int = 2
    short_utt_max_gap: float = 1.0

    short_utt_texts: tuple[str, ...] = (
        "че", "чё", "а", "эм", "угу", "ага", "да", "нет", "что", "чего",
    )


@dataclass
class StereoCfg:
    threshold: float = 0.98
    rms_diff_db: float = 1.0

@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    cut: CutCfg = field(default_factory=CutCfg)
    silence: SilenceCfg = field(default_factory=SilenceCfg)
    turns: TurnsCfg = field(default_factory=TurnsCfg)
    asr: ASRCfg = field(default_factory=ASRCfg)
    pyannote: PyannoteCfg = field(default_factory=PyannoteCfg)
    role: RoleCfg = field(default_factory=RoleCfg)
    stereo: StereoCfg = field(default_factory=StereoCfg)

CFG = Config()
