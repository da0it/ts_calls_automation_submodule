from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from razdel import sentenize, tokenize as razdel_tokenize

# Список частых служебных слов, они могут удаляться при включенном параметре drop_stopwords=True в конфигурационном файле
STOP_WORDS = {
    "и","а","но","или","да","нет","это","в","на","к","ко","по","за","для","из","у","мы","вы","он","она","они",
    "я","ты","же","бы","ли","то","вот","там","тут","еще","ещё","уже","ну","ок","ладно","понятно","спасибо"
}

# Регулярные выражения для удаления коротких бессодержательных реплик
FILLER_PATTERNS = [
    r"^\s*(ал(е|ё)|алло)\s*[.!?]?\s*$",
    r"^\s*(да|да-да|угу|ага|мм+|мгм)\s*[.!?]?\s*$",
    r"^\s*(понятно|ясно|окей|хорошо)\s*[.!?]?\s*$",
    r"^\s*(спасибо)\s*[.!?]?\s*$",
]

# Регулярное выражение для удаления префикса Speaker при работе с диаризацией
SPEAKER_PREFIX_RE = re.compile(r"^\s*(speaker\s*\d+\s*:\s*)", re.IGNORECASE)

# Маскирование Email, чисел и телефонных номеров
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{8,}\d)")
NUM_RE = re.compile(r"\b\d+\b")

WS_RE = re.compile(r"\s+")
PUNCT_SPACES_RE = re.compile(r"\s+([,.!?;:])")
MULTI_DOTS_RE = re.compile(r"\.{2,}")

# Класс данных, отвечающий за настройки предобработки текста
@dataclass
class PreprocessConfig:
    model_text_mode: str = "canonical"
    drop_fillers: bool = True
    drop_stopwords: bool = False

    dedupe: bool = True
    dedupe_window: int = 2

    max_chars: int = 4000
    keep_timestamps: bool = True

    do_tokenize: bool = True

    keep_special_tokens: bool = True


# Класс данных для хранения результата предобработки текста.
# canonical_text — основная нормализованная версия текста звонка;
# model_text — текст, который фактически передается на вход модели.
@dataclass
class PreprocessResult:
    canonical_text: str
    model_text: str
    lines: List[str]
    sentences: List[str]
    tokens: List[str]
    meta: Dict[str, Any]

# Функция предназначена для базовой очистки текста: убирает пробелы по краям, заменяет ё на е, приводит
# текст к нижнему регистру и т.д.
def normalize_text(text: str) -> str:
    t = text.strip()
    t = SPEAKER_PREFIX_RE.sub("", t)
    t = t.replace("Ё", "Е").replace("ё", "е")
    t = t.lower()

    t = EMAIL_RE.sub(" <email> ", t)
    t = PHONE_RE.sub(" <phone> ", t)
    t = NUM_RE.sub(" <num> ", t)

    t = MULTI_DOTS_RE.sub(".", t)
    t = WS_RE.sub(" ", t)
    t = PUNCT_SPACES_RE.sub(r"\1", t)

    return t.strip()

# Функция предназначена для проверки содержательности реплики. Проверка идет по предварительно определенному
# массиву FILLER_PATTERNS
def is_filler(text: str) -> bool:
    t = text.strip().lower()
    for pat in FILLER_PATTERNS:
        if re.match(pat, t, flags=re.IGNORECASE):
            return True
    return False

# Функция удаляет близкие повторы фраз, например, "не могу войти не могу войти" - вторая строка будет удалена.
# Функция смотрит только на ближайшее окно, а не на весь транскрипт
def dedupe_nearby(texts: List[str], window: int = 2) -> List[str]:
    out: List[str] = []
    recent: List[str] = []
    for t in texts:
        if t in recent:
            continue
        out.append(t)
        recent.append(t)
        if len(recent) > window:
            recent.pop(0)
    return out

# Функция используется при ошибке в tokenize_ru
def _fallback_tokenize(norm_text: str) -> List[str]:
    return re.findall(r"<[a-z_]+>|[a-zа-я0-9]+", norm_text, flags=re.IGNORECASE)

# Функция используется для токенизации русского текста через библиотеку razdel
def tokenize_ru(norm_text: str, keep_special_tokens: bool = True) -> List[str]:
    try:
        toks = [t.text for t in razdel_tokenize(norm_text)]
    except Exception:
        toks = _fallback_tokenize(norm_text)

    out: List[str] = []
    for tok in toks:
        tok = tok.strip()
        if not tok:
            continue
        if re.fullmatch(r"\W+", tok):
            continue
        if not keep_special_tokens and tok.startswith("<") and tok.endswith(">"):
            continue
        out.append(tok)
    return out

# Разбивает текст на предложения используя библиотечный метод razdel.sentenize. Если не получилось, возвращает
# весь текст как одно предложение
def split_sentences(norm_text: str) -> List[str]:
    try:
        return [s.text.strip() for s in sentenize(norm_text) if s.text.strip()]
    except Exception:
        return [norm_text] if norm_text.strip() else []

# Функция удаляет стоп-слова из списка токенов, по умолчанию в конфиге отключено.
def drop_stopwords(items: List[str]) -> List[str]:
    return [x for x in items if x not in STOP_WORDS]

# Функция формирует текст для входа в модель в одном из режимов: 
# Canonical: Возвращает текст с временными метками и строками
# Token: Возвращает токены (слова) через пробел
# Plain или Normalized: удаляет временные метки и собирает текст в одну строку
def build_model_text(
    canonical_text: str,
    tokens: List[str],
    *,
    mode: str,
) -> str:
    mode_norm = str(mode or "canonical").strip().lower()
    if mode_norm == "tokens":
        return " ".join(tokens).strip() or canonical_text
    if mode_norm in {"plain", "normalized"}:
        plain_text = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", canonical_text, flags=re.MULTILINE)
        plain_text = WS_RE.sub(" ", plain_text).strip()
        return plain_text or canonical_text
    return canonical_text

# Главная функция предобработки.
# Принимает сегменты звонка, выполняет фильтрацию, нормализацию и агрегацию текста,
# формирует каноническое представление и текст для модели.
# Дополнительно вычисляет предложения, токены и метаданные обработки.
# Возвращает объект PreprocessResult.
def build_canonical(
    segments: List[Tuple[float, str, Optional[str]]],
    cfg: Optional[PreprocessConfig] = None,
) -> PreprocessResult:
    cfg = cfg or PreprocessConfig()

    lines: List[str] = []
    raw_kept = 0
    raw_dropped = 0

    for start, raw, _role in segments:
        if not raw:
            raw_dropped += 1
            continue
        if cfg.drop_fillers and is_filler(raw):
            raw_dropped += 1
            continue

        norm = normalize_text(raw)
        if not norm:
            raw_dropped += 1
            continue

        if cfg.keep_timestamps:
            mm = int(max(0, start)) // 60
            ss = int(max(0, start)) % 60
            lines.append(f"[{mm:02d}:{ss:02d}] {norm}")
        else:
            lines.append(norm)

        raw_kept += 1

    if cfg.dedupe:
        lines = dedupe_nearby(lines, window=cfg.dedupe_window)

    canonical_text = "\n".join(lines)
    if len(canonical_text) > cfg.max_chars:
        canonical_text = canonical_text[: cfg.max_chars]

    no_ts_text = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", canonical_text, flags=re.MULTILINE)
    sentences = split_sentences(no_ts_text)

    tokens: List[str] = []
    if cfg.do_tokenize:
        tokens = tokenize_ru(no_ts_text, keep_special_tokens=cfg.keep_special_tokens)

    if cfg.drop_stopwords and tokens:
        tokens = drop_stopwords(tokens)

    model_text = build_model_text(
        canonical_text,
        tokens,
        mode=cfg.model_text_mode,
    )

    meta: Dict[str, Any] = {
        "raw_kept": raw_kept,
        "raw_dropped": raw_dropped,
        "chars": len(canonical_text),
        "model_text_chars": len(model_text),
        "tokens_n": len(tokens),
        "sentences_n": len(sentences),
        "model_text_mode": cfg.model_text_mode,
        "keep_timestamps": cfg.keep_timestamps,
        "dedupe": cfg.dedupe,
        "do_tokenize": cfg.do_tokenize,
        "drop_stopwords": cfg.drop_stopwords,
    }

    return PreprocessResult(
        canonical_text=canonical_text,
        model_text=model_text,
        lines=lines,
        sentences=sentences,
        tokens=tokens,
        meta=meta,
    )
