from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from razdel import sentenize, tokenize as razdel_tokenize


logger = logging.getLogger(__name__)

_STANZA_PIPELINE = None

STOP_WORDS = {
    "и","а","но","или","да","нет","это","в","на","к","ко","по","за","для","из","у","мы","вы","он","она","они",
    "я","ты","же","бы","ли","то","вот","там","тут","еще","ещё","уже","ну","ок","ладно","понятно","спасибо"
}

FILLER_PATTERNS = [
    r"^\s*(ал(е|ё)|алло)\s*[.!?]?\s*$",
    r"^\s*(да|да-да|угу|ага|мм+|мгм)\s*[.!?]?\s*$",
    r"^\s*(понятно|ясно|окей|хорошо)\s*[.!?]?\s*$",
    r"^\s*(спасибо)\s*[.!?]?\s*$",
]

SPEAKER_PREFIX_RE = re.compile(r"^\s*(speaker\s*\d+\s*:\s*)", re.IGNORECASE)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{8,}\d)")

NUM_RE = re.compile(r"\b\d+\b")

WS_RE = re.compile(r"\s+")
PUNCT_SPACES_RE = re.compile(r"\s+([,.!?;:])")
MULTI_DOTS_RE = re.compile(r"\.{2,}")

@dataclass
class PreprocessConfig:
    backend: str = "stanza"
    model_text_mode: str = "canonical"
    drop_fillers: bool = True
    drop_stopwords: bool = False

    dedupe: bool = True
    dedupe_window: int = 2

    max_chars: int = 4000
    keep_timestamps: bool = True

    do_tokenize: bool = True
    do_lemmatize: bool = True

    keep_special_tokens: bool = True
    stanza_resources_dir: str = ""


@dataclass
class PreprocessResult:
    canonical_text: str
    model_text: str
    lines: List[str]
    sentences: List[str]
    tokens: List[str]
    lemmas: List[str]
    meta: Dict[str, Any]


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


def is_filler(text: str) -> bool:
    t = text.strip().lower()
    for pat in FILLER_PATTERNS:
        if re.match(pat, t, flags=re.IGNORECASE):
            return True
    return False


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


def _fallback_tokenize(norm_text: str) -> List[str]:
    return re.findall(r"<[a-z_]+>|[a-zа-я0-9]+", norm_text, flags=re.IGNORECASE)


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


def lemmatize(norm_text: str, keep_special_tokens: bool = True) -> Tuple[List[str], List[str]]:
    tokens, lemmas, _backend = lemmatize_with_backend(
        norm_text,
        backend="stanza",
        keep_special_tokens=keep_special_tokens,
        stanza_resources_dir="",
    )
    return tokens, lemmas


def lemmatize_with_backend(
    norm_text: str,
    *,
    backend: str,
    keep_special_tokens: bool,
    stanza_resources_dir: str,
) -> Tuple[List[str], List[str], str]:
    backend_norm = str(backend or "stanza").strip().lower()
    if backend_norm == "none":
        tokens = tokenize_ru(norm_text, keep_special_tokens=keep_special_tokens)
        return tokens, tokens[:], "none"
    tokens, lemmas = _lemmatize_stanza(
        norm_text,
        keep_special_tokens=keep_special_tokens,
        stanza_resources_dir=stanza_resources_dir,
    )
    if tokens:
        return tokens, lemmas, "stanza"
    logger.warning("Stanza lemmatization unavailable, falling back to token text")
    fallback = tokenize_ru(norm_text, keep_special_tokens=keep_special_tokens)
    return fallback, fallback[:], "none"

def _lemmatize_stanza(
    norm_text: str,
    *,
    keep_special_tokens: bool,
    stanza_resources_dir: str,
) -> Tuple[List[str], List[str]]:
    pipeline = _ensure_stanza(stanza_resources_dir)
    if pipeline is None:
        return [], []

    try:
        doc = pipeline(norm_text)
    except Exception as exc:
        logger.warning("Stanza failed while processing text: %s", exc)
        return [], []

    tokens: List[str] = []
    lemmas: List[str] = []
    for sentence in doc.sentences:
        for word in sentence.words:
            tok = str(getattr(word, "text", "") or "").strip()
            lem = str(getattr(word, "lemma", "") or tok).strip()
            if not tok:
                continue
            if not keep_special_tokens and tok.startswith("<") and tok.endswith(">"):
                continue
            tokens.append(tok)
            lemmas.append(lem or tok)
    return tokens, lemmas

def _ensure_stanza(stanza_resources_dir: str):
    global _STANZA_PIPELINE
    if _STANZA_PIPELINE is not None:
        return _STANZA_PIPELINE
    try:
        import stanza
        from stanza.pipeline.core import DownloadMethod
    except Exception as exc:
        logger.warning("Failed to import stanza: %s", exc)
        return None

    kwargs: Dict[str, Any] = {
        "lang": "ru",
        "processors": "tokenize,pos,lemma",
        "download_method": DownloadMethod.REUSE_RESOURCES,
        "use_gpu": False,
    }
    resources_dir = str(stanza_resources_dir or "").strip()
    if resources_dir:
        kwargs["dir"] = resources_dir

    try:
        _STANZA_PIPELINE = stanza.Pipeline(**kwargs)
    except Exception as exc:
        logger.warning("Failed to initialize stanza pipeline: %s", exc)
        return None
    return _STANZA_PIPELINE


def split_sentences(norm_text: str) -> List[str]:
    try:
        return [s.text.strip() for s in sentenize(norm_text) if s.text.strip()]
    except Exception:
        return [norm_text] if norm_text.strip() else []


def drop_stopwords(items: List[str]) -> List[str]:
    return [x for x in items if x not in STOP_WORDS]


def build_model_text(
    canonical_text: str,
    tokens: List[str],
    lemmas: List[str],
    *,
    mode: str,
) -> str:
    mode_norm = str(mode or "canonical").strip().lower()
    if mode_norm == "lemmas":
        return " ".join(lemmas).strip() or canonical_text
    if mode_norm == "tokens":
        return " ".join(tokens).strip() or canonical_text
    if mode_norm in {"plain", "normalized"}:
        plain_text = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", canonical_text, flags=re.MULTILINE)
        plain_text = WS_RE.sub(" ", plain_text).strip()
        return plain_text or canonical_text
    return canonical_text


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
    lemmas: List[str] = []
    actual_backend = "none"

    if cfg.do_lemmatize:
        tokens, lemmas, actual_backend = lemmatize_with_backend(
            no_ts_text,
            backend=cfg.backend,
            keep_special_tokens=cfg.keep_special_tokens,
            stanza_resources_dir=cfg.stanza_resources_dir,
        )
    elif cfg.do_tokenize:
        tokens = tokenize_ru(no_ts_text, keep_special_tokens=cfg.keep_special_tokens)
        lemmas = tokens[:]
        actual_backend = "none"

    if cfg.drop_stopwords and tokens:
        tokens = drop_stopwords(tokens)
        lemmas = drop_stopwords(lemmas)

    model_text = build_model_text(
        canonical_text,
        tokens,
        lemmas,
        mode=cfg.model_text_mode,
    )

    meta: Dict[str, Any] = {
        "raw_kept": raw_kept,
        "raw_dropped": raw_dropped,
        "chars": len(canonical_text),
        "model_text_chars": len(model_text),
        "tokens_n": len(tokens),
        "lemmas_n": len(lemmas),
        "sentences_n": len(sentences),
        "backend_requested": cfg.backend,
        "backend_used": actual_backend,
        "model_text_mode": cfg.model_text_mode,
        "keep_timestamps": cfg.keep_timestamps,
        "dedupe": cfg.dedupe,
        "do_lemmatize": cfg.do_lemmatize,
        "drop_stopwords": cfg.drop_stopwords,
    }

    return PreprocessResult(
        canonical_text=canonical_text,
        model_text=model_text,
        lines=lines,
        sentences=sentences,
        tokens=tokens,
        lemmas=lemmas,
        meta=meta,
    )
