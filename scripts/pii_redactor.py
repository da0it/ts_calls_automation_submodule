#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Match, Tuple


PHONE_RE = re.compile(
    r"(?:(?:\+7|8)\s*(?:\(\s*\d{3}\s*\)|\d{3})\s*[\- ]?\s*\d{3}\s*[\- ]?\s*\d{2}\s*[\- ]?\s*\d{2})"
)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
TG_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{4,32}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
LONG_ID_RE = re.compile(r"\b\d{6,}\b")
SNILS_RE = re.compile(r"\b\d{3}-\d{3}-\d{3}\s*\d{2}\b")
PASSPORT_RE = re.compile(r"\b\d{4}\s*\d{6}\b")
INN_RE = re.compile(r"\b(?:инн)\s*[:№]?\s*(?:\d{10}|\d{12})\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{2}[./-]\d{2}[./-]\d{4}\b")
NAME_INTRO_RE = re.compile(
    r"(?P<prefix>\b(?:меня зовут|это|с вами говорит|я)\s+)"
    r"(?P<name>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})",
    re.IGNORECASE,
)


@dataclass
class RedactionConfig:
    mode: str = "balanced"  # balanced | strict


def _normalize(kind: str, value: str) -> str:
    v = str(value or "").strip()
    if kind in {"PHONE", "CARD", "LONG_ID", "SNILS", "PASSPORT", "INN"}:
        return re.sub(r"\D+", "", v)
    return v.lower()


def _replace_pattern(
    text: str,
    pattern: re.Pattern[str],
    kind: str,
    counters: Dict[str, int],
    mapping: Dict[str, Dict[str, str]],
) -> Tuple[str, int]:
    replaced = 0

    def _sub(m: Match[str]) -> str:
        nonlocal replaced
        raw = m.group(0)
        key = _normalize(kind, raw)
        if not key:
            return raw
        if kind not in mapping:
            mapping[kind] = {}
        if key not in mapping[kind]:
            counters[kind] = counters.get(kind, 0) + 1
            mapping[kind][key] = f"[{kind}_{counters[kind]}]"
        replaced += 1
        return mapping[kind][key]

    return pattern.sub(_sub, text), replaced


def _replace_name_intro(
    text: str,
    counters: Dict[str, int],
    mapping: Dict[str, Dict[str, str]],
) -> Tuple[str, int]:
    replaced = 0

    def _sub(m: Match[str]) -> str:
        nonlocal replaced
        prefix = m.group("prefix")
        raw = m.group("name")
        key = _normalize("PERSON", raw)
        if not key:
            return m.group(0)
        if "PERSON" not in mapping:
            mapping["PERSON"] = {}
        if key not in mapping["PERSON"]:
            counters["PERSON"] = counters.get("PERSON", 0) + 1
            mapping["PERSON"][key] = f"[PERSON_{counters['PERSON']}]"
        replaced += 1
        return f"{prefix}{mapping['PERSON'][key]}"

    return NAME_INTRO_RE.sub(_sub, text), replaced


def redact_text(
    text: str,
    *,
    config: RedactionConfig,
    counters: Dict[str, int],
    mapping: Dict[str, Dict[str, str]],
) -> Tuple[str, Dict[str, int]]:
    s = str(text or "")
    local: Dict[str, int] = {}

    def _apply(pattern: re.Pattern[str], kind: str) -> None:
        nonlocal s
        s, n = _replace_pattern(s, pattern, kind, counters, mapping)
        if n:
            local[kind] = local.get(kind, 0) + n

    # High precision entities first.
    _apply(EMAIL_RE, "EMAIL")
    _apply(PHONE_RE, "PHONE")
    _apply(URL_RE, "URL")
    _apply(TG_RE, "HANDLE")
    _apply(SNILS_RE, "SNILS")
    _apply(PASSPORT_RE, "PASSPORT")
    _apply(CARD_RE, "CARD")
    _apply(INN_RE, "INN")
    _apply(DATE_RE, "DATE")

    s, n_names = _replace_name_intro(s, counters, mapping)
    if n_names:
        local["PERSON"] = local.get("PERSON", 0) + n_names

    if config.mode == "strict":
        _apply(LONG_ID_RE, "LONG_ID")

    return s, local


def redact_segments(
    segments: List[Dict[str, object]],
    *,
    mode: str = "balanced",
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    cfg = RedactionConfig(mode=str(mode or "balanced").strip().lower())
    counters: Dict[str, int] = {}
    mapping: Dict[str, Dict[str, str]] = {}
    hits: Dict[str, int] = {}
    out: List[Dict[str, object]] = []

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        item = dict(seg)
        text = str(item.get("text") or "")
        redacted, local = redact_text(text, config=cfg, counters=counters, mapping=mapping)
        item["text"] = redacted
        out.append(item)
        for k, v in local.items():
            hits[k] = hits.get(k, 0) + int(v)

    report = {
        "mode": cfg.mode,
        "entities_masked_total": int(sum(hits.values())),
        "entities_masked_by_type": {k: int(v) for k, v in sorted(hits.items())},
        "unique_entities_by_type": {k: len(v) for k, v in sorted(mapping.items())},
    }
    return out, report
