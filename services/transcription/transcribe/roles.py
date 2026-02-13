from __future__ import annotations
from typing import Any, Dict, List, Tuple
import re

from transcribe.config import CFG


def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _count_hits(text: str, phrases: Tuple[str, ...]) -> int:
    t = _norm_text(text)
    hits = 0
    for p in phrases:
        if p and p in t:
            hits += 1
    return hits


def infer_role_map_from_segments(
    segments: List[Dict[str, Any]],
    intro_window_sec: float | None = None,
) -> Dict[str, str]:
    """
    segments: [{"start","end","speaker","text", ...}, ...] (роль ещё не проставлена)
    Возвращает mapping speaker->role: "ответчик"/"звонящий"/"спикер"/"ivr"
    """
    intro_window_sec = CFG.role.intro_window_sec if intro_window_sec is None else intro_window_sec

    # Соберём спикеров
    speakers = sorted({s.get("speaker") for s in segments if s.get("speaker") is not None})
    if not speakers:
        return {}
    if len(speakers) == 1:
        return {speakers[0]: "спикер"}

    # Агрегируем статистики
    stats: Dict[str, Dict[str, float]] = {spk: {
        "total": 0.0,
        "early": 0.0,
        "first": 1e9,
        "ans_hits": 0.0,
        "call_hits": 0.0,
        "ivr_hits": 0.0,
    } for spk in speakers}

    for seg in segments:
        spk = seg.get("speaker")
        if spk not in stats:
            continue
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        dur = max(0.0, e - s)
        stats[spk]["total"] += dur
        stats[spk]["first"] = min(stats[spk]["first"], s)

        # early overlap with [0, intro_window_sec]
        early_end = min(e, intro_window_sec)
        if s < intro_window_sec and early_end > s:
            stats[spk]["early"] += (early_end - s)

        text = seg.get("text", "") or ""
        stats[spk]["ans_hits"] += _count_hits(text, CFG.role.answerer_phrases)
        stats[spk]["call_hits"] += _count_hits(text, CFG.role.caller_phrases)
        stats[spk]["ivr_hits"] += _count_hits(text, CFG.role.ivr_phrases)

    # 1) Выделим возможный IVR: много ivr_hits и почти всё в начале
    # (делаем мягко: если явно IVR — пометим)
    role_map: Dict[str, str] = {}
    for spk in speakers:
        ivr_score = stats[spk]["ivr_hits"]
        if ivr_score >= 2 and stats[spk]["early"] > 0.0 and stats[spk]["total"] < 30.0:
            role_map[spk] = "ivr"

    # Кандидаты (не ivr)
    cand = [spk for spk in speakers if role_map.get(spk) != "ivr"]
    if len(cand) == 1:
        # один реальный говорящий + ivr
        role_map[cand[0]] = "спикер"
        return role_map

    # Нормировки
    max_early = max(stats[spk]["early"] for spk in cand) or 1.0
    min_first = min(stats[spk]["first"] for spk in cand)
    max_first = max(stats[spk]["first"] for spk in cand)
    first_span = (max_first - min_first) or 1.0

    # 2) Скорая функция “ответчик”
    def answerer_score(spk: str) -> float:
        early = stats[spk]["early"] / max_early
        # чем позже начал — тем хуже (0..1)
        late = (stats[spk]["first"] - min_first) / first_span
        ans = stats[spk]["ans_hits"]
        call = stats[spk]["call_hits"]
        # простая формула
        return (
            CFG.role.early_weight * early
            - CFG.role.first_start_weight * late
            + 0.9 * ans
            - 0.6 * call
        )

    scored = sorted(((answerer_score(spk), spk) for spk in cand), reverse=True)
    best_score, best_spk = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1e9

    # уверенность как разница
    confidence = best_score - second_score

    # 3) Назначаем роли
    # Если не уверены — fallback: кто раньше начал говорить = ответчик
    if confidence < CFG.role.min_confidence:
        best_spk = min(cand, key=lambda spk: stats[spk]["first"])

    role_map.setdefault(best_spk, "ответчик")

    # “звонящий” = лучший по caller_score среди остальных (или просто другой)
    others = [spk for spk in cand if spk != best_spk]
    if others:
        caller_spk = max(others, key=lambda spk: (stats[spk]["call_hits"], stats[spk]["total"]))
        role_map.setdefault(caller_spk, "звонящий")
        for spk in others:
            if spk != caller_spk:
                role_map.setdefault(spk, "спикер")

    return role_map
