from __future__ import annotations
from typing import Any, Dict, List, Tuple
import re

from transcribe_logic.config import CFG


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
        "id_hits": 0.0,
        "question_hits": 0.0,
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
        # Номер заказа/договора/счёта — сильный сигнал звонящего.
        stats[spk]["id_hits"] += 1.0 if re.search(r"\b\d{5,}\b", text) else 0.0
        stats[spk]["question_hits"] += text.count("?")

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
    intro_is_reliable = min_first <= CFG.role.intro_reliable_until_sec

    # 2) Скорая функция “ответчик”
    def answerer_score(spk: str) -> float:
        early = stats[spk]["early"] / max_early
        # чем позже начал — тем хуже (0..1)
        late = (stats[spk]["first"] - min_first) / first_span
        ans = stats[spk]["ans_hits"]
        call = stats[spk]["call_hits"]
        q = stats[spk]["question_hits"]
        ids = stats[spk]["id_hits"]
        early_weight = CFG.role.early_weight if intro_is_reliable else 0.15
        first_start_weight = CFG.role.first_start_weight if intro_is_reliable else 0.05
        # простая формула
        return (
            early_weight * early
            - first_start_weight * late
            + 0.9 * ans
            - 0.9 * call
            - 0.2 * q
            - 0.5 * ids
        )

    def caller_score(spk: str) -> float:
        # caller-речь обычно содержит запрос/проблему и доменные слова ("заказ", "купила", ...)
        return (
            stats[spk]["call_hits"]
            - 0.6 * stats[spk]["ans_hits"]
            + 0.5 * stats[spk]["question_hits"]
            + 0.8 * stats[spk]["id_hits"]
        )

    caller_scored = sorted(((caller_score(spk), spk) for spk in cand), reverse=True)
    strong_caller = False
    if len(caller_scored) >= 2:
        strong_caller = (caller_scored[0][0] - caller_scored[1][0]) >= 0.8

    if strong_caller:
        caller_spk = caller_scored[0][1]
        role_map.setdefault(caller_spk, "звонящий")
        rest = [spk for spk in cand if spk != caller_spk]
        if rest:
            answerer_spk = max(rest, key=answerer_score)
            role_map.setdefault(answerer_spk, "ответчик")
            for spk in rest:
                if spk != answerer_spk:
                    role_map.setdefault(spk, "спикер")
        return role_map

    scored = sorted(((answerer_score(spk), spk) for spk in cand), reverse=True)
    best_score, best_spk = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1e9

    # уверенность как разница
    confidence = best_score - second_score

    # 3) Назначаем роли
    # Если не уверены — fallback: кто раньше начал говорить = ответчик
    if confidence < CFG.role.min_confidence:
        # при плохой уверенности предпочтём контекст звонящего/ответчика, а не "кто заговорил первым"
        best_spk = max(cand, key=lambda spk: (stats[spk]["ans_hits"] - stats[spk]["call_hits"], -stats[spk]["first"]))

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


def _segment_role_scores(text: str) -> Tuple[float, float, float]:
    t = _norm_text(text)
    caller = float(_count_hits(t, CFG.role.caller_phrases))
    answerer = float(_count_hits(t, CFG.role.answerer_phrases))
    ivr = float(_count_hits(t, CFG.role.ivr_phrases))

    if re.search(r"\b\d{5,}\b", t):
        caller += 1.0
    caller += min(2.0, t.count("?") * 0.5)

    # Operator/service wording boost.
    if re.search(r"\b(сервис|служба|оператор|центр|заявк[аи]|слушаю)\b", t):
        answerer += 0.8
    # Client/request wording boost.
    if re.search(r"\b(заказ|купил|купила|установк|доставк|гарант|номер)\b", t):
        caller += 0.8

    return caller, answerer, ivr


def assign_roles_to_segments(
    segments: List[Dict[str, Any]],
    speaker_role_map: Dict[str, str],
) -> Dict[str, str]:
    """
    Назначает role в каждом сегменте с учетом speaker role map,
    но допускает override на уровне конкретной реплики.
    Возвращает обновленный speaker->role map (по длительности сегментов).
    """
    per_speaker_dur: Dict[str, Dict[str, float]] = {}

    for seg in segments:
        spk = seg.get("speaker")
        base_role = speaker_role_map.get(spk, "спикер")
        text = str(seg.get("text", "") or "")
        caller_score, answerer_score, ivr_score = _segment_role_scores(text)

        role = base_role
        if ivr_score >= 2.0:
            role = "ivr"
        elif caller_score - answerer_score >= 1.2:
            role = "звонящий"
        elif answerer_score - caller_score >= 1.2:
            role = "ответчик"

        seg["role"] = role

        if spk is None:
            continue
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        dur = max(0.0, e - s)
        if spk not in per_speaker_dur:
            per_speaker_dur[spk] = {}
        per_speaker_dur[spk][role] = per_speaker_dur[spk].get(role, 0.0) + dur

    updated_map: Dict[str, str] = {}
    for spk, role_dur in per_speaker_dur.items():
        if not role_dur:
            updated_map[spk] = speaker_role_map.get(spk, "спикер")
            continue
        best_role = max(
            role_dur.items(),
            key=lambda item: (item[1], 1 if item[0] == speaker_role_map.get(spk) else 0),
        )[0]
        updated_map[spk] = best_role

    for spk, role in speaker_role_map.items():
        updated_map.setdefault(spk, role)

    return updated_map
