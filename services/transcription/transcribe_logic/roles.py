from __future__ import annotations
from typing import Any, Dict, List, Tuple
import re

from transcribe_logic.config import CFG

ROLE_CALLER = "звонящий"
ROLE_ANSWERER = "ответчик"
ROLE_UNKNOWN = "не определено"
_ALLOWED_ROLES = {ROLE_CALLER, ROLE_ANSWERER, ROLE_UNKNOWN}


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


def _sanitize_role(role: str | None) -> str:
    if role in _ALLOWED_ROLES:
        return str(role)
    return ROLE_UNKNOWN


def _segment_duration(seg: Dict[str, Any]) -> float:
    s = float(seg.get("start", 0.0))
    e = float(seg.get("end", s))
    return max(0.0, e - s)


def _is_short_utterance(seg: Dict[str, Any]) -> bool:
    text = _norm_text(str(seg.get("text", "") or ""))
    if not text:
        return True
    words = [w for w in text.split(" ") if w]
    dur = _segment_duration(seg)
    if dur <= CFG.role.short_utt_max_dur and len(words) <= CFG.role.short_utt_max_words:
        return True
    if text in CFG.role.short_utt_texts:
        return True
    return False


def _opening_role_hint(text: str) -> Tuple[float, float]:
    """
    Возвращает (answerer_bonus, caller_bonus) для стартовых фраз.
    """
    t = _norm_text(text)
    if not t:
        return 0.0, 0.0

    answerer_bonus = 0.0
    caller_bonus = 0.0

    if re.search(r"\b(здравств|добрый день|добрый вечер|доброе утро)\b", t):
        answerer_bonus += 0.3
    if re.search(r"\b(чем .*могу .*помочь|слушаю вас|как .*помочь)\b", t):
        answerer_bonus += 1.2
    if re.search(r"\b(компания|служба|поддержк|оператор)\b", t):
        answerer_bonus += 0.5

    if re.search(r"\b(я звоню|меня зовут|представляю|интересует|подскажите)\b", t):
        caller_bonus += 0.6
    if re.search(r"\b(вопрос|нужно|хочу|проблем|ошибк|заказ)\b", t):
        caller_bonus += 0.4

    return answerer_bonus, caller_bonus


def _opening_sentence_score(text: str) -> float:
    t = _norm_text(text)
    if not t:
        return 0.0

    score = float(_count_hits(t, CFG.role.opening_phrases))
    if re.search(r"\b(меня зовут|вас приветствует)\b", t):
        score += 0.9
    if re.search(r"\b(чем .*могу .*помочь|как .*могу .*помочь)\b", t):
        score += 1.1
    if re.search(r"\b(из компании|компания|служба поддержки|оператор)\b", t):
        score += 0.6
    if re.search(r"\b(это .* из|беспокоит .* из)\b", t):
        score += 0.4
    return score


def _detect_opening_speakers(
    segments: List[Dict[str, Any]],
    speakers: List[Any],
) -> List[Any]:
    if not segments or not speakers:
        return []

    max_start = float(max(1.0, CFG.role.opening_max_start_sec))
    opening_stats: Dict[Any, Dict[str, float]] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if spk not in speakers:
            continue
        start = float(seg.get("start", 0.0))
        if start > max_start:
            continue
        if _is_short_utterance(seg):
            continue
        text = str(seg.get("text", "") or "")
        score = _opening_sentence_score(text)
        if score <= 0.0:
            continue

        bucket = opening_stats.setdefault(spk, {"best": 0.0, "sum": 0.0, "count": 0.0, "first": 1e9})
        bucket["best"] = max(bucket["best"], score)
        bucket["sum"] += score
        bucket["count"] += 1.0
        bucket["first"] = min(bucket["first"], start)

    if not opening_stats:
        return []

    scored: List[Tuple[float, Any]] = []
    for spk, st in opening_stats.items():
        # "best opening sentence" — главный сигнал; sum/count — вспомогательные.
        composite = st["best"] + 0.2 * st["sum"] + 0.15 * st["count"]
        scored.append((composite, spk))
    scored.sort(reverse=True)

    best_score = scored[0][0]
    if best_score < float(CFG.role.opening_min_score):
        return []

    threshold = max(
        float(CFG.role.opening_min_score),
        best_score - float(CFG.role.opening_near_best_delta),
    )
    return [spk for score, spk in scored if score >= threshold]


def _speaker_map_from_segments(
    segments: List[Dict[str, Any]],
    fallback_map: Dict[str, str],
) -> Dict[str, str]:
    per_speaker_dur: Dict[str, Dict[str, float]] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if spk is None:
            continue
        role = _sanitize_role(seg.get("role"))
        dur = _segment_duration(seg)
        bucket = per_speaker_dur.setdefault(spk, {})
        bucket[role] = bucket.get(role, 0.0) + dur

    updated_map: Dict[str, str] = {}
    for spk, role_dur in per_speaker_dur.items():
        if not role_dur:
            updated_map[spk] = _sanitize_role(fallback_map.get(spk))
            continue
        base_role = _sanitize_role(fallback_map.get(spk))
        best_role = max(
            role_dur.items(),
            key=lambda item: (item[1], 1 if item[0] == base_role else 0),
        )[0]
        updated_map[spk] = _sanitize_role(best_role)

    for spk, role in fallback_map.items():
        updated_map.setdefault(spk, _sanitize_role(role))
    return updated_map


def infer_role_map_from_segments(
    segments: List[Dict[str, Any]],
    intro_window_sec: float | None = None,
) -> Dict[str, str]:
    """
    segments: [{"start","end","speaker","text", ...}, ...] (роль ещё не проставлена)
    Возвращает mapping speaker->role: "ответчик"/"звонящий"/"не определено".
    """
    intro_window_sec = CFG.role.intro_window_sec if intro_window_sec is None else intro_window_sec

    # Соберём спикеров
    speakers = sorted({s.get("speaker") for s in segments if s.get("speaker") is not None})
    if not speakers:
        return {}
    if len(speakers) == 1:
        return {speakers[0]: ROLE_UNKNOWN}

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
        dur = _segment_duration(seg)
        e = s + dur
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

        # Старт звонка несет максимальный сигнал о роли.
        if s <= intro_window_sec:
            ans_bonus, call_bonus = _opening_role_hint(text)
            pos_weight = max(0.35, 1.0 - (s / max(1e-6, intro_window_sec)))
            stats[spk]["ans_hits"] += ans_bonus * pos_weight
            stats[spk]["call_hits"] += call_bonus * pos_weight

    # 1) Если явно IVR-like реплики, не пытаемся насильно относить к caller/answerer.
    role_map: Dict[str, str] = {}
    forced_unknown = set()
    for spk in speakers:
        ivr_score = stats[spk]["ivr_hits"]
        if ivr_score >= 2 and stats[spk]["early"] > 0.0 and stats[spk]["total"] < 30.0:
            role_map[spk] = ROLE_UNKNOWN
            forced_unknown.add(spk)
            continue
        # Слишком короткая речь без лексических сигналов -> не определено.
        weak_signal = (
            stats[spk]["ans_hits"] <= 0.0
            and stats[spk]["call_hits"] <= 0.0
            and stats[spk]["question_hits"] <= 0.0
            and stats[spk]["id_hits"] <= 0.0
        )
        if stats[spk]["total"] < CFG.role.min_role_turn and weak_signal:
            role_map[spk] = ROLE_UNKNOWN
            forced_unknown.add(spk)

    cand = [spk for spk in speakers if spk not in forced_unknown]
    if not cand:
        return role_map
    if len(cand) == 1:
        role_map[cand[0]] = ROLE_UNKNOWN
        return role_map

    opening_agents = _detect_opening_speakers(segments, cand)
    if opening_agents and len(opening_agents) < len(cand):
        opening_set = set(opening_agents)
        for spk in cand:
            if spk in opening_set:
                role_map.setdefault(spk, ROLE_ANSWERER)
            else:
                # В call-centre сценарии остальные чаще являются клиентами.
                # Но если по статистике speaker выглядит как "агент", оставляем unknown.
                ans = stats[spk]["ans_hits"]
                call = stats[spk]["call_hits"]
                if ans > call + 1.5:
                    role_map.setdefault(spk, ROLE_UNKNOWN)
                else:
                    role_map.setdefault(spk, ROLE_CALLER)
        for spk in speakers:
            role_map.setdefault(spk, ROLE_UNKNOWN)
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
        role_map.setdefault(caller_spk, ROLE_CALLER)
        rest = [spk for spk in cand if spk != caller_spk]
        if rest:
            answerer_spk = max(rest, key=answerer_score)
            role_map.setdefault(answerer_spk, ROLE_ANSWERER)
            for spk in rest:
                if spk != answerer_spk:
                    role_map.setdefault(spk, ROLE_UNKNOWN)
        for spk in speakers:
            role_map.setdefault(spk, ROLE_UNKNOWN)
        return role_map

    scored = sorted(((answerer_score(spk), spk) for spk in cand), reverse=True)
    best_score, best_spk = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1e9

    # уверенность как разница
    confidence = best_score - second_score

    # 3) Назначаем роли
    # Если не уверены — не гадаем, ставим "не определено".
    if confidence < CFG.role.min_confidence:
        for spk in cand:
            role_map.setdefault(spk, ROLE_UNKNOWN)
        for spk in speakers:
            role_map.setdefault(spk, ROLE_UNKNOWN)
        return role_map

    role_map.setdefault(best_spk, ROLE_ANSWERER)

    # “звонящий” = лучший по caller_score среди остальных.
    others = [spk for spk in cand if spk != best_spk]
    if others:
        caller_spk = max(others, key=lambda spk: (stats[spk]["call_hits"], stats[spk]["total"]))
        role_map.setdefault(caller_spk, ROLE_CALLER)
        for spk in others:
            if spk != caller_spk:
                role_map.setdefault(spk, ROLE_UNKNOWN)

    for spk in speakers:
        role_map.setdefault(spk, ROLE_UNKNOWN)

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
    Назначает role в каждом сегменте с учетом speaker role map.
    Роли ограничены: "звонящий"/"ответчик"/"не определено".
    Возвращает обновленный speaker->role map (по длительности сегментов).
    """
    seg_debug: List[Tuple[float, float, float, float]] = []

    for seg in segments:
        spk = seg.get("speaker")
        base_role = _sanitize_role(speaker_role_map.get(spk))
        text = str(seg.get("text", "") or "")
        caller_score, answerer_score, ivr_score = _segment_role_scores(text)
        dur = _segment_duration(seg)
        is_short = _is_short_utterance(seg)

        role = base_role
        delta = caller_score - answerer_score
        strong_delta = 0.9 if dur >= 2.5 else 1.2
        weak_delta = 0.35 if dur >= 2.5 else 0.5
        if ivr_score >= 2.0:
            role = ROLE_UNKNOWN
        elif is_short and base_role != ROLE_UNKNOWN and abs(delta) < strong_delta:
            role = base_role
        elif delta >= strong_delta:
            role = ROLE_CALLER
        elif delta <= -strong_delta:
            role = ROLE_ANSWERER
        elif abs(delta) >= weak_delta and base_role != ROLE_UNKNOWN:
            role = base_role
        else:
            role = ROLE_UNKNOWN

        seg["role"] = role
        seg_debug.append((caller_score, answerer_score, ivr_score, delta))

    updated_map = _speaker_map_from_segments(segments, speaker_role_map)

    # Второй проход: стабилизируем сегменты вокруг доминирующей роли speaker-а,
    # если у сегмента нет сильных противоположных сигналов.
    for i, seg in enumerate(segments):
        spk = seg.get("speaker")
        if spk is None:
            continue
        dominant = _sanitize_role(updated_map.get(spk))
        if dominant == ROLE_UNKNOWN:
            continue

        caller_score, answerer_score, ivr_score, delta = seg_debug[i]
        current = _sanitize_role(seg.get("role"))
        strong_opposite = (
            (dominant == ROLE_CALLER and delta <= -1.6)
            or (dominant == ROLE_ANSWERER and delta >= 1.6)
        )
        if ivr_score >= 1.5 or strong_opposite:
            continue
        if current == ROLE_UNKNOWN or abs(delta) < 1.0:
            seg["role"] = dominant

    return _speaker_map_from_segments(segments, updated_map)
