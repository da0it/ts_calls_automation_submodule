from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Priority = Literal["low", "medium", "high", "critical"]

# Минимальная единица звонка. Каждый транскрипт разделен на сегменты, которые формируют полноценный текст диалога
@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    role: Optional[str]
    text: str

# Вход системы анализа. Содержит в себе список всех сегментов, уникальный идентификатор и любые дополнительные данные (meta)
# В мета может храниться, например, номер телефона, id клиента и т.д.
@dataclass
class CallInput:
    call_id: str
    segments: List[Segment] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

# Класс данных хранит в себе обоснование итогового решения модели. Благодаря ему можно проверить, на каком из сегментов
# модель приняла то или иное решение
@dataclass
class Evidence:
    text: str
    timestamp: str

# Результат классификации цели звонка
@dataclass
class IntentResult:
    intent_id: str
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    notes: str = ""


# Финальный результат всей логики интеллектуальной обработки, помимо классифицированной цели обращения содержит
# приоритет, группу для назначения заявки. Raw может содержать полезные поля, такие как время обработки,
# используемое устройство для получения предсказания (ЦПУ, ГПУ), информацию для отладки
@dataclass
class AIAnalysis:
    intent: IntentResult
    priority: Priority
    suggested_targets: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
