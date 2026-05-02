from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Priority = Literal["low", "medium", "high", "critical"]

@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    role: Optional[str]
    text: str

@dataclass
class CallInput:
    call_id: str
    segments: List[Segment] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Evidence:
    text: str
    timestamp: str

@dataclass
class IntentResult:
    intent_id: str
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    notes: str = ""

@dataclass
class AIAnalysis:
    intent: IntentResult
    priority: Priority
    suggested_targets: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
