# services/entity-extraction/extractor/models.py
from pydantic import BaseModel
from typing import List, Optional


class Segment(BaseModel):
    """Сегмент диалога"""
    start: float
    end: float
    speaker: str
    role: Optional[str] = None
    text: str


class ExtractedEntity(BaseModel):
    """Извлеченная сущность"""
    type: str  # person, phone, email, order_id, etc.
    value: str
    confidence: float
    context: str


class Entities(BaseModel):
    """Все извлеченные сущности"""
    persons: List[ExtractedEntity] = []
    phones: List[ExtractedEntity] = []
    emails: List[ExtractedEntity] = []
    order_ids: List[ExtractedEntity] = []
    account_ids: List[ExtractedEntity] = []
    money_amounts: List[ExtractedEntity] = []
    dates: List[ExtractedEntity] = []