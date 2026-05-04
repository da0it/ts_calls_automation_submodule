# services/entity-extraction/extractor/models.py
from pydantic import BaseModel
from typing import List, Optional

# Сегмент диалога из расшифровки звонка
class Segment(BaseModel):

    # Время начала сегмента в секундах
    start: float

    # Время окончания сегмента в секундах
    end: float

    # Идентификатор говорящего
    speaker: str

    # Роль говорящего, если определена
    role: Optional[str] = None

    # Текст сегмента
    text: str

# Одна извлеченная сущность
class ExtractedEntity(BaseModel):

    # Тип сущности: person, phone, email, order_id и т.д.
    type: str

    # Найденное значение сущности
    value: str

    # Уверенность извлечения
    confidence: float

    # Фрагмент текста, в котором найдена сущность
    context: str

# Набор всех сущностей, извлеченных из транскрипта звонка
class Entities(BaseModel):

    # Найденные имена
    persons: List[ExtractedEntity] = []

    # Найденные номера телефонов
    phones: List[ExtractedEntity] = []

    # Найденные emails
    emails: List[ExtractedEntity] = []

    # Найденные номера заказов
    order_ids: List[ExtractedEntity] = []

    # Найденные идентификаторы аккаунтов
    account_ids: List[ExtractedEntity] = []

    # Найденные денежные суммы
    money_amounts: List[ExtractedEntity] = []

    # Найденные даты
    dates: List[ExtractedEntity] = []