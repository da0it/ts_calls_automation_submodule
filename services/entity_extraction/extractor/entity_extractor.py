# services/entity-extraction/extractor/entity_extractor.py
from __future__ import annotations
import re
import logging
from typing import List, Dict, Any, Optional
from deeppavlov import build_model, configs

from .models import Entities, ExtractedEntity, Segment

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Извлечение сущностей из текста диалога используя DeepPavlov NER"""
    
    def __init__(self, use_ner: bool = True):
        """
        Args:
            use_ner: Использовать DeepPavlov NER для извлечения персон и организаций
        """
        self.use_ner = use_ner
        self.ner_model = None
        
        if use_ner:
            try:
                logger.info("Loading DeepPavlov NER model (ner_rus_bert)...")
                # Загружаем предобученную модель для русского NER
                self.ner_model = build_model(configs.ner.ner_rus_bert, download=True, install=True)
                logger.info("DeepPavlov NER model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load DeepPavlov NER: {e}, will use regex only")
                self.ner_model = None
    
    def extract(self, segments: List[Segment]) -> Entities:
        """
        Извлекает сущности из сегментов диалога
        
        Args:
            segments: Список сегментов с полями start, end, speaker, role, text
            
        Returns:
            Entities с извлеченными данными
        """
        # Собираем текст от клиента + полный текст диалога
        caller_text = []
        all_text = []
        caller_contexts = []  # Для сохранения контекста каждого сегмента
        
        for seg in segments:
            all_text.append(seg.text)
            if seg.role == "звонящий":
                text = seg.text
                caller_text.append(text)
                caller_contexts.append({
                    "text": text,
                    "start": seg.start,
                    "end": seg.end
                })
        
        full_text = " ".join(caller_text)
        full_text_all = " ".join(all_text)
        # Если роли были определены плохо и caller пуст, не теряем сущности.
        source_text = full_text if full_text.strip() else full_text_all
        
        entities = Entities()
        
        # 1. Извлекаем персоны и организации через DeepPavlov NER
        if self.ner_model:
            ner_entities = self._extract_ner_entities(source_text, caller_contexts)
            entities.persons = ner_entities.get("persons", [])
            # Можно добавить organizations если нужно
        else:
            # Fallback: простое извлечение имен через regex
            entities.persons = self._extract_persons_regex(source_text)
        
        # 2. Извлекаем телефоны (regex)
        entities.phones = self._extract_phones(full_text_all)
        
        # 3. Извлекаем emails (regex)
        entities.emails = self._extract_emails(full_text_all)
        
        # 4. Извлекаем номера заказов (regex + эвристики)
        entities.order_ids = self._extract_order_ids(full_text_all)
        
        # 5. Извлекаем ID аккаунтов
        entities.account_ids = self._extract_account_ids(full_text_all)
        
        # 6. Извлекаем суммы денег
        entities.money_amounts = self._extract_money(full_text_all)
        
        # 7. Извлекаем даты
        entities.dates = self._extract_dates(full_text_all)
        
        logger.info(f"Extracted entities: {len(entities.persons)} persons, "
                   f"{len(entities.phones)} phones, {len(entities.emails)} emails, "
                   f"{len(entities.money_amounts)} money amounts")
        
        return entities
    
    def _extract_ner_entities(
        self, 
        text: str, 
        contexts: List[Dict]
    ) -> Dict[str, List[ExtractedEntity]]:
        """
        Извлечение сущностей через DeepPavlov NER
        
        DeepPavlov NER возвращает теги в формате BIO:
        - B-PER: начало имени персоны
        - I-PER: продолжение имени персоны
        - B-ORG: начало названия организации
        - I-ORG: продолжение названия организации
        - B-LOC: начало локации
        - I-LOC: продолжение локации
        - O: не сущность
        """
        try:
            # DeepPavlov NER принимает список текстов
            # Возвращает ([tokens], [tags])
            result = self.ner_model([text])
            tokens = result[0][0]  # Первый текст, первый элемент
            tags = result[1][0]    # Первый текст, второй элемент
            
            entities = {
                "persons": [],
                "organizations": [],
                "locations": []
            }
            
            # Собираем сущности из BIO-тегов
            current_entity = None
            current_tokens = []
            current_type = None
            
            for token, tag in zip(tokens, tags):
                if tag.startswith("B-"):
                    # Начало новой сущности
                    if current_tokens:
                        # Сохраняем предыдущую
                        entity = self._create_entity_from_tokens(
                            current_tokens, current_type, text
                        )
                        if entity:
                            entities[self._map_tag_to_type(current_type)].append(entity)
                    
                    current_tokens = [token]
                    current_type = tag[2:]  # PER, ORG, LOC
                    
                elif tag.startswith("I-") and current_tokens:
                    # Продолжение текущей сущности
                    current_tokens.append(token)
                    
                else:
                    # Тег O или несовпадение типа - завершаем текущую
                    if current_tokens:
                        entity = self._create_entity_from_tokens(
                            current_tokens, current_type, text
                        )
                        if entity:
                            entities[self._map_tag_to_type(current_type)].append(entity)
                    current_tokens = []
                    current_type = None
            
            # Не забываем последнюю сущность
            if current_tokens:
                entity = self._create_entity_from_tokens(
                    current_tokens, current_type, text
                )
                if entity:
                    entities[self._map_tag_to_type(current_type)].append(entity)
            
            logger.info(f"DeepPavlov NER found: {len(entities['persons'])} persons, "
                       f"{len(entities['organizations'])} orgs, "
                       f"{len(entities['locations'])} locations")
            
            return entities
            
        except Exception as e:
            logger.error(f"DeepPavlov NER extraction failed: {e}")
            return {"persons": [], "organizations": [], "locations": []}
    
    def _create_entity_from_tokens(
        self, 
        tokens: List[str], 
        entity_type: str,
        full_text: str
    ) -> Optional[ExtractedEntity]:
        """Создает ExtractedEntity из токенов"""
        if not tokens:
            return None
        
        # Собираем значение
        value = " ".join(tokens).strip()
        
        # Фильтруем шум (односимвольные, цифры и т.д.)
        if len(value) < 2 or value.isdigit():
            return None
        
        # Ищем контекст в полном тексте
        try:
            pos = full_text.lower().find(value.lower())
            if pos >= 0:
                start = max(0, pos - 30)
                end = min(len(full_text), pos + len(value) + 30)
                context = full_text[start:end]
            else:
                context = value
        except:
            context = value
        
        # Confidence зависит от длины и типа
        confidence = 0.85
        if len(tokens) > 2:  # Полное имя (Иван Иванович Иванов)
            confidence = 0.9
        if entity_type == "PER" and len(value.split()) >= 2:  # Имя + Фамилия
            confidence = 0.95
        
        return ExtractedEntity(
            type="person" if entity_type == "PER" else "organization",
            value=value,
            confidence=confidence,
            context=context
        )
    
    def _map_tag_to_type(self, tag: str) -> str:
        """Маппинг BIO-тегов в типы сущностей"""
        mapping = {
            "PER": "persons",
            "ORG": "organizations",
            "LOC": "locations"
        }
        return mapping.get(tag, "persons")
    
    def _extract_persons_regex(self, text: str) -> List[ExtractedEntity]:
        """
        Fallback: простое извлечение имен через regex
        Ищет паттерны типа "меня зовут Иван", "я Петр Сидоров"
        """
        patterns = [
            r'(?:меня зовут|я|это)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})',
            r'([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)',  # Имя Фамилия
        ]
        
        persons = []
        seen = set()  # Для дедупликации
        
        for pattern in patterns:
            for m in re.finditer(pattern, text):
                name = m.group(1).strip()
                
                # Фильтр: имя должно быть >= 2 слов или известное имя
                if name in seen:
                    continue
                
                if len(name.split()) >= 2 or self._is_common_name(name):
                    start = max(0, m.start() - 30)
                    end = min(len(text), m.end() + 30)
                    context = text[start:end]
                    
                    persons.append(ExtractedEntity(
                        type="person",
                        value=name,
                        confidence=0.7,  # Ниже чем у NER
                        context=context
                    ))
                    seen.add(name)
        
        return persons
    
    def _is_common_name(self, name: str) -> bool:
        """Проверка на распространенные имена"""
        common_names = {
            "Александр", "Алексей", "Андрей", "Анна", "Борис", "Василий",
            "Виктор", "Владимир", "Дмитрий", "Евгений", "Елена", "Игорь",
            "Иван", "Ирина", "Константин", "Мария", "Михаил", "Наталья",
            "Николай", "Ольга", "Павел", "Петр", "Сергей", "Татьяна"
        }
        return name in common_names
    
    def _extract_phones(self, text: str) -> List[ExtractedEntity]:
        """Извлечение телефонов (regex)"""
        # Паттерн для российских номеров
        pattern = r"""
            (?:
                \+?7\s?[\-]?
                (?:\(\d{3}\)|\d{3})
                \s?[\-]?\d{3}\s?[\-]?\d{2}\s?[\-]?\d{2}
            )|
            (?:
                8\s?[\-]?
                (?:\(\d{3}\)|\d{3})
                \s?[\-]?\d{3}\s?[\-]?\d{2}\s?[\-]?\d{2}
            )
        """
        phones = []
        seen = set()
        
        for m in re.finditer(pattern, text, re.VERBOSE):
            phone = m.group(0)
            # Нормализуем (убираем пробелы, скобки)
            normalized = re.sub(r"[\s\-\(\)]", "", phone)
            
            if normalized in seen:
                continue
            
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            context = text[start:end]
            
            phones.append(ExtractedEntity(
                type="phone",
                value=normalized,
                confidence=0.95,
                context=context
            ))
            seen.add(normalized)
        
        return phones
    
    def _extract_emails(self, text: str) -> List[ExtractedEntity]:
        """Извлечение email адресов"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = []
        seen = set()
        
        for m in re.finditer(pattern, text):
            email = m.group(0).lower()
            
            if email in seen:
                continue
            
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            context = text[start:end]
            
            emails.append(ExtractedEntity(
                type="email",
                value=email,
                confidence=0.95,
                context=context
            ))
            seen.add(email)
        
        return emails
    
    def _extract_order_ids(self, text: str) -> List[ExtractedEntity]:
        """
        Извлечение номеров заказов
        Примеры: "заказ 12345", "номер заказа АБ-123456"
        """
        patterns = [
            # "номер заказа 123456", "заказ на установку 756.13.632"
            r'(?:номер\s+заказа?|заказ(?:\s+на\s+\w+){0,3})[^A-ZА-Я0-9]{0,8}([A-ZА-Я]{0,3}\-?\d[\d\.\-\s]{2,20}\d)',
            # english fallback
            r'(?:order(?:\s+number)?)[:\s№#-]{0,8}([A-Z]{0,3}\-?\d[\d\.\-\s]{2,20}\d)',
        ]
        
        order_ids = []
        seen = set()
        
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                raw_order_id = m.group(1).strip().rstrip(".,;:!?")
                # Нормализуем "756.13.632" -> "75613632", "AB-1234" -> "AB1234"
                order_id = re.sub(r'[^A-ZА-Я0-9]', '', raw_order_id.upper())
                if len(order_id) < 4 or not re.search(r'\d', order_id):
                    continue
                
                if order_id in seen:
                    continue
                
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 30)
                context = text[start:end]
                
                order_ids.append(ExtractedEntity(
                    type="order_id",
                    value=order_id,
                    confidence=0.8,
                    context=context
                ))
                seen.add(order_id)
        
        return order_ids
    
    def _extract_account_ids(self, text: str) -> List[ExtractedEntity]:
        """Извлечение ID аккаунтов"""
        patterns = [
            r'(?:аккаунт|account|лицевой счет)[:\s№#]*([A-ZА-Я0-9\-]{4,15})',
            r'(?:ID|ид)[:\s№#]*([A-ZА-Я0-9\-]{4,15})',
        ]
        
        account_ids = []
        seen = set()
        
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                acc_id = m.group(1)
                
                if acc_id in seen:
                    continue
                
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 30)
                context = text[start:end]
                
                account_ids.append(ExtractedEntity(
                    type="account_id",
                    value=acc_id,
                    confidence=0.8,
                    context=context
                ))
                seen.add(acc_id)
        
        return account_ids
    
    def _extract_money(self, text: str) -> List[ExtractedEntity]:
        """Извлечение денежных сумм"""
        pattern = r"""
            (\d{1,3}(?:[\s\u00A0]?\d{3})*(?:[.,]\d{1,2})?)
            \s*
            (?:руб(?:лей|ля|\.)?|₽|р\.|рублей?|
               доллар(?:ов|а)?|USD|\$|
               евро|EUR|€)
        """
        
        money = []
        seen = set()
        
        for m in re.finditer(pattern, text, re.VERBOSE | re.IGNORECASE):
            amount = m.group(1).replace(" ", "").replace("\u00A0", "")
            
            if amount in seen:
                continue
            
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            context = text[start:end]
            
            money.append(ExtractedEntity(
                type="money",
                value=amount,
                confidence=0.9,
                context=context
            ))
            seen.add(amount)
        
        return money
    
    def _extract_dates(self, text: str) -> List[ExtractedEntity]:
        """Извлечение дат"""
        patterns = [
            r'\b(\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4})\b',  # 01.12.2024
            r'\b(\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+\d{4})?)\b',
        ]
        
        dates = []
        seen = set()
        
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                date = m.group(1)
                
                if date in seen:
                    continue
                
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 30)
                context = text[start:end]
                
                dates.append(ExtractedEntity(
                    type="date",
                    value=date,
                    confidence=0.85,
                    context=context
                ))
                seen.add(date)
        
        return dates
