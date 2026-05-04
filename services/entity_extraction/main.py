import logging
import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# EntityExtractor — класс, извлекающий сущности
from extractor.entity_extractor import EntityExtractor

# Entities — модель ответа с найденными сущностями, Segment — модель одного сегмента транскрипции.
from extractor.models import Entities, Segment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Создание FastAPI - приложения сервиса для извлечения сущностей
app = FastAPI(
    title="Entity Extraction Service",
    description="NER сервис для извлечения сущностей из диалогов",
    version="1.0.0",
)

# Глобальный объект извлекателя сущностей:
extractor: Optional[EntityExtractor] = None
startup_error = ""

# Функция читает булеву настройку из переменной окружения
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# Функция формирует статус сервиса для проверки активности healthcheck.
def _service_status() -> dict:

    # Если извлекатель еще не создан, возвращается статус starting и ready - false 
    # (сервис еще запускается или не успел инициализироваться)
    if extractor is None:
        return {
            "status": "starting",
            "service": "entity-extraction",
            "mode": "starting" if not startup_error else "regex_only",
            "ner_loaded": False,
            "ready": False,
            "startup_error": startup_error,
        }

    # Проверка работоспособнособности ner модели. Если она не загружена - degraded присваивается значение 1. Сервис работает в режиме
    # Регулярных выражений
    degraded = extractor.ner_model is None
    return {
        "status": "degraded" if degraded else "healthy",
        "service": "entity-extraction",
        "mode": extractor.mode,
        "ner_loaded": not degraded,
        "ready": True,
        "startup_error": extractor.startup_error or startup_error,
    }

# Функция, выполняющаяся при старте FastAPI-приложения
@app.on_event("startup")
async def startup_event():
    global extractor, startup_error
    logger.info("Starting Entity Extraction Service...")

    # Чтение настроек запуска из переменных окружения
    # Включает или отключает использование DeepPavlov NER.
    use_ner = _env_bool("ENTITY_USE_NER", True)

    # Разрешает скачивание модели при запуске.
    allow_download = _env_bool("ENTITY_NER_DOWNLOAD_ON_STARTUP", False)

    # Разрешает установку зависимостей при запуске.
    allow_install = _env_bool("ENTITY_NER_INSTALL_ON_STARTUP", False)

    try:
        logger.info(
            "Initializing EntityExtractor: use_ner=%s download_on_startup=%s install_on_startup=%s",
            use_ner,
            allow_download,
            allow_install,
        )

        # Создание экземпляра EntityExtractor
        extractor = EntityExtractor(
            use_ner=use_ner,
            allow_download=allow_download,
            allow_install=allow_install,
        )
        startup_error = extractor.startup_error
        if extractor.ner_model is not None:
            logger.info("Entity Extraction Service is ready with DeepPavlov NER")
        else:
            logger.warning(
                "Entity Extraction Service is ready in regex-only mode. startup_error=%s",
                startup_error or "none",
            )
    except Exception as exc:
        startup_error = str(exc)
        logger.error(
            "Failed to initialize EntityExtractor with NER, falling back to regex-only mode: %s",
            exc,
            exc_info=True,
        )
        extractor = EntityExtractor(use_ner=False)
        logger.warning("Entity Extraction Service started in forced regex-only mode")

# Блок разрешает CORS-запросы.
app.add_middleware(
    CORSMiddleware,

    # Разрешает запросы с любых доменов.
    allow_origins=["*"],
    allow_credentials=True,

    # Разрешает любые HTTP-методы.
    allow_methods=["*"],

    # Разрешает любые заголовки.
    allow_headers=["*"],
)

# Pydantic-модель запроса и ответа
class ExtractRequest(BaseModel):
    segments: list[dict]

# Модель ответа. Ответ будет содержать объект Entities
class ExtractResponse(BaseModel):
    entities: Entities

# POST-эндпоинт для извлечения сущностей. Он принимает ExtractRequest и возвращает ExtractResponse.
@app.post("/api/extract-entities", response_model=ExtractResponse)
async def extract_entities(request: ExtractRequest):
    if extractor is None:
        raise HTTPException(status_code=503, detail="service is starting")

    try:
        logger.info("Extracting entities from %d segments", len(request.segments))

        # Каждый словарь из запроса преобразуется в Pydantic-модель Segment.
        segments = [Segment(**seg) for seg in request.segments]

        # Основная логика извлечения сущностей
        entities = extractor.extract(segments)
        logger.info(
            "Extracted: %d persons, %d phones, %d emails",
            len(entities.persons),
            len(entities.phones),
            len(entities.emails),
        )

        # Возвращается HTTP-ответ с найденными сущностями
        return ExtractResponse(entities=entities)
    except Exception as exc:
        logger.error("Error extracting entities: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

# Эндпоинт для проверки состояния сервиса. Возвращает результат _service_status().
@app.get("/health")
async def health():
    return _service_status()


if __name__ == "__main__":
    import uvicorn


# Запуск через uvicorn. Запускается только если файл запущен напрямую. Если файл импортируется как модуль, этот блок не выполнится.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
        log_level="info",
    )
