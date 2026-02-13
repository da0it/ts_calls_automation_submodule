# services/entity-extraction/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
import logging
import sys

from extractor.entity_extractor import EntityExtractor
from extractor.models import Segment, Entities

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# State для хранения экстрактора
class AppState:
    extractor: Optional[EntityExtractor] = None

state = AppState()

# Создание приложения
app = FastAPI(
    title="Entity Extraction Service",
    description="NER сервис для извлечения сущностей из диалогов",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Entity Extraction Service...")
    try:
        logger.info("Initializing EntityExtractor (this may take a while)...")
        state.extractor = EntityExtractor(use_ner=True)
        logger.info("✓ EntityExtractor initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize EntityExtractor: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Entity Extraction Service...")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response модели
class ExtractRequest(BaseModel):
    segments: List[dict]  # List of {start, end, speaker, role, text}

    class Config:
        schema_extra = {
            "example": {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "speaker": "SPEAKER_00",
                        "role": "звонящий",
                        "text": "Здравствуйте, меня зовут Иван Петров"
                    }
                ]
            }
        }


class ExtractResponse(BaseModel):
    entities: Entities


# Эндпоинты
@app.post("/api/extract-entities", response_model=ExtractResponse)
async def extract_entities(request: ExtractRequest):
    """
    Извлекает сущности из сегментов диалога
    """
    if state.extractor is None:
        raise HTTPException(
            status_code=503,
            detail="EntityExtractor not initialized. Service is starting up."
        )
    
    try:
        logger.info(f"Extracting entities from {len(request.segments)} segments")
        
        # Конвертируем в модели
        segments = [Segment(**seg) for seg in request.segments]
        
        # Извлекаем сущности
        entities = state.extractor.extract(segments)
        
        logger.info(f"Extracted: {len(entities.persons)} persons, "
                   f"{len(entities.phones)} phones, {len(entities.emails)} emails")
        
        return ExtractResponse(entities=entities)
        
    except Exception as e:
        logger.error(f"Error extracting entities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    is_ready = (
        state.extractor is not None 
        and state.extractor.ner_model is not None
    )
    return {
        "status": "healthy" if is_ready else "starting",
        "service": "entity-extraction",
        "ner_loaded": is_ready,
        "ready": is_ready
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Entity Extraction Service",
        "version": "1.0.0",
        "description": "NER сервис для извлечения сущностей из транскрибированных диалогов",
        "endpoints": {
            "extract": "POST /api/extract-entities",
            "health": "GET /health",
            "docs": "GET /docs",
            "redoc": "GET /redoc"
        },
        "status": "ready" if state.extractor is not None else "starting"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5001,
        log_level="info"
    )