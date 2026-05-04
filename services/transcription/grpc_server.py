#!/usr/bin/env python
# services/transcription/grpc_server.py

# gRPC-сервер сервиса транскрибации.
# Принимает аудиофайл в запросе, сохраняет его во временный файл,
# запускает pipeline транскрибации WhisperX и возвращает результат в формате protobuf.

# Аннотации типов
from __future__ import annotations

# Работа с переменными окружения
import os

# Для добавления директорий в путь импорта Python
import sys

# Необходим для создания временных файлов
import tempfile

# Используется для измерения времени обработки запроса
import time

# Логи сервера
import logging

# Удобная работа с путями
from pathlib import Path

# Необходим для работы ThreadPoolExecutor, через который gRPC-сервер обрабатывает запросы в пуле потоков.
from concurrent import futures

# Добавляем текущую директорию в путь Python
CURRENT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "grpc_gen"))

import grpc

# Импортирование сгенерированных protobuf файлов
try:
    from grpc_gen import call_processing_pb2 as pb2
    from grpc_gen import call_processing_pb2_grpc as pb2_grpc
    print("✓ Imported protobuf modules from grpc_gen")
except ImportError as e:
    print(f"✗ Failed to import protobuf modules: {e}")
    print(f"Make sure grpc_gen/__init__.py exists")
    sys.exit(1)

# Импортирование логики транскрибации
try:
    from transcribe_logic.pipeline import transcribe_with_roles
    print("✓ Imported transcribe_with_roles")
except ImportError as e:
    print(f"✗ Failed to import transcribe_with_roles: {e}")
    print(f"Current directory: {CURRENT_DIR}")
    print(f"Files: {list(CURRENT_DIR.glob('*'))}")
    sys.exit(1)

# Импорт функции для загрузки модели из кэша перед обработкой первого поступившего запроса
try:
    from transcribe_logic.whisperx_runtime import warmup_whisperx_runtime
except ImportError:
    warmup_whisperx_runtime = None

# Импорт функции для получения устройства, на котором будет работать сервис whisperX
try:
    from transcribe_logic.config import get_whisperx_device_from_env
except ImportError:
    get_whisperx_device_from_env = None

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Функция читает переменную окружения и преобразует её в bool. 
# Если переменная окружения не задана, возвращается значение по умолчанию.
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# Функция собирает TLS-credentials для gRPC-сервера, если TLS включён через переменные окружения.
def _build_grpc_server_credentials(prefix: str):
    tls_enabled = _env_bool(f"{prefix}_TLS_ENABLED", _env_bool("GRPC_TLS_ENABLED", False))
    if not tls_enabled:
        return None
    
    # Если TLS включен, читаются пути к сертификату и приватному ключу
    cert_file = os.getenv(f"{prefix}_TLS_CERT_FILE", "").strip()
    key_file = os.getenv(f"{prefix}_TLS_KEY_FILE", "").strip()

    # Если TLS включён, но путь к сертификату или ключу не задан, функция выбрасывает ошибку.
    if not cert_file or not key_file:
        raise RuntimeError(
            f"{prefix}_TLS_ENABLED=1 but cert/key path is empty "
            f"({prefix}_TLS_CERT_FILE, {prefix}_TLS_KEY_FILE)"
        )

    with open(cert_file, "rb") as cert_f:
        cert_data = cert_f.read()
    with open(key_file, "rb") as key_f:
        key_data = key_f.read()

    return grpc.ssl_server_credentials(((key_data, cert_data),))

# Класс реализует gRPC-сервис транскрибации. Наследуется от pb2_grpc.TranscriptionServiceServicer, сгенерированного из .proto файла
# Принимает аудио в запросе, запускает WhisperX конвейер и возвращает транскрипт в protobuf-формате.
class TranscriptionServicer(pb2_grpc.TranscriptionServiceServicer):
    
    def __init__(self):
        logger.info("Initializing TranscriptionServicer")
        self._maybe_warmup_whisperx()

    # Метод опционально выполняет предварительную загрузку моделей WhisperX в зависимости от заданного параметра WHISPERX_PRELOAD
    def _maybe_warmup_whisperx(self) -> None:
        preload = _env_bool("WHISPERX_PRELOAD", False)
        if not preload:
            return
        if warmup_whisperx_runtime is None:
            logger.warning("WHISPERX_PRELOAD=1 but warmup_whisperx_runtime import failed.")
            return

        try:
            device = (
                get_whisperx_device_from_env()
                if get_whisperx_device_from_env is not None
                else os.getenv("WHISPERX_DEVICE", "auto")
            )
            logger.info("WhisperX preload enabled: warming up persistent runtime on device=%s...", device)

            # После определения устройства исполнения, и проверки условия, запускается метод предварительной загрузки с параметрами 
            # загружаемой модели
            warmup_whisperx_runtime(
                model=os.getenv("WHISPERX_MODEL", "large-v3"),
                language=os.getenv("WHISPERX_LANGUAGE", "ru"),
                device=device,
                compute_type=os.getenv("WHISPERX_COMPUTE_TYPE", "int8"),
                vad_method=os.getenv("WHISPERX_VAD_METHOD", "silero").strip().lower(),
            )
            logger.info("WhisperX preload completed.")
        except Exception as exc:
            logger.warning("WhisperX preload failed, continuing without warmup: %s", exc)
    
    # Функция выполняет запрос на транскрибацию. На вход поступает объект protobuf-запроса с аудио, именем файла и call_id (request)
    # gRPC-контекст, через который можно выставить код и описание ошибки (context)
    def Transcribe(self, request, context):

        # Время начала обработки запроса
        start_time = time.time()

        # Определение расширения исходного файла
        file_ext = Path(request.filename).suffix.lower() or ".mp3"
        logger.info(
            "Received request: call_id=%s ext=%s bytes=%d",
            request.call_id,
            file_ext,
            len(request.audio or b""),
        )

        # Если в запросе нет аудиоданных, сервер возвращает gRPC-ошибку
        if not request.audio:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("audio is required")
            return pb2.TranscribeResponse()
        
        temp_file = None
        try:
            # Аудио сохраняется во временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(request.audio)
                temp_file = tmp.name
            logger.info("Temporary audio file created for call_id=%s", request.call_id)
            
            # Вызывается функция транскрибации
            result = transcribe_with_roles(
                audio_path=temp_file,
            )
            
            # Создается ответ
            transcript = pb2.Transcript()
            transcript.call_id = request.call_id
            
            # Добавляются сегменты
            for seg in result.get("segments", []):
                proto_seg = transcript.segments.add()
                proto_seg.start = float(seg.get("start", 0))
                proto_seg.end = float(seg.get("end", 0))
                proto_seg.speaker = seg.get("speaker", "")
                proto_seg.text = seg.get("text", "")
            
            # Добавляются метаданные
            metadata = {
                "mode": result.get("mode", "whisperx"),
                "input": result.get("input", ""),
                "note": result.get("note", ""),
                "processing_time_seconds": str(round(time.time() - start_time, 2))
            }
            
            # Метаданные конвертируются в google.protobuf.Struct
            for key, value in metadata.items():
                transcript.metadata[key] = value
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f}s, segments: {len(transcript.segments)}")
            
            return pb2.TranscribeResponse(transcript=transcript)
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Transcription failed: {str(e)}")
            return pb2.TranscribeResponse()
            
        finally:
            # Удаляем временный файл
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
                logger.debug("Temporary audio file deleted for call_id=%s", request.call_id)

# Основная функция gRPC-сервера. Отвечает за запуск сервера.
def serve():

    # Параметры сервера из переменных окружения
    host = os.getenv("GRPC_HOST", "0.0.0.0")
    port = int(os.getenv("TRANSCRIPTION_GRPC_PORT", os.getenv("GRPC_PORT", "50051")))
    max_workers = int(os.getenv("GRPC_MAX_WORKERS", 4))
    
    # Создается сервер
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        maximum_concurrent_rpcs=max_workers
    )
    
    # Добавляется сервис
    pb2_grpc.add_TranscriptionServiceServicer_to_server(
        TranscriptionServicer(), server
    )
    
    # Привязка к порту
    server_address = f"{host}:{port}"
    creds = _build_grpc_server_credentials("TRANSCRIPTION_GRPC")
    if creds is not None:
        server.add_secure_port(server_address, creds)
        logger.info("Transcription gRPC TLS enabled")
    else:
        server.add_insecure_port(server_address)
        logger.info("Transcription gRPC running without TLS")
    
    logger.info(f"Starting transcription server on {server_address}")
    logger.info(f"Max workers: {max_workers}")
    
    # Запуск
    server.start()
    logger.info("Server is ready to accept requests")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)


if __name__ == "__main__":
    serve()
