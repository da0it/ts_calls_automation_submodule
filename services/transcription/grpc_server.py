#!/usr/bin/env python
# services/transcription/grpc_server.py

from __future__ import annotations

import os
import sys
import tempfile
import time
import logging
from pathlib import Path
from concurrent import futures

# Добавляем текущую директорию в путь Python
CURRENT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(CURRENT_DIR))

import grpc

# Импортируем сгенерированные protobuf файлы
try:
    from grpc_gen import call_processing_pb2 as pb2
    from grpc_gen import call_processing_pb2_grpc as pb2_grpc
    print("✓ Imported protobuf modules from grpc_gen")
except ImportError as e:
    print(f"✗ Failed to import protobuf modules: {e}")
    print(f"Make sure grpc_gen/__init__.py exists")
    sys.exit(1)

# Импортируем вашу логику транскрибации
try:
    from transcribe_logic.pipeline import transcribe_with_roles
    print("✓ Imported transcribe_with_roles")
except ImportError as e:
    print(f"✗ Failed to import transcribe_with_roles: {e}")
    print(f"Current directory: {CURRENT_DIR}")
    print(f"Files: {list(CURRENT_DIR.glob('*'))}")
    sys.exit(1)

try:
    from transcribe_logic.whisperx_runtime import warmup_whisperx_runtime
except ImportError:
    warmup_whisperx_runtime = None

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_grpc_server_credentials(prefix: str):
    tls_enabled = _env_bool(f"{prefix}_TLS_ENABLED", _env_bool("GRPC_TLS_ENABLED", False))
    if not tls_enabled:
        return None

    cert_file = os.getenv(f"{prefix}_TLS_CERT_FILE", "").strip()
    key_file = os.getenv(f"{prefix}_TLS_KEY_FILE", "").strip()
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


class TranscriptionServicer(pb2_grpc.TranscriptionServiceServicer):
    """gRPC сервис для транскрибации на WhisperX."""
    
    def __init__(self):
        logger.info("Initializing TranscriptionServicer")
        # Путь нужен только для whisperx+nemo diarization backend.
        self.whisper_repo_dir = os.getenv(
            "WHISPER_REPO_DIR", 
            os.path.expanduser("~/whisper-diarization")
        )
        logger.info(f"WhisperX NeMo repo: {self.whisper_repo_dir}")

        self._maybe_warmup_whisperx()

    def _maybe_warmup_whisperx(self) -> None:
        preload = _env_bool("WHISPERX_PRELOAD", False)
        persistent = _env_bool("WHISPERX_PERSISTENT", True)
        if not preload or not persistent:
            return
        if warmup_whisperx_runtime is None:
            logger.warning("WHISPERX_PRELOAD=1 but warmup_whisperx_runtime import failed.")
            return

        try:
            logger.info("WhisperX preload enabled: warming up persistent runtime...")
            warmup_whisperx_runtime(
                model=os.getenv("WHISPERX_MODEL", "large-v3"),
                language=os.getenv("WHISPERX_LANGUAGE", "ru"),
                device=os.getenv("WHISPERX_DEVICE", "cpu"),
                compute_type=os.getenv("WHISPERX_COMPUTE_TYPE", "int8"),
                vad_method=os.getenv("WHISPERX_VAD_METHOD", "silero").strip().lower(),
                diarize=_env_bool("WHISPERX_DIARIZE", True),
                diarization_backend=os.getenv("WHISPERX_DIARIZATION_BACKEND", "pyannote").strip().lower(),
                diarize_model=os.getenv("WHISPERX_DIARIZE_MODEL", "pyannote/speaker-diarization-3.1"),
                nemo_repo_dir=os.getenv("WHISPER_REPO_DIR", self.whisper_repo_dir),
                hf_token=os.getenv("HF_TOKEN"),
            )
            logger.info("WhisperX preload completed.")
        except Exception as exc:
            logger.warning("WhisperX preload failed, continuing without warmup: %s", exc)
    
    def _convert_to_proto_segments(self, segments: list) -> list:
        """Конвертирует сегменты в protobuf формат"""
        proto_segments = []
        for seg in segments:
            proto_seg = pb2.Segment(
                start=float(seg.get("start", 0)),
                end=float(seg.get("end", 0)),
                speaker=seg.get("speaker", ""),
                role=seg.get("role", ""),
                text=seg.get("text", "")
            )
            proto_segments.append(proto_seg)
        return proto_segments
    
    def Transcribe(self, request, context):
        """
        Обрабатывает запрос на транскрибацию
        """
        start_time = time.time()
        logger.info(f"Received request: call_id={request.call_id}, filename={request.filename}")

        if not request.audio:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("audio is required")
            return pb2.TranscribeResponse()
        
        temp_file = None
        try:
            # Сохраняем аудио во временный файл
            file_ext = Path(request.filename).suffix or ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(request.audio)
                temp_file = tmp.name
            
            logger.info(f"Saved audio to temporary file: {temp_file} ({len(request.audio)} bytes)")
            
            # Вызываем функцию транскрибации
            result = transcribe_with_roles(
                audio_path=temp_file,
                whisper_repo_dir=self.whisper_repo_dir,
            )
            
            # Создаем ответ
            transcript = pb2.Transcript()
            transcript.call_id = request.call_id
            
            # Добавляем сегменты
            for seg in result.get("segments", []):
                proto_seg = transcript.segments.add()
                proto_seg.start = float(seg.get("start", 0))
                proto_seg.end = float(seg.get("end", 0))
                proto_seg.speaker = seg.get("speaker", "")
                proto_seg.role = seg.get("role", "")
                proto_seg.text = seg.get("text", "")
            
            # Добавляем role_mapping
            role_mapping = result.get("role_mapping", {})
            for key, value in role_mapping.items():
                transcript.role_mapping[key] = value
            
            # Добавляем метаданные
            metadata = {
                "mode": result.get("mode", "whisperx"),
                "input": result.get("input", ""),
                "note": result.get("note", ""),
                "processing_time_seconds": str(round(time.time() - start_time, 2))
            }
            
            # Конвертируем метаданные в google.protobuf.Struct
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
                logger.debug(f"Deleted temporary file: {temp_file}")


def serve():
    """Запускает gRPC сервер"""
    # Параметры сервера из переменных окружения
    host = os.getenv("GRPC_HOST", "0.0.0.0")
    port = int(os.getenv("TRANSCRIPTION_GRPC_PORT", os.getenv("GRPC_PORT", "50051")))
    max_workers = int(os.getenv("GRPC_MAX_WORKERS", 4))
    
    # Создаем сервер
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        maximum_concurrent_rpcs=max_workers
    )
    
    # Добавляем сервис
    pb2_grpc.add_TranscriptionServiceServicer_to_server(
        TranscriptionServicer(), server
    )
    
    # Привязываем к порту
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
    logger.info(
        "WhisperX NeMo repo: %s",
        os.getenv("WHISPER_REPO_DIR", os.path.expanduser("~/whisper-diarization")),
    )
    
    # Запускаем
    server.start()
    logger.info("Server is ready to accept requests")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)


if __name__ == "__main__":
    serve()
