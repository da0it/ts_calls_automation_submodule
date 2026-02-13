from __future__ import annotations

import logging
import os
import sys
import tempfile
import uuid
from concurrent import futures
from pathlib import Path
from typing import Optional

import grpc
from google.protobuf import struct_pb2

sys.path.insert(0, str(Path(__file__).resolve().parent / "grpc_gen"))
import call_processing_pb2 as pb2
import call_processing_pb2_grpc as pb2_grpc
from transcribe.pipeline import transcribe_with_roles


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("transcription-grpc")


class TranscriptionService(pb2_grpc.TranscriptionServiceServicer):
    def __init__(self, model_name: str, hf_token: Optional[str] = None) -> None:
        self.model_name = model_name
        self.hf_token = hf_token

    def Transcribe(self, request: pb2.TranscribeRequest, context: grpc.ServicerContext) -> pb2.TranscribeResponse:
        if not request.audio:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("audio is required")
            return pb2.TranscribeResponse()

        suffix = Path(request.filename or "audio.wav").suffix or ".wav"
        call_id = request.call_id or f"call_{uuid.uuid4().hex[:12]}"

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(request.audio)
                tmp_audio_path = tmp_file.name

            result = transcribe_with_roles(
                tmp_audio_path,
                gigaam_model_name=self.model_name,
                hf_token=self.hf_token,
            )
        except Exception as exc:
            logger.exception("transcription failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"transcription failed: {exc}")
            return pb2.TranscribeResponse()
        finally:
            if "tmp_audio_path" in locals():
                try:
                    os.remove(tmp_audio_path)
                except OSError:
                    pass

        segments = [
            pb2.Segment(
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                speaker=str(item.get("speaker", "")),
                role=str(item.get("role", "")),
                text=str(item.get("text", "")),
            )
            for item in result.get("segments", [])
        ]

        metadata = struct_pb2.Struct()
        metadata.update(
            {
                "mode": result.get("mode", ""),
                "input": result.get("input", request.filename or ""),
                "note": result.get("note", ""),
            }
        )

        transcript = pb2.Transcript(
            call_id=call_id,
            segments=segments,
            role_mapping={
                str(k): str(v) for k, v in (result.get("role_mapping") or {}).items()
            },
            metadata=metadata,
        )
        return pb2.TranscribeResponse(transcript=transcript)


def serve() -> None:
    port = os.getenv("TRANSCRIPTION_GRPC_PORT", "50051")
    model_name = os.getenv("GIGAAM_MODEL_NAME", "v3_e2e_rnnt")
    hf_token = os.getenv("HF_TOKEN")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_TranscriptionServiceServicer_to_server(
        TranscriptionService(model_name=model_name, hf_token=hf_token),
        server,
    )

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    server.start()
    logger.info("Transcription gRPC server listening on %s", listen_addr)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
