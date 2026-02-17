from __future__ import annotations

import json
import logging
import os
import sys
from concurrent import futures
from pathlib import Path
from threading import RLock
from typing import Any, Dict

import grpc

sys.path.insert(0, str(Path(__file__).resolve().parent / "grpc_gen"))
import grpc_gen.call_processing_pb2 as pb2
import grpc_gen.call_processing_pb2_grpc as pb2_grpc
from routing.ai_analyzer import RubertEmbeddingAnalyzer
from routing.models import CallInput, Segment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("router-grpc")


def load_intents(intents_path: Path) -> Dict[str, Dict[str, Any]]:
    with intents_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("intents payload must be a JSON object")
    return payload


class RoutingService(pb2_grpc.RoutingServiceServicer):
    def __init__(self, intents_path: Path, intents: Dict[str, Dict[str, Any]], analyzer: RubertEmbeddingAnalyzer) -> None:
        self.intents_path = intents_path
        self.intents = intents
        self._intents_mtime = intents_path.stat().st_mtime if intents_path.exists() else 0.0
        self._lock = RLock()
        self.analyzer = analyzer

    def _get_intents(self) -> Dict[str, Dict[str, Any]]:
        try:
            current_mtime = self.intents_path.stat().st_mtime
        except OSError:
            current_mtime = 0.0

        with self._lock:
            if current_mtime <= self._intents_mtime:
                return self.intents

            try:
                loaded = load_intents(self.intents_path)
            except Exception as exc:
                logger.warning("failed to reload intents from %s: %s", self.intents_path, exc)
                return self.intents

            self.intents = loaded
            self._intents_mtime = current_mtime
            logger.info("reloaded intents config from %s (%d intents)", self.intents_path, len(self.intents))
            return self.intents

    def Route(self, request: pb2.RouteRequest, context: grpc.ServicerContext) -> pb2.RouteResponse:
        if len(request.segments) == 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("segments are required")
            return pb2.RouteResponse()

        try:
            segments = [
                Segment(
                    start=seg.start,
                    end=seg.end,
                    speaker=seg.speaker,
                    role=seg.role or None,
                    text=seg.text,
                )
                for seg in request.segments
            ]

            call = CallInput(
                call_id=request.call_id or "unknown-call",
                segments=segments,
                meta={},
            )

            analysis = self.analyzer.analyze(call, self._get_intents())

            suggested_group = ""
            for target in analysis.suggested_targets:
                if target.get("type") == "group":
                    suggested_group = str(target.get("id", ""))
                    break

            priority = str(analysis.priority)
            if priority == "normal":
                priority = "medium"

            response = pb2.RouteResponse(
                routing=pb2.Routing(
                    intent_id=analysis.intent.intent_id,
                    intent_confidence=float(analysis.intent.confidence),
                    priority=priority,
                    suggested_group=suggested_group,
                )
            )
            return response
        except Exception as exc:
            logger.exception("routing failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"routing failed: {exc}")
            return pb2.RouteResponse()


def serve() -> None:
    port = os.getenv("ROUTER_GRPC_PORT", "50052")
    model_name = os.getenv("ROUTER_MODEL_NAME", "DeepPavlov/rubert-base-cased")
    min_confidence = float(os.getenv("ROUTER_MIN_CONFIDENCE", "0.55"))
    intents_path = Path(
        os.getenv("ROUTER_INTENTS_PATH", str(Path(__file__).parent / "configs" / "intents.json"))
    )

    intents = load_intents(intents_path)
    logger.info("loaded intents config from %s (%d intents)", intents_path, len(intents))
    analyzer = RubertEmbeddingAnalyzer(
        model_name=model_name,
        min_confidence=min_confidence,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_RoutingServiceServicer_to_server(
        RoutingService(intents_path=intents_path, intents=intents, analyzer=analyzer),
        server,
    )

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    server.start()
    logger.info("Routing gRPC server listening on %s", listen_addr)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
