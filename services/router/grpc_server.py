from __future__ import annotations

import json
import logging
import os
import sys
from concurrent import futures
from pathlib import Path

import grpc

sys.path.insert(0, str(Path(__file__).resolve().parent / "grpc_gen"))
import call_processing_pb2 as pb2
import call_processing_pb2_grpc as pb2_grpc
from routing.ai_analyzer import RubertEmbeddingAnalyzer
from routing.models import CallInput, Segment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("router-grpc")


def load_intents() -> dict:
    intents_path = Path(__file__).parent / "configs" / "intents.json"
    with intents_path.open("r", encoding="utf-8") as file:
        return json.load(file)


class RoutingService(pb2_grpc.RoutingServiceServicer):
    def __init__(self, intents: dict, analyzer: RubertEmbeddingAnalyzer) -> None:
        self.intents = intents
        self.analyzer = analyzer

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

            analysis = self.analyzer.analyze(call, self.intents)

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

    intents = load_intents()
    analyzer = RubertEmbeddingAnalyzer(
        model_name=model_name,
        min_confidence=min_confidence,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_RoutingServiceServicer_to_server(
        RoutingService(intents=intents, analyzer=analyzer),
        server,
    )

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    server.start()
    logger.info("Routing gRPC server listening on %s", listen_addr)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
