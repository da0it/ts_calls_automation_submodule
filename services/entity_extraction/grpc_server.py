from __future__ import annotations

import logging
import os
import sys
from concurrent import futures
from pathlib import Path

import grpc

import grpc_gen.call_processing_pb2 as pb2
import grpc_gen.call_processing_pb2_grpc as pb2_grpc
from extractor.entity_extractor import EntityExtractor
from extractor.models import Segment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("entity-grpc")


def to_proto_entities(entities) -> pb2.Entities:
    def convert(items: list) -> list[pb2.ExtractedEntity]:
        out: list[pb2.ExtractedEntity] = []
        for item in items:
            out.append(
                pb2.ExtractedEntity(
                    type=item.type,
                    value=item.value,
                    confidence=float(item.confidence),
                    context=item.context,
                )
            )
        return out

    return pb2.Entities(
        persons=convert(entities.persons),
        phones=convert(entities.phones),
        emails=convert(entities.emails),
        order_ids=convert(entities.order_ids),
        account_ids=convert(entities.account_ids),
        money_amounts=convert(entities.money_amounts),
        dates=convert(entities.dates),
    )


class EntityService(pb2_grpc.EntityExtractionServiceServicer):
    def __init__(self, extractor: EntityExtractor) -> None:
        self.extractor = extractor

    def ExtractEntities(
        self,
        request: pb2.ExtractEntitiesRequest,
        context: grpc.ServicerContext,
    ) -> pb2.ExtractEntitiesResponse:
        if len(request.segments) == 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("segments are required")
            return pb2.ExtractEntitiesResponse()

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

            entities = self.extractor.extract(segments)
            return pb2.ExtractEntitiesResponse(entities=to_proto_entities(entities))
        except Exception as exc:
            logger.exception("entity extraction failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"entity extraction failed: {exc}")
            return pb2.ExtractEntitiesResponse()


def serve() -> None:
    port = os.getenv("ENTITY_GRPC_PORT", "50053")

    logger.info("Initializing EntityExtractor...")
    extractor = EntityExtractor(use_ner=True)
    logger.info("EntityExtractor initialized")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_EntityExtractionServiceServicer_to_server(
        EntityService(extractor=extractor),
        server,
    )

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    server.start()
    logger.info("Entity gRPC server listening on %s", listen_addr)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
