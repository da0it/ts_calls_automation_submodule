from __future__ import annotations

import json
import logging
import os
import sys
from concurrent import futures
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock, Thread
from typing import Any, Dict, Optional

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


class RoutingService(pb2_grpc.RoutingServiceServicer):
    def __init__(
        self,
        intents_path: Path,
        intents: Dict[str, Dict[str, Any]],
        analyzer: RubertEmbeddingAnalyzer,
        *,
        feedback_path: str,
        tuned_model_path: str,
        train_defaults: Dict[str, Any],
    ) -> None:
        self.intents_path = intents_path
        self.intents = intents
        self._intents_mtime = intents_path.stat().st_mtime if intents_path.exists() else 0.0
        self._lock = RLock()
        self.analyzer = analyzer

        self.feedback_path = feedback_path
        self.tuned_model_path = tuned_model_path
        self.train_defaults = dict(train_defaults)

        self._train_lock = RLock()
        self._training_in_progress = False

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

    def get_model_status(self) -> Dict[str, Any]:
        with self._train_lock:
            training = self._training_in_progress

        intents = self._get_intents()
        tuned_status = self.analyzer.get_training_status(intents)

        return {
            "service": "router-admin",
            "training_in_progress": training,
            "intents_count": len(intents),
            "tuned_model": tuned_status,
        }

    def reload_tuned_model(self) -> Dict[str, Any]:
        self.analyzer.reload_tuned_head_from_disk()
        return self.get_model_status()

    def train_tuned_model(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}

        with self._train_lock:
            if self._training_in_progress:
                raise RuntimeError("training already in progress")
            self._training_in_progress = True

        try:
            intents = self._get_intents()

            epochs = _as_int(payload.get("epochs"), self.train_defaults.get("epochs", 3), 1, 12)
            batch_size = _as_int(payload.get("batch_size"), self.train_defaults.get("batch_size", 16), 4, 64)
            random_seed = _as_int(payload.get("random_seed"), self.train_defaults.get("random_seed", 42), 1, 2**31 - 1)
            learning_rate = _as_float(payload.get("learning_rate"), self.train_defaults.get("learning_rate", 2e-5), 1e-6, 1e-3)
            val_ratio = _as_float(payload.get("val_ratio"), self.train_defaults.get("val_ratio", 0.2), 0.0, 0.5)
            feedback_path = str(payload.get("feedback_path") or self.feedback_path)
            output_path = str(payload.get("output_path") or self.tuned_model_path)

            report = self.analyzer.train_tuned_head(
                intents,
                feedback_path=feedback_path,
                output_path=output_path,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_ratio=val_ratio,
                random_seed=random_seed,
            )
            return {
                "ok": True,
                "report": report,
                "status": self.get_model_status(),
            }
        finally:
            with self._train_lock:
                self._training_in_progress = False


def _as_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    return max(low, min(high, parsed))


def _as_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    return max(low, min(high, parsed))


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_admin_handler(service: RoutingService, admin_token: str):
    class RouterAdminHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if not self._authorize():
                return

            if self.path == "/admin/model/status":
                _json_response(self, HTTPStatus.OK, service.get_model_status())
                return

            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._authorize():
                return

            payload = self._read_json_body()
            if payload is None:
                return

            if self.path == "/admin/model/train":
                try:
                    result = service.train_tuned_model(payload)
                    _json_response(self, HTTPStatus.OK, result)
                except Exception as exc:
                    logger.exception("admin train failed")
                    _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            if self.path == "/admin/model/reload":
                try:
                    result = service.reload_tuned_model()
                    _json_response(self, HTTPStatus.OK, result)
                except Exception as exc:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Admin-Token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def log_message(self, fmt: str, *args) -> None:
            logger.info("admin %s - %s", self.address_string(), fmt % args)

        def _read_json_body(self) -> Optional[Dict[str, Any]]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                return {}

            raw = self.rfile.read(content_length)
            if not raw:
                return {}

            try:
                parsed = json.loads(raw.decode("utf-8"))
                if isinstance(parsed, dict):
                    return parsed
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "request body must be a JSON object"})
                return None
            except Exception:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "invalid json body"})
                return None

        def _authorize(self) -> bool:
            if not admin_token:
                return True

            auth_header = self.headers.get("Authorization", "")
            token_header = self.headers.get("X-Admin-Token", "")
            bearer = ""
            if auth_header.startswith("Bearer "):
                bearer = auth_header[len("Bearer ") :].strip()

            if bearer == admin_token or token_header == admin_token:
                return True

            _json_response(self, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return False

    return RouterAdminHandler


def serve() -> None:
    port = os.getenv("ROUTER_GRPC_PORT", "50052")
    model_name = os.getenv("ROUTER_MODEL_NAME", "ai-forever/ruBert-base")
    min_confidence = float(os.getenv("ROUTER_MIN_CONFIDENCE", "0.55"))
    intents_path = Path(
        os.getenv("ROUTER_INTENTS_PATH", str(Path(__file__).parent / "configs" / "intents.json"))
    )

    feedback_path = os.getenv("ROUTER_FEEDBACK_PATH", str(Path(__file__).parent / "configs" / "routing_feedback.jsonl"))
    tuned_model_path = os.getenv("ROUTER_TUNED_MODEL_PATH", str(Path(__file__).parent / "configs" / "router_tuned_head.pt"))
    finetuned_enabled = os.getenv("ROUTER_FINETUNED_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
    finetuned_model_path = os.getenv(
        "ROUTER_FINETUNED_MODEL_PATH",
        str(Path(__file__).parent / "configs" / "router_finetuned_model"),
    )
    finetuned_learning_rate = float(os.getenv("ROUTER_FINETUNED_LR", "2e-5"))
    finetuned_epochs = int(os.getenv("ROUTER_FINETUNED_EPOCHS", "3"))
    finetuned_batch_size = int(os.getenv("ROUTER_FINETUNED_BATCH_SIZE", "16"))
    finetuned_max_length = int(os.getenv("ROUTER_FINETUNED_MAX_LENGTH", "256"))
    finetuned_weight_decay = float(os.getenv("ROUTER_FINETUNED_WEIGHT_DECAY", "0.01"))

    train_defaults = {
        "epochs": int(os.getenv("ROUTER_TRAIN_EPOCHS", "3")),
        "batch_size": int(os.getenv("ROUTER_TRAIN_BATCH_SIZE", "16")),
        "learning_rate": float(os.getenv("ROUTER_TRAIN_LR", "2e-5")),
        "val_ratio": float(os.getenv("ROUTER_TRAIN_VAL_RATIO", "0.2")),
        "random_seed": int(os.getenv("ROUTER_TRAIN_SEED", "42")),
    }

    intents = load_intents(intents_path)
    logger.info("loaded intents config from %s (%d intents)", intents_path, len(intents))
    analyzer = RubertEmbeddingAnalyzer(
        model_name=model_name,
        min_confidence=min_confidence,
        tuned_model_path=tuned_model_path,
        finetuned_enabled=finetuned_enabled,
        finetuned_model_path=finetuned_model_path,
        finetuned_learning_rate=finetuned_learning_rate,
        finetuned_epochs=finetuned_epochs,
        finetuned_batch_size=finetuned_batch_size,
        finetuned_max_length=finetuned_max_length,
        finetuned_weight_decay=finetuned_weight_decay,
    )

    routing_service = RoutingService(
        intents_path=intents_path,
        intents=intents,
        analyzer=analyzer,
        feedback_path=feedback_path,
        tuned_model_path=tuned_model_path,
        train_defaults=train_defaults,
    )

    admin_server = None
    admin_port = os.getenv("ROUTER_ADMIN_PORT", "").strip()
    admin_token = os.getenv("ROUTER_ADMIN_TOKEN", "").strip()
    if admin_port:
        admin_addr = ("0.0.0.0", int(admin_port))
        admin_handler = make_admin_handler(routing_service, admin_token)
        admin_server = ThreadingHTTPServer(admin_addr, admin_handler)
        admin_thread = Thread(target=admin_server.serve_forever, daemon=True)
        admin_thread.start()
        logger.info("Routing admin server listening on http://%s:%s", admin_addr[0], admin_port)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_RoutingServiceServicer_to_server(routing_service, server)

    listen_addr = f"[::]:{port}"
    creds = _build_grpc_server_credentials("ROUTER_GRPC")
    if creds is not None:
        server.add_secure_port(listen_addr, creds)
        logger.info("Routing gRPC TLS enabled")
    else:
        server.add_insecure_port(listen_addr)
        logger.info("Routing gRPC running without TLS")
    server.start()
    logger.info("Routing gRPC server listening on %s", listen_addr)

    try:
        server.wait_for_termination()
    finally:
        if admin_server is not None:
            admin_server.shutdown()
            admin_server.server_close()


if __name__ == "__main__":
    serve()
