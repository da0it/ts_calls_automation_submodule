# gRPC Call Processing Pipeline

Реализована цепочка через gRPC:

1. Поступление аудио
2. Транскрибация + диаризация
3. Маршрутизация
4. Формирование тикета

## Контракт

Общий protobuf-контракт: `proto/call_processing.proto`

## Сервисы

- `services/transcription/grpc_server.py` (`TRANSCRIPTION_GRPC_PORT`, default `50051`)
- `services/router/grpc_server.py` (`ROUTER_GRPC_PORT`, default `50052`)
- `services/ticket_creation` gRPC сервер (`GRPC_PORT`, default `50054`)
- `services/orchestrator` gRPC сервер (`GRPC_PORT`, default `9000`)

## Конфиги

Пример переменных:

- `configs/transcription.env`
- `configs/routing.env`
- `configs/ticket.env`
- `configs/orchestrator.env`

## Security notes

- CORS теперь ограничивается списком `CORS_ALLOWED_ORIGINS` (по умолчанию localhost-ориджины).
- Для админа доступен аудит действий: `GET /api/v1/audit/events` (только admin JWT).
- В ticket service PII в описание тикета по умолчанию отключены:
  `TICKET_INCLUDE_PII_IN_DESCRIPTION=0`.
- В orchestrator/transcription убраны лишние детали из логов (имена файлов/пути временных файлов).

## Go проверка

```bash
cd services/orchestrator && go test ./...
cd services/ticket_creation && go test ./...
```

## Linux deployment

Production-like Linux deployment (systemd + bootstrap scripts):

- `/Users/dmitrii/ts_calls_automation_submodule/deploy/linux/DEPLOY.md`

## Docker deployment

Full stack in Docker Compose:

- `/Users/dmitrii/ts_calls_automation_submodule/deploy/docker/DEPLOY.md`

16.02.2026
