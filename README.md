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

## Go проверка

```bash
cd services/orchestrator && go test ./...
cd services/ticket_creation && go test ./...
```

16.02.2026