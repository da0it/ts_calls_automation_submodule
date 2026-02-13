# Orchestrator Service

Главный сервис, который оркестрирует цепочку:

`Аудио -> Транскрибация+диаризация -> Маршрутизация -> Формирование тикета`

## Транспорт

- HTTP API (совместимость): `POST /api/v1/process-call`
- gRPC API: `callprocessing.v1.OrchestratorService/ProcessCall`

## gRPC зависимости

Orchestrator вызывает по gRPC:

- `TranscriptionService` (`TRANSCRIPTION_GRPC_ADDR`, default `localhost:50051`)
- `RoutingService` (`ROUTING_GRPC_ADDR`, default `localhost:50052`)
- `TicketService` (`TICKET_GRPC_ADDR`, default `localhost:50054`)

## Локальный запуск

```bash
cd services/orchestrator
go mod download
go run cmd/server/main.go
```

Порты:

- HTTP: `HTTP_PORT` (default `8000`)
- gRPC: `GRPC_PORT` (default `9000`)
