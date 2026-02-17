# Stage 1: Сборка
FROM golang:1.23-alpine AS builder

RUN apk add --no-cache git

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /ticket-service ./cmd/server

# Stage 2: Финальный образ
FROM alpine:3.19

RUN apk add --no-cache ca-certificates tzdata

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

WORKDIR /app

COPY --from=builder /ticket-service .
COPY migrations/ ./migrations/

USER appuser

EXPOSE 8080

ENTRYPOINT ["./ticket-service"]
