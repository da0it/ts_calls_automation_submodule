// internal/clients/routing_client.go
package clients

import (
	"context"
	"fmt"
	"time"

	callprocessingv1 "orchestrator/internal/gen"

	"google.golang.org/grpc"
)

// Структура RoutingClient хранит все, что нужно для вызова router-service
type RoutingClient struct {

	// gRPC-подключение к router-service
	conn *grpc.ClientConn

	// Сгенерированный protobuf-клиент
	client callprocessingv1.RoutingServiceClient
}

// Функция-конструктор. Создает клиента router-service. Принимает на вход адрес сервиса маршрутизации, возвращает клиент.
func NewRoutingClient(addr string) (*RoutingClient, error) {

	// Создание gRPC-подключения через вспомогательную функцию в grpc_dial.go
	conn, err := grpcConnForService(addr, "ROUTING_GRPC")
	if err != nil {
		return nil, fmt.Errorf("dial routing grpc: %w", err)
	}

	// Если подключение создано, возвращается новый RoutingClient
	return &RoutingClient{
		conn:   conn,
		client: callprocessingv1.NewRoutingServiceClient(conn),
	}, nil
}

// Структура описывает запрос на маршрутизацию в JSON-формате
type RoutingRequest struct {
	CallID   string    `json:"call_id"`
	Segments []Segment `json:"segments"`
}

// Результат маршрутизации, который client возвращает orchestrator
type RoutingResponse struct {
	IntentID         string             `json:"intent_id"`
	IntentConfidence float64            `json:"intent_confidence"`
	Priority         string             `json:"priority"`
	SuggestedGroup   string             `json:"suggested_group,omitempty"`
	SpamCheck        *SpamCheckResponse `json:"spam_check,omitempty"`
}

// Структура описывает результат проверки на спам
type SpamCheckResponse struct {
	Status         string  `json:"status"`
	PredictedLabel string  `json:"predicted_label,omitempty"`
	Confidence     float64 `json:"confidence"`
	ThresholdLow   float64 `json:"threshold_low"`
	ThresholdHigh  float64 `json:"threshold_high"`
	Reason         string  `json:"reason,omitempty"`
	Skipped        bool    `json:"skipped,omitempty"`
	Backend        string  `json:"backend,omitempty"`
}

// Публичный метод для обычной маршрутизации.
func (c *RoutingClient) Route(callID string, segments []Segment) (*RoutingResponse, error) {
	return c.route(callID, segments, false)
}

// Метод также выполняет маршрутизацию, но просит router-service пропустить проверку на спам
func (c *RoutingClient) RouteSkippingSpam(callID string, segments []Segment) (*RoutingResponse, error) {
	return c.route(callID, segments, true)
}

// Внутренний метод route содержит общую логику gRPC-вызова
func (c *RoutingClient) route(callID string, segments []Segment, skipSpamGate bool) (*RoutingResponse, error) {

	// Создается контекст с таймаутом 60 секунд
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Создается список protobuf-сегментов.
	pbSegments := make([]*callprocessingv1.Segment, 0, len(segments))
	for _, seg := range segments {

		// Каждый внутренний Segment преобразуется в protobuf-сегмент
		pbSegments = append(pbSegments, &callprocessingv1.Segment{
			Start:   seg.Start,
			End:     seg.End,
			Speaker: seg.Speaker,
			Role:    seg.Role,
			Text:    seg.Text,
		})
	}

	// gRPC-вызов router-service
	resp, err := c.client.Route(ctx, &callprocessingv1.RouteRequest{
		CallId:       callID,
		Segments:     pbSegments,
		SkipSpamGate: skipSpamGate,
	})
	if err != nil {
		return nil, fmt.Errorf("routing rpc: %w", err)
	}

	if resp.GetRouting() == nil {
		return nil, fmt.Errorf("routing rpc: empty response")
	}

	// Protobuf-структура spam-check преобразуется во внутреннюю структуру клиента.
	var spamCheck *SpamCheckResponse
	if raw := resp.GetRouting().GetSpamCheck(); raw != nil {
		spamCheck = &SpamCheckResponse{
			Status:         raw.GetStatus(),
			PredictedLabel: raw.GetPredictedLabel(),
			Confidence:     raw.GetConfidence(),
			ThresholdLow:   raw.GetThresholdLow(),
			ThresholdHigh:  raw.GetThresholdHigh(),
			Reason:         raw.GetReason(),
			Skipped:        raw.GetSkipped(),
			Backend:        raw.GetBackend(),
		}
	}

	// Метод возвращает результат маршрутизации
	return &RoutingResponse{
		IntentID:         resp.GetRouting().GetIntentId(),
		IntentConfidence: resp.GetRouting().GetIntentConfidence(),
		Priority:         resp.GetRouting().GetPriority(),
		SuggestedGroup:   resp.GetRouting().GetSuggestedGroup(),
		SpamCheck:        spamCheck,
	}, nil
}

// Закрытие gRPC-соединения
func (c *RoutingClient) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}
