// internal/services/orchestrator_service.go
package services

import (
	"fmt"
	"log"
	"time"

	"orchestrator/internal/clients"
)

type OrchestratorService struct {
	transcriptionClient  *clients.TranscriptionClient
	routingClient        *clients.RoutingClient
	ticketClient         *clients.TicketClient
	notificationClient   *clients.NotificationClient
}

func NewOrchestratorService(
	transcriptionClient *clients.TranscriptionClient,
	routingClient *clients.RoutingClient,
	ticketClient *clients.TicketClient,
	notificationClient *clients.NotificationClient,
) *OrchestratorService {
	return &OrchestratorService{
		transcriptionClient:  transcriptionClient,
		routingClient:        routingClient,
		ticketClient:         ticketClient,
		notificationClient:   notificationClient,
	}
}

type ProcessCallResult struct {
	CallID         string                         `json:"call_id"`
	Transcript     *clients.TranscriptionResponse `json:"transcript"`
	Routing        *clients.RoutingResponse       `json:"routing"`
	Entities       *clients.Entities              `json:"entities"`
	Ticket         *clients.TicketCreated         `json:"ticket"`
	Notification   *clients.NotificationResult    `json:"notification,omitempty"`
	ProcessingTime map[string]float64             `json:"processing_time"`
	TotalTime      float64                        `json:"total_time"`
}

// ProcessCall обрабатывает аудио звонка через все модули
func (s *OrchestratorService) ProcessCall(audioPath string) (*ProcessCallResult, error) {
	startTime := time.Now()
	processingTime := make(map[string]float64)

	log.Printf("Starting call processing for audio: %s", audioPath)

	// 1. Транскрибация
	log.Println("Step 1/4: Transcribing audio...")
	stepStart := time.Now()
	transcript, err := s.transcriptionClient.Transcribe(audioPath)
	if err != nil {
		return nil, fmt.Errorf("transcription failed: %w", err)
	}
	processingTime["transcription"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Transcription completed in %.2fs (found %d segments)",
		processingTime["transcription"], len(transcript.Segments))

	// 2. Маршрутизация
	log.Println("Step 2/4: Routing call...")
	stepStart = time.Now()
	routing, err := s.routingClient.Route(transcript.CallID, transcript.Segments)
	if err != nil {
		return nil, fmt.Errorf("routing failed: %w", err)
	}
	processingTime["routing"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Routing completed in %.2fs (intent: %s, priority: %s)",
		processingTime["routing"], routing.IntentID, routing.Priority)

	entities := &clients.Entities{}

	// 3. Создание тикета
	log.Println("Step 3/4: Creating ticket...")
	stepStart = time.Now()
	ticket, err := s.ticketClient.CreateTicket(transcript, routing, entities)
	if err != nil {
		return nil, fmt.Errorf("ticket creation failed: %w", err)
	}
	processingTime["ticket_creation"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Ticket created in %.2fs (ID: %s, URL: %s)",
		processingTime["ticket_creation"], ticket.TicketID, ticket.URL)

	// 4. Отправка уведомлений (non-fatal)
	log.Println("Step 4/4: Sending notifications...")
	stepStart = time.Now()
	var notification *clients.NotificationResult
	notification, err = s.notificationClient.SendNotification(transcript, routing, entities, ticket)
	if err != nil {
		log.Printf("⚠ Notification sending failed (non-fatal): %v", err)
		notification = &clients.NotificationResult{Success: false}
	}
	processingTime["notification"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Notifications sent in %.2fs (success: %v)",
		processingTime["notification"], notification.Success)

	totalTime := time.Since(startTime).Seconds()
	log.Printf("Call processing completed successfully in %.2fs", totalTime)

	return &ProcessCallResult{
		CallID:         transcript.CallID,
		Transcript:     transcript,
		Routing:        routing,
		Entities:       entities,
		Ticket:         ticket,
		Notification:   notification,
		ProcessingTime: processingTime,
		TotalTime:      totalTime,
	}, nil
}

// HealthCheck проверяет доступность всех сервисов
func (s *OrchestratorService) HealthCheck() map[string]string {
	// TODO: можно добавить проверку health endpoints всех сервисов
	return map[string]string{
		"orchestrator":  "healthy",
		"transcription": "unknown",
		"routing":       "unknown",
		"ticket":        "unknown",
	}
}
