// internal/services/orchestrator_service.go
package services

import (
	"fmt"
	"log"
	"time"

	"orchestrator/internal/clients"
)

type OrchestratorService struct {
	transcriptionClient *clients.TranscriptionClient
	routingClient       *clients.RoutingClient
	ticketClient        *clients.TicketClient
	notificationClient  *clients.NotificationClient
	entityClient        *clients.EntityClient
}

func NewOrchestratorService(
	transcriptionClient *clients.TranscriptionClient,
	routingClient *clients.RoutingClient,
	ticketClient *clients.TicketClient,
	notificationClient *clients.NotificationClient,
	entityClient *clients.EntityClient,
) *OrchestratorService {
	return &OrchestratorService{
		transcriptionClient: transcriptionClient,
		routingClient:       routingClient,
		ticketClient:        ticketClient,
		notificationClient:  notificationClient,
		entityClient:        entityClient,
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

func emptyEntities() *clients.Entities {
	return &clients.Entities{
		Persons:      []clients.ExtractedEntity{},
		Phones:       []clients.ExtractedEntity{},
		Emails:       []clients.ExtractedEntity{},
		OrderIDs:     []clients.ExtractedEntity{},
		AccountIDs:   []clients.ExtractedEntity{},
		MoneyAmounts: []clients.ExtractedEntity{},
		Dates:        []clients.ExtractedEntity{},
	}
}

func normalizeEntities(e *clients.Entities) *clients.Entities {
	if e == nil {
		return emptyEntities()
	}
	if e.Persons == nil {
		e.Persons = []clients.ExtractedEntity{}
	}
	if e.Phones == nil {
		e.Phones = []clients.ExtractedEntity{}
	}
	if e.Emails == nil {
		e.Emails = []clients.ExtractedEntity{}
	}
	if e.OrderIDs == nil {
		e.OrderIDs = []clients.ExtractedEntity{}
	}
	if e.AccountIDs == nil {
		e.AccountIDs = []clients.ExtractedEntity{}
	}
	if e.MoneyAmounts == nil {
		e.MoneyAmounts = []clients.ExtractedEntity{}
	}
	if e.Dates == nil {
		e.Dates = []clients.ExtractedEntity{}
	}
	return e
}

// ProcessCall обрабатывает аудио звонка через все модули
func (s *OrchestratorService) ProcessCall(audioPath string) (*ProcessCallResult, error) {
	startTime := time.Now()
	processingTime := make(map[string]float64)

	log.Printf("Starting call processing for audio: %s", audioPath)

	// 1. Транскрибация
	log.Println("Step 1/5: Transcribing audio...")
	stepStart := time.Now()
	transcript, err := s.transcriptionClient.Transcribe(audioPath)
	if err != nil {
		return nil, fmt.Errorf("transcription failed: %w", err)
	}
	processingTime["transcription"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Transcription completed in %.2fs (found %d segments)",
		processingTime["transcription"], len(transcript.Segments))

	// 2. Маршрутизация
	log.Println("Step 2/5: Routing call...")
	stepStart = time.Now()
	routing, err := s.routingClient.Route(transcript.CallID, transcript.Segments)
	if err != nil {
		return nil, fmt.Errorf("routing failed: %w", err)
	}
	processingTime["routing"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Routing completed in %.2fs (intent: %s, priority: %s)",
		processingTime["routing"], routing.IntentID, routing.Priority)

	// 3. Извлечение сущностей (non-fatal)
	log.Println("Step 3/5: Extracting entities...")
	stepStart = time.Now()
	entities := emptyEntities()
	if s.entityClient != nil {
		extracted, extractErr := s.entityClient.Extract(transcript.Segments)
		if extractErr != nil {
			log.Printf("⚠ Entity extraction failed (non-fatal): %v", extractErr)
		} else {
			entities = normalizeEntities(extracted)
		}
	}
	processingTime["entity_extraction"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Entity extraction completed in %.2fs (order_ids: %d, phones: %d, emails: %d)",
		processingTime["entity_extraction"], len(entities.OrderIDs), len(entities.Phones), len(entities.Emails))

	// 4. Создание тикета
	log.Println("Step 4/5: Creating ticket...")
	stepStart = time.Now()
	ticket, err := s.ticketClient.CreateTicket(transcript, routing, entities)
	if err != nil {
		return nil, fmt.Errorf("ticket creation failed: %w", err)
	}
	processingTime["ticket_creation"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Ticket created in %.2fs (ID: %s, URL: %s)",
		processingTime["ticket_creation"], ticket.TicketID, ticket.URL)

	// 5. Отправка уведомлений (non-fatal)
	log.Println("Step 5/5: Sending notifications...")
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
