// internal/services/orchestrator_service.go
package services

import (
	"fmt"
	"log"
	"strings"
	"time"

	"orchestrator/internal/clients"
)

const (
	ProcessStatusCompleted             = "completed"
	ProcessStatusAwaitingRoutingReview = "awaiting_routing_review"
	ProcessStatusSpamBlocked           = "spam_blocked"
	ProcessStatusNoSpeech              = "no_speech"
	spamIntentID                       = "spam.call"
	legacySpamIntentID                 = "spam"
)

type OrchestratorService struct {
	transcriptionClient              *clients.TranscriptionClient
	routingClient                    *clients.RoutingClient
	ticketClient                     *clients.TicketClient
	entityClient                     *clients.EntityClient
	routingReviewConfidenceThreshold float64
}

func NewOrchestratorService(
	transcriptionClient *clients.TranscriptionClient,
	routingClient *clients.RoutingClient,
	ticketClient *clients.TicketClient,
	entityClient *clients.EntityClient,
	routingReviewConfidenceThreshold float64,
) *OrchestratorService {
	if routingReviewConfidenceThreshold < 0.0 {
		routingReviewConfidenceThreshold = 0.0
	}
	if routingReviewConfidenceThreshold > 1.0 {
		routingReviewConfidenceThreshold = 1.0
	}
	return &OrchestratorService{
		transcriptionClient:              transcriptionClient,
		routingClient:                    routingClient,
		ticketClient:                     ticketClient,
		entityClient:                     entityClient,
		routingReviewConfidenceThreshold: routingReviewConfidenceThreshold,
	}
}

type ProcessCallResult struct {
	QueueID           string                         `json:"queue_id,omitempty"`
	CallID            string                         `json:"call_id"`
	Status            string                         `json:"status"`
	Transcript        *clients.TranscriptionResponse `json:"transcript"`
	Routing           *clients.RoutingResponse       `json:"routing"`
	SpamCheck         *clients.SpamCheckResponse     `json:"spam_check,omitempty"`
	Entities          *clients.Entities              `json:"entities"`
	Ticket            *clients.TicketCreated         `json:"ticket"`
	ProcessingTime    map[string]float64             `json:"processing_time"`
	TotalTime         float64                        `json:"total_time"`
	RequestReceivedAt string                         `json:"request_received_at,omitempty"`
	ProcessedAt       string                         `json:"processed_at,omitempty"`
}

type ContinueAfterRoutingReviewInput struct {
	CallID         string
	SourceFilename string
	Decision       string
	Transcript     *clients.TranscriptionResponse
	Routing        *clients.RoutingResponse
}

type ContinueAfterSpamOverrideInput struct {
	CallID         string
	SourceFilename string
	Decision       string
	Transcript     *clients.TranscriptionResponse
	Routing        *clients.RoutingResponse
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

func isSpamBlocked(spamCheck *clients.SpamCheckResponse) bool {
	if spamCheck == nil {
		return false
	}
	return spamCheck.Status == "block" || spamCheck.Status == "review"
}

func isSpamIntent(intentID string) bool {
	raw := strings.ToLower(strings.TrimSpace(intentID))
	return raw == spamIntentID || raw == legacySpamIntentID
}

func isSpamBlockedRouting(routing *clients.RoutingResponse) bool {
	if routing == nil {
		return false
	}
	return isSpamIntent(routing.IntentID) || isSpamBlocked(routing.SpamCheck)
}

func isSpamConflictReview(spamCheck *clients.SpamCheckResponse) bool {
	if spamCheck == nil {
		return false
	}
	if spamCheck.Status != "review" {
		return false
	}
	return strings.HasPrefix(strings.TrimSpace(spamCheck.Reason), "spam_nonspam_conflict:")
}

func isRoutingReviewRequired(routing *clients.RoutingResponse, threshold float64) bool {
	if routing == nil || threshold <= 0.0 {
		return false
	}
	confidence := routing.IntentConfidence
	return confidence >= 0.0 && confidence < threshold
}

func cloneSpamCheck(spamCheck *clients.SpamCheckResponse) *clients.SpamCheckResponse {
	if spamCheck == nil {
		return nil
	}
	copied := *spamCheck
	return &copied
}

func isTranscriptEmpty(transcript *clients.TranscriptionResponse) bool {
	if transcript == nil {
		return true
	}
	return len(transcript.Segments) == 0
}

func ensureTranscriptCallID(transcript *clients.TranscriptionResponse, fallback string) string {
	if transcript == nil {
		if fallback == "" {
			return "unknown-call"
		}
		return fallback
	}
	if transcript.CallID == "" {
		transcript.CallID = fallback
	}
	if transcript.CallID == "" {
		transcript.CallID = "unknown-call"
	}
	return transcript.CallID
}

func buildProcessResult(
	status string,
	transcript *clients.TranscriptionResponse,
	routing *clients.RoutingResponse,
	entities *clients.Entities,
	ticket *clients.TicketCreated,
	startTime time.Time,
	processingTime map[string]float64,
) *ProcessCallResult {
	return &ProcessCallResult{
		CallID:         ensureTranscriptCallID(transcript, ""),
		Status:         status,
		Transcript:     transcript,
		Routing:        routing,
		SpamCheck:      cloneSpamCheck(spamCheckFromRouting(routing)),
		Entities:       normalizeEntities(entities),
		Ticket:         ticket,
		ProcessingTime: processingTime,
		TotalTime:      time.Since(startTime).Seconds(),
	}
}

func spamCheckFromRouting(routing *clients.RoutingResponse) *clients.SpamCheckResponse {
	if routing == nil {
		return nil
	}
	return routing.SpamCheck
}

func (s *OrchestratorService) routeTranscript(transcript *clients.TranscriptionResponse) (*clients.RoutingResponse, error) {
	if transcript == nil {
		return nil, fmt.Errorf("transcript is required")
	}
	return s.routingClient.Route(transcript.CallID, transcript.Segments)
}

func (s *OrchestratorService) routeTranscriptSkippingSpam(transcript *clients.TranscriptionResponse) (*clients.RoutingResponse, error) {
	if transcript == nil {
		return nil, fmt.Errorf("transcript is required")
	}
	return s.routingClient.RouteSkippingSpam(transcript.CallID, transcript.Segments)
}

func (s *OrchestratorService) completeNonSpamCall(
	transcript *clients.TranscriptionResponse,
	routing *clients.RoutingResponse,
	status string,
	startTime time.Time,
	processingTime map[string]float64,
) (*ProcessCallResult, error) {
	log.Println("Step 3/4: Extracting entities...")
	stepStart := time.Now()
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

	log.Println("Step 4/4: Creating ticket...")
	stepStart = time.Now()
	ticket, err := s.ticketClient.CreateTicket(transcript, routing, entities)
	if err != nil {
		return nil, fmt.Errorf("ticket creation failed: %w", err)
	}
	processingTime["ticket_creation"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Ticket created in %.2fs (ID: %s, URL: %s)",
		processingTime["ticket_creation"], ticket.TicketID, ticket.URL)

	result := buildProcessResult(status, transcript, routing, entities, ticket, startTime, processingTime)
	log.Printf("Call processing completed successfully in %.2fs", result.TotalTime)
	return result, nil
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

	if isTranscriptEmpty(transcript) {
		result := buildProcessResult(ProcessStatusNoSpeech, transcript, nil, nil, nil, startTime, processingTime)
		log.Printf("Call processing finished without routing: no speech/segments detected in %.2fs", result.TotalTime)
		return result, nil
	}

	// 2. Маршрутизация
	log.Println("Step 2/5: Routing call...")
	stepStart = time.Now()
	routing, err := s.routeTranscript(transcript)
	if err != nil {
		return nil, fmt.Errorf("routing failed: %w", err)
	}
	processingTime["routing"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Routing completed in %.2fs (intent: %s, priority: %s)",
		processingTime["routing"], routing.IntentID, routing.Priority)

	if isSpamConflictReview(routing.SpamCheck) {
		result := buildProcessResult(ProcessStatusAwaitingRoutingReview, transcript, routing, nil, nil, startTime, processingTime)
		log.Printf(
			"Call processing paused for manual routing review due to spam/non-spam conflict in %.2fs (intent=%s confidence=%.3f)",
			result.TotalTime,
			routing.IntentID,
			routing.IntentConfidence,
		)
		return result, nil
	}
	if isRoutingReviewRequired(routing, s.routingReviewConfidenceThreshold) {
		result := buildProcessResult(ProcessStatusAwaitingRoutingReview, transcript, routing, nil, nil, startTime, processingTime)
		log.Printf(
			"Call processing paused for manual routing review in %.2fs (confidence=%.3f threshold=%.3f)",
			result.TotalTime,
			routing.IntentConfidence,
			s.routingReviewConfidenceThreshold,
		)
		return result, nil
	}
	if isSpamBlockedRouting(routing) {
		result := buildProcessResult(ProcessStatusSpamBlocked, transcript, routing, nil, nil, startTime, processingTime)
		log.Printf("Call classified as spam in %.2fs", result.TotalTime)
		return result, nil
	}
	return s.completeNonSpamCall(transcript, routing, ProcessStatusCompleted, startTime, processingTime)
}

func (s *OrchestratorService) ContinueAfterRoutingReview(input ContinueAfterRoutingReviewInput) (*ProcessCallResult, error) {
	startTime := time.Now()
	processingTime := make(map[string]float64)
	decision := strings.ToLower(strings.TrimSpace(input.Decision))

	if input.Transcript == nil {
		return nil, fmt.Errorf("transcript is required")
	}
	if input.Routing == nil {
		return nil, fmt.Errorf("routing is required")
	}
	if decision != "accepted" && decision != "rejected" {
		return nil, fmt.Errorf("decision must be accepted or rejected")
	}
	callID := ensureTranscriptCallID(input.Transcript, input.CallID)

	log.Printf(
		"Completing call after manual routing review: call_id=%s decision=%s intent=%s confidence=%.3f",
		callID,
		decision,
		input.Routing.IntentID,
		input.Routing.IntentConfidence,
	)

	return s.completeNonSpamCall(input.Transcript, input.Routing, ProcessStatusCompleted, startTime, processingTime)
}

func (s *OrchestratorService) ContinueAfterSpamOverride(input ContinueAfterSpamOverrideInput) (*ProcessCallResult, error) {
	startTime := time.Now()
	processingTime := make(map[string]float64)
	decision := strings.ToLower(strings.TrimSpace(input.Decision))

	if input.Transcript == nil {
		return nil, fmt.Errorf("transcript is required")
	}
	if input.Routing == nil {
		return nil, fmt.Errorf("routing is required")
	}
	if decision != "accepted" {
		return nil, fmt.Errorf("spam override supports only accepted decision")
	}
	if !isSpamIntent(input.Routing.IntentID) {
		return nil, fmt.Errorf("spam override requires current spam routing")
	}

	callID := ensureTranscriptCallID(input.Transcript, input.CallID)
	log.Printf("Continuing call after spam override: call_id=%s rerouting without spam intent", callID)

	stepStart := time.Now()
	rerouted, err := s.routeTranscriptSkippingSpam(input.Transcript)
	if err != nil {
		return nil, fmt.Errorf("routing after spam override failed: %w", err)
	}
	processingTime["routing"] = time.Since(stepStart).Seconds()
	log.Printf(
		"✓ Routing after spam override completed in %.2fs (intent: %s, priority: %s)",
		processingTime["routing"],
		rerouted.IntentID,
		rerouted.Priority,
	)

	if isRoutingReviewRequired(rerouted, s.routingReviewConfidenceThreshold) {
		result := buildProcessResult(ProcessStatusAwaitingRoutingReview, input.Transcript, rerouted, nil, nil, startTime, processingTime)
		log.Printf(
			"Call processing paused for manual routing review after spam override in %.2fs (confidence=%.3f threshold=%.3f)",
			result.TotalTime,
			rerouted.IntentConfidence,
			s.routingReviewConfidenceThreshold,
		)
		return result, nil
	}
	if isSpamIntent(rerouted.IntentID) {
		return nil, fmt.Errorf("routing after spam override returned spam intent again")
	}

	return s.completeNonSpamCall(input.Transcript, rerouted, ProcessStatusCompleted, startTime, processingTime)
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
