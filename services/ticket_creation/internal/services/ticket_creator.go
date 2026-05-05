// internal/services/ticket_creator.go
package services

import (
	"fmt"
	"log"
	"strings"
	"time"

	"ticket_module/internal/adapters"
	"ticket_module/internal/clients"
	"ticket_module/internal/database"
	"ticket_module/internal/models"
)


type TicketCreatorService struct {
	pythonClient            *clients.PythonClient
	summarizer              TicketSummarizer
	ticketAdapter           adapters.TicketSystemAdapter
	repository              *database.TicketRepository
	includePIIInDescription bool
}

func NewTicketCreatorService(
	pythonClient *clients.PythonClient,
	summarizer TicketSummarizer,
	ticketAdapter adapters.TicketSystemAdapter,
	repository *database.TicketRepository,
	includePIIInDescription bool,
) *TicketCreatorService {
	return &TicketCreatorService{
		pythonClient:            pythonClient,
		summarizer:              summarizer,
		ticketAdapter:           ticketAdapter,
		repository:              repository,
		includePIIInDescription: includePIIInDescription,
	}
}

// CreateTicket основной метод создания тикета
func (s *TicketCreatorService) CreateTicket(req *models.CreateTicketRequest) (*models.TicketCreated, error) {
	log.Printf("Creating ticket for call_id: %s, intent: %s",
		req.Transcript.CallID, req.Routing.IntentID)

	// 1. Используем уже извлеченные сущности, если orchestrator их передал.
	entities := req.Entities
	if entities == nil {
		entities = &models.Entities{}
	}
	if req.Entities == nil && s.pythonClient != nil {
		extracted, err := s.pythonClient.ExtractEntities(req.Transcript.Segments)
		if err != nil {
			log.Printf("Warning: Entity extraction failed: %v", err)
		} else {
			entities = extracted
		}
	}
	log.Printf("Extracted entities: %d persons, %d phones, %d emails",
		len(entities.Persons), len(entities.Phones), len(entities.Emails))

	// 2. Генерируем заголовок и описание через LLM
	summary, err := s.summarizer.GenerateSummary(
		req.Transcript.Segments,
		req.Routing.IntentID,
		req.Routing.Priority,
		entities,
	)
	if err != nil {
		return nil, fmt.Errorf("generate summary: %w", err)
	}
	log.Printf("Generated ticket summary successfully")

	// 3. Формируем черновик тикета
	draft := s.buildTicketDraft(req, summary, entities)
	payload := buildTicketSystemPayload(req, draft, summary, entities)

	// 4. Создаем тикет в внешней системе (Mock/Jira/Redmine)
	created, err := s.ticketAdapter.CreateTicket(payload)
	if err != nil {
		return nil, fmt.Errorf("create ticket in external system: %w", err)
	}
	log.Printf("Created ticket in external system: %s (%s)", created.ExternalID, created.System)

	// 5. Сохраняем в БД
	if s.repository != nil {
		if err := s.repository.CreateTicket(draft, created); err != nil {
			// Логируем ошибку, но не фейлим весь процесс
			log.Printf("Warning: Failed to save ticket to DB: %v", err)
		}
	}

	log.Printf("Ticket created successfully: %s", created.TicketID)
	return created, nil
}

// buildTicketDraft формирует черновик тикета из данных
func (s *TicketCreatorService) buildTicketDraft(
	req *models.CreateTicketRequest,
	summary *models.TicketSummary,
	entities *models.Entities,
) *models.TicketDraft {

	// Определяем assignee на основе routing
	assigneeType := "group"
	assigneeID := req.Routing.SuggestedGroup
	if assigneeID == "" {
		assigneeID = "support"
	}

	// Генерируем теги
	tags := []string{req.Routing.IntentID}
	if req.Routing.Priority == "high" || req.Routing.Priority == "critical" {
		tags = append(tags, "urgent")
	}

	// Добавляем извлеченные сущности в описание только если это явно разрешено.
	description := composeTicketDescription(summary)
	if s.includePIIInDescription {
		description = appendEntityDetails(description, entities)
	}

	return &models.TicketDraft{
		Title:            buildTicketTitle(req.Routing.IntentID),
		Description:      description,
		Priority:         req.Routing.Priority,
		AssigneeType:     assigneeType,
		AssigneeID:       assigneeID,
		Tags:             tags,
		CallID:           req.Transcript.CallID,
		AudioURL:         req.AudioURL,
		IntentID:         req.Routing.IntentID,
		IntentConfidence: req.Routing.IntentConfidence,
		Entities:         entities,
	}
}

func composeTicketDescription(summary *models.TicketSummary) string {
	if summary == nil {
		return "Автоматически созданный тикет из транскрипции звонка."
	}

	sections := make([]string, 0, 4)
	if description := strings.TrimSpace(summary.Description); description != "" {
		sections = append(sections, description)
	}
	if len(summary.KeyPoints) > 0 {
		sections = append(sections, "Ключевые моменты:\n- "+strings.Join(summary.KeyPoints, "\n- "))
	}
	if suggestedSolution := strings.TrimSpace(summary.SuggestedSolution); suggestedSolution != "" {
		sections = append(sections, "Предлагаемое решение:\n"+suggestedSolution)
	}
	if urgencyReason := strings.TrimSpace(summary.UrgencyReason); urgencyReason != "" {
		sections = append(sections, "Причина срочности:\n"+urgencyReason)
	}
	if len(sections) == 0 {
		return "Автоматически созданный тикет из транскрипции звонка."
	}
	return strings.Join(sections, "\n\n")
}

func buildTicketTitle(intentID string) string {
	intentID = collapseWhitespace(intentID)
	if intentID == "" {
		intentID = "общий запрос"
	}
	return truncateRunes("Обращение: "+intentID, maxTicketTitleRunes)
}

func appendEntityDetails(description string, entities *models.Entities) string {
	if entities == nil {
		return description
	}

	items := make([]string, 0, 5)
	if len(entities.Persons) > 0 {
		items = append(items, fmt.Sprintf("Имя: %s", entities.Persons[0].Value))
	}
	if len(entities.Phones) > 0 {
		items = append(items, fmt.Sprintf("Телефон: %s", entities.Phones[0].Value))
	}
	if len(entities.Emails) > 0 {
		items = append(items, fmt.Sprintf("Email: %s", entities.Emails[0].Value))
	}
	if len(entities.OrderIDs) > 0 {
		items = append(items, fmt.Sprintf("Номер заказа: %s", entities.OrderIDs[0].Value))
	}
	if len(entities.AccountIDs) > 0 {
		items = append(items, fmt.Sprintf("Аккаунт: %s", entities.AccountIDs[0].Value))
	}
	if len(items) == 0 {
		return description
	}

	extraInfo := "Доп. информация:\n- " + strings.Join(items, "\n- ")
	if strings.TrimSpace(description) == "" {
		return extraInfo
	}
	return description + "\n\n" + extraInfo
}

func buildTicketSystemPayload(
	req *models.CreateTicketRequest,
	draft *models.TicketDraft,
	summary *models.TicketSummary,
	entities *models.Entities,
) *models.TicketSystemPayload {
	requestCopy := models.CreateTicketRequest{
		Transcript: req.Transcript,
		Routing:    req.Routing,
		Entities:   entities,
		AudioURL:   req.AudioURL,
	}

	return &models.TicketSystemPayload{
		Service: models.TicketServiceMetadata{
			Source:        "ts_calls_automation",
			Component:     "ticket_creation",
			SchemaVersion: "v1",
			SentAt:        time.Now().UTC(),
			CallID:        req.Transcript.CallID,
			IntentID:      req.Routing.IntentID,
			Priority:      req.Routing.Priority,
		},
		Request: requestCopy,
		Summary: summary,
		Draft:   draft,
	}
}

// GetTicket получает информацию о тикете
func (s *TicketCreatorService) GetTicket(ticketID string) (*models.TicketRecord, error) {
	return s.repository.GetTicket(ticketID)
}

// ListTickets получает список тикетов
func (s *TicketCreatorService) ListTickets(filters map[string]interface{}, limit, offset int) ([]models.TicketRecord, error) {
	return s.repository.ListTickets(filters, limit, offset)
}

// UpdateTicketStatus обновляет статус тикета
func (s *TicketCreatorService) UpdateTicketStatus(ticketID, status string) error {
	return s.repository.UpdateTicketStatus(ticketID, status)
}

// GetStats получает статистику по тикетам
func (s *TicketCreatorService) GetStats() (map[string]interface{}, error) {
	return s.repository.GetTicketStats()
}
