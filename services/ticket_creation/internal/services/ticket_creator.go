// internal/services/ticket_creator.go
package services

import (
    "fmt"
    "log"
    
    "ticket_module/internal/adapters"
    "ticket_module/internal/clients"
    "ticket_module/internal/database"
    "ticket_module/internal/models"
)

type TicketCreatorService struct {
    pythonClient  *clients.PythonClient
    summarizer    *ClaudeSummarizer
    ticketAdapter adapters.TicketSystemAdapter
    repository    *database.TicketRepository
}

func NewTicketCreatorService(
    pythonClient *clients.PythonClient,
    summarizer *ClaudeSummarizer,
    ticketAdapter adapters.TicketSystemAdapter,
    repository *database.TicketRepository,
) *TicketCreatorService {
    return &TicketCreatorService{
        pythonClient:  pythonClient,
        summarizer:    summarizer,
        ticketAdapter: ticketAdapter,
        repository:    repository,
    }
}

// CreateTicket основной метод создания тикета
func (s *TicketCreatorService) CreateTicket(req *models.CreateTicketRequest) (*models.TicketCreated, error) {
    log.Printf("Creating ticket for call_id: %s, intent: %s", 
        req.Transcript.CallID, req.Routing.IntentID)
    
    // 1. Извлекаем сущности через Python NER сервис
    entities, err := s.pythonClient.ExtractEntities(req.Transcript.Segments)
    if err != nil {
        log.Printf("Warning: Entity extraction failed: %v", err)
        entities = &models.Entities{} // Продолжаем с пустыми сущностями
    }
    log.Printf("Extracted entities: %d persons, %d phones, %d emails",
        len(entities.Persons), len(entities.Phones), len(entities.Emails))
    
    // 2. Генерируем заголовок и описание через Claude API
    summary, err := s.summarizer.GenerateSummary(
        req.Transcript.Segments,
        req.Routing.IntentID,
        req.Routing.Priority,
        entities,
    )
    if err != nil {
        return nil, fmt.Errorf("generate summary: %w", err)
    }
    log.Printf("Generated summary: %s", summary.Title)
    
    // 3. Формируем черновик тикета
    draft := s.buildTicketDraft(req, summary, entities)
    
    // 4. Создаем тикет в внешней системе (Mock/Jira/Redmine)
    created, err := s.ticketAdapter.CreateTicket(draft)
    if err != nil {
        return nil, fmt.Errorf("create ticket in external system: %w", err)
    }
    log.Printf("Created ticket in external system: %s (%s)", created.ExternalID, created.System)
    
    // 5. Сохраняем в БД
    if err := s.repository.CreateTicket(draft, created); err != nil {
        // Логируем ошибку, но не фейлим весь процесс
        log.Printf("Warning: Failed to save ticket to DB: %v", err)
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
        assigneeID = "default_support"
    }
    
    // Генерируем теги
    tags := []string{req.Routing.IntentID}
    if req.Routing.Priority == "high" || req.Routing.Priority == "critical" {
        tags = append(tags, "urgent")
    }
    
    // Добавляем извлеченные сущности в описание
    description := summary.Description
    if len(entities.Persons) > 0 {
        description += fmt.Sprintf("\n\nКлиент: %s", entities.Persons[0].Value)
    }
    if len(entities.Phones) > 0 {
        description += fmt.Sprintf("\nТелефон: %s", entities.Phones[0].Value)
    }
    if len(entities.OrderIDs) > 0 {
        description += fmt.Sprintf("\nНомер заказа: %s", entities.OrderIDs[0].Value)
    }
    
    return &models.TicketDraft{
        Title:            summary.Title,
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