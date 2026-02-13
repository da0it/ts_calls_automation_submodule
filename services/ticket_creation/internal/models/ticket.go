// internal/models/ticket.go
package models

import "time"

// Segment представляет сегмент диалога из транскрипции
type Segment struct {
    Start   float64 `json:"start"`
    End     float64 `json:"end"`
    Speaker string  `json:"speaker"`
    Role    string  `json:"role"`
    Text    string  `json:"text"`
}

// TranscriptData данные транскрипции
type TranscriptData struct {
    CallID      string              `json:"call_id"`
    Segments    []Segment           `json:"segments"`
    RoleMapping map[string]string   `json:"role_mapping"`
    Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// RoutingData данные маршрутизации
type RoutingData struct {
    IntentID         string  `json:"intent_id"`
    IntentConfidence float64 `json:"intent_confidence"`
    Priority         string  `json:"priority"`
    SuggestedGroup   string  `json:"suggested_group,omitempty"`
}

// ExtractedEntity извлеченная сущность
type ExtractedEntity struct {
    Type       string  `json:"type"`       // person, phone, email, order_id, etc.
    Value      string  `json:"value"`
    Confidence float64 `json:"confidence"`
    Context    string  `json:"context"`
}

// Entities все извлеченные сущности
type Entities struct {
    Persons      []ExtractedEntity `json:"persons"`
    Phones       []ExtractedEntity `json:"phones"`
    Emails       []ExtractedEntity `json:"emails"`
    OrderIDs     []ExtractedEntity `json:"order_ids"`
    AccountIDs   []ExtractedEntity `json:"account_ids"`
    MoneyAmounts []ExtractedEntity `json:"money_amounts"`
    Dates        []ExtractedEntity `json:"dates"`
}

// TicketSummary сгенерированное описание тикета
type TicketSummary struct {
    Title             string   `json:"title"`
    Description       string   `json:"description"`
    KeyPoints         []string `json:"key_points"`
    SuggestedSolution string   `json:"suggested_solution,omitempty"`
    UrgencyReason     string   `json:"urgency_reason,omitempty"`
}

// TicketDraft черновик тикета
type TicketDraft struct {
    Title        string                 `json:"title"`
    Description  string                 `json:"description"`
    Priority     string                 `json:"priority"`
    AssigneeType string                 `json:"assignee_type"` // user, group, queue
    AssigneeID   string                 `json:"assignee_id"`
    Tags         []string               `json:"tags"`
    CustomFields map[string]interface{} `json:"custom_fields,omitempty"`
    
    // Связи
    CallID        string `json:"call_id,omitempty"`
    AudioURL      string `json:"audio_url,omitempty"`
    TranscriptURL string `json:"transcript_url,omitempty"`
    
    // Метаданные
    IntentID         string    `json:"intent_id,omitempty"`
    IntentConfidence float64   `json:"intent_confidence,omitempty"`
    Entities         *Entities `json:"entities,omitempty"`
}

// TicketCreated результат создания тикета
type TicketCreated struct {
    TicketID   string    `json:"ticket_id"`
    ExternalID string    `json:"external_id"` // ID в Jira/Redmine
    URL        string    `json:"url"`
    System     string    `json:"system"` // jira, redmine, mock
    CreatedAt  time.Time `json:"created_at"`
}

// TicketRecord запись в БД
type TicketRecord struct {
    ID               int64     `db:"id"`
    TicketID         string    `db:"ticket_id"`
    ExternalID       string    `db:"external_id"`
    CallID           string    `db:"call_id"`
    Title            string    `db:"title"`
    Description      string    `db:"description"`
    Priority         string    `db:"priority"`
    Status           string    `db:"status"`
    AssigneeType     string    `db:"assignee_type"`
    AssigneeID       string    `db:"assignee_id"`
    IntentID         *string   `db:"intent_id"`
    IntentConfidence *float64  `db:"intent_confidence"`
    EntitiesJSON     string    `db:"entities_json"`
    URL              *string   `db:"url"`
    System           string    `db:"system"`
    CreatedAt        time.Time `db:"created_at"`
    UpdatedAt        time.Time `db:"updated_at"`
}

// CreateTicketRequest запрос на создание тикета
type CreateTicketRequest struct {
    Transcript TranscriptData `json:"transcript"`
    Routing    RoutingData    `json:"routing"`
    AudioURL   string         `json:"audio_url,omitempty"`
}

// CreateTicketResponse ответ на создание тикета
type CreateTicketResponse struct {
    Success bool           `json:"success"`
    Ticket  *TicketCreated `json:"ticket,omitempty"`
    Error   string         `json:"error,omitempty"`
}