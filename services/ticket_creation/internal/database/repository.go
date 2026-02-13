// internal/database/repository.go
package database

import (
    "database/sql"
    "encoding/json"
    "fmt"
    "time"
    
    "github.com/google/uuid"
    "ticket_module/internal/models"
)

type TicketRepository struct {
    db *Database
}

func NewTicketRepository(db *Database) *TicketRepository {
    return &TicketRepository{db: db}
}

// CreateTicket создает новую запись о тикете в БД
func (r *TicketRepository) CreateTicket(draft *models.TicketDraft, created *models.TicketCreated) error {
    ticketID := uuid.New().String()
    
    // Сериализуем entities в JSON
    entitiesJSON := "{}"
    if draft.Entities != nil {
        data, err := json.Marshal(draft.Entities)
        if err != nil {
            return fmt.Errorf("marshal entities: %w", err)
        }
        entitiesJSON = string(data)
    }
    
    query := `
        INSERT INTO tickets (
            ticket_id, external_id, call_id,
            title, description, priority, status,
            assignee_type, assignee_id,
            intent_id, intent_confidence,
            entities_json, url, system
        ) VALUES (
            $1, $2, $3,
            $4, $5, $6, $7,
            $8, $9,
            $10, $11,
            $12, $13, $14
        )
        RETURNING id, created_at, updated_at
    `
    
    var id int64
    var createdAt, updatedAt time.Time
    
    err := r.db.DB.QueryRow(
        query,
        ticketID,
        created.ExternalID,
        draft.CallID,
        draft.Title,
        draft.Description,
        draft.Priority,
        "open", // default status
        draft.AssigneeType,
        draft.AssigneeID,
        draft.IntentID,
        draft.IntentConfidence,
        entitiesJSON,
        created.URL,
        created.System,
    ).Scan(&id, &createdAt, &updatedAt)
    
    if err != nil {
        return fmt.Errorf("insert ticket: %w", err)
    }
    
    created.TicketID = ticketID
    created.CreatedAt = createdAt
    
    return nil
}

// GetTicket получает тикет по ticket_id
func (r *TicketRepository) GetTicket(ticketID string) (*models.TicketRecord, error) {
    query := `
        SELECT 
            id, ticket_id, external_id, call_id,
            title, description, priority, status,
            assignee_type, assignee_id,
            intent_id, intent_confidence,
            entities_json, url, system,
            created_at, updated_at
        FROM tickets
        WHERE ticket_id = $1
    `
    
    var record models.TicketRecord
    err := r.db.DB.Get(&record, query, ticketID)
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, fmt.Errorf("ticket not found")
        }
        return nil, fmt.Errorf("query ticket: %w", err)
    }
    
    return &record, nil
}

// GetTicketByCallID получает тикет по call_id
func (r *TicketRepository) GetTicketByCallID(callID string) (*models.TicketRecord, error) {
    query := `
        SELECT 
            id, ticket_id, external_id, call_id,
            title, description, priority, status,
            assignee_type, assignee_id,
            intent_id, intent_confidence,
            entities_json, url, system,
            created_at, updated_at
        FROM tickets
        WHERE call_id = $1
        ORDER BY created_at DESC
        LIMIT 1
    `
    
    var record models.TicketRecord
    err := r.db.DB.Get(&record, query, callID)
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, fmt.Errorf("ticket not found for call_id: %s", callID)
        }
        return nil, fmt.Errorf("query ticket: %w", err)
    }
    
    return &record, nil
}

// ListTickets получает список тикетов с фильтрацией
func (r *TicketRepository) ListTickets(filters map[string]interface{}, limit, offset int) ([]models.TicketRecord, error) {
    query := `
        SELECT 
            id, ticket_id, external_id, call_id,
            title, description, priority, status,
            assignee_type, assignee_id,
            intent_id, intent_confidence,
            entities_json, url, system,
            created_at, updated_at
        FROM tickets
        WHERE 1=1
    `
    
    args := []interface{}{}
    argIdx := 1
    
    // Добавляем фильтры
    if status, ok := filters["status"].(string); ok && status != "" {
        query += fmt.Sprintf(" AND status = $%d", argIdx)
        args = append(args, status)
        argIdx++
    }
    
    if priority, ok := filters["priority"].(string); ok && priority != "" {
        query += fmt.Sprintf(" AND priority = $%d", argIdx)
        args = append(args, priority)
        argIdx++
    }
    
    if assigneeID, ok := filters["assignee_id"].(string); ok && assigneeID != "" {
        query += fmt.Sprintf(" AND assignee_id = $%d", argIdx)
        args = append(args, assigneeID)
        argIdx++
    }
    
    // Сортировка и пагинация
    query += " ORDER BY created_at DESC"
    query += fmt.Sprintf(" LIMIT $%d OFFSET $%d", argIdx, argIdx+1)
    args = append(args, limit, offset)
    
    var records []models.TicketRecord
    err := r.db.DB.Select(&records, query, args...)
    if err != nil {
        return nil, fmt.Errorf("query tickets: %w", err)
    }
    
    return records, nil
}

// UpdateTicketStatus обновляет статус тикета
func (r *TicketRepository) UpdateTicketStatus(ticketID, status string) error {
    query := `
        UPDATE tickets
        SET status = $1, updated_at = NOW()
        WHERE ticket_id = $2
    `
    
    result, err := r.db.DB.Exec(query, status, ticketID)
    if err != nil {
        return fmt.Errorf("update status: %w", err)
    }
    
    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return fmt.Errorf("get rows affected: %w", err)
    }
    
    if rowsAffected == 0 {
        return fmt.Errorf("ticket not found")
    }
    
    return nil
}

// GetTicketStats получает статистику по тикетам
func (r *TicketRepository) GetTicketStats() (map[string]interface{}, error) {
    query := `
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN status = 'open' THEN 1 END) as open,
            COUNT(CASE WHEN status = 'in_progress' THEN 1 END) as in_progress,
            COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved,
            COUNT(CASE WHEN status = 'closed' THEN 1 END) as closed,
            COUNT(CASE WHEN priority = 'high' THEN 1 END) as high_priority,
            COUNT(CASE WHEN priority = 'critical' THEN 1 END) as critical_priority
        FROM tickets
    `
    
    var stats struct {
        Total            int `db:"total"`
        Open             int `db:"open"`
        InProgress       int `db:"in_progress"`
        Resolved         int `db:"resolved"`
        Closed           int `db:"closed"`
        HighPriority     int `db:"high_priority"`
        CriticalPriority int `db:"critical_priority"`
    }
    
    err := r.db.DB.Get(&stats, query)
    if err != nil {
        return nil, fmt.Errorf("query stats: %w", err)
    }
    
    return map[string]interface{}{
        "total":             stats.Total,
        "open":              stats.Open,
        "in_progress":       stats.InProgress,
        "resolved":          stats.Resolved,
        "closed":            stats.Closed,
        "high_priority":     stats.HighPriority,
        "critical_priority": stats.CriticalPriority,
    }, nil
}