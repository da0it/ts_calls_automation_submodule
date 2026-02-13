// internal/adapters/ticket_system.go
package adapters

import "ticket_module/internal/models"

// TicketSystemAdapter интерфейс для разных тикет-систем
type TicketSystemAdapter interface {
    CreateTicket(draft *models.TicketDraft) (*models.TicketCreated, error)
    GetTicket(externalID string) (*models.TicketCreated, error)
    UpdateTicket(externalID string, update map[string]interface{}) error
}