// internal/adapters/ticket_system.go
package adapters

import "ticket_module/internal/models"

// TicketSystemAdapter интерфейс для разных тикет-систем. Определяет обязательный метод CreateTicket
type TicketSystemAdapter interface {
	CreateTicket(payload *models.TicketSystemPayload) (*models.TicketCreated, error)
}
