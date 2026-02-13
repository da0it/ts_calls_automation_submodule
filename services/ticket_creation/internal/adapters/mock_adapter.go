// internal/adapters/mock_adapter.go
package adapters

import (
    "fmt"
    "time"
    
    "github.com/google/uuid"
    "ticket_module/internal/models"
)

type MockAdapter struct{}

func NewMockAdapter() *MockAdapter {
    return &MockAdapter{}
}

func (a *MockAdapter) CreateTicket(draft *models.TicketDraft) (*models.TicketCreated, error) {
    ticketID := uuid.New().String()
    externalID := fmt.Sprintf("MOCK-%d", time.Now().Unix())
    
    return &models.TicketCreated{
        TicketID:   ticketID,
        ExternalID: externalID,
        URL:        fmt.Sprintf("http://mock-system/tickets/%s", externalID),
        System:     "mock",
        CreatedAt:  time.Now(),
    }, nil
}

func (a *MockAdapter) GetTicket(externalID string) (*models.TicketCreated, error) {
    return &models.TicketCreated{
        ExternalID: externalID,
        URL:        fmt.Sprintf("http://mock-system/tickets/%s", externalID),
        System:     "mock",
    }, nil
}

func (a *MockAdapter) UpdateTicket(externalID string, update map[string]interface{}) error {
    return nil
}