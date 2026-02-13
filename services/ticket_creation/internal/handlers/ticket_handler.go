// internal/handlers/ticket_handler.go
package handlers

import (
    "log"
    "net/http"
    "strconv"
    
    "github.com/gin-gonic/gin"
    "ticket_module/internal/models"
    "ticket_module/internal/services"
)

type TicketHandler struct {
    service *services.TicketCreatorService
}

func NewTicketHandler(service *services.TicketCreatorService) *TicketHandler {
    return &TicketHandler{service: service}
}

// CreateTicket godoc
// @Summary Создать тикет из транскрипции и маршрутизации
// @Tags tickets
// @Accept json
// @Produce json
// @Param request body models.CreateTicketRequest true "Данные для создания тикета"
// @Success 200 {object} models.CreateTicketResponse
// @Failure 400 {object} models.CreateTicketResponse
// @Failure 500 {object} models.CreateTicketResponse
// @Router /api/tickets [post]
func (h *TicketHandler) CreateTicket(c *gin.Context) {
    var req models.CreateTicketRequest
    
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, models.CreateTicketResponse{
            Success: false,
            Error:   "Invalid request: " + err.Error(),
        })
        return
    }
    
    // Валидация
    if req.Transcript.CallID == "" {
        c.JSON(http.StatusBadRequest, models.CreateTicketResponse{
            Success: false,
            Error:   "call_id is required",
        })
        return
    }
    
    if len(req.Transcript.Segments) == 0 {
        c.JSON(http.StatusBadRequest, models.CreateTicketResponse{
            Success: false,
            Error:   "segments are required",
        })
        return
    }
    
    // Создаем тикет
    ticket, err := h.service.CreateTicket(&req)
    if err != nil {
        log.Printf("Error creating ticket: %v", err)
        c.JSON(http.StatusInternalServerError, models.CreateTicketResponse{
            Success: false,
            Error:   "Failed to create ticket: " + err.Error(),
        })
        return
    }
    
    c.JSON(http.StatusOK, models.CreateTicketResponse{
        Success: true,
        Ticket:  ticket,
    })
}

// GetTicket godoc
// @Summary Получить тикет по ID
// @Tags tickets
// @Produce json
// @Param id path string true "Ticket ID"
// @Success 200 {object} models.TicketRecord
// @Failure 404 {object} map[string]string
// @Router /api/tickets/{id} [get]
func (h *TicketHandler) GetTicket(c *gin.Context) {
    ticketID := c.Param("id")
    
    ticket, err := h.service.GetTicket(ticketID)
    if err != nil {
        c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, ticket)
}

// ListTickets godoc
// @Summary Получить список тикетов
// @Tags tickets
// @Produce json
// @Param status query string false "Фильтр по статусу"
// @Param priority query string false "Фильтр по приоритету"
// @Param assignee_id query string false "Фильтр по assignee"
// @Param limit query int false "Лимит" default(50)
// @Param offset query int false "Offset" default(0)
// @Success 200 {array} models.TicketRecord
// @Router /api/tickets [get]
func (h *TicketHandler) ListTickets(c *gin.Context) {
    filters := make(map[string]interface{})
    
    if status := c.Query("status"); status != "" {
        filters["status"] = status
    }
    if priority := c.Query("priority"); priority != "" {
        filters["priority"] = priority
    }
    if assigneeID := c.Query("assignee_id"); assigneeID != "" {
        filters["assignee_id"] = assigneeID
    }
    
    limit := 50
    if l := c.Query("limit"); l != "" {
        if val, err := strconv.Atoi(l); err == nil && val > 0 {
            limit = val
        }
    }
    
    offset := 0
    if o := c.Query("offset"); o != "" {
        if val, err := strconv.Atoi(o); err == nil && val >= 0 {
            offset = val
        }
    }
    
    tickets, err := h.service.ListTickets(filters, limit, offset)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, tickets)
}

// UpdateTicketStatus godoc
// @Summary Обновить статус тикета
// @Tags tickets
// @Accept json
// @Produce json
// @Param id path string true "Ticket ID"
// @Param request body map[string]string true "Новый статус"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Router /api/tickets/{id}/status [patch]
func (h *TicketHandler) UpdateTicketStatus(c *gin.Context) {
    ticketID := c.Param("id")
    
    var req struct {
        Status string `json:"status" binding:"required"`
    }
    
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "status is required"})
        return
    }
    
    // Валидация статуса
    validStatuses := map[string]bool{
        "open": true, "in_progress": true, "resolved": true, "closed": true,
    }
    if !validStatuses[req.Status] {
        c.JSON(http.StatusBadRequest, gin.H{"error": "invalid status"})
        return
    }
    
    if err := h.service.UpdateTicketStatus(ticketID, req.Status); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{"message": "status updated", "status": req.Status})
}

// GetStats godoc
// @Summary Получить статистику по тикетам
// @Tags tickets
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /api/tickets/stats [get]
func (h *TicketHandler) GetStats(c *gin.Context) {
    stats, err := h.service.GetStats()
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, stats)
}

// Health проверка здоровья сервиса
func (h *TicketHandler) Health(c *gin.Context) {
    c.JSON(http.StatusOK, gin.H{
        "status":  "healthy",
        "service": "ticket-service",
    })
}