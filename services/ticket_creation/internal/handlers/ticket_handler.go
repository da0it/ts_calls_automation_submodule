// internal/handlers/ticket_handler.go
package handlers

import (
	"log"
	"net/http"
	"strconv"

	"ticket_module/internal/models"
	"ticket_module/internal/services"

	"github.com/gin-gonic/gin"
)

// Структура TicketHandler
type TicketHandler struct {

	// Поле service хранит указатель на объект сервиса создания тикетов
	service *services.TicketCreatorService
}

// Конструктор хандлера. На вход принимает указатель на сервис. Возвращает указатель на TicketHandler
func NewTicketHandler(service *services.TicketCreatorService) *TicketHandler {
	return &TicketHandler{service: service}
}

// CreateTicket - Метод структуры TicketHandler. Принимает на вход gin.Context — это объект текущего HTTP-запроса.
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

// Метод получает один тикет по ID. Принимает *gin.Context
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

	// Хандлер вызывает сервисный слой и просит получить тикет по ID
	ticket, err := h.service.GetTicket(ticketID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, ticket)
}

// Метод возвращает список тикетов с фильтрами и пагинацией. Принимает на вход *gin.Context
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

	// c.Query("status") читает query-параметр из UR
	if status := c.Query("status"); status != "" {
		filters["status"] = status
	}

	// Фильтр по приоритету
	if priority := c.Query("priority"); priority != "" {
		filters["priority"] = priority
	}

	// Фильтр по исполнителю
	if assigneeID := c.Query("assignee_id"); assigneeID != "" {
		filters["assignee_id"] = assigneeID
	}

	// Лимит количества записей
	limit := 50
	if l := c.Query("limit"); l != "" {
		if val, err := strconv.Atoi(l); err == nil && val > 0 {
			limit = val
		}
	}

	// Смещение списка. По умолчанию список начинается с первой записи
	offset := 0
	if o := c.Query("offset"); o != "" {
		if val, err := strconv.Atoi(o); err == nil && val >= 0 {
			offset = val
		}
	}

	// Получение списка тикетов
	tickets, err := h.service.ListTickets(filters, limit, offset)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Возвращается массив тикетов
	c.JSON(http.StatusOK, tickets)
}

// Метод обновляет статус тикета. Принимает *gin.Context
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

	// Локальная анонимная структура. Структура ожидает JSON такого вида: {"status": "in_progress"}
	var req struct {

		// Поле String хранит новый статус
		Status string `json:"status" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "status is required"})
		return
	}

	// Валидация статуса (проверка допустимых статусов)
	validStatuses := map[string]bool{
		"open": true, "in_progress": true, "resolved": true, "closed": true,
	}
	if !validStatuses[req.Status] {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid status"})
		return
	}

	// Если статус валидный, хандлер вызывает сервис:
	if err := h.service.UpdateTicketStatus(ticketID, req.Status); err != nil {

		// Передаются: ID тикета, новый статус
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Успешный ответ
	c.JSON(http.StatusOK, gin.H{"message": "status updated", "status": req.Status})
}

// метод получает статистику по тикетам.Например, сервис может вернуть: количество всех тикетов, количество открытых,
// количество закрытых, распределение по приоритетам, распределение по статусам.
// GetStats godoc
// @Summary Получить статистику по тикетам
// @Tags tickets
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /api/tickets/stats [get]
func (h *TicketHandler) GetStats(c *gin.Context) {

	// Хандлер вызывает:
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
