package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"notification_sender/internal/services"
)

type NotificationHandler struct {
	service *services.NotificationService
}

func NewNotificationHandler(service *services.NotificationService) *NotificationHandler {
	return &NotificationHandler{service: service}
}

func (h *NotificationHandler) Health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "healthy", "service": "notification_sender"})
}
