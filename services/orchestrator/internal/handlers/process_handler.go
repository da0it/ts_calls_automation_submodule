package handlers

import (
	"log"
	"os"

	"github.com/gin-gonic/gin"
	"orchestrator/internal/models"
	"orchestrator/internal/services"
)

var allowedAudioFormats = map[string]struct{}{
	".mp3":  {},
	".wav":  {},
	".m4a":  {},
	".flac": {},
	".ogg":  {},
}

type ProcessHandler struct {
	orchestrator           *services.OrchestratorService
	callQueueService       *services.CallQueueService
	appSettingsService     *services.AppSettingsService
	routingConfigService   *services.RoutingConfigService
	routingFeedbackService *services.RoutingFeedbackService
	routingModelService    *services.RoutingModelService
	auditService           *services.AuditService
	uploadDir              string
}

func NewProcessHandler(
	orchestrator *services.OrchestratorService,
	callQueueService *services.CallQueueService,
	appSettingsService *services.AppSettingsService,
	routingConfigService *services.RoutingConfigService,
	routingFeedbackService *services.RoutingFeedbackService,
	routingModelService *services.RoutingModelService,
	auditService *services.AuditService,
) *ProcessHandler {
	uploadDir := "./uploads"
	if err := os.MkdirAll(uploadDir, 0750); err != nil {
		log.Printf("Failed to create upload directory %s: %v", uploadDir, err)
	}

	return &ProcessHandler{
		orchestrator:           orchestrator,
		callQueueService:       callQueueService,
		appSettingsService:     appSettingsService,
		routingConfigService:   routingConfigService,
		routingFeedbackService: routingFeedbackService,
		routingModelService:    routingModelService,
		auditService:           auditService,
		uploadDir:              uploadDir,
	}
}

func (h *ProcessHandler) writeAudit(
	c *gin.Context,
	eventType string,
	resourceType string,
	resourceID string,
	outcome string,
	details map[string]interface{},
) {
	if h.auditService == nil {
		return
	}

	var actorUserID *int64
	actorUsername := ""
	actorRole := ""
	if userVal, ok := c.Get("user"); ok {
		if user, castOK := userVal.(*models.User); castOK && user != nil {
			actorUserID = &user.ID
			actorUsername = user.Username
			actorRole = string(user.Role)
		}
	}

	if err := h.auditService.LogEvent(services.AuditEvent{
		RequestID:     c.GetString("request_id"),
		ActorUserID:   actorUserID,
		ActorUsername: actorUsername,
		ActorRole:     actorRole,
		EventType:     eventType,
		ResourceType:  resourceType,
		ResourceID:    resourceID,
		Outcome:       outcome,
		Details:       details,
		IPAddress:     c.ClientIP(),
		UserAgent:     c.GetHeader("User-Agent"),
	}); err != nil {
		log.Printf("Failed to write audit event (%s): %v", eventType, err)
	}
}

// Health godoc
// @Summary Health check
// @Description Проверка доступности оркестратора и зависимых сервисов
// @Tags health
// @Produce json
// @Success 200 {object} map[string]string
// @Router /health [get]
func (h *ProcessHandler) Health(c *gin.Context) {
	status := h.orchestrator.HealthCheck()
	c.JSON(200, status)
}

// Root godoc
// @Summary API Information
// @Description Информация об Orchestrator API
// @Tags info
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /api/info [get]
func (h *ProcessHandler) Root(c *gin.Context) {
	c.JSON(200, gin.H{
		"service":     "Ticket System Orchestrator",
		"version":     "1.0.0",
		"description": "Оркестрирует обработку звонков через все модули системы",
		"endpoints": gin.H{
			"process_call":     "POST /api/v1/process-call",
			"calls":            "GET /api/v1/calls",
			"app_settings":     "GET /api/v1/app-settings, PUT /api/v1/app-settings (admin)",
			"routing_config":   "GET /api/v1/routing-config",
			"routing_feedback": "POST /api/v1/routing-feedback",
			"routing_review":   "POST /api/v1/routing-review",
			"routing_model":    "GET /api/v1/routing-model/status",
			"calls_admin":      "DELETE /api/v1/calls, DELETE /api/v1/calls/:id (admin)",
			"audit_events":     "GET /api/v1/audit/events (admin)",
			"health":           "GET /health",
			"docs":             "GET /docs (если включен Swagger)",
		},
		"pipeline": []string{
			"1. Transcription + Diarization",
			"2. Routing (intent classification, spam detection, low-confidence review)",
			"3. Entity Extraction",
			"4. Ticket Creation",
		},
	})
}
