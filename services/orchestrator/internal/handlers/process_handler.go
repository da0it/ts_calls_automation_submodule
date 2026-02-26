// internal/handlers/process_handler.go
package handlers

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"
	"orchestrator/internal/models"
	"orchestrator/internal/services"
)

type ProcessHandler struct {
	orchestrator           *services.OrchestratorService
	routingConfigService   *services.RoutingConfigService
	routingFeedbackService *services.RoutingFeedbackService
	routingModelService    *services.RoutingModelService
	auditService           *services.AuditService
	uploadDir              string
}

func envBool(name string, def bool) bool {
	raw := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
	if raw == "" {
		return def
	}
	switch raw {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return def
	}
}

func NewProcessHandler(
	orchestrator *services.OrchestratorService,
	routingConfigService *services.RoutingConfigService,
	routingFeedbackService *services.RoutingFeedbackService,
	routingModelService *services.RoutingModelService,
	auditService *services.AuditService,
) *ProcessHandler {
	// Создаём директорию для загрузки файлов
	uploadDir := "./uploads"
	os.MkdirAll(uploadDir, 0755)

	return &ProcessHandler{
		orchestrator:           orchestrator,
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

// ProcessCall godoc
// @Summary Обработать аудио звонка
// @Description Загружает аудио файл, транскрибирует, определяет интент и создает тикет
// @Tags calls
// @Accept multipart/form-data
// @Produce json
// @Param audio formData file true "Audio file (mp3, wav, m4a)"
// @Success 200 {object} services.ProcessCallResult
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /api/v1/process-call [post]
func (h *ProcessHandler) ProcessCall(c *gin.Context) {
	// 1. Получаем загруженный файл
	file, err := c.FormFile("audio")
	if err != nil {
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason": "missing_audio",
		})
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "audio file is required",
		})
		return
	}

	log.Printf("Received audio file: %s (%.2f MB)", file.Filename, float64(file.Size)/1024/1024)

	// Валидация формата
	ext := filepath.Ext(file.Filename)
	allowedFormats := map[string]bool{
		".mp3":  true,
		".wav":  true,
		".m4a":  true,
		".flac": true,
		".ogg":  true,
	}
	if !allowedFormats[ext] {
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason":     "unsupported_audio_format",
			"audio_ext":  ext,
			"audio_size": file.Size,
		})
		c.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf("unsupported audio format: %s (allowed: mp3, wav, m4a, flac, ogg)", ext),
		})
		return
	}

	// 2. Сохраняем файл
	requestID := c.GetString("request_id")
	if requestID == "" {
		requestID = "no_request_id"
	}
	filename := fmt.Sprintf("%s_%s", requestID, file.Filename)
	audioPath := filepath.Join(h.uploadDir, filename)

	if err := c.SaveUploadedFile(file, audioPath); err != nil {
		log.Printf("Failed to save file: %v", err)
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason":     "save_upload_failed",
			"audio_ext":  ext,
			"audio_size": file.Size,
		})
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "failed to save audio file",
		})
		return
	}

	log.Printf("Audio saved to: %s", audioPath)
	deleteAfterProcess := envBool("ORCH_DELETE_UPLOADED_AUDIO_AFTER_PROCESS", true)
	if deleteAfterProcess {
		defer func() {
			if rmErr := os.Remove(audioPath); rmErr != nil && !os.IsNotExist(rmErr) {
				log.Printf("Failed to remove uploaded audio %s: %v", audioPath, rmErr)
			}
		}()
	}

	// 3. Запускаем обработку
	result, err := h.orchestrator.ProcessCall(audioPath)
	if err != nil {
		log.Printf("Processing failed: %v", err)

		// Удаляем файл при ошибке, даже если cleanup on success выключен.
		_ = os.Remove(audioPath)
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason":     "pipeline_failed",
			"audio_ext":  ext,
			"audio_size": file.Size,
		})

		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("processing failed: %v", err),
		})
		return
	}
	if result == nil {
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason": "empty_pipeline_result",
		})
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "processing failed: empty pipeline result",
		})
		return
	}

	// 4. Опционально: удаляем файл после обработки
	// os.Remove(audioPath)
	segmentsCount := 0
	intentID := ""
	priority := ""
	suggestedGroup := ""
	if result.Transcript != nil {
		segmentsCount = len(result.Transcript.Segments)
	}
	if result.Routing != nil {
		intentID = result.Routing.IntentID
		priority = result.Routing.Priority
		suggestedGroup = result.Routing.SuggestedGroup
	}
	h.writeAudit(c, "call.process", "call", result.CallID, "success", map[string]interface{}{
		"audio_ext":       ext,
		"audio_size":      file.Size,
		"segments_count":  segmentsCount,
		"intent_id":       intentID,
		"priority":        priority,
		"suggested_group": suggestedGroup,
	})

	c.JSON(http.StatusOK, result)
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
	c.JSON(http.StatusOK, status)
}

// Root godoc
// @Summary API Information
// @Description Информация об Orchestrator API
// @Tags info
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /api/info [get]
func (h *ProcessHandler) Root(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"service":     "Ticket System Orchestrator",
		"version":     "1.0.0",
		"description": "Оркестрирует обработку звонков через все модули системы",
		"endpoints": gin.H{
			"process_call":     "POST /api/v1/process-call",
			"routing_config":   "GET/PUT /api/v1/routing-config",
			"routing_groups":   "POST/DELETE /api/v1/routing-config/groups",
			"routing_intents":  "POST/DELETE /api/v1/routing-config/intents",
			"routing_feedback": "POST /api/v1/routing-feedback",
			"routing_model":    "GET /api/v1/routing-model/status, POST /api/v1/routing-model/reload, POST /api/v1/routing-model/train, POST /api/v1/routing-model/train-csv",
			"health":           "GET /health",
			"docs":             "GET /docs (если включен Swagger)",
		},
		"pipeline": []string{
			"1. Transcription + Diarization",
			"2. Routing (RuBERT Intent Classification)",
			"3. Ticket Creation",
		},
	})
}
