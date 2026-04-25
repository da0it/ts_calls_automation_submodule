// internal/handlers/process_handler.go
package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"orchestrator/internal/clients"
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
	spamFeedbackService    *services.SpamFeedbackService
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

type reviewSpamCheckPayload struct {
	Status         string  `json:"status"`
	PredictedLabel string  `json:"predicted_label"`
	Confidence     float64 `json:"confidence"`
	ThresholdLow   float64 `json:"threshold_low"`
	ThresholdHigh  float64 `json:"threshold_high"`
	Reason         string  `json:"reason"`
	Backend        string  `json:"backend"`
}

type reviewRoutingPayload struct {
	IntentID         string  `json:"intent_id"`
	IntentConfidence float64 `json:"intent_confidence"`
	Priority         string  `json:"priority"`
	SuggestedGroup   string  `json:"suggested_group"`
}

func isAllowedAudioExt(ext string) bool {
	_, ok := allowedAudioFormats[ext]
	return ok
}

func buildTranscript(callID string, payload spamReviewTranscriptPayload) *clients.TranscriptionResponse {
	transcript := &clients.TranscriptionResponse{
		CallID:      strings.TrimSpace(payload.CallID),
		Segments:    payload.Segments,
		RoleMapping: payload.RoleMapping,
		Metadata:    payload.Metadata,
	}
	if transcript.CallID == "" {
		transcript.CallID = strings.TrimSpace(callID)
	}
	if transcript.Metadata == nil {
		transcript.Metadata = map[string]interface{}{}
	}
	return transcript
}

func buildSpamCheck(payload reviewSpamCheckPayload) *clients.SpamCheckResponse {
	if payload.Status == "" &&
		payload.PredictedLabel == "" &&
		payload.Confidence == 0 &&
		payload.ThresholdLow == 0 &&
		payload.ThresholdHigh == 0 &&
		payload.Reason == "" &&
		payload.Backend == "" {
		return nil
	}

	return &clients.SpamCheckResponse{
		Status:         payload.Status,
		PredictedLabel: payload.PredictedLabel,
		Confidence:     payload.Confidence,
		ThresholdLow:   payload.ThresholdLow,
		ThresholdHigh:  payload.ThresholdHigh,
		Reason:         payload.Reason,
		Backend:        payload.Backend,
	}
}

func normalizePriority(raw string) string {
	value := strings.ToLower(strings.TrimSpace(raw))
	if value == "" || value == "normal" {
		return "medium"
	}
	return value
}

func buildSuggestedReview(result *services.ProcessCallResult) map[string]interface{} {
	intentID := ""
	priority := "medium"
	group := ""
	if result != nil && result.Routing != nil {
		intentID = strings.TrimSpace(result.Routing.IntentID)
		priority = normalizePriority(result.Routing.Priority)
		group = strings.TrimSpace(result.Routing.SuggestedGroup)
	}
	if intentID == "" {
		intentID = "misc.triage"
	}
	return map[string]interface{}{
		"decision":    "pending",
		"intentId":    intentID,
		"priority":    priority,
		"group":       group,
		"errorType":   "none",
		"comment":     "",
		"completedAt": "",
	}
}

func mapString(value interface{}) string {
	if value == nil {
		return ""
	}
	if text, ok := value.(string); ok {
		return strings.TrimSpace(text)
	}
	return strings.TrimSpace(fmt.Sprintf("%v", value))
}

func mapObject(value interface{}) map[string]interface{} {
	if item, ok := value.(map[string]interface{}); ok && item != nil {
		return item
	}
	return nil
}

func (h *ProcessHandler) currentSLAMinutes() int {
	if h.appSettingsService == nil {
		return 15
	}
	settings, err := h.appSettingsService.Get()
	if err != nil || settings == nil || settings.SLAMinutes <= 0 {
		return 15
	}
	return settings.SLAMinutes
}

func processResultMap(result *services.ProcessCallResult) (map[string]interface{}, error) {
	raw, err := json.Marshal(result)
	if err != nil {
		return nil, err
	}
	payload := map[string]interface{}{}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	delete(payload, "queue_id")
	return payload, nil
}

func hasTicket(raw map[string]interface{}) bool {
	return mapString(raw["ticket_id"]) != "" ||
		mapString(raw["external_id"]) != "" ||
		mapString(raw["url"]) != "" ||
		mapString(raw["system"]) != "" ||
		mapString(raw["created_at"]) != ""
}

func automaticStopTime(status string, ticket map[string]interface{}, processedAt string) string {
	if status == services.ProcessStatusSpamBlocked || status == services.ProcessStatusNoSpeech {
		return processedAt
	}
	if hasTicket(ticket) {
		if createdAt := mapString(ticket["created_at"]); createdAt != "" {
			return createdAt
		}
		return processedAt
	}
	return ""
}

func (h *ProcessHandler) buildQueueCallRecord(
	result *services.ProcessCallResult,
	sourceFilename string,
	queueID string,
	existing map[string]interface{},
	review map[string]interface{},
) (map[string]interface{}, error) {
	raw, err := processResultMap(result)
	if err != nil {
		return nil, fmt.Errorf("marshal queue payload: %w", err)
	}

	routing := mapObject(raw["routing"])
	ticket := mapObject(raw["ticket"])
	spamCheck := mapObject(raw["spam_check"])
	if spamCheck == nil && routing != nil {
		spamCheck = mapObject(routing["spam_check"])
	}

	requestReceivedAt := strings.TrimSpace(result.RequestReceivedAt)
	if requestReceivedAt == "" {
		requestReceivedAt = mapString(existing["createdAt"])
	}
	if requestReceivedAt == "" {
		requestReceivedAt = time.Now().UTC().Format(time.RFC3339Nano)
	}

	processedAt := strings.TrimSpace(result.ProcessedAt)
	if processedAt == "" {
		processedAt = time.Now().UTC().Format(time.RFC3339Nano)
	}

	suggestedIntentID := ""
	suggestedGroup := ""
	priority := "medium"
	confidence := 0.0
	if routing != nil {
		suggestedIntentID = mapString(routing["intent_id"])
		suggestedGroup = mapString(routing["suggested_group"])
		priority = normalizePriority(mapString(routing["priority"]))
		if value, ok := routing["intent_confidence"].(float64); ok {
			confidence = value
		}
	}

	if review == nil {
		if existingReview := mapObject(existing["review"]); existingReview != nil {
			review = existingReview
		} else {
			review = map[string]interface{}{
				"decision":    "pending",
				"intentId":    suggestedIntentID,
				"priority":    priority,
				"group":       suggestedGroup,
				"errorType":   "none",
				"comment":     "",
				"completedAt": "",
			}
		}
	}

	stopTime := automaticStopTime(strings.TrimSpace(result.Status), ticket, processedAt)
	if mapString(review["completedAt"]) == "" && stopTime != "" {
		review["completedAt"] = stopTime
	}

	slaMinutes := h.currentSLAMinutes()
	slaStartedAt := requestReceivedAt
	if existingSLA := mapObject(existing["sla"]); existingSLA != nil {
		if value := mapString(existingSLA["startedAt"]); value != "" {
			slaStartedAt = value
		}
		if value, ok := existingSLA["limitMinutes"].(float64); ok && int(value) > 0 {
			slaMinutes = int(value)
		}
	}

	callID := strings.TrimSpace(result.CallID)
	if callID == "" && result.Transcript != nil {
		callID = strings.TrimSpace(result.Transcript.CallID)
	}

	record := map[string]interface{}{
		"id":             strings.TrimSpace(queueID),
		"sourceFilename": strings.TrimSpace(sourceFilename),
		"createdAt":      requestReceivedAt,
		"processedAt":    processedAt,
		"callId":         callID,
		"status":         strings.TrimSpace(result.Status),
		"transcript":     raw["transcript"],
		"ticket":         raw["ticket"],
		"raw":            raw,
		"spamCheck":      spamCheck,
		"review":         review,
		"aiSuggestion": map[string]interface{}{
			"intentId":   suggestedIntentID,
			"confidence": confidence,
			"priority":   priority,
			"group":      suggestedGroup,
		},
		"sla": map[string]interface{}{
			"limitMinutes": slaMinutes,
			"startedAt":    slaStartedAt,
			"stoppedAt":    stopTime,
		},
	}

	if record["id"] == "" {
		delete(record, "id")
	}
	if existing != nil && existing["lastFeedback"] != nil {
		record["lastFeedback"] = existing["lastFeedback"]
	}
	return record, nil
}

func (h *ProcessHandler) saveQueueRecord(record map[string]interface{}) string {
	if h.callQueueService == nil || record == nil {
		return ""
	}
	saved, err := h.callQueueService.Save(record)
	if err != nil {
		log.Printf("Failed to save call queue record: %v", err)
		return ""
	}
	return mapString(saved["id"])
}

func (h *ProcessHandler) updateQueueReview(queueID string, review map[string]interface{}, feedback interface{}) {
	if h.callQueueService == nil || strings.TrimSpace(queueID) == "" {
		return
	}

	record, err := h.callQueueService.Get(queueID)
	if err != nil {
		log.Printf("Failed to load call queue record %s: %v", queueID, err)
		return
	}
	record["id"] = queueID
	if review != nil {
		record["review"] = review
	}
	if feedback != nil {
		record["lastFeedback"] = feedback
	}
	if _, err := h.callQueueService.Save(record); err != nil {
		log.Printf("Failed to update call queue record %s: %v", queueID, err)
	}
}

func NewProcessHandler(
	orchestrator *services.OrchestratorService,
	callQueueService *services.CallQueueService,
	appSettingsService *services.AppSettingsService,
	routingConfigService *services.RoutingConfigService,
	routingFeedbackService *services.RoutingFeedbackService,
	spamFeedbackService *services.SpamFeedbackService,
	routingModelService *services.RoutingModelService,
	auditService *services.AuditService,
) *ProcessHandler {
	// Создаём директорию для загрузки файлов
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
		spamFeedbackService:    spamFeedbackService,
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
	requestReceivedAt := time.Now().UTC()

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

	originalName := filepath.Base(file.Filename)
	ext := strings.ToLower(filepath.Ext(originalName))
	log.Printf(
		"Received audio upload request_id=%s ext=%s size_mb=%.2f",
		c.GetString("request_id"),
		ext,
		float64(file.Size)/1024/1024,
	)

	if !isAllowedAudioExt(ext) {
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

	if file.Size == 0 {
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason":     "empty_audio_file",
			"audio_ext":  ext,
			"audio_size": file.Size,
		})
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "audio file is empty",
		})
		return
	}

	// 2. Сохраняем файл
	requestID := c.GetString("request_id")
	if requestID == "" {
		requestID = "no_request_id"
	}
	filename := fmt.Sprintf("%s_%d%s", requestID, time.Now().UnixNano(), ext)
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

	log.Printf("Saved uploaded audio request_id=%s", requestID)
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

	result.RequestReceivedAt = requestReceivedAt.Format(time.RFC3339Nano)
	result.ProcessedAt = time.Now().UTC().Format(time.RFC3339Nano)
	if record, buildErr := h.buildQueueCallRecord(result, originalName, "", nil, nil); buildErr != nil {
		log.Printf("Failed to build call queue record: %v", buildErr)
	} else {
		result.QueueID = h.saveQueueRecord(record)
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
		"status":          result.Status,
		"intent_id":       intentID,
		"priority":        priority,
		"suggested_group": suggestedGroup,
	})

	c.JSON(http.StatusOK, result)
}

type spamReviewTranscriptPayload struct {
	CallID      string                 `json:"call_id"`
	Segments    []clients.Segment      `json:"segments"`
	RoleMapping map[string]string      `json:"role_mapping"`
	Metadata    map[string]interface{} `json:"metadata"`
}

type spamOverrideRequest struct {
	QueueID        string                      `json:"queue_id"`
	CallID         string                      `json:"call_id"`
	SourceFilename string                      `json:"source_filename"`
	Transcript     spamReviewTranscriptPayload `json:"transcript"`
	SpamCheck      reviewSpamCheckPayload      `json:"spam_check"`
}

type routingReviewRequest struct {
	QueueID        string                      `json:"queue_id"`
	CallID         string                      `json:"call_id"`
	SourceFilename string                      `json:"source_filename"`
	Decision       string                      `json:"decision"`
	Transcript     spamReviewTranscriptPayload `json:"transcript"`
	Routing        reviewRoutingPayload        `json:"routing"`
	SpamCheck      reviewSpamCheckPayload      `json:"spam_check"`
}

func (h *ProcessHandler) OverrideSpamBlock(c *gin.Context) {
	var payload spamOverrideRequest
	if err := c.ShouldBindJSON(&payload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	transcript := buildTranscript(payload.CallID, payload.Transcript)
	spamCheck := buildSpamCheck(payload.SpamCheck)

	if h.spamFeedbackService != nil {
		if _, err := h.spamFeedbackService.SaveDecision(services.SpamFeedbackRequest{
			CallID:         transcript.CallID,
			SourceFilename: payload.SourceFilename,
			Decision:       "not_spam",
			SpamCheck: services.SpamFeedbackMeta{
				Status:         payload.SpamCheck.Status,
				PredictedLabel: payload.SpamCheck.PredictedLabel,
				Confidence:     payload.SpamCheck.Confidence,
				ThresholdLow:   payload.SpamCheck.ThresholdLow,
				ThresholdHigh:  payload.SpamCheck.ThresholdHigh,
				Reason:         payload.SpamCheck.Reason,
				Backend:        payload.SpamCheck.Backend,
			},
		}); err != nil {
			log.Printf("Failed to save spam override feedback: %v", err)
		}
	}

	result, err := h.orchestrator.ContinueAfterSpamBlock(services.ContinueAfterSpamBlockInput{
		CallID:         payload.CallID,
		SourceFilename: payload.SourceFilename,
		Transcript:     transcript,
		SpamCheck:      spamCheck,
	})
	if err != nil {
		h.writeAudit(c, "call.spam_override", "call", transcript.CallID, "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	h.writeAudit(c, "call.spam_override", "call", transcript.CallID, "success", map[string]interface{}{
		"status": result.Status,
	})
	result.ProcessedAt = time.Now().UTC().Format(time.RFC3339Nano)
	existing := map[string]interface{}{}
	if payload.QueueID != "" && h.callQueueService != nil {
		if current, getErr := h.callQueueService.Get(payload.QueueID); getErr == nil {
			existing = current
		}
	}
	review := buildSuggestedReview(result)
	if record, buildErr := h.buildQueueCallRecord(result, payload.SourceFilename, payload.QueueID, existing, review); buildErr != nil {
		log.Printf("Failed to build spam override call queue record: %v", buildErr)
	} else {
		result.QueueID = h.saveQueueRecord(record)
	}

	c.JSON(http.StatusOK, result)
}

func (h *ProcessHandler) ResolveRoutingReview(c *gin.Context) {
	var payload routingReviewRequest
	if err := c.ShouldBindJSON(&payload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	transcript := buildTranscript(payload.CallID, payload.Transcript)

	routing := &clients.RoutingResponse{
		IntentID:         strings.TrimSpace(payload.Routing.IntentID),
		IntentConfidence: payload.Routing.IntentConfidence,
		Priority:         strings.TrimSpace(payload.Routing.Priority),
		SuggestedGroup:   strings.TrimSpace(payload.Routing.SuggestedGroup),
	}
	routing.SpamCheck = buildSpamCheck(payload.SpamCheck)

	result, err := h.orchestrator.ContinueAfterRoutingReview(services.ContinueAfterRoutingReviewInput{
		CallID:         payload.CallID,
		SourceFilename: payload.SourceFilename,
		Decision:       payload.Decision,
		Transcript:     transcript,
		Routing:        routing,
	})
	if err != nil {
		h.writeAudit(c, "call.routing_review", "call", transcript.CallID, "failed", map[string]interface{}{
			"decision": payload.Decision,
			"reason":   err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	h.writeAudit(c, "call.routing_review", "call", transcript.CallID, "success", map[string]interface{}{
		"decision":        payload.Decision,
		"status":          result.Status,
		"intent_id":       routing.IntentID,
		"priority":        routing.Priority,
		"suggested_group": routing.SuggestedGroup,
	})
	result.ProcessedAt = time.Now().UTC().Format(time.RFC3339Nano)
	existing := map[string]interface{}{}
	if payload.QueueID != "" && h.callQueueService != nil {
		if current, getErr := h.callQueueService.Get(payload.QueueID); getErr == nil {
			existing = current
		}
	}
	review := map[string]interface{}{
		"decision":    strings.ToLower(strings.TrimSpace(payload.Decision)),
		"intentId":    strings.TrimSpace(payload.Routing.IntentID),
		"priority":    normalizePriority(payload.Routing.Priority),
		"group":       strings.TrimSpace(payload.Routing.SuggestedGroup),
		"errorType":   "none",
		"comment":     "",
		"completedAt": "",
	}
	if record, buildErr := h.buildQueueCallRecord(result, payload.SourceFilename, payload.QueueID, existing, review); buildErr != nil {
		log.Printf("Failed to build routing review call queue record: %v", buildErr)
	} else {
		result.QueueID = h.saveQueueRecord(record)
	}

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
			"calls":            "GET /api/v1/calls",
			"app_settings":     "GET /api/v1/app-settings, PUT /api/v1/app-settings (admin)",
			"routing_config":   "GET /api/v1/routing-config",
			"routing_feedback": "POST /api/v1/routing-feedback",
			"spam_override":    "POST /api/v1/spam-override",
			"routing_review":   "POST /api/v1/routing-review",
			"routing_model":    "GET /api/v1/routing-model/status",
			"calls_admin":      "DELETE /api/v1/calls, DELETE /api/v1/calls/:id (admin)",
			"audit_events":     "GET /api/v1/audit/events (admin)",
			"health":           "GET /health",
			"docs":             "GET /docs (если включен Swagger)",
		},
		"pipeline": []string{
			"1. Transcription + Diarization",
			"2. Spam Gate (allow / block, blocked calls can be continued manually)",
			"3. Routing (RuBERT Intent Classification + low-confidence review)",
			"4. Entity Extraction",
			"5. Ticket Creation",
		},
	})
}
