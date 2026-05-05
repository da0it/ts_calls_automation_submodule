package handlers

import (
	"log"
	"net/http"
	"strings"
	"time"

	"orchestrator/internal/clients"
	"orchestrator/internal/services"

	"github.com/gin-gonic/gin"
)

// Константы spam-класса обращений
const (
	reviewSpamIntentID       = "spam.call"
	reviewLegacySpamIntentID = "spam"
)

// Часть JSON-запроса, которая описывает результат проверки на спам.
type reviewSpamCheckPayload struct {
	Status         string  `json:"status"`
	PredictedLabel string  `json:"predicted_label"`
	Confidence     float64 `json:"confidence"`
	ThresholdLow   float64 `json:"threshold_low"`
	ThresholdHigh  float64 `json:"threshold_high"`
	Reason         string  `json:"reason"`
	Backend        string  `json:"backend"`
}

// Блок маршрутизации.
type reviewRoutingPayload struct {
	IntentID         string  `json:"intent_id"`
	IntentConfidence float64 `json:"intent_confidence"`
	Priority         string  `json:"priority"`
	SuggestedGroup   string  `json:"suggested_group"`
}

// Транскрипция звонка, которую нужно передать обратно в оркестратор
type spamReviewTranscriptPayload struct {
	CallID      string                 `json:"call_id"`
	Segments    []clients.Segment      `json:"segments"`
	RoleMapping map[string]string      `json:"role_mapping"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// Основной JSON-запрос для ручной проверки.
type routingReviewRequest struct {
	QueueID        string                      `json:"queue_id"`
	CallID         string                      `json:"call_id"`
	SourceFilename string                      `json:"source_filename"`
	Decision       string                      `json:"decision"`
	Transcript     spamReviewTranscriptPayload `json:"transcript"`
	Routing        reviewRoutingPayload        `json:"routing"`
	SpamCheck      reviewSpamCheckPayload      `json:"spam_check"`
}

// Метод проверяет, пустой ли блок spam-check
func (p reviewSpamCheckPayload) isEmpty() bool {
	return p.Status == "" &&
		p.PredictedLabel == "" &&
		p.Confidence == 0 &&
		p.ThresholdLow == 0 &&
		p.ThresholdHigh == 0 &&
		p.Reason == "" &&
		p.Backend == ""
}

// Функция собирает внутреннюю структуру транскрипции из JSON-запроса
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

// Функция преобразует JSON-блок spam-check во внутреннюю структуру clients.SpamCheckResponse
func buildSpamCheck(payload reviewSpamCheckPayload) *clients.SpamCheckResponse {
	if payload.isEmpty() {
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

// Функция пытается загрузить существующую запись очереди
func (h *ProcessHandler) loadExistingQueueRecord(queueID string) map[string]interface{} {
	if queueID == "" || h.callQueueService == nil {
		return map[string]interface{}{}
	}
	current, err := h.callQueueService.Get(queueID)
	if err != nil {
		return map[string]interface{}{}
	}
	return current
}

// Функция собирает внутренний результат маршрутизации из JSON-запроса
func buildRouting(payload routingReviewRequest) *clients.RoutingResponse {
	return &clients.RoutingResponse{
		IntentID:         strings.TrimSpace(payload.Routing.IntentID),
		IntentConfidence: payload.Routing.IntentConfidence,
		Priority:         strings.TrimSpace(payload.Routing.Priority),
		SuggestedGroup:   strings.TrimSpace(payload.Routing.SuggestedGroup),
		SpamCheck:        buildSpamCheck(payload.SpamCheck),
	}
}

// Функция проверяет, является ли цель обращения спамом
func isReviewSpamIntent(intentID string) bool {
	raw := strings.ToLower(strings.TrimSpace(intentID))
	return raw == reviewSpamIntentID || raw == reviewLegacySpamIntentID
}

// Функция формирует review-блок после решения оператора.
func buildCompletedReview(payload routingReviewRequest) map[string]interface{} {
	return map[string]interface{}{
		"decision":    strings.ToLower(strings.TrimSpace(payload.Decision)),
		"intentId":    strings.TrimSpace(payload.Routing.IntentID),
		"priority":    normalizePriority(payload.Routing.Priority),
		"group":       strings.TrimSpace(payload.Routing.SuggestedGroup),
		"errorType":   "none",
		"comment":     "",
		"completedAt": "",
	}
}

// Финальный общий метод для обоих сценариев:обычный routing review и spam override.
func (h *ProcessHandler) finalizeReviewResult(
	c *gin.Context,
	result *services.ProcessCallResult,
	sourceFilename string,
	queueID string,
	existing map[string]interface{},
	review map[string]interface{},
	reviewKind string,
) {
	result.ProcessedAt = time.Now().UTC().Format(time.RFC3339Nano)
	if record, err := h.buildQueueCallRecord(result, sourceFilename, queueID, existing, review); err != nil {
		log.Printf("Failed to build %s call queue record: %v", reviewKind, err)
	} else {
		result.QueueID = h.saveQueueRecord(record)
	}
	c.JSON(http.StatusOK, result)
}

// HTTP endpoint, который принимает решение ручной проверки.
func (h *ProcessHandler) ResolveRoutingReview(c *gin.Context) {

	// Чтение json
	var payload routingReviewRequest
	if err := c.ShouldBindJSON(&payload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	// JSON-запрос приводится к виду внутренни структур, которые понимает оркестратор.
	transcript := buildTranscript(payload.CallID, payload.Transcript)
	routing := buildRouting(payload)

	// Эта ветка срабатывает, если: оператор принял решение accepted; выбранный класс является спамом
	// То есть ручная проверка подтверждает, что звонок действительно спам.
	if strings.ToLower(strings.TrimSpace(payload.Decision)) == "accepted" && isReviewSpamIntent(routing.IntentID) {

		// Вызов оркестратора
		result, err := h.orchestrator.ContinueAfterSpamOverride(services.ContinueAfterSpamOverrideInput{
			CallID:         payload.CallID,
			SourceFilename: payload.SourceFilename,
			Decision:       payload.Decision,
			Transcript:     transcript,
			Routing:        routing,
		})
		if err != nil {
			h.writeAudit(c, "call.spam_override", "call", transcript.CallID, "failed", map[string]interface{}{
				"decision": payload.Decision,
				"reason":   err.Error(),
			})
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		h.writeAudit(c, "call.spam_override", "call", transcript.CallID, "success", map[string]interface{}{
			"decision": payload.Decision,
			"status":   result.Status,
			"intent_id": func() string {
				if result.Routing == nil {
					return ""
				}
				return result.Routing.IntentID
			}(),
		})

		// Обновление review после результата
		// Сначала берется существующая запись в очереди, потом создается completed review.
		existing := h.loadExistingQueueRecord(payload.QueueID)
		review := buildCompletedReview(payload)
		if result.Routing != nil {
			review["intentId"] = strings.TrimSpace(result.Routing.IntentID)
			review["priority"] = normalizePriority(result.Routing.Priority)
			review["group"] = strings.TrimSpace(result.Routing.SuggestedGroup)
		}
		h.finalizeReviewResult(c, result, payload.SourceFilename, payload.QueueID, existing, review, "spam override")
		return
	}

	// Если условие spam override не сработало, выполняется обычная ветка:
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

	existing := h.loadExistingQueueRecord(payload.QueueID)
	review := buildCompletedReview(payload)
	h.finalizeReviewResult(c, result, payload.SourceFilename, payload.QueueID, existing, review, "routing review")
}
