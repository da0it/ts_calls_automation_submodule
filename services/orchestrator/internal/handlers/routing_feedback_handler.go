package handlers

import (
	"net/http"
	"time"

	"orchestrator/internal/services"

	"github.com/gin-gonic/gin"
)

// Структура JSON-запроса, который приходит от интерфейса проверки
type routingFeedbackRequest struct {
	QueueID            string                               `json:"queue_id"`
	CallID             string                               `json:"call_id"`
	SourceFilename     string                               `json:"source_filename"`
	Decision           string                               `json:"decision"`
	ErrorType          string                               `json:"error_type"`
	Comment            string                               `json:"comment"`
	TranscriptText     string                               `json:"transcript_text"`
	TranscriptSegments []services.FeedbackTranscriptSegment `json:"transcript_segments"`
	TrainingSample     string                               `json:"training_sample"`

	// Блок AI - то, что предложила модель.
	AI struct {
		IntentID   string  `json:"intent_id"`
		Confidence float64 `json:"confidence"`
		Priority   string  `json:"priority"`
		Group      string  `json:"group"`
	} `json:"ai"`

	// Финальное решение оператора
	Final struct {
		IntentID string `json:"intent_id"`
		Priority string `json:"priority"`
		Group    string `json:"group"`
	} `json:"final"`
}

// Основной HTTP-хэндлер для сохранения результата в фидбек
func (h *ProcessHandler) SaveRoutingFeedback(c *gin.Context) {
	if h.routingFeedbackService == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "routing feedback service is not configured"})
		return
	}

	var payload routingFeedbackRequest
	if err := c.ShouldBindJSON(&payload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	// Входной JSON преобразуется в структуру сервисного слоя services.RoutingFeedbackRequest. Сохраняется запись фидбека
	record, err := h.routingFeedbackService.SaveFeedback(services.RoutingFeedbackRequest{
		CallID:             payload.CallID,
		SourceFilename:     payload.SourceFilename,
		Decision:           payload.Decision,
		ErrorType:          payload.ErrorType,
		Comment:            payload.Comment,
		TranscriptText:     payload.TranscriptText,
		TranscriptSegments: payload.TranscriptSegments,
		TrainingSample:     payload.TrainingSample,
		AI: services.FeedbackAISuggestion{
			IntentID:   payload.AI.IntentID,
			Confidence: payload.AI.Confidence,
			Priority:   payload.AI.Priority,
			Group:      payload.AI.Group,
		},
		Final: services.FeedbackFinalRouting{
			IntentID: payload.Final.IntentID,
			Priority: payload.Final.Priority,
			Group:    payload.Final.Group,
		},
	})
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	h.updateQueueReview(payload.QueueID, map[string]interface{}{
		"decision":    payload.Decision,
		"intentId":    payload.Final.IntentID,
		"priority":    normalizePriority(payload.Final.Priority),
		"group":       payload.Final.Group,
		"errorType":   payload.ErrorType,
		"comment":     payload.Comment,
		"completedAt": time.Now().UTC().Format(time.RFC3339Nano),
	}, record)

	c.JSON(http.StatusOK, record)
}
