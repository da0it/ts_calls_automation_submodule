package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"orchestrator/internal/services"
)

type routingFeedbackRequest struct {
	CallID             string                               `json:"call_id"`
	SourceFilename     string                               `json:"source_filename"`
	Decision           string                               `json:"decision"`
	ErrorType          string                               `json:"error_type"`
	Comment            string                               `json:"comment"`
	TranscriptText     string                               `json:"transcript_text"`
	TranscriptSegments []services.FeedbackTranscriptSegment `json:"transcript_segments"`
	TrainingSample     string                               `json:"training_sample"`
	AI                 struct {
		IntentID   string  `json:"intent_id"`
		Confidence float64 `json:"confidence"`
		Priority   string  `json:"priority"`
		Group      string  `json:"group"`
	} `json:"ai"`
	Final struct {
		IntentID string `json:"intent_id"`
		Priority string `json:"priority"`
		Group    string `json:"group"`
	} `json:"final"`
}

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

	c.JSON(http.StatusOK, record)
}
