package handlers

import (
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"orchestrator/internal/clients"
	"orchestrator/internal/services"
)

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

func (p reviewSpamCheckPayload) isEmpty() bool {
	return p.Status == "" &&
		p.PredictedLabel == "" &&
		p.Confidence == 0 &&
		p.ThresholdLow == 0 &&
		p.ThresholdHigh == 0 &&
		p.Reason == "" &&
		p.Backend == ""
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

func buildRouting(payload routingReviewRequest) *clients.RoutingResponse {
	return &clients.RoutingResponse{
		IntentID:         strings.TrimSpace(payload.Routing.IntentID),
		IntentConfidence: payload.Routing.IntentConfidence,
		Priority:         strings.TrimSpace(payload.Routing.Priority),
		SuggestedGroup:   strings.TrimSpace(payload.Routing.SuggestedGroup),
		SpamCheck:        buildSpamCheck(payload.SpamCheck),
	}
}

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

	existing := h.loadExistingQueueRecord(payload.QueueID)
	review := buildSuggestedReview(result)
	h.finalizeReviewResult(c, result, payload.SourceFilename, payload.QueueID, existing, review, "spam override")
}

func (h *ProcessHandler) ResolveRoutingReview(c *gin.Context) {
	var payload routingReviewRequest
	if err := c.ShouldBindJSON(&payload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	transcript := buildTranscript(payload.CallID, payload.Transcript)
	routing := buildRouting(payload)

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
