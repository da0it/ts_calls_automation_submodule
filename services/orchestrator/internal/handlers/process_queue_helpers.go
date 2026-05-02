package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	"orchestrator/internal/services"
)

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

func defaultReview(intentID string, priority string, group string) map[string]interface{} {
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

func reviewOrExisting(
	review map[string]interface{},
	existing map[string]interface{},
	intentID string,
	priority string,
	group string,
) map[string]interface{} {
	if review != nil {
		return review
	}
	if existingReview := mapObject(existing["review"]); existingReview != nil {
		return existingReview
	}
	return defaultReview(intentID, priority, group)
}

func requestReceivedAt(result *services.ProcessCallResult, existing map[string]interface{}) string {
	value := strings.TrimSpace(result.RequestReceivedAt)
	if value == "" {
		value = mapString(existing["createdAt"])
	}
	if value == "" {
		value = time.Now().UTC().Format(time.RFC3339Nano)
	}
	return value
}

func processedAt(result *services.ProcessCallResult) string {
	value := strings.TrimSpace(result.ProcessedAt)
	if value == "" {
		value = time.Now().UTC().Format(time.RFC3339Nano)
	}
	return value
}

func callIDFromResult(result *services.ProcessCallResult) string {
	callID := strings.TrimSpace(result.CallID)
	if callID == "" && result.Transcript != nil {
		callID = strings.TrimSpace(result.Transcript.CallID)
	}
	return callID
}

func routingSuggestion(routing map[string]interface{}) (string, string, string, float64) {
	if routing == nil {
		return "", "", "medium", 0
	}

	intentID := mapString(routing["intent_id"])
	group := mapString(routing["suggested_group"])
	priority := normalizePriority(mapString(routing["priority"]))
	confidence := 0.0
	if value, ok := routing["intent_confidence"].(float64); ok {
		confidence = value
	}
	return intentID, group, priority, confidence
}

func resolveSLA(existing map[string]interface{}, fallbackStartedAt string, fallbackLimitMinutes int) (string, int) {
	startedAt := fallbackStartedAt
	limitMinutes := fallbackLimitMinutes
	if existingSLA := mapObject(existing["sla"]); existingSLA != nil {
		if value := mapString(existingSLA["startedAt"]); value != "" {
			startedAt = value
		}
		if value, ok := existingSLA["limitMinutes"].(float64); ok && int(value) > 0 {
			limitMinutes = int(value)
		}
	}
	return startedAt, limitMinutes
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

	requestAt := requestReceivedAt(result, existing)
	doneAt := processedAt(result)
	suggestedIntentID, suggestedGroup, priority, confidence := routingSuggestion(routing)
	review = reviewOrExisting(review, existing, suggestedIntentID, priority, suggestedGroup)

	stopTime := automaticStopTime(strings.TrimSpace(result.Status), ticket, doneAt)
	if mapString(review["completedAt"]) == "" && stopTime != "" {
		review["completedAt"] = stopTime
	}

	slaStartedAt, slaMinutes := resolveSLA(existing, requestAt, h.currentSLAMinutes())

	record := map[string]interface{}{
		"id":             strings.TrimSpace(queueID),
		"sourceFilename": strings.TrimSpace(sourceFilename),
		"createdAt":      requestAt,
		"processedAt":    doneAt,
		"callId":         callIDFromResult(result),
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
