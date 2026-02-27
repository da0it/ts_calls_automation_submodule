package handlers

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
)

func (h *ProcessHandler) ListAuditEvents(c *gin.Context) {
	if h.auditService == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "audit service is not configured"})
		return
	}

	limit := parseQueryInt(c.Query("limit"), 100)
	offset := parseQueryInt(c.Query("offset"), 0)
	eventType := strings.TrimSpace(c.Query("event_type"))
	outcome := strings.TrimSpace(c.Query("outcome"))
	resourceType := strings.TrimSpace(c.Query("resource_type"))

	events, err := h.auditService.ListEvents(limit, offset, eventType, outcome, resourceType)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	h.writeAudit(c, "audit.events.list", "audit", "", "success", map[string]interface{}{
		"limit":         limit,
		"offset":        offset,
		"event_type":    eventType,
		"outcome":       outcome,
		"resource_type": resourceType,
		"result_count":  len(events),
	})

	c.JSON(http.StatusOK, gin.H{
		"events": events,
		"limit":  limit,
		"offset": offset,
		"count":  len(events),
	})
}

func parseQueryInt(raw string, def int) int {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return def
	}
	return value
}
