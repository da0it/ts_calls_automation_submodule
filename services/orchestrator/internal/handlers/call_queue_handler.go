package handlers

import (
	"database/sql"
	"net/http"

	"github.com/gin-gonic/gin"
)

// Функция получает список звонков из очереди.
func (h *ProcessHandler) ListCalls(c *gin.Context) {
	if h.callQueueService == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "call queue service is not configured"})
		return
	}

	limit := parseQueryInt(c.Query("limit"), 200)
	offset := parseQueryInt(c.Query("offset"), 0)

	calls, err := h.callQueueService.List(limit, offset)
	if err != nil {
		h.writeAudit(c, "call.queue.list", "call_queue", "", "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	h.writeAudit(c, "call.queue.list", "call_queue", "", "success", map[string]interface{}{
		"limit":        limit,
		"offset":       offset,
		"result_count": len(calls),
	})
	c.JSON(http.StatusOK, gin.H{
		"calls":  calls,
		"limit":  limit,
		"offset": offset,
		"count":  len(calls),
	})
}

// Функция DeleteCall удаляет один звонок по id
func (h *ProcessHandler) DeleteCall(c *gin.Context) {
	if h.callQueueService == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "call queue service is not configured"})
		return
	}

	id := c.Param("id")
	if err := h.callQueueService.Delete(id); err != nil {
		outcome := "failed"
		status := http.StatusBadRequest
		if err == sql.ErrNoRows {
			status = http.StatusNotFound
		}
		h.writeAudit(c, "call.queue.delete", "call_queue", id, outcome, map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(status, gin.H{"error": err.Error()})
		return
	}

	h.writeAudit(c, "call.queue.delete", "call_queue", id, "success", map[string]interface{}{})
	c.JSON(http.StatusOK, gin.H{"message": "call deleted"})
}

// Функция ClearCalls полностью очищает очередь звонков.
func (h *ProcessHandler) ClearCalls(c *gin.Context) {
	if h.callQueueService == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "call queue service is not configured"})
		return
	}

	removed, err := h.callQueueService.Clear()
	if err != nil {
		h.writeAudit(c, "call.queue.clear", "call_queue", "", "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	h.writeAudit(c, "call.queue.clear", "call_queue", "", "success", map[string]interface{}{
		"removed_count": removed,
	})
	c.JSON(http.StatusOK, gin.H{"message": "call queue cleared", "removed_count": removed})
}
