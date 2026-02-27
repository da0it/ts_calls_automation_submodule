package handlers

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"orchestrator/internal/services"
)

type createGroupRequest struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Description string `json:"description"`
}

type createIntentRequest struct {
	ID           string   `json:"id"`
	Title        string   `json:"title"`
	Description  string   `json:"description"`
	Examples     []string `json:"examples"`
	DefaultGroup string   `json:"default_group"`
	Priority     string   `json:"priority"`
	Tags         []string `json:"tags"`
	Keywords     []string `json:"keywords"`
}

func (h *ProcessHandler) GetRoutingConfig(c *gin.Context) {
	catalog, err := h.routingConfigService.GetCatalog()
	if err != nil {
		h.writeAudit(c, "routing.config.get", "routing_config", "", "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	h.writeAudit(c, "routing.config.get", "routing_config", "", "success", map[string]interface{}{
		"groups_count":  len(catalog.Groups),
		"intents_count": len(catalog.Intents),
	})
	c.JSON(http.StatusOK, catalog)
}

func (h *ProcessHandler) UpdateRoutingConfig(c *gin.Context) {
	var payload services.RoutingCatalog
	if err := c.ShouldBindJSON(&payload); err != nil {
		h.writeAudit(c, "routing.config.update", "routing_config", "", "failed", map[string]interface{}{
			"reason": "invalid_payload",
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	catalog, err := h.routingConfigService.ReplaceCatalog(&payload)
	if err != nil {
		h.writeAudit(c, "routing.config.update", "routing_config", "", "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	h.writeAudit(c, "routing.config.update", "routing_config", "", "success", map[string]interface{}{
		"groups_count":  len(catalog.Groups),
		"intents_count": len(catalog.Intents),
	})
	c.JSON(http.StatusOK, catalog)
}

func (h *ProcessHandler) CreateRoutingGroup(c *gin.Context) {
	var payload createGroupRequest
	if err := c.ShouldBindJSON(&payload); err != nil {
		h.writeAudit(c, "routing.group.create", "routing_group", payload.ID, "failed", map[string]interface{}{
			"reason": "invalid_payload",
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	catalog, err := h.routingConfigService.AddGroup(services.RoutingGroup{
		ID:          payload.ID,
		Title:       payload.Title,
		Description: payload.Description,
	})
	if err != nil {
		h.writeAudit(c, "routing.group.create", "routing_group", payload.ID, "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	h.writeAudit(c, "routing.group.create", "routing_group", payload.ID, "success", map[string]interface{}{})
	c.JSON(http.StatusOK, catalog)
}

func (h *ProcessHandler) DeleteRoutingGroup(c *gin.Context) {
	groupID := strings.TrimSpace(c.Param("id"))
	catalog, err := h.routingConfigService.DeleteGroup(groupID)
	if err != nil {
		h.writeAudit(c, "routing.group.delete", "routing_group", groupID, "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		status := http.StatusBadRequest
		if strings.Contains(err.Error(), "not found") {
			status = http.StatusNotFound
		}
		c.JSON(status, gin.H{"error": err.Error()})
		return
	}
	h.writeAudit(c, "routing.group.delete", "routing_group", groupID, "success", map[string]interface{}{})
	c.JSON(http.StatusOK, catalog)
}

func (h *ProcessHandler) CreateRoutingIntent(c *gin.Context) {
	var payload createIntentRequest
	if err := c.ShouldBindJSON(&payload); err != nil {
		h.writeAudit(c, "routing.intent.create", "routing_intent", payload.ID, "failed", map[string]interface{}{
			"reason": "invalid_payload",
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	catalog, err := h.routingConfigService.AddIntent(services.RoutingIntent{
		ID:           payload.ID,
		Title:        payload.Title,
		Description:  payload.Description,
		Examples:     payload.Examples,
		DefaultGroup: payload.DefaultGroup,
		Priority:     payload.Priority,
		Tags:         payload.Tags,
		Keywords:     payload.Keywords,
	})
	if err != nil {
		h.writeAudit(c, "routing.intent.create", "routing_intent", payload.ID, "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	h.writeAudit(c, "routing.intent.create", "routing_intent", payload.ID, "success", map[string]interface{}{})
	c.JSON(http.StatusOK, catalog)
}

func (h *ProcessHandler) DeleteRoutingIntent(c *gin.Context) {
	intentID := strings.TrimSpace(c.Param("id"))
	catalog, err := h.routingConfigService.DeleteIntent(intentID)
	if err != nil {
		h.writeAudit(c, "routing.intent.delete", "routing_intent", intentID, "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		status := http.StatusBadRequest
		if strings.Contains(err.Error(), "not found") {
			status = http.StatusNotFound
		}
		c.JSON(status, gin.H{"error": err.Error()})
		return
	}
	h.writeAudit(c, "routing.intent.delete", "routing_intent", intentID, "success", map[string]interface{}{})
	c.JSON(http.StatusOK, catalog)
}
