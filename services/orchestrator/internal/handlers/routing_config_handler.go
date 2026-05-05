package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// HTTP-хэндлер Gin.
func (h *ProcessHandler) GetRoutingConfig(c *gin.Context) {

	// Хэндлер обращается к сервису маршрутизации. routingConfigService отвечает за получение справочника маршрутизации
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
