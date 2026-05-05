package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// Структура входного JSON-запроса для обновления настроек.
type updateAppSettingsRequest struct {
	SLAMinutes int `json:"sla_minutes"`
}

// Метод GetAppSettings обрабатывает HTTP-запрос на получение настроек приложения.
func (h *ProcessHandler) GetAppSettings(c *gin.Context) {

	// Проверка, что сервис настроек подключен
	if h.appSettingsService == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "app settings service is not configured"})
		return
	}

	// Хэндлер обращается к сервисному слою для получения настроек. В случае ошибки это записывается в аудит.
	settings, err := h.appSettingsService.Get()
	if err != nil {
		h.writeAudit(c, "app.settings.get", "app_settings", "", "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	h.writeAudit(c, "app.settings.get", "app_settings", "", "success", map[string]interface{}{
		"sla_minutes": settings.SLAMinutes,
	})
	c.JSON(http.StatusOK, settings)
}

// Метод обновляет настройки приложения. В данном случае — значение SLA.
func (h *ProcessHandler) UpdateAppSettings(c *gin.Context) {
	if h.appSettingsService == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "app settings service is not configured"})
		return
	}

	var payload updateAppSettingsRequest
	if err := c.ShouldBindJSON(&payload); err != nil {
		h.writeAudit(c, "app.settings.update", "app_settings", "", "failed", map[string]interface{}{
			"reason": "invalid_payload",
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	// Здесь вызывается сервисный слой и передаётся новое значение SLA.
	settings, err := h.appSettingsService.UpdateSLA(payload.SLAMinutes)
	if err != nil {
		h.writeAudit(c, "app.settings.update", "app_settings", "", "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	h.writeAudit(c, "app.settings.update", "app_settings", "", "success", map[string]interface{}{
		"sla_minutes": settings.SLAMinutes,
	})
	c.JSON(http.StatusOK, settings)
}
