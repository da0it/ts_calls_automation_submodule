// internal/handlers/process_handler.go
package handlers

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/gin-gonic/gin"
	"orchestrator/internal/services"
)

type ProcessHandler struct {
	orchestrator         *services.OrchestratorService
	routingConfigService *services.RoutingConfigService
	uploadDir            string
}

func NewProcessHandler(
	orchestrator *services.OrchestratorService,
	routingConfigService *services.RoutingConfigService,
) *ProcessHandler {
	// Создаём директорию для загрузки файлов
	uploadDir := "./uploads"
	os.MkdirAll(uploadDir, 0755)

	return &ProcessHandler{
		orchestrator:         orchestrator,
		routingConfigService: routingConfigService,
		uploadDir:            uploadDir,
	}
}

// ProcessCall godoc
// @Summary Обработать аудио звонка
// @Description Загружает аудио файл, транскрибирует, определяет интент и создает тикет
// @Tags calls
// @Accept multipart/form-data
// @Produce json
// @Param audio formData file true "Audio file (mp3, wav, m4a)"
// @Success 200 {object} services.ProcessCallResult
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /api/v1/process-call [post]
func (h *ProcessHandler) ProcessCall(c *gin.Context) {
	// 1. Получаем загруженный файл
	file, err := c.FormFile("audio")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "audio file is required",
		})
		return
	}

	log.Printf("Received audio file: %s (%.2f MB)", file.Filename, float64(file.Size)/1024/1024)

	// Валидация формата
	ext := filepath.Ext(file.Filename)
	allowedFormats := map[string]bool{
		".mp3":  true,
		".wav":  true,
		".m4a":  true,
		".flac": true,
		".ogg":  true,
	}
	if !allowedFormats[ext] {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf("unsupported audio format: %s (allowed: mp3, wav, m4a, flac, ogg)", ext),
		})
		return
	}

	// 2. Сохраняем файл
	requestID := c.GetString("request_id")
	if requestID == "" {
		requestID = "no_request_id"
	}
	filename := fmt.Sprintf("%s_%s", requestID, file.Filename)
	audioPath := filepath.Join(h.uploadDir, filename)

	if err := c.SaveUploadedFile(file, audioPath); err != nil {
		log.Printf("Failed to save file: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "failed to save audio file",
		})
		return
	}

	log.Printf("Audio saved to: %s", audioPath)

	// 3. Запускаем обработку
	result, err := h.orchestrator.ProcessCall(audioPath)
	if err != nil {
		log.Printf("Processing failed: %v", err)

		// Удаляем файл при ошибке
		os.Remove(audioPath)

		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("processing failed: %v", err),
		})
		return
	}

	// 4. Опционально: удаляем файл после обработки
	// os.Remove(audioPath)

	c.JSON(http.StatusOK, result)
}

// Health godoc
// @Summary Health check
// @Description Проверка доступности оркестратора и зависимых сервисов
// @Tags health
// @Produce json
// @Success 200 {object} map[string]string
// @Router /health [get]
func (h *ProcessHandler) Health(c *gin.Context) {
	status := h.orchestrator.HealthCheck()
	c.JSON(http.StatusOK, status)
}

// Root godoc
// @Summary API Information
// @Description Информация об Orchestrator API
// @Tags info
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /api/info [get]
func (h *ProcessHandler) Root(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"service":     "Ticket System Orchestrator",
		"version":     "1.0.0",
		"description": "Оркестрирует обработку звонков через все модули системы",
		"endpoints": gin.H{
			"process_call":    "POST /api/v1/process-call",
			"routing_config":  "GET/PUT /api/v1/routing-config",
			"routing_groups":  "POST/DELETE /api/v1/routing-config/groups",
			"routing_intents": "POST/DELETE /api/v1/routing-config/intents",
			"health":          "GET /health",
			"docs":            "GET /docs (если включен Swagger)",
		},
		"pipeline": []string{
			"1. Transcription + Diarization",
			"2. Routing (RuBERT Intent Classification)",
			"3. Ticket Creation",
		},
	})
}
