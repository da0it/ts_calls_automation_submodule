package handlers

import (
	"fmt"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"orchestrator/internal/services"
)

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
	requestReceivedAt := time.Now().UTC()

	file, originalName, ext, ok := h.loadAudioUpload(c)
	if !ok {
		return
	}

	audioPath, cleanup, ok := h.saveAudioUpload(c, file, ext)
	if !ok {
		return
	}
	if cleanup != nil {
		defer cleanup()
	}

	result, ok := h.runProcessPipeline(c, audioPath, ext, file.Size)
	if !ok {
		return
	}

	h.finalizeProcessResult(result, requestReceivedAt, originalName)
	h.writeProcessSuccessAudit(c, result, ext, file.Size)
	c.JSON(http.StatusOK, result)
}

func (h *ProcessHandler) loadAudioUpload(c *gin.Context) (*multipart.FileHeader, string, string, bool) {
	file, err := c.FormFile("audio")
	if err != nil {
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason": "missing_audio",
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": "audio file is required"})
		return nil, "", "", false
	}

	originalName := filepath.Base(file.Filename)
	ext := strings.ToLower(filepath.Ext(originalName))
	log.Printf(
		"Received audio upload request_id=%s ext=%s size_mb=%.2f",
		c.GetString("request_id"),
		ext,
		float64(file.Size)/1024/1024,
	)

	if !isAllowedAudioExt(ext) {
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason":     "unsupported_audio_format",
			"audio_ext":  ext,
			"audio_size": file.Size,
		})
		c.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf("unsupported audio format: %s (allowed: mp3, wav, m4a, flac, ogg)", ext),
		})
		return nil, "", "", false
	}

	if file.Size == 0 {
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason":     "empty_audio_file",
			"audio_ext":  ext,
			"audio_size": file.Size,
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": "audio file is empty"})
		return nil, "", "", false
	}

	return file, originalName, ext, true
}

func (h *ProcessHandler) saveAudioUpload(c *gin.Context, file *multipart.FileHeader, ext string) (string, func(), bool) {
	requestID := c.GetString("request_id")
	if requestID == "" {
		requestID = "no_request_id"
	}

	filename := fmt.Sprintf("%s_%d%s", requestID, time.Now().UnixNano(), ext)
	audioPath := filepath.Join(h.uploadDir, filename)
	if err := c.SaveUploadedFile(file, audioPath); err != nil {
		log.Printf("Failed to save file: %v", err)
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason":     "save_upload_failed",
			"audio_ext":  ext,
			"audio_size": file.Size,
		})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to save audio file"})
		return "", nil, false
	}

	log.Printf("Saved uploaded audio request_id=%s", requestID)
	if !envBool("ORCH_DELETE_UPLOADED_AUDIO_AFTER_PROCESS", true) {
		return audioPath, nil, true
	}

	cleanup := func() {
		if rmErr := os.Remove(audioPath); rmErr != nil && !os.IsNotExist(rmErr) {
			log.Printf("Failed to remove uploaded audio %s: %v", audioPath, rmErr)
		}
	}
	return audioPath, cleanup, true
}

func (h *ProcessHandler) runProcessPipeline(
	c *gin.Context,
	audioPath string,
	ext string,
	audioSize int64,
) (*services.ProcessCallResult, bool) {
	result, err := h.orchestrator.ProcessCall(audioPath)
	if err != nil {
		log.Printf("Processing failed: %v", err)
		_ = os.Remove(audioPath)
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason":     "pipeline_failed",
			"audio_ext":  ext,
			"audio_size": audioSize,
		})
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("processing failed: %v", err),
		})
		return nil, false
	}
	if result == nil {
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason": "empty_pipeline_result",
		})
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "processing failed: empty pipeline result",
		})
		return nil, false
	}
	return result, true
}

func (h *ProcessHandler) finalizeProcessResult(
	result *services.ProcessCallResult,
	requestReceivedAt time.Time,
	originalName string,
) {
	result.RequestReceivedAt = requestReceivedAt.Format(time.RFC3339Nano)
	result.ProcessedAt = time.Now().UTC().Format(time.RFC3339Nano)
	if record, err := h.buildQueueCallRecord(result, originalName, "", nil, nil); err != nil {
		log.Printf("Failed to build call queue record: %v", err)
	} else {
		result.QueueID = h.saveQueueRecord(record)
	}
}

func (h *ProcessHandler) writeProcessSuccessAudit(
	c *gin.Context,
	result *services.ProcessCallResult,
	ext string,
	audioSize int64,
) {
	segmentsCount := 0
	intentID := ""
	priority := ""
	suggestedGroup := ""
	if result.Transcript != nil {
		segmentsCount = len(result.Transcript.Segments)
	}
	if result.Routing != nil {
		intentID = result.Routing.IntentID
		priority = result.Routing.Priority
		suggestedGroup = result.Routing.SuggestedGroup
	}

	h.writeAudit(c, "call.process", "call", result.CallID, "success", map[string]interface{}{
		"audio_ext":       ext,
		"audio_size":      audioSize,
		"segments_count":  segmentsCount,
		"status":          result.Status,
		"intent_id":       intentID,
		"priority":        priority,
		"suggested_group": suggestedGroup,
	})
}

func isAllowedAudioExt(ext string) bool {
	_, ok := allowedAudioFormats[ext]
	return ok
}

func envBool(name string, def bool) bool {
	raw := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
	if raw == "" {
		return def
	}
	switch raw {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return def
	}
}
