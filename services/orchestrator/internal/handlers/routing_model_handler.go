package handlers

import (
	"net/http"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"orchestrator/internal/services"
)

type routingModelTrainPayload struct {
	Epochs       *int     `json:"epochs"`
	BatchSize    *int     `json:"batch_size"`
	LearningRate *float64 `json:"learning_rate"`
	ValRatio     *float64 `json:"val_ratio"`
	RandomSeed   *int     `json:"random_seed"`
	FeedbackPath *string  `json:"feedback_path"`
}

func (h *ProcessHandler) GetRoutingModelStatus(c *gin.Context) {
	if h.routingModelService == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "routing model service is not configured"})
		return
	}

	status, err := h.routingModelService.GetStatus()
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, status)
}

func (h *ProcessHandler) ReloadRoutingModel(c *gin.Context) {
	if h.routingModelService == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "routing model service is not configured"})
		return
	}

	status, err := h.routingModelService.Reload()
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, status)
}

func (h *ProcessHandler) TrainRoutingModel(c *gin.Context) {
	if h.routingModelService == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "routing model service is not configured"})
		return
	}

	var payload routingModelTrainPayload
	if c.Request.ContentLength > 0 {
		if err := c.ShouldBindJSON(&payload); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
			return
		}
	}

	result, err := h.routingModelService.Train(services.RoutingModelTrainRequest{
		Epochs:       payload.Epochs,
		BatchSize:    payload.BatchSize,
		LearningRate: payload.LearningRate,
		ValRatio:     payload.ValRatio,
		RandomSeed:   payload.RandomSeed,
		FeedbackPath: payload.FeedbackPath,
	})
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, result)
}

func (h *ProcessHandler) TrainRoutingModelCSV(c *gin.Context) {
	if h.routingModelService == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "routing model service is not configured"})
		return
	}

	fileHeader, err := c.FormFile("dataset")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "csv dataset file is required (form field: dataset)"})
		return
	}
	if !strings.EqualFold(filepath.Ext(fileHeader.Filename), ".csv") {
		c.JSON(http.StatusBadRequest, gin.H{"error": "dataset must be a .csv file"})
		return
	}

	file, err := fileHeader.Open()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to open uploaded csv"})
		return
	}
	defer file.Close()

	trainReq := services.RoutingModelTrainRequest{
		Epochs:       parseOptionalIntForm(c, "epochs"),
		BatchSize:    parseOptionalIntForm(c, "batch_size"),
		LearningRate: parseOptionalFloatForm(c, "learning_rate"),
		ValRatio:     parseOptionalFloatForm(c, "val_ratio"),
		RandomSeed:   parseOptionalIntForm(c, "random_seed"),
	}

	result, err := h.routingModelService.TrainFromCSV(file, fileHeader.Filename, trainReq)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, result)
}

func parseOptionalIntForm(c *gin.Context, key string) *int {
	raw := strings.TrimSpace(c.PostForm(key))
	if raw == "" {
		return nil
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return nil
	}
	return &v
}

func parseOptionalFloatForm(c *gin.Context, key string) *float64 {
	raw := strings.TrimSpace(c.PostForm(key))
	if raw == "" {
		return nil
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return nil
	}
	return &v
}
