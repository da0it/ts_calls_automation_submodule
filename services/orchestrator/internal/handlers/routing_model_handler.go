package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"orchestrator/internal/services"
)

type routingModelTrainPayload struct {
	Epochs       *int     `json:"epochs"`
	BatchSize    *int     `json:"batch_size"`
	LearningRate *float64 `json:"learning_rate"`
	ValRatio     *float64 `json:"val_ratio"`
	RandomSeed   *int     `json:"random_seed"`
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
	})
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, result)
}
