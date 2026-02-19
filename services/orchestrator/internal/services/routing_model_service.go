package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type RoutingModelTrainRequest struct {
	Epochs       *int     `json:"epochs,omitempty"`
	BatchSize    *int     `json:"batch_size,omitempty"`
	LearningRate *float64 `json:"learning_rate,omitempty"`
	ValRatio     *float64 `json:"val_ratio,omitempty"`
	RandomSeed   *int     `json:"random_seed,omitempty"`
}

type RoutingModelService struct {
	baseURL    string
	adminToken string
	client     *http.Client
}

func NewRoutingModelService(baseURL, adminToken string, timeout time.Duration) *RoutingModelService {
	url := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if url == "" {
		return nil
	}
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	return &RoutingModelService{
		baseURL:    url,
		adminToken: strings.TrimSpace(adminToken),
		client: &http.Client{
			Timeout: timeout,
		},
	}
}

func (s *RoutingModelService) GetStatus() (map[string]any, error) {
	return s.requestJSON(http.MethodGet, "/admin/model/status", nil)
}

func (s *RoutingModelService) Train(req RoutingModelTrainRequest) (map[string]any, error) {
	return s.requestJSON(http.MethodPost, "/admin/model/train", req)
}

func (s *RoutingModelService) requestJSON(method, path string, payload any) (map[string]any, error) {
	if s == nil {
		return nil, fmt.Errorf("routing model service is not configured")
	}

	var body io.Reader
	if payload != nil {
		raw, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal request body: %w", err)
		}
		body = bytes.NewReader(raw)
	}

	req, err := http.NewRequest(method, s.baseURL+path, body)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if s.adminToken != "" {
		req.Header.Set("Authorization", "Bearer "+s.adminToken)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request router admin: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	var parsed map[string]any
	if len(respBody) > 0 {
		if err := json.Unmarshal(respBody, &parsed); err != nil {
			return nil, fmt.Errorf("decode response json: %w", err)
		}
	} else {
		parsed = map[string]any{}
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		if msg, ok := parsed["error"].(string); ok && strings.TrimSpace(msg) != "" {
			return nil, fmt.Errorf("%s", msg)
		}
		return nil, fmt.Errorf("router admin http %d", resp.StatusCode)
	}

	return parsed, nil
}
