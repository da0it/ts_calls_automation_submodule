// internal/clients/python_client.go
package clients

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"ticket_module/internal/models"
)

type extractResponse struct {
	Entities models.Entities `json:"entities"`
}

type PythonClient struct {
	baseURL    string
	httpClient *http.Client
}

func NewPythonClient(baseURL string) *PythonClient {
	return &PythonClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// ExtractEntities вызывает Python NER сервис для извлечения сущностей
func (c *PythonClient) ExtractEntities(segments []models.Segment) (*models.Entities, error) {
	reqBody := map[string]interface{}{
		"segments": segments,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(
		c.baseURL+"/api/extract-entities",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	var wrapped extractResponse
	if err := json.Unmarshal(respBody, &wrapped); err == nil {
		// Typical shape: {"entities": {...}}
		if wrapped.Entities.Persons != nil ||
			wrapped.Entities.Phones != nil ||
			wrapped.Entities.Emails != nil ||
			wrapped.Entities.OrderIDs != nil ||
			wrapped.Entities.AccountIDs != nil ||
			wrapped.Entities.MoneyAmounts != nil ||
			wrapped.Entities.Dates != nil {
			return &wrapped.Entities, nil
		}
	}

	// Backward compatibility: raw entities payload
	var direct models.Entities
	if err := json.Unmarshal(respBody, &direct); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &direct, nil
}
