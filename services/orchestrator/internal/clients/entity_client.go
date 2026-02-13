// internal/clients/entity_client.go
package clients

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type EntityClient struct {
	baseURL    string
	httpClient *http.Client
}

func NewEntityClient(baseURL string) *EntityClient {
	return &EntityClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

type ExtractedEntity struct {
	Type       string  `json:"type"`
	Value      string  `json:"value"`
	Confidence float64 `json:"confidence"`
	Context    string  `json:"context"`
}

type Entities struct {
	Persons      []ExtractedEntity `json:"persons"`
	Phones       []ExtractedEntity `json:"phones"`
	Emails       []ExtractedEntity `json:"emails"`
	OrderIDs     []ExtractedEntity `json:"order_ids"`
	AccountIDs   []ExtractedEntity `json:"account_ids"`
	MoneyAmounts []ExtractedEntity `json:"money_amounts"`
	Dates        []ExtractedEntity `json:"dates"`
}

type EntityRequest struct {
	Segments []Segment `json:"segments"`
}

type EntityResponse struct {
	Entities Entities `json:"entities"`
}

func (c *EntityClient) Extract(segments []Segment) (*Entities, error) {
	reqBody := EntityRequest{
		Segments: segments,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := c.baseURL + "/api/extract-entities"
	resp, err := c.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("entity service returned %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result EntityResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &result.Entities, nil
}