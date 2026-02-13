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
    
    var entities models.Entities
    if err := json.NewDecoder(resp.Body).Decode(&entities); err != nil {
        return nil, fmt.Errorf("decode response: %w", err)
    }
    
    return &entities, nil
}