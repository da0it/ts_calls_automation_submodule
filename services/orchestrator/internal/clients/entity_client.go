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

// EntityClient — это клиент для обращения к сервису извлечения сущностей
type EntityClient struct {

	// Базовый адрес entity-service
	baseURL string

	// HTTP-клиент Go, через который отправляются запросы
	httpClient *http.Client
}

// Функция-конструктор. Принимает адрес сервиса и возвращает готовый клиент
func NewEntityClient(baseURL string) *EntityClient {
	return &EntityClient{
		baseURL: baseURL,

		// Создаётся HTTP-клиент с таймаутом 30 секунд. Если entity-service не ответит за 30 секунд, запрос завершится ошибкой.
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Структура описывает одну найденную сущность.
type ExtractedEntity struct {
	Type       string  `json:"type"`
	Value      string  `json:"value"`
	Confidence float64 `json:"confidence"`
	Context    string  `json:"context"`
}

// Entities — набор всех найденных сущностей, сгруппированных по типам
type Entities struct {
	Persons      []ExtractedEntity `json:"persons"`
	Phones       []ExtractedEntity `json:"phones"`
	Emails       []ExtractedEntity `json:"emails"`
	OrderIDs     []ExtractedEntity `json:"order_ids"`
	AccountIDs   []ExtractedEntity `json:"account_ids"`
	MoneyAmounts []ExtractedEntity `json:"money_amounts"`
	Dates        []ExtractedEntity `json:"dates"`
}

// Тело запроса, которое модуль управления отправляет в entity-service
type EntityRequest struct {
	Segments []Segment `json:"segments"`
}

// Структура ответа от entity-service.
type EntityResponse struct {
	Entities Entities `json:"entities"`
}

// Главный метод файла. Отправляет сегменты транскрипции в entity-service и возвращает найденные сущности
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
