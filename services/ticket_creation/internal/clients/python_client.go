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

// Вспомогательная структура для разбора ответа Python-сервиса
type extractResponse struct {
	Entities models.Entities `json:"entities"`
}

// HTTP-клиент для обращения к Python-сервису NER/entity-extraction
type PythonClient struct {

	// Базовый адрес Python-сервиса.
	baseURL    string

	// HTTP-клиент с настроенным таймаутом
	httpClient *http.Client
}

// Конструктор клиента. Принимает адрес python-сервиса и возвращает готовый PythonClient
func NewPythonClient(baseURL string) *PythonClient {

	// Создаётся HTTP-клиент с таймаутом 30 секунд
	return &PythonClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// ExtractEntities. Главный метод клиента. Вызывает Python NER сервис для извлечения сущностей.
// Принимает список сегментов транскрипции и возвращает *models.Entities
func (c *PythonClient) ExtractEntities(segments []models.Segment) (*models.Entities, error) {
	reqBody := map[string]interface{}{
		"segments": segments,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	// Отправка HTTP-запроса
	resp, err := c.httpClient.Post(

		// URL Python-сервиса
		c.baseURL+"/api/extract-entities",

		// Content-Type
		"application/json",

		// Тело запроса
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}

	// Тело ответа обязательно закрывается в конце функции
	defer resp.Body.Close()

	// Проверка HTTP-статуса (ожидается 200 ok)
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// Чтение успешного ответа. Если статус 200 OK, тело ответа читается полностью.
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	// Код пытается разобрать ответ в стандартный формат:
	var wrapped extractResponse
	if err := json.Unmarshal(respBody, &wrapped); err == nil {
		// Проверка, что Unmarshal выдал результат, который от него ожидается
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

	// Если ответ не подошёл под формат: {"entities": {...}} код пробует разобрать его как старый формат — напрямую Entities.
	// Старый формат: {"persons": [], "phones": [], ...}
	var direct models.Entities
	if err := json.Unmarshal(respBody, &direct); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &direct, nil
}
