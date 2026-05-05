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

// Структура хранит настройки для обращения к admin API router-service
type RoutingModelService struct {

	// Базовый адрес router admin API
	baseURL string

	// Токен для авторизации в admin API router-service
	adminToken string

	// HTTP-клиент с настроенным timeout.
	client *http.Client
}

// Функция-конструктор создает сервис для работы с admin API router-service
func NewRoutingModelService(baseURL, adminToken string, timeout time.Duration, datasetDir string) *RoutingModelService {
	url := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if url == "" {
		return nil
	}
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}

	// Возвращает готовый сервис
	return &RoutingModelService{
		baseURL:    url,
		adminToken: strings.TrimSpace(adminToken),
		client: &http.Client{
			Timeout: timeout,
		},
	}
}

// Метод получает статус модели маршрутизации.
func (s *RoutingModelService) GetStatus() (map[string]any, error) {
	return s.requestJSON(http.MethodGet, "/admin/model/status", nil)
}

// Универсальный метод для выполнения HTTP-запросов к router admin API
func (s *RoutingModelService) requestJSON(method, path string, payload any) (map[string]any, error) {
	if s == nil {
		return nil, fmt.Errorf("routing model service is not configured")
	}

	// Подготовка тела запроса
	var body io.Reader
	if payload != nil {

		// Сериализация payload
		raw, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal request body: %w", err)
		}
		body = bytes.NewReader(raw)
	}

	// Создаётся HTTP-запрос
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

	// Выполнение запроса
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request router admin: %w", err)
	}
	defer resp.Body.Close()

	// Чтение тела ответа
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	// Если тело ответа не пустое, оно декодируется в: map[string]any
	var parsed map[string]any
	if len(respBody) > 0 {
		if err := json.Unmarshal(respBody, &parsed); err != nil {
			return nil, fmt.Errorf("decode response json: %w", err)
		}
	} else {
		parsed = map[string]any{}
	}

	// Обработка http-ошибок
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		if msg, ok := parsed["error"].(string); ok && strings.TrimSpace(msg) != "" {
			return nil, fmt.Errorf("%s", msg)
		}
		return nil, fmt.Errorf("router admin http %d", resp.StatusCode)
	}

	return parsed, nil
}
