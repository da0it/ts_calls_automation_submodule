package adapters

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"ticket_module/internal/models"
)

// Структура конфигурации адаптера SimpleOne
type SimpleOneAdapterConfig struct {

	// URL endpoint, куда нужно отправлять запрос на создание заявки
	EndpointURL string

	// Токен авторизации
	BearerToken string

	// Таймаут HTTP-запроса
	Timeout     time.Duration
}

// Структура адаптера для тикет-системы SimpleOne
type SimpleOneAdapter struct {

	// Адрес API SimpleOne
	endpointURL string

	// Токен авторизации
	bearerToken string

	// HTTP-клиент с таймаутом
	httpClient  *http.Client
}

// Конструктор адаптера. Принимает конфиг и возвращает *SimpleOneAdapter если все успешно, и error если конфигурация некорректна
func NewSimpleOneAdapter(cfg SimpleOneAdapterConfig) (*SimpleOneAdapter, error) {

	// URL очищается от пробелов по краям
	endpointURL := strings.TrimSpace(cfg.EndpointURL)
	if endpointURL == "" {
		return nil, fmt.Errorf("simpleone endpoint URL is required")
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 30 * time.Second
	}

	// Создается объект адаптера
	return &SimpleOneAdapter{
		endpointURL: endpointURL,
		bearerToken: strings.TrimSpace(cfg.BearerToken),
		httpClient:  &http.Client{Timeout: cfg.Timeout},
	}, nil
}

// Главный метод адаптера. Принимает внутренний payload (структуру с содержанием) тикет-сервиса и возвращает результат создания 
func (a *SimpleOneAdapter) CreateTicket(payload *models.TicketSystemPayload) (*models.TicketCreated, error) {
	if a == nil {
		return nil, fmt.Errorf("simpleone adapter is not configured")
	}
	if payload == nil {
		return nil, fmt.Errorf("ticket payload is required")
	}
	if payload.Draft == nil {
		return nil, fmt.Errorf("ticket payload draft is required")
	}

	// Внутренний payload преобразуется в формат, который будет отправлен в SimpleOne
	body, err := json.Marshal(buildSimpleOnePayload(payload))
	if err != nil {
		return nil, fmt.Errorf("marshal simpleone payload: %w", err)
	}

	// Создание HTTP POST-запроса. Аргументы:  метод POST, URL SimpleOne, тело запроса с JSON
	req, err := http.NewRequest(http.MethodPost, a.endpointURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create simpleone request: %w", err)
	}

	// Заголовки запроса
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "ts-calls-automation/ticket-service")
	if a.bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+a.bearerToken)
	}

	// Отправка запроса
	resp, err := a.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send simpleone request: %w", err)
	}
	defer resp.Body.Close()

	// Чтение ответа
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read simpleone response: %w", err)
	}

	// Проверка статуса HTTP
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("simpleone response %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	return buildSimpleOneCreated(respBody, payload), nil
}

// Функция преобразует внутренний TicketSystemPayload в JSON-совместимую map для отправки в SimpleOne.
// Принимает на вход Payload. Возвращает словарь
func buildSimpleOnePayload(payload *models.TicketSystemPayload) map[string]interface{} {
	draft := payload.Draft
	request := payload.Request
	summary := payload.Summary

	body := map[string]interface{}{
		"title":               draft.Title,
		"description":         draft.Description,
		"priority":            draft.Priority,
		"assignee_type":       draft.AssigneeType,
		"assignee_id":         draft.AssigneeID,
		"tags":                draft.Tags,
		"call_id":             draft.CallID,
		"audio_url":           draft.AudioURL,
		"transcript_url":      draft.TranscriptURL,
		"intent_id":           draft.IntentID,
		"intent_confidence":   draft.IntentConfidence,
		"custom_fields":       draft.CustomFields,
		"problem_summary":     collapseWhitespace(simpleOneProblemSummary(summary, draft)),
		"transcript_text":     simpleOneTranscriptText(request.Transcript.Segments),
		"transcript_segments": request.Transcript.Segments,
		"entities":            draft.Entities,
		"service":             payload.Service,
		"request":             request,
		"summary":             summary,
		"draft":               draft,
	}

	// Источник файла
	if sourceFile := simpleOneSourceFile(request.Transcript.Metadata); sourceFile != "" {
		body["source_file"] = sourceFile
	}

	return body
}

// Функция выбирает текст краткого описания проблемы
func simpleOneProblemSummary(summary *models.TicketSummary, draft *models.TicketDraft) string {
	if summary != nil {
		if value := strings.TrimSpace(summary.Description); value != "" {
			return value
		}
	}
	if draft != nil {
		return strings.TrimSpace(draft.Description)
	}
	return ""
}

// Функция преобразует список сегментов транскрипции в обычный многострочный текст
func simpleOneTranscriptText(segments []models.Segment) string {
	if len(segments) == 0 {
		return ""
	}

	// Создаётся срез строк. Емкость заранее задаётся равной количеству сегментов
	lines := make([]string, 0, len(segments))

	// Цикл проходит по всем сегментам
	for _, segment := range segments {
		text := collapseWhitespace(segment.Text)
		if text == "" {
			continue
		}

		label := collapseWhitespace(segment.Role)
		if label == "" {
			label = collapseWhitespace(segment.Speaker)
		}
		if label != "" {
			lines = append(lines, label+": "+text)
		} else {
			lines = append(lines, text)
		}
	}

	return strings.Join(lines, "\n")
}

// Функция пытается достать из метаданных имя исходного файла
func simpleOneSourceFile(metadata map[string]interface{}) string {
	if len(metadata) == 0 {
		return ""
	}
	if raw, ok := metadata["source_file"]; ok {
		if value, ok := raw.(string); ok {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

// Функция необходима для нормализации пробелов в строке
func collapseWhitespace(value string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
}

// Функция преобразует ответ SimpleOne во внутреннюю модель TicketCreated.
func buildSimpleOneCreated(respBody []byte, payload *models.TicketSystemPayload) *models.TicketCreated {

	// Сначала создаётся базовый результат
	created := &models.TicketCreated{
		TicketID:   simpleOneAckID(payload),
		ExternalID: simpleOneAckID(payload),
		System:     "simpleone",
		CreatedAt:  time.Now().UTC(),
	}
	if strings.TrimSpace(string(respBody)) == "" {
		return created
	}

	// Если тело не пустое, код пытается разобрать его как JSON.
	var decoded map[string]interface{}
	if err := json.Unmarshal(respBody, &decoded); err != nil {
		return created
	}

	// Функция simpleOneFindString пытается найти ID тикета в ответе SimpleOne.
	if id := simpleOneFindString(decoded, "external_id", "id", "number", "ticket_id"); id != "" {
		created.TicketID = id
		created.ExternalID = id
	}

	// Аналогично ищется ссылка на запись
	if url := simpleOneFindString(decoded, "record_url", "url", "recordUrl", "href", "link"); url != "" {
		created.URL = url
	}
	return created
}

// Функция рекурсивно ищет строковое значение по одному из ключей в произвольном JSON-объекте
func simpleOneFindString(node interface{}, keys ...string) string {

	// Проверка фактического значения переданного node.
	switch value := node.(type) {

	// Сначала функция проверяет нужные ключи на текущем уровне:
	case map[string]interface{}:
		for _, key := range keys {
			if raw, ok := value[key]; ok {
				if text, ok := raw.(string); ok && strings.TrimSpace(text) != "" {
					return strings.TrimSpace(text)
				}
			}
		}

		// Если на текущем уровне ничего не найдено, функция идёт глубже:
		for _, raw := range value {
			if text := simpleOneFindString(raw, keys...); text != "" {
				return text
			}
		}
	case []interface{}:
		for _, item := range value {
			if text := simpleOneFindString(item, keys...); text != "" {
				return text
			}
		}
	}
	return ""
}

// Функция формирует технический ID подтверждения, если SimpleOne не вернул нормальный ID тикета
func simpleOneAckID(payload *models.TicketSystemPayload) string {
	if payload != nil && payload.Service.CallID != "" {
		return "simpleone-ack-" + payload.Service.CallID
	}
	return "simpleone-ack"
}
