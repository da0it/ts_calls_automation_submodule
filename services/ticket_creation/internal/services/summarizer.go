// internal/services/summarizer.go
package services

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

const (
	anthropicAPIURL     = "https://api.anthropic.com/v1/messages"
	anthropicAPIVersion = "2023-06-01"
	claudeModel         = "claude-sonnet-4-5-20250929"
)

type ClaudeSummarizer struct {
	apiKey     string
	httpClient *http.Client
}

func NewClaudeSummarizer(apiKey string) *ClaudeSummarizer {
	return &ClaudeSummarizer{
		apiKey: apiKey,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

// claudeRequest описывает запрос к Anthropic Messages API
type claudeRequest struct {
	Model     string          `json:"model"`
	MaxTokens int             `json:"max_tokens"`
	System    string          `json:"system"`
	Messages  []claudeMessage `json:"messages"`
}

type claudeMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// claudeResponse описывает ответ от Anthropic Messages API
type claudeResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type claudeErrorResponse struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// GenerateSummary генерирует заголовок и описание тикета через Claude API
func (s *ClaudeSummarizer) GenerateSummary(
	segments []models.Segment,
	intentID string,
	priority string,
	entities *models.Entities,
) (*models.TicketSummary, error) {
	if s.apiKey == "" {
		return s.fallbackSummary(segments, intentID, priority), nil
	}

	transcript := formatTranscript(segments)
	entitiesInfo := formatEntities(entities)

	systemPrompt := `Ты — ассистент для генерации тикетов в системе поддержки клиентов.
На вход получаешь транскрипцию звонка, определённый intent и извлечённые сущности.
Сгенерируй краткий заголовок тикета и структурированное описание.

Ответ верни СТРОГО в JSON формате:
{
  "title": "Краткий заголовок тикета (до 100 символов)",
  "description": "Подробное описание проблемы клиента",
  "key_points": ["ключевой момент 1", "ключевой момент 2"],
  "suggested_solution": "Предложение по решению, если очевидно",
  "urgency_reason": "Причина срочности, если приоритет high/critical"
}

Правила:
- Заголовок должен быть кратким и информативным
- Описание должно содержать суть обращения
- Используй русский язык
- Не включай персональные данные клиента в заголовок
- Если приоритет high или critical, обязательно укажи urgency_reason`

	userMessage := fmt.Sprintf(`Транскрипция звонка:
%s

Intent: %s
Приоритет: %s

Извлечённые сущности:
%s

Сгенерируй тикет в JSON формате.`, transcript, intentID, priority, entitiesInfo)

	req := claudeRequest{
		Model:     claudeModel,
		MaxTokens: 1024,
		System:    systemPrompt,
		Messages: []claudeMessage{
			{Role: "user", Content: userMessage},
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", anthropicAPIURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", s.apiKey)
	httpReq.Header.Set("anthropic-version", anthropicAPIVersion)

	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp claudeErrorResponse
		if json.Unmarshal(respBody, &errResp) == nil {
			return nil, fmt.Errorf("claude API error (%d): %s - %s",
				resp.StatusCode, errResp.Error.Type, errResp.Error.Message)
		}
		return nil, fmt.Errorf("claude API error (%d): %s", resp.StatusCode, string(respBody))
	}

	var claudeResp claudeResponse
	if err := json.Unmarshal(respBody, &claudeResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	if len(claudeResp.Content) == 0 {
		return nil, fmt.Errorf("empty response from Claude API")
	}

	// Извлекаем текст ответа
	text := claudeResp.Content[0].Text

	// Парсим JSON из ответа
	summary, err := parseSummaryJSON(text)
	if err != nil {
		return nil, fmt.Errorf("parse summary: %w", err)
	}

	return summary, nil
}

// parseSummaryJSON извлекает и парсит JSON из ответа Claude
func parseSummaryJSON(text string) (*models.TicketSummary, error) {
	// Пытаемся найти JSON в ответе (может быть обёрнут в markdown code block)
	jsonStr := text
	if idx := strings.Index(text, "{"); idx >= 0 {
		if endIdx := strings.LastIndex(text, "}"); endIdx > idx {
			jsonStr = text[idx : endIdx+1]
		}
	}

	var summary models.TicketSummary
	if err := json.Unmarshal([]byte(jsonStr), &summary); err != nil {
		return nil, fmt.Errorf("unmarshal summary JSON: %w (raw: %s)", err, jsonStr)
	}

	if summary.Title == "" {
		return nil, fmt.Errorf("empty title in summary")
	}

	return &summary, nil
}

// fallbackSummary генерирует базовое описание без API (когда ключ не задан)
func (s *ClaudeSummarizer) fallbackSummary(
	segments []models.Segment,
	intentID string,
	priority string,
) *models.TicketSummary {
	var clientTexts []string
	for _, seg := range segments {
		if seg.Role == "client" {
			clientTexts = append(clientTexts, seg.Text)
		}
	}

	title := fmt.Sprintf("Обращение: %s", intentID)
	if len(title) > 100 {
		title = title[:97] + "..."
	}

	description := "Автоматически созданный тикет из транскрипции звонка.\n\n"
	if len(clientTexts) > 0 {
		firstMessage := clientTexts[0]
		if len(firstMessage) > 200 {
			firstMessage = firstMessage[:197] + "..."
		}
		description += fmt.Sprintf("Первое сообщение клиента: %s", firstMessage)
	}

	summary := &models.TicketSummary{
		Title:       title,
		Description: description,
	}

	if priority == "high" || priority == "critical" {
		summary.UrgencyReason = fmt.Sprintf("Приоритет: %s", priority)
	}

	return summary
}

// formatTranscript форматирует сегменты в читаемый текст
func formatTranscript(segments []models.Segment) string {
	var sb strings.Builder
	for _, seg := range segments {
		role := seg.Role
		if role == "" {
			role = seg.Speaker
		}
		sb.WriteString(fmt.Sprintf("[%s]: %s\n", role, seg.Text))
	}
	return sb.String()
}

// formatEntities форматирует сущности в читаемый текст
func formatEntities(entities *models.Entities) string {
	if entities == nil {
		return "Не извлечены"
	}

	var parts []string
	for _, p := range entities.Persons {
		parts = append(parts, fmt.Sprintf("Имя: %s", p.Value))
	}
	for _, p := range entities.Phones {
		parts = append(parts, fmt.Sprintf("Телефон: %s", p.Value))
	}
	for _, e := range entities.Emails {
		parts = append(parts, fmt.Sprintf("Email: %s", e.Value))
	}
	for _, o := range entities.OrderIDs {
		parts = append(parts, fmt.Sprintf("Заказ: %s", o.Value))
	}
	for _, a := range entities.AccountIDs {
		parts = append(parts, fmt.Sprintf("Аккаунт: %s", a.Value))
	}
	for _, m := range entities.MoneyAmounts {
		parts = append(parts, fmt.Sprintf("Сумма: %s", m.Value))
	}
	for _, d := range entities.Dates {
		parts = append(parts, fmt.Sprintf("Дата: %s", d.Value))
	}

	if len(parts) == 0 {
		return "Не извлечены"
	}

	return strings.Join(parts, "\n")
}
