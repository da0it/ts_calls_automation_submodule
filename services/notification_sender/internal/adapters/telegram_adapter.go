package adapters

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type TelegramAdapter struct {
	botToken string
	chatIDs  []string
	client   *http.Client
}

func NewTelegramAdapter(botToken, chatIDs string) *TelegramAdapter {
	var ids []string
	for _, id := range strings.Split(chatIDs, ",") {
		id = strings.TrimSpace(id)
		if id != "" {
			ids = append(ids, id)
		}
	}
	return &TelegramAdapter{
		botToken: botToken,
		chatIDs:  ids,
		client:   &http.Client{Timeout: 10 * time.Second},
	}
}

func (a *TelegramAdapter) Name() string { return "telegram" }

func (a *TelegramAdapter) Send(payload *NotificationPayload) error {
	if len(a.chatIDs) == 0 {
		return fmt.Errorf("no telegram chat IDs configured")
	}

	url := fmt.Sprintf("https://api.telegram.org/bot%s/sendMessage", a.botToken)

	var lastErr error
	for _, chatID := range a.chatIDs {
		body := map[string]interface{}{
			"chat_id":    chatID,
			"text":       payload.Markdown,
			"parse_mode": "Markdown",
		}
		jsonBody, _ := json.Marshal(body)

		resp, err := a.client.Post(url, "application/json", bytes.NewReader(jsonBody))
		if err != nil {
			lastErr = fmt.Errorf("telegram send to %s: %w", chatID, err)
			continue
		}
		resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("telegram send to %s: status %d", chatID, resp.StatusCode)
		}
	}
	return lastErr
}
