package adapters

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type SlackAdapter struct {
	webhookURL string
	client     *http.Client
}

func NewSlackAdapter(webhookURL string) *SlackAdapter {
	return &SlackAdapter{
		webhookURL: webhookURL,
		client:     &http.Client{Timeout: 10 * time.Second},
	}
}

func (a *SlackAdapter) Name() string { return "slack" }

func (a *SlackAdapter) Send(payload *NotificationPayload) error {
	if a.webhookURL == "" {
		return fmt.Errorf("no slack webhook URL configured")
	}

	body := map[string]string{"text": payload.Markdown}
	jsonBody, _ := json.Marshal(body)

	resp, err := a.client.Post(a.webhookURL, "application/json", bytes.NewReader(jsonBody))
	if err != nil {
		return fmt.Errorf("slack webhook: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("slack webhook: status %d", resp.StatusCode)
	}
	return nil
}
