package adapters

import "log"

// LogAdapter prints notifications to stdout — useful for local development
type LogAdapter struct{}

func NewLogAdapter() *LogAdapter { return &LogAdapter{} }

func (a *LogAdapter) Name() string { return "log" }

func (a *LogAdapter) Send(payload *NotificationPayload) error {
	log.Printf("[NOTIFICATION] Subject: %s", payload.Subject)
	log.Printf("[NOTIFICATION] Body:\n%s", payload.BodyText)
	return nil
}
