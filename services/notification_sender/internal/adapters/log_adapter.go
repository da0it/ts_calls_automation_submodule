package adapters

import (
	"log"
	"os"
	"strings"
)

// LogAdapter prints notifications to stdout — useful for local development
type LogAdapter struct{}

func NewLogAdapter() *LogAdapter { return &LogAdapter{} }

func (a *LogAdapter) Name() string { return "log" }

func (a *LogAdapter) Send(payload *NotificationPayload) error {
	log.Printf("[NOTIFICATION] Subject: %s", payload.Subject)
	if envBool("LOG_ADAPTER_INCLUDE_BODY", false) {
		log.Printf("[NOTIFICATION] Body:\n%s", payload.BodyText)
	} else {
		log.Printf("[NOTIFICATION] Body omitted (set LOG_ADAPTER_INCLUDE_BODY=1 to include)")
	}
	return nil
}

func envBool(name string, def bool) bool {
	raw := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
	if raw == "" {
		return def
	}
	switch raw {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return def
	}
}
