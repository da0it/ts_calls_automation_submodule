package config

import (
	"log"
	"os"
	"strings"

	"github.com/joho/godotenv"
)

type Config struct {
	HTTPPort        string
	GRPCPort        string
	GRPCTLSEnabled  bool
	GRPCTLSCertFile string
	GRPCTLSKeyFile  string

	// Enabled channels (comma-separated: "email,telegram,slack,log")
	EnabledChannels []string

	// Email (SMTP)
	SMTPHost     string
	SMTPPort     string
	SMTPUser     string
	SMTPPassword string
	SMTPFrom     string
	EmailTo      string

	// Telegram
	TelegramBotToken string
	TelegramChatID   string

	// Slack / Mattermost
	SlackWebhookURL string
}

func Load() *Config {
	_ = godotenv.Load()

	channels := getEnv("NOTIFICATION_CHANNELS", "log")

	cfg := &Config{
		HTTPPort:         getEnv("HTTP_PORT", "8085"),
		GRPCPort:         getEnv("GRPC_PORT", "50055"),
		GRPCTLSEnabled:   envBool("NOTIFICATION_GRPC_TLS_ENABLED", false),
		GRPCTLSCertFile:  getEnv("NOTIFICATION_GRPC_TLS_CERT_FILE", ""),
		GRPCTLSKeyFile:   getEnv("NOTIFICATION_GRPC_TLS_KEY_FILE", ""),
		EnabledChannels:  parseList(channels),
		SMTPHost:         getEnv("SMTP_HOST", "localhost"),
		SMTPPort:         getEnv("SMTP_PORT", "587"),
		SMTPUser:         getEnv("SMTP_USER", ""),
		SMTPPassword:     getEnv("SMTP_PASSWORD", ""),
		SMTPFrom:         getEnv("SMTP_FROM", "noreply@example.com"),
		EmailTo:          getEnv("EMAIL_TO", ""),
		TelegramBotToken: getEnv("TELEGRAM_BOT_TOKEN", ""),
		TelegramChatID:   getEnv("TELEGRAM_CHAT_ID", ""),
		SlackWebhookURL:  getEnv("SLACK_WEBHOOK_URL", ""),
	}

	log.Printf("Notification config loaded: HTTP=%s gRPC=%s (tls=%v) channels=%v",
		cfg.HTTPPort, cfg.GRPCPort, cfg.GRPCTLSEnabled, cfg.EnabledChannels)

	return cfg
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func parseList(s string) []string {
	var result []string
	for _, item := range strings.Split(s, ",") {
		item = strings.TrimSpace(item)
		if item != "" {
			result = append(result, item)
		}
	}
	return result
}

func envBool(key string, defaultValue bool) bool {
	value := strings.TrimSpace(strings.ToLower(getEnv(key, "")))
	if value == "" {
		return defaultValue
	}
	switch value {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return defaultValue
	}
}
