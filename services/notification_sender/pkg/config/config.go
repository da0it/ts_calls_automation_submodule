package config

import (
	"log"
	"os"
	"strings"

	"github.com/joho/godotenv"
)

type Config struct {
	HTTPPort string
	GRPCPort string

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

	log.Printf("Notification config loaded: HTTP=%s gRPC=%s channels=%v",
		cfg.HTTPPort, cfg.GRPCPort, cfg.EnabledChannels)

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
