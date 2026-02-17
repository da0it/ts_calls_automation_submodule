// pkg/config/config.go
package config

import (
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	HTTPPort string
	GRPCPort string

	// gRPC адреса сервисов
	TranscriptionGRPCAddr     string
	RoutingGRPCAddr           string
	TicketGRPCAddr            string
	NotificationGRPCAddr      string
	EntityServiceURL          string
	RoutingIntentsPath        string
	RoutingGroupsPath         string
	RoutingFeedbackPath       string
	RoutingAutoLearn          bool
	RoutingAutoLearnLimit     int
	RouterAdminURL            string
	RouterAdminToken          string
	RouterAdminTimeoutSeconds int
}

func Load() *Config {
	_ = godotenv.Load()

	cfg := &Config{
		HTTPPort:                  getEnv("HTTP_PORT", getEnv("SERVER_PORT", "8000")),
		GRPCPort:                  getEnv("GRPC_PORT", "9000"),
		TranscriptionGRPCAddr:     getEnv("TRANSCRIPTION_GRPC_ADDR", "localhost:50051"),
		RoutingGRPCAddr:           getEnv("ROUTING_GRPC_ADDR", "localhost:50052"),
		TicketGRPCAddr:            getEnv("TICKET_GRPC_ADDR", "localhost:50054"),
		NotificationGRPCAddr:      getEnv("NOTIFICATION_GRPC_ADDR", "localhost:50055"),
		EntityServiceURL:          getEnv("ENTITY_SERVICE_URL", "http://localhost:5001"),
		RoutingIntentsPath:        getEnv("ROUTING_INTENTS_PATH", "../router/configs/intents.json"),
		RoutingGroupsPath:         getEnv("ROUTING_GROUPS_PATH", "../router/configs/groups.json"),
		RoutingFeedbackPath:       getEnv("ROUTING_FEEDBACK_PATH", "./data/routing_feedback.jsonl"),
		RoutingAutoLearn:          getEnv("ROUTING_AUTO_LEARN", "1") == "1",
		RoutingAutoLearnLimit:     getEnvInt("ROUTING_AUTO_LEARN_LIMIT", 50),
		RouterAdminURL:            getEnv("ROUTER_ADMIN_URL", "http://localhost:8082"),
		RouterAdminToken:          getEnv("ROUTER_ADMIN_TOKEN", ""),
		RouterAdminTimeoutSeconds: getEnvInt("ROUTER_ADMIN_TIMEOUT_SECONDS", 600),
	}

	log.Printf("Orchestrator config loaded:")
	log.Printf("  - HTTP port: %s", cfg.HTTPPort)
	log.Printf("  - gRPC port: %s", cfg.GRPCPort)
	log.Printf("  - Transcription gRPC: %s", cfg.TranscriptionGRPCAddr)
	log.Printf("  - Routing gRPC: %s", cfg.RoutingGRPCAddr)
	log.Printf("  - Ticket gRPC: %s", cfg.TicketGRPCAddr)
	log.Printf("  - Notification gRPC: %s", cfg.NotificationGRPCAddr)
	log.Printf("  - Entity service URL: %s", cfg.EntityServiceURL)
	log.Printf("  - Routing intents path: %s", cfg.RoutingIntentsPath)
	log.Printf("  - Routing groups path: %s", cfg.RoutingGroupsPath)
	log.Printf("  - Routing feedback path: %s", cfg.RoutingFeedbackPath)
	log.Printf("  - Routing auto learn: %v", cfg.RoutingAutoLearn)
	log.Printf("  - Routing auto learn limit: %d", cfg.RoutingAutoLearnLimit)
	log.Printf("  - Router admin URL: %s", cfg.RouterAdminURL)
	log.Printf("  - Router admin timeout (sec): %d", cfg.RouterAdminTimeoutSeconds)

	return cfg
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		var parsed int
		_, err := fmt.Sscanf(value, "%d", &parsed)
		if err == nil && parsed > 0 {
			return parsed
		}
	}
	return defaultValue
}
