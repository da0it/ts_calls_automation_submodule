// pkg/config/config.go
package config

import (
	"log"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	HTTPPort string
	GRPCPort string

	// gRPC адреса сервисов
	TranscriptionGRPCAddr string
	RoutingGRPCAddr       string
	TicketGRPCAddr        string
	NotificationGRPCAddr  string
	EntityServiceURL      string
	RoutingIntentsPath    string
	RoutingGroupsPath     string
}

func Load() *Config {
	_ = godotenv.Load()

	cfg := &Config{
		HTTPPort:              getEnv("HTTP_PORT", getEnv("SERVER_PORT", "8000")),
		GRPCPort:              getEnv("GRPC_PORT", "9000"),
		TranscriptionGRPCAddr: getEnv("TRANSCRIPTION_GRPC_ADDR", "localhost:50051"),
		RoutingGRPCAddr:       getEnv("ROUTING_GRPC_ADDR", "localhost:50052"),
		TicketGRPCAddr:        getEnv("TICKET_GRPC_ADDR", "localhost:50054"),
		NotificationGRPCAddr:  getEnv("NOTIFICATION_GRPC_ADDR", "localhost:50055"),
		EntityServiceURL:      getEnv("ENTITY_SERVICE_URL", "http://localhost:5001"),
		RoutingIntentsPath:    getEnv("ROUTING_INTENTS_PATH", "../router/configs/intents.json"),
		RoutingGroupsPath:     getEnv("ROUTING_GROUPS_PATH", "../router/configs/groups.json"),
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

	return cfg
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
