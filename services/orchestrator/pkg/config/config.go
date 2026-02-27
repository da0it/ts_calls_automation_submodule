// pkg/config/config.go
package config

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/joho/godotenv"
)

type Config struct {
	HTTPPort           string
	GRPCPort           string
	HTTPTLSEnabled     bool
	HTTPTLSCertFile    string
	HTTPTLSKeyFile     string
	GRPCTLSEnabled     bool
	GRPCTLSCertFile    string
	GRPCTLSKeyFile     string
	CORSAllowedOrigins string

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

	// Auth / DB
	DatabaseURL    string
	JWTSecret      string
	JWTExpiryHours int
	AdminUsername  string
	AdminPassword  string
}

func Load() *Config {
	_ = godotenv.Load()

	cfg := &Config{
		HTTPPort:                  getEnv("HTTP_PORT", getEnv("SERVER_PORT", "8000")),
		GRPCPort:                  getEnv("GRPC_PORT", "9000"),
		HTTPTLSEnabled:            getEnvBool("HTTP_TLS_ENABLED", false),
		HTTPTLSCertFile:           getEnv("HTTP_TLS_CERT_FILE", ""),
		HTTPTLSKeyFile:            getEnv("HTTP_TLS_KEY_FILE", ""),
		GRPCTLSEnabled:            getEnvBool("ORCH_GRPC_TLS_ENABLED", false),
		GRPCTLSCertFile:           getEnv("ORCH_GRPC_TLS_CERT_FILE", ""),
		GRPCTLSKeyFile:            getEnv("ORCH_GRPC_TLS_KEY_FILE", ""),
		CORSAllowedOrigins:        getEnv("CORS_ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:3000"),
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

		DatabaseURL:    getEnv("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/tickets?sslmode=disable"),
		JWTSecret:      getEnv("JWT_SECRET", ""),
		JWTExpiryHours: getEnvInt("JWT_EXPIRY_HOURS", 24),
		AdminUsername:  getEnv("ADMIN_USERNAME", "admin"),
		AdminPassword:  getEnv("ADMIN_PASSWORD", ""),
	}

	if cfg.JWTSecret == "" {
		log.Fatal("JWT_SECRET is required")
	}

	log.Printf("Orchestrator config loaded:")
	log.Printf("  - HTTP port: %s", cfg.HTTPPort)
	log.Printf("  - gRPC port: %s", cfg.GRPCPort)
	log.Printf("  - HTTP TLS enabled: %v", cfg.HTTPTLSEnabled)
	log.Printf("  - gRPC TLS enabled: %v", cfg.GRPCTLSEnabled)
	log.Printf("  - CORS allowed origins: %s", cfg.CORSAllowedOrigins)
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
	log.Printf("  - Database URL: %s", cfg.DatabaseURL)
	log.Printf("  - JWT expiry hours: %d", cfg.JWTExpiryHours)
	log.Printf("  - Admin username: %s", cfg.AdminUsername)

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

func getEnvBool(key string, defaultValue bool) bool {
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
