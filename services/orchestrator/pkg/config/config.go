// pkg/config/config.go
package config

import (
	"log"
	"os"
	"strconv"
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
	TranscriptionGRPCAddr            string
	RoutingGRPCAddr                  string
	TicketGRPCAddr                   string
	TicketRPCTimeoutSeconds          int
	EntityServiceURL                 string
	RoutingReviewConfidenceThreshold float64
	RoutingIntentsPath               string
	RoutingGroupsPath                string
	RoutingFeedbackPath              string
	RoutingAutoLearn                 bool
	RoutingAutoLearnLimit            int
	RouterAdminURL                   string
	RouterAdminToken                 string
	RouterAdminTimeoutSeconds        int

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
		HTTPPort:                         getEnv("HTTP_PORT", getEnv("SERVER_PORT", "8000")),
		GRPCPort:                         getEnv("GRPC_PORT", "9000"),
		HTTPTLSEnabled:                   getEnvBool("HTTP_TLS_ENABLED", false),
		HTTPTLSCertFile:                  getEnv("HTTP_TLS_CERT_FILE", ""),
		HTTPTLSKeyFile:                   getEnv("HTTP_TLS_KEY_FILE", ""),
		GRPCTLSEnabled:                   getEnvBool("ORCH_GRPC_TLS_ENABLED", false),
		GRPCTLSCertFile:                  getEnv("ORCH_GRPC_TLS_CERT_FILE", ""),
		GRPCTLSKeyFile:                   getEnv("ORCH_GRPC_TLS_KEY_FILE", ""),
		CORSAllowedOrigins:               getEnv("CORS_ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:3000"),
		TranscriptionGRPCAddr:            getEnv("TRANSCRIPTION_GRPC_ADDR", "localhost:50051"),
		RoutingGRPCAddr:                  getEnv("ROUTING_GRPC_ADDR", "localhost:50052"),
		TicketGRPCAddr:                   getEnv("TICKET_GRPC_ADDR", "localhost:50054"),
		TicketRPCTimeoutSeconds:          getEnvInt("TICKET_RPC_TIMEOUT_SECONDS", 300),
		EntityServiceURL:                 getEnv("ENTITY_SERVICE_URL", "http://localhost:5001"),
		RoutingReviewConfidenceThreshold: getEnvFloat("ROUTING_REVIEW_CONFIDENCE_THRESHOLD", getEnvFloat("ROUTER_MIN_CONFIDENCE", 0.8)),
		RoutingIntentsPath:               getEnv("ROUTING_INTENTS_PATH", "../router/configs/intents.json"),
		RoutingGroupsPath:                getEnv("ROUTING_GROUPS_PATH", "../router/configs/groups.json"),
		RoutingFeedbackPath:              getEnv("ROUTING_FEEDBACK_PATH", "./data/routing_feedback.jsonl"),
		RoutingAutoLearn:                 getEnv("ROUTING_AUTO_LEARN", "1") == "1",
		RoutingAutoLearnLimit:            getEnvInt("ROUTING_AUTO_LEARN_LIMIT", 50),
		RouterAdminURL:                   getEnv("ROUTER_ADMIN_URL", "http://localhost:8082"),
		RouterAdminToken:                 getEnv("ROUTER_ADMIN_TOKEN", ""),
		RouterAdminTimeoutSeconds:        getEnvInt("ROUTER_ADMIN_TIMEOUT_SECONDS", 600),

		DatabaseURL:    getEnv("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/tickets?sslmode=disable"),
		JWTSecret:      getEnv("JWT_SECRET", ""),
		JWTExpiryHours: getEnvInt("JWT_EXPIRY_HOURS", 24),
		AdminUsername:  getEnv("ADMIN_USERNAME", "admin"),
		AdminPassword:  getEnv("ADMIN_PASSWORD", ""),
	}

	if cfg.JWTSecret == "" {
		log.Fatal("JWT_SECRET is required")
	}

	logConfig(cfg)
	return cfg
}

func logConfig(cfg *Config) {
	log.Println("Orchestrator config loaded:")
	items := []struct {
		name  string
		value interface{}
	}{
		{"HTTP port", cfg.HTTPPort},
		{"gRPC port", cfg.GRPCPort},
		{"HTTP TLS enabled", cfg.HTTPTLSEnabled},
		{"gRPC TLS enabled", cfg.GRPCTLSEnabled},
		{"CORS allowed origins", cfg.CORSAllowedOrigins},
		{"Transcription gRPC", cfg.TranscriptionGRPCAddr},
		{"Routing gRPC", cfg.RoutingGRPCAddr},
		{"Ticket gRPC", cfg.TicketGRPCAddr},
		{"Ticket RPC timeout (sec)", cfg.TicketRPCTimeoutSeconds},
		{"Entity service URL", cfg.EntityServiceURL},
		{"Routing review confidence threshold", cfg.RoutingReviewConfidenceThreshold},
		{"Routing intents path", cfg.RoutingIntentsPath},
		{"Routing groups path", cfg.RoutingGroupsPath},
		{"Routing feedback path", cfg.RoutingFeedbackPath},
		{"Routing auto learn", cfg.RoutingAutoLearn},
		{"Routing auto learn limit", cfg.RoutingAutoLearnLimit},
		{"Router admin URL", cfg.RouterAdminURL},
		{"Router admin timeout (sec)", cfg.RouterAdminTimeoutSeconds},
		{"Database URL", cfg.DatabaseURL},
		{"JWT expiry hours", cfg.JWTExpiryHours},
		{"Admin username", cfg.AdminUsername},
	}
	for _, item := range items {
		log.Printf("  - %s: %v", item.name, item.value)
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return defaultValue
	}
	parsed, err := strconv.Atoi(value)
	if err == nil && parsed > 0 {
		return parsed
	}
	return defaultValue
}

func getEnvFloat(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		parsed, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
		if err == nil {
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
