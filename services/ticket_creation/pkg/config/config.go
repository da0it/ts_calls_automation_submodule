// pkg/config/config.go
package config

import (
	"os"
	"strconv"
	"strings"
)

type Config struct {
	// Server
	ServerPort      string
	GRPCPort        string
	GRPCTLSEnabled  bool
	GRPCTLSCertFile string
	GRPCTLSKeyFile  string

	// Database
	DatabaseURL string

	// Python services
	PythonNERServiceURL string

	// Claude API
	AnthropicAPIKey string

	// Ticket systems
	TicketSystem  string //so, mock
	JiraURL       string
	JiraUser      string
	JiraAPIToken  string
	RedmineURL    string
	RedmineAPIKey string
}

func Load() *Config {
	return &Config{
		ServerPort:          getEnv("SERVER_PORT", "8080"),
		GRPCPort:            getEnv("GRPC_PORT", "50054"),
		GRPCTLSEnabled:      getEnvBool("TICKET_GRPC_TLS_ENABLED", false),
		GRPCTLSCertFile:     getEnv("TICKET_GRPC_TLS_CERT_FILE", ""),
		GRPCTLSKeyFile:      getEnv("TICKET_GRPC_TLS_KEY_FILE", ""),
		DatabaseURL:         getEnv("DATABASE_URL", "postgres://localhost/tickets?sslmode=disable"),
		PythonNERServiceURL: getEnv("PYTHON_NER_SERVICE_URL", "http://localhost:5000"),
		AnthropicAPIKey:     getEnv("ANTHROPIC_API_KEY", ""),
		TicketSystem:        getEnv("TICKET_SYSTEM", "mock"),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intVal, err := strconv.Atoi(value); err == nil {
			return intVal
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
