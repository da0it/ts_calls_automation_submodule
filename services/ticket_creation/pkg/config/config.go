// pkg/config/config.go
package config

import (
	"os"
	"strconv"
	"strings"
)

// Структура со всеми настройками сервиса создания тикетов
type Config struct {
	// Блок настроек сервер (настройки HTTP/gRPC-сервера)
	// Порт HTTP-сервера
	ServerPort         string

	// Порт gRPC-сервера
	GRPCPort           string

	// Включен ли TLS для gRPC
	GRPCTLSEnabled     bool

	// Путь к TLS-сертификату
	GRPCTLSCertFile    string

	// Путь к приватному ключу TLS
	GRPCTLSKeyFile     string

	// Список разрешенных оригинов для CORS, например: http://localhost:8000,http://localhost:3000
	CORSAllowedOrigins string

	// Блок настройки базы данных
	// Строка подключения к базе данных.
	DatabaseURL string

	// Питон-сервисы
	// URL Python-сервиса извлечения сущностей
	PythonNERServiceURL string

	// Блок настроек больших языковых моделей (LLM)
	// Таймаут запроса к LLM в секундах
	LLMRequestTimeoutSeconds int

	// Адрес Ollama API
	OllamaBaseURL            string

	// Название модели Ollama
	OllamaModel              string

	// Температура генерации. Чем выше температура, тем более вариативным может быть ответ
	OllamaTemperature        float64

	// Ограничение на количество генерируемых токенов.
	OllamaNumPredict         int

	// Блок тикет-систем
	// Выбирает бэкенд создания заявки
	TicketSystem         string

	// Адрес API SimpleOne
	SimpleOneEndpointURL string

	// Токен авторизации SimpleOne
	SimpleOneBearerToken string

	// Таймаут запроса к SimpleOne в секундах
	SimpleOneTimeoutSecs int

	// PII-настройка
	TicketIncludePIIInDescription bool
}

// Load создаёт и возвращает указатель на заполненную структуру Config.
func Load() *Config {
	return &Config{
		ServerPort:                    getEnv("SERVER_PORT", "8080"),
		GRPCPort:                      getEnv("GRPC_PORT", "50054"),
		GRPCTLSEnabled:                getEnvBool("TICKET_GRPC_TLS_ENABLED", false),
		GRPCTLSCertFile:               getEnv("TICKET_GRPC_TLS_CERT_FILE", ""),
		GRPCTLSKeyFile:                getEnv("TICKET_GRPC_TLS_KEY_FILE", ""),
		CORSAllowedOrigins:            getEnv("CORS_ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:3000"),
		DatabaseURL:                   getEnv("DATABASE_URL", "postgres://localhost/tickets?sslmode=disable"),
		PythonNERServiceURL:           getEnv("PYTHON_NER_SERVICE_URL", "http://localhost:5000"),
		LLMRequestTimeoutSeconds:      getEnvInt("LLM_REQUEST_TIMEOUT_SECONDS", 180),
		OllamaBaseURL:                 getEnv("OLLAMA_BASE_URL", "http://localhost:11434"),
		OllamaModel:                   getEnv("OLLAMA_MODEL", "gemma"),
		OllamaTemperature:             getEnvFloat("OLLAMA_TEMPERATURE", 0.0),
		OllamaNumPredict:              getEnvInt("OLLAMA_NUM_PREDICT", 48),
		TicketSystem:                  getEnv("TICKET_SYSTEM", "mock"),
		SimpleOneEndpointURL:          getEnv("SIMPLEONE_ENDPOINT_URL", ""),
		SimpleOneBearerToken:          getEnv("SIMPLEONE_BEARER_TOKEN", ""),
		SimpleOneTimeoutSecs:          getEnvInt("SIMPLEONE_TIMEOUT_SECONDS", 30),
		TicketIncludePIIInDescription: getEnvBool("TICKET_INCLUDE_PII_IN_DESCRIPTION", false),
	}
}

// Функция читает строковую переменную окружения. Если переменная окружения существует и не пустая, возвращается её значение.
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// Эта функция читает переменную окружения как int
func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intVal, err := strconv.Atoi(value); err == nil {
			return intVal
		}
	}
	return defaultValue
}

// Функция читает переменную окружения как float64
func getEnvFloat(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if floatVal, err := strconv.ParseFloat(value, 64); err == nil {
			return floatVal
		}
	}
	return defaultValue
}

// Функция читает boolean-переменную окружения
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
