// cmd/server/main.go
package main

import (
	"context"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"ticket_module/internal/adapters"
	"ticket_module/internal/clients"
	"ticket_module/internal/database"
	callprocessingv1 "ticket_module/internal/gen"
	"ticket_module/internal/handlers"
	"ticket_module/internal/services"
	"ticket_module/pkg/config"
)

func main() {
	// Загрузка конфигурации
	cfg := config.Load()

	// Подключение к БД
	db, err := database.NewDatabase(cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()
	log.Println("Connected to database")

	// Инициализация репозитория
	ticketRepo := database.NewTicketRepository(db)

	// Инициализация клиентов
	pythonClient := clients.NewPythonClient(cfg.PythonNERServiceURL)
	summarizer := services.NewClaudeSummarizer(cfg.AnthropicAPIKey)

	// Выбор адаптера тикет-системы
	var ticketAdapter adapters.TicketSystemAdapter
	switch cfg.TicketSystem {
	case "mock":
		ticketAdapter = adapters.NewMockAdapter()
		log.Println("Using Mock ticket adapter")
	default:
		ticketAdapter = adapters.NewMockAdapter()
		log.Printf("Unknown ticket system '%s', using Mock", cfg.TicketSystem)
	}

	// Инициализация сервиса
	ticketService := services.NewTicketCreatorService(
		pythonClient,
		summarizer,
		ticketAdapter,
		ticketRepo,
		cfg.TicketIncludePIIInDescription,
	)

	// Инициализация обработчиков
	ticketHandler := handlers.NewTicketHandler(ticketService)
	ticketGRPCHandler := handlers.NewTicketGRPCHandler(ticketService)

	// HTTP router
	router := setupRouter(ticketHandler, cfg)
	httpAddr := ":" + cfg.ServerPort
	httpSrv := &http.Server{
		Addr:    httpAddr,
		Handler: router,
	}

	// gRPC server
	grpcAddr := ":" + cfg.GRPCPort
	lis, err := net.Listen("tcp", grpcAddr)
	if err != nil {
		log.Fatalf("Failed to listen gRPC: %v", err)
	}

	grpcOptions := make([]grpc.ServerOption, 0, 1)
	if cfg.GRPCTLSEnabled {
		creds, tlsErr := credentials.NewServerTLSFromFile(cfg.GRPCTLSCertFile, cfg.GRPCTLSKeyFile)
		if tlsErr != nil {
			log.Fatalf("Failed to configure ticket gRPC TLS: %v", tlsErr)
		}
		grpcOptions = append(grpcOptions, grpc.Creds(creds))
	}
	grpcSrv := grpc.NewServer(grpcOptions...)
	callprocessingv1.RegisterTicketServiceServer(grpcSrv, ticketGRPCHandler)

	log.Printf("Starting ticket HTTP service on %s", httpAddr)
	grpcMode := "insecure"
	if cfg.GRPCTLSEnabled {
		grpcMode = "tls"
	}
	log.Printf("Starting ticket gRPC service on %s (%s)", grpcAddr, grpcMode)

	go func() {
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start HTTP server: %v", err)
		}
	}()

	go func() {
		if err := grpcSrv.Serve(lis); err != nil {
			log.Fatalf("Failed to start gRPC server: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down ticket service...")

	grpcSrv.GracefulStop()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(ctx); err != nil {
		log.Fatalf("HTTP server forced to shutdown: %v", err)
	}
}

func setupRouter(h *handlers.TicketHandler, cfg *config.Config) *gin.Engine {
	// В production используй gin.SetMode(gin.ReleaseMode)
	if os.Getenv("GIN_MODE") == "" {
		gin.SetMode(gin.DebugMode)
	}

	router := gin.Default()

	// Middleware
	router.Use(gin.Recovery())
	router.Use(corsMiddleware(cfg.CORSAllowedOrigins))

	// Health check
	router.GET("/health", h.Health)

	// API routes
	api := router.Group("/api")
	{
		// Создание тикета
		api.POST("/tickets", h.CreateTicket)

		// Получение тикета
		api.GET("/tickets/:id", h.GetTicket)

		// Список тикетов
		api.GET("/tickets", h.ListTickets)

		// Обновление статуса
		api.PATCH("/tickets/:id/status", h.UpdateTicketStatus)

		// Статистика
		api.GET("/tickets/stats", h.GetStats)
	}

	return router
}

func corsMiddleware(allowedOriginsRaw string) gin.HandlerFunc {
	allowedOrigins := parseAllowedOrigins(allowedOriginsRaw)
	allowAny := allowedOrigins["*"]

	return func(c *gin.Context) {
		setSecurityHeaders(c)
		origin := strings.TrimSpace(c.GetHeader("Origin"))
		if allowAny {
			c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		} else if origin != "" && allowedOrigins[origin] {
			c.Writer.Header().Set("Access-Control-Allow-Origin", origin)
			c.Writer.Header().Set("Vary", "Origin")
		}
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}

func parseAllowedOrigins(raw string) map[string]bool {
	out := make(map[string]bool)
	for _, item := range strings.Split(raw, ",") {
		origin := strings.TrimSpace(item)
		if origin == "" {
			continue
		}
		out[origin] = true
	}
	return out
}

func setSecurityHeaders(c *gin.Context) {
	c.Writer.Header().Set("X-Content-Type-Options", "nosniff")
	c.Writer.Header().Set("X-Frame-Options", "DENY")
	c.Writer.Header().Set("Referrer-Policy", "no-referrer")
}
