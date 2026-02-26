// cmd/server/main.go
package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	_ "github.com/lib/pq"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"orchestrator/internal/clients"
	callprocessingv1 "orchestrator/internal/gen"
	"orchestrator/internal/handlers"
	"orchestrator/internal/middleware"
	"orchestrator/internal/models"
	"orchestrator/internal/services"
	"orchestrator/pkg/config"
)

func main() {
	// Загрузка конфигурации
	cfg := config.Load()

	// Подключение к БД
	db, err := sql.Open("postgres", cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("Failed to open database: %v", err)
	}
	defer db.Close()

	if err := db.Ping(); err != nil {
		log.Fatalf("Failed to ping database: %v", err)
	}
	log.Println("✓ Database connected")

	// User service + миграция + seed
	userService := services.NewUserService(db)
	if err := userService.Migrate(); err != nil {
		log.Fatalf("Failed to run users migration: %v", err)
	}
	auditService := services.NewAuditService(db)
	if err := auditService.Migrate(); err != nil {
		log.Fatalf("Failed to run audit migration: %v", err)
	}
	if err := userService.SeedAdmin(cfg.AdminUsername, cfg.AdminPassword); err != nil {
		log.Fatalf("Failed to seed admin: %v", err)
	}

	// Инициализация клиентов
	transcriptionClient, err := clients.NewTranscriptionClient(cfg.TranscriptionGRPCAddr)
	if err != nil {
		log.Fatalf("Failed to initialize transcription client: %v", err)
	}
	defer transcriptionClient.Close()

	routingClient, err := clients.NewRoutingClient(cfg.RoutingGRPCAddr)
	if err != nil {
		log.Fatalf("Failed to initialize routing client: %v", err)
	}
	defer routingClient.Close()

	ticketClient, err := clients.NewTicketClient(cfg.TicketGRPCAddr)
	if err != nil {
		log.Fatalf("Failed to initialize ticket client: %v", err)
	}
	defer ticketClient.Close()

	notificationClient, err := clients.NewNotificationClient(cfg.NotificationGRPCAddr)
	if err != nil {
		log.Fatalf("Failed to initialize notification client: %v", err)
	}
	defer notificationClient.Close()

	entityClient := clients.NewEntityClient(cfg.EntityServiceURL)

	log.Println("✓ All clients initialized")

	// Инициализация оркестратора
	orchestrator := services.NewOrchestratorService(
		transcriptionClient,
		routingClient,
		ticketClient,
		notificationClient,
		entityClient,
	)

	log.Println("✓ Orchestrator service initialized")

	routingConfigService := services.NewRoutingConfigService(
		cfg.RoutingIntentsPath,
		cfg.RoutingGroupsPath,
	)
	routingFeedbackService := services.NewRoutingFeedbackService(
		cfg.RoutingFeedbackPath,
		cfg.RoutingAutoLearn,
		cfg.RoutingAutoLearnLimit,
		routingConfigService,
	)
	routingModelService := services.NewRoutingModelService(
		cfg.RouterAdminURL,
		cfg.RouterAdminToken,
		time.Duration(cfg.RouterAdminTimeoutSeconds)*time.Second,
		filepath.Dir(cfg.RoutingFeedbackPath),
	)

	// Инициализация handlers
	processHandler := handlers.NewProcessHandler(orchestrator, routingConfigService, routingFeedbackService, routingModelService, auditService)
	authHandler := handlers.NewAuthHandler(userService, cfg.JWTSecret, cfg.JWTExpiryHours, auditService)
	grpcHandler := handlers.NewProcessGRPCHandler(orchestrator)

	// Auth middleware
	authMw := middleware.AuthRequired(cfg.JWTSecret, userService)
	adminMw := middleware.RequireRole(models.RoleAdmin)

	// HTTP router
	router := setupRouter(processHandler, authHandler, authMw, adminMw)

	httpAddr := ":" + cfg.HTTPPort
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
			log.Fatalf("Failed to configure gRPC TLS: %v", tlsErr)
		}
		grpcOptions = append(grpcOptions, grpc.Creds(creds))
	}
	grpcSrv := grpc.NewServer(grpcOptions...)
	callprocessingv1.RegisterOrchestratorServiceServer(grpcSrv, grpcHandler)

	httpScheme := "http"
	if cfg.HTTPTLSEnabled {
		httpScheme = "https"
	}
	grpcMode := "insecure"
	if cfg.GRPCTLSEnabled {
		grpcMode = "tls"
	}
	log.Printf("Starting Orchestrator HTTP on %s://0.0.0.0%s", httpScheme, httpAddr)
	log.Printf("Starting Orchestrator gRPC on 0.0.0.0%s (%s)", grpcAddr, grpcMode)
	log.Printf("Ready to process calls!")

	go func() {
		var serveErr error
		if cfg.HTTPTLSEnabled {
			serveErr = httpSrv.ListenAndServeTLS(cfg.HTTPTLSCertFile, cfg.HTTPTLSKeyFile)
		} else {
			serveErr = httpSrv.ListenAndServe()
		}
		if serveErr != nil && serveErr != http.ErrServerClosed {
			log.Fatalf("Failed to start HTTP server: %v", serveErr)
		}
	}()

	go func() {
		if err := grpcSrv.Serve(lis); err != nil {
			log.Fatalf("Failed to start gRPC server: %v", err)
		}
	}()

	// Ожидание сигнала завершения
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	grpcSrv.GracefulStop()

	if err := httpSrv.Shutdown(ctx); err != nil {
		log.Fatal("HTTP server forced to shutdown:", err)
	}

	log.Println("Orchestrator exited")
}

func setupRouter(
	h *handlers.ProcessHandler,
	auth *handlers.AuthHandler,
	authMw gin.HandlerFunc,
	adminMw gin.HandlerFunc,
) *gin.Engine {
	// Production mode
	if os.Getenv("GIN_MODE") == "" {
		gin.SetMode(gin.DebugMode)
	}

	router := gin.Default()

	// Middleware
	router.Use(gin.Recovery())
	router.Use(corsMiddleware())
	router.Use(requestIDMiddleware())

	// Limit upload size (500 MB)
	router.MaxMultipartMemory = 500 << 20

	// Public routes
	router.GET("/", func(c *gin.Context) {
		c.File("./web/index.html")
	})
	router.GET("/api/info", h.Root)
	router.GET("/health", h.Health)

	// Auth (public)
	router.POST("/api/v1/auth/login", auth.Login)
	router.POST("/api/v1/auth/register", auth.Register)

	// Authenticated routes (operator + admin)
	api := router.Group("/api/v1")
	api.Use(authMw)
	{
		api.GET("/auth/me", auth.Me)
		api.POST("/process-call", h.ProcessCall)
		api.GET("/routing-config", h.GetRoutingConfig)
		api.POST("/routing-feedback", h.SaveRoutingFeedback)
		api.GET("/routing-model/status", h.GetRoutingModelStatus)

		// Admin-only routes
		admin := api.Group("")
		admin.Use(adminMw)
		{
			admin.PUT("/routing-config", h.UpdateRoutingConfig)
			admin.POST("/routing-config/groups", h.CreateRoutingGroup)
			admin.DELETE("/routing-config/groups/:id", h.DeleteRoutingGroup)
			admin.POST("/routing-config/intents", h.CreateRoutingIntent)
			admin.DELETE("/routing-config/intents/:id", h.DeleteRoutingIntent)
			admin.POST("/routing-model/reload", h.ReloadRoutingModel)
			admin.POST("/routing-model/train", h.TrainRoutingModel)
			admin.POST("/routing-model/train-csv", h.TrainRoutingModelCSV)

			// User management
			admin.GET("/users", auth.ListUsers)
			admin.POST("/users", auth.CreateUser)
			admin.POST("/users/:id/approve", auth.ApproveUser)
			admin.POST("/users/:id/deactivate", auth.DeactivateUser)
			admin.DELETE("/users/:id", auth.DeleteUser)
		}
	}

	return router
}

func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}

func requestIDMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		requestID := c.GetHeader("X-Request-ID")
		if requestID == "" {
			requestID = generateRequestID()
		}

		c.Set("request_id", requestID)
		c.Writer.Header().Set("X-Request-ID", requestID)
		c.Next()
	}
}

func generateRequestID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}
