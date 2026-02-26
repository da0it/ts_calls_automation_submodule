package main

import (
	"context"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"notification_sender/internal/adapters"
	callprocessingv1 "notification_sender/internal/gen"
	"notification_sender/internal/handlers"
	"notification_sender/internal/services"
	"notification_sender/pkg/config"
)

func main() {
	cfg := config.Load()

	// Build enabled channel adapters
	channels := buildChannels(cfg)
	log.Printf("Enabled %d notification channel(s)", len(channels))

	// Initialize service
	notificationService := services.NewNotificationService(channels)

	// Handlers
	httpHandler := handlers.NewNotificationHandler(notificationService)
	grpcHandler := handlers.NewNotificationGRPCHandler(notificationService)

	// HTTP server (Gin)
	router := setupRouter(httpHandler)
	httpAddr := ":" + cfg.HTTPPort
	httpSrv := &http.Server{Addr: httpAddr, Handler: router}

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
			log.Fatalf("Failed to configure notification gRPC TLS: %v", tlsErr)
		}
		grpcOptions = append(grpcOptions, grpc.Creds(creds))
	}
	grpcSrv := grpc.NewServer(grpcOptions...)
	callprocessingv1.RegisterNotificationServiceServer(grpcSrv, grpcHandler)

	log.Printf("Starting notification HTTP service on %s", httpAddr)
	grpcMode := "insecure"
	if cfg.GRPCTLSEnabled {
		grpcMode = "tls"
	}
	log.Printf("Starting notification gRPC service on %s (%s)", grpcAddr, grpcMode)

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

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down notification service...")
	grpcSrv.GracefulStop()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(ctx); err != nil {
		log.Fatalf("HTTP server forced to shutdown: %v", err)
	}
	log.Println("Notification service exited")
}

func buildChannels(cfg *config.Config) []adapters.ChannelAdapter {
	var channels []adapters.ChannelAdapter
	for _, ch := range cfg.EnabledChannels {
		switch ch {
		case "email":
			channels = append(channels, adapters.NewEmailAdapter(
				cfg.SMTPHost, cfg.SMTPPort, cfg.SMTPUser,
				cfg.SMTPPassword, cfg.SMTPFrom, cfg.EmailTo,
			))
		case "telegram":
			channels = append(channels, adapters.NewTelegramAdapter(
				cfg.TelegramBotToken, cfg.TelegramChatID,
			))
		case "slack":
			channels = append(channels, adapters.NewSlackAdapter(cfg.SlackWebhookURL))
		case "log":
			channels = append(channels, adapters.NewLogAdapter())
		default:
			log.Printf("Unknown notification channel: %s, skipping", ch)
		}
	}
	if len(channels) == 0 {
		log.Println("No channels configured, falling back to log adapter")
		channels = append(channels, adapters.NewLogAdapter())
	}
	return channels
}

func setupRouter(h *handlers.NotificationHandler) *gin.Engine {
	if os.Getenv("GIN_MODE") == "" {
		gin.SetMode(gin.DebugMode)
	}
	router := gin.Default()
	router.Use(gin.Recovery())
	router.GET("/health", h.Health)
	return router
}
