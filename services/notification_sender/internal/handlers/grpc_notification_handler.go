package handlers

import (
	"context"

	callprocessingv1 "notification_sender/internal/gen"
	"notification_sender/internal/services"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type NotificationGRPCHandler struct {
	callprocessingv1.UnimplementedNotificationServiceServer
	service *services.NotificationService
}

func NewNotificationGRPCHandler(service *services.NotificationService) *NotificationGRPCHandler {
	return &NotificationGRPCHandler{service: service}
}

func (h *NotificationGRPCHandler) SendNotification(
	ctx context.Context,
	req *callprocessingv1.SendNotificationRequest,
) (*callprocessingv1.SendNotificationResponse, error) {
	if req.GetTicket() == nil {
		return nil, status.Error(codes.InvalidArgument, "ticket is required")
	}

	success, results := h.service.SendNotification(req)

	protoResults := make([]*callprocessingv1.NotificationChannel, 0, len(results))
	for _, r := range results {
		protoResults = append(protoResults, &callprocessingv1.NotificationChannel{
			Type:    r.ChannelName,
			Success: r.Success,
			Error:   r.Error,
		})
	}

	return &callprocessingv1.SendNotificationResponse{
		Success: success,
		Results: protoResults,
	}, nil
}
