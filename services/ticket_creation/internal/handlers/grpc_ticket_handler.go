package handlers

import (
	"context"

	callprocessingv1 "ticket_module/internal/gen"
	"ticket_module/internal/models"
	"ticket_module/internal/services"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type TicketGRPCHandler struct {
	callprocessingv1.UnimplementedTicketServiceServer
	service *services.TicketCreatorService
}

func NewTicketGRPCHandler(service *services.TicketCreatorService) *TicketGRPCHandler {
	return &TicketGRPCHandler{service: service}
}

func (h *TicketGRPCHandler) CreateTicket(ctx context.Context, req *callprocessingv1.CreateTicketRequest) (*callprocessingv1.CreateTicketResponse, error) {
	if req.GetTranscript() == nil {
		return nil, status.Error(codes.InvalidArgument, "transcript is required")
	}
	if len(req.GetTranscript().GetSegments()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "segments are required")
	}

	segments := make([]models.Segment, 0, len(req.GetTranscript().GetSegments()))
	for _, seg := range req.GetTranscript().GetSegments() {
		segments = append(segments, models.Segment{
			Start:   seg.GetStart(),
			End:     seg.GetEnd(),
			Speaker: seg.GetSpeaker(),
			Role:    seg.GetRole(),
			Text:    seg.GetText(),
		})
	}

	createReq := &models.CreateTicketRequest{
		Transcript: models.TranscriptData{
			CallID:      req.GetTranscript().GetCallId(),
			Segments:    segments,
			RoleMapping: req.GetTranscript().GetRoleMapping(),
		},
		Routing: models.RoutingData{
			IntentID:         req.GetRouting().GetIntentId(),
			IntentConfidence: req.GetRouting().GetIntentConfidence(),
			Priority:         req.GetRouting().GetPriority(),
			SuggestedGroup:   req.GetRouting().GetSuggestedGroup(),
		},
		AudioURL: req.GetAudioUrl(),
	}

	created, err := h.service.CreateTicket(createReq)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "create ticket: %v", err)
	}

	return &callprocessingv1.CreateTicketResponse{
		Ticket: &callprocessingv1.TicketCreated{
			TicketId:   created.TicketID,
			ExternalId: created.ExternalID,
			Url:        created.URL,
			System:     created.System,
			CreatedAt:  timestamppb.New(created.CreatedAt),
		},
	}, nil
}
