package handlers

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"orchestrator/internal/clients"
	callprocessingv1 "orchestrator/internal/gen"
	"orchestrator/internal/services"

	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type ProcessGRPCHandler struct {
	callprocessingv1.UnimplementedOrchestratorServiceServer
	orchestrator *services.OrchestratorService
}

func NewProcessGRPCHandler(orchestrator *services.OrchestratorService) *ProcessGRPCHandler {
	return &ProcessGRPCHandler{
		orchestrator: orchestrator,
	}
}

func (h *ProcessGRPCHandler) ProcessCall(ctx context.Context, req *callprocessingv1.ProcessCallRequest) (*callprocessingv1.ProcessCallResponse, error) {
	if len(req.GetAudio()) == 0 {
		return nil, fmt.Errorf("audio is required")
	}

	suffix := filepath.Ext(req.GetFilename())
	if suffix == "" {
		suffix = ".wav"
	}

	tmpFile, err := os.CreateTemp("", "orchestrator-audio-*"+suffix)
	if err != nil {
		return nil, fmt.Errorf("create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()

	if _, err := tmpFile.Write(req.GetAudio()); err != nil {
		tmpFile.Close()
		os.Remove(tmpPath)
		return nil, fmt.Errorf("write temp audio: %w", err)
	}
	tmpFile.Close()
	defer os.Remove(tmpPath)

	result, err := h.orchestrator.ProcessCall(tmpPath)
	if err != nil {
		return nil, err
	}

	if req.GetCallId() != "" {
		result.CallID = req.GetCallId()
		if result.Transcript != nil {
			result.Transcript.CallID = req.GetCallId()
		}
	}

	return &callprocessingv1.ProcessCallResponse{
		CallId:         result.CallID,
		Transcript:     transcriptToProto(result.Transcript),
		Routing:        routingToProto(result.Routing),
		Entities:       entitiesToProto(result.Entities),
		Ticket:         ticketToProto(result.Ticket),
		ProcessingTime: result.ProcessingTime,
		TotalTime:      result.TotalTime,
	}, nil
}

func transcriptToProto(transcript *clients.TranscriptionResponse) *callprocessingv1.Transcript {
	if transcript == nil {
		return nil
	}

	segments := make([]*callprocessingv1.Segment, 0, len(transcript.Segments))
	for _, seg := range transcript.Segments {
		segments = append(segments, &callprocessingv1.Segment{
			Start:   seg.Start,
			End:     seg.End,
			Speaker: seg.Speaker,
			Role:    seg.Role,
			Text:    seg.Text,
		})
	}

	metadata, err := structpb.NewStruct(transcript.Metadata)
	if err != nil {
		metadata = &structpb.Struct{}
	}

	return &callprocessingv1.Transcript{
		CallId:      transcript.CallID,
		Segments:    segments,
		RoleMapping: transcript.RoleMapping,
		Metadata:    metadata,
	}
}

func routingToProto(routing *clients.RoutingResponse) *callprocessingv1.Routing {
	if routing == nil {
		return nil
	}

	return &callprocessingv1.Routing{
		IntentId:         routing.IntentID,
		IntentConfidence: routing.IntentConfidence,
		Priority:         routing.Priority,
		SuggestedGroup:   routing.SuggestedGroup,
	}
}

func entitiesToProto(entities *clients.Entities) *callprocessingv1.Entities {
	if entities == nil {
		return &callprocessingv1.Entities{}
	}

	return &callprocessingv1.Entities{
		Persons:      entityListToProto(entities.Persons),
		Phones:       entityListToProto(entities.Phones),
		Emails:       entityListToProto(entities.Emails),
		OrderIds:     entityListToProto(entities.OrderIDs),
		AccountIds:   entityListToProto(entities.AccountIDs),
		MoneyAmounts: entityListToProto(entities.MoneyAmounts),
		Dates:        entityListToProto(entities.Dates),
	}
}

func entityListToProto(items []clients.ExtractedEntity) []*callprocessingv1.ExtractedEntity {
	out := make([]*callprocessingv1.ExtractedEntity, 0, len(items))
	for _, item := range items {
		out = append(out, &callprocessingv1.ExtractedEntity{
			Type:       item.Type,
			Value:      item.Value,
			Confidence: item.Confidence,
			Context:    item.Context,
		})
	}
	return out
}

func ticketToProto(ticket *clients.TicketCreated) *callprocessingv1.TicketCreated {
	if ticket == nil {
		return nil
	}

	createdAt := ticket.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now()
	}

	return &callprocessingv1.TicketCreated{
		TicketId:   ticket.TicketID,
		ExternalId: ticket.ExternalID,
		Url:        ticket.URL,
		System:     ticket.System,
		CreatedAt:  timestamppb.New(createdAt),
	}
}
