// internal/clients/ticket_client.go
package clients

import (
	"context"
	"fmt"
	"time"

	callprocessingv1 "orchestrator/internal/gen"

	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/structpb"
)

type TicketClient struct {
	conn   *grpc.ClientConn
	client callprocessingv1.TicketServiceClient
}

func NewTicketClient(addr string) (*TicketClient, error) {
	conn, err := grpcConnForService(addr, "TICKET_GRPC")
	if err != nil {
		return nil, fmt.Errorf("dial ticket grpc: %w", err)
	}

	return &TicketClient{
		conn:   conn,
		client: callprocessingv1.NewTicketServiceClient(conn),
	}, nil
}

type TranscriptData struct {
	CallID      string                 `json:"call_id"`
	Segments    []Segment              `json:"segments"`
	RoleMapping map[string]string      `json:"role_mapping"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

type RoutingData struct {
	IntentID         string  `json:"intent_id"`
	IntentConfidence float64 `json:"intent_confidence"`
	Priority         string  `json:"priority"`
	SuggestedGroup   string  `json:"suggested_group,omitempty"`
}

type CreateTicketRequest struct {
	Transcript TranscriptData `json:"transcript"`
	Routing    RoutingData    `json:"routing"`
	AudioURL   string         `json:"audio_url,omitempty"`
}

type TicketCreated struct {
	TicketID   string    `json:"ticket_id"`
	ExternalID string    `json:"external_id"`
	URL        string    `json:"url"`
	System     string    `json:"system"`
	CreatedAt  time.Time `json:"created_at"`
}

type CreateTicketResponse struct {
	Success bool           `json:"success"`
	Ticket  *TicketCreated `json:"ticket,omitempty"`
	Error   string         `json:"error,omitempty"`
}

func (c *TicketClient) CreateTicket(transcript *TranscriptionResponse, routing *RoutingResponse, entities *Entities) (*TicketCreated, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	pbSegments := make([]*callprocessingv1.Segment, 0, len(transcript.Segments))
	for _, seg := range transcript.Segments {
		pbSegments = append(pbSegments, &callprocessingv1.Segment{
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

	resp, err := c.client.CreateTicket(ctx, &callprocessingv1.CreateTicketRequest{
		Transcript: &callprocessingv1.Transcript{
			CallId:      transcript.CallID,
			Segments:    pbSegments,
			RoleMapping: transcript.RoleMapping,
			Metadata:    metadata,
		},
		Routing: &callprocessingv1.Routing{
			IntentId:         routing.IntentID,
			IntentConfidence: routing.IntentConfidence,
			Priority:         routing.Priority,
			SuggestedGroup:   routing.SuggestedGroup,
		},
		Entities: toProtoEntities(entities),
	})
	if err != nil {
		return nil, fmt.Errorf("ticket rpc: %w", err)
	}

	if resp.GetTicket() == nil {
		return nil, fmt.Errorf("ticket rpc: empty response")
	}

	createdAt := time.Now()
	if resp.GetTicket().GetCreatedAt() != nil {
		createdAt = resp.GetTicket().GetCreatedAt().AsTime()
	}

	return &TicketCreated{
		TicketID:   resp.GetTicket().GetTicketId(),
		ExternalID: resp.GetTicket().GetExternalId(),
		URL:        resp.GetTicket().GetUrl(),
		System:     resp.GetTicket().GetSystem(),
		CreatedAt:  createdAt,
	}, nil
}

func (c *TicketClient) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

func toProtoEntities(entities *Entities) *callprocessingv1.Entities {
	if entities == nil {
		return &callprocessingv1.Entities{}
	}

	return &callprocessingv1.Entities{
		Persons:      toProtoEntityList(entities.Persons),
		Phones:       toProtoEntityList(entities.Phones),
		Emails:       toProtoEntityList(entities.Emails),
		OrderIds:     toProtoEntityList(entities.OrderIDs),
		AccountIds:   toProtoEntityList(entities.AccountIDs),
		MoneyAmounts: toProtoEntityList(entities.MoneyAmounts),
		Dates:        toProtoEntityList(entities.Dates),
	}
}

func toProtoEntityList(items []ExtractedEntity) []*callprocessingv1.ExtractedEntity {
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
