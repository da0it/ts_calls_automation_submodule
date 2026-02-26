// internal/clients/notification_client.go
package clients

import (
	"context"
	"fmt"
	"time"

	callprocessingv1 "orchestrator/internal/gen"

	"google.golang.org/grpc"
)

type NotificationClient struct {
	conn   *grpc.ClientConn
	client callprocessingv1.NotificationServiceClient
}

func NewNotificationClient(addr string) (*NotificationClient, error) {
	conn, err := grpcConnForService(addr, "NOTIFICATION_GRPC")
	if err != nil {
		return nil, fmt.Errorf("dial notification grpc: %w", err)
	}

	return &NotificationClient{
		conn:   conn,
		client: callprocessingv1.NewNotificationServiceClient(conn),
	}, nil
}

type NotificationResult struct {
	Success  bool            `json:"success"`
	Channels []ChannelResult `json:"channels"`
}

type ChannelResult struct {
	Type    string `json:"type"`
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}

func (c *NotificationClient) SendNotification(
	transcript *TranscriptionResponse,
	routing *RoutingResponse,
	entities *Entities,
	ticket *TicketCreated,
) (*NotificationResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
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

	resp, err := c.client.SendNotification(ctx, &callprocessingv1.SendNotificationRequest{
		CallId: transcript.CallID,
		Transcript: &callprocessingv1.Transcript{
			CallId:      transcript.CallID,
			Segments:    pbSegments,
			RoleMapping: transcript.RoleMapping,
		},
		Routing: &callprocessingv1.Routing{
			IntentId:         routing.IntentID,
			IntentConfidence: routing.IntentConfidence,
			Priority:         routing.Priority,
			SuggestedGroup:   routing.SuggestedGroup,
		},
		Entities: toProtoEntities(entities),
		Ticket: &callprocessingv1.TicketCreated{
			TicketId:   ticket.TicketID,
			ExternalId: ticket.ExternalID,
			Url:        ticket.URL,
			System:     ticket.System,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("notification rpc: %w", err)
	}

	result := &NotificationResult{Success: resp.GetSuccess()}
	for _, ch := range resp.GetResults() {
		result.Channels = append(result.Channels, ChannelResult{
			Type:    ch.GetType(),
			Success: ch.GetSuccess(),
			Error:   ch.GetError(),
		})
	}
	return result, nil
}

func (c *NotificationClient) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}
