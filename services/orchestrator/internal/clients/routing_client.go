// internal/clients/routing_client.go
package clients

import (
	"context"
	"fmt"
	"time"

	callprocessingv1 "orchestrator/internal/gen"

	"google.golang.org/grpc"
)

type RoutingClient struct {
	conn   *grpc.ClientConn
	client callprocessingv1.RoutingServiceClient
}

func NewRoutingClient(addr string) (*RoutingClient, error) {
	conn, err := grpcConnForService(addr, "ROUTING_GRPC")
	if err != nil {
		return nil, fmt.Errorf("dial routing grpc: %w", err)
	}

	return &RoutingClient{
		conn:   conn,
		client: callprocessingv1.NewRoutingServiceClient(conn),
	}, nil
}

type RoutingRequest struct {
	CallID   string    `json:"call_id"`
	Segments []Segment `json:"segments"`
}

type RoutingResponse struct {
	IntentID         string  `json:"intent_id"`
	IntentConfidence float64 `json:"intent_confidence"`
	Priority         string  `json:"priority"`
	SuggestedGroup   string  `json:"suggested_group,omitempty"`
}

func (c *RoutingClient) Route(callID string, segments []Segment) (*RoutingResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	pbSegments := make([]*callprocessingv1.Segment, 0, len(segments))
	for _, seg := range segments {
		pbSegments = append(pbSegments, &callprocessingv1.Segment{
			Start:   seg.Start,
			End:     seg.End,
			Speaker: seg.Speaker,
			Role:    seg.Role,
			Text:    seg.Text,
		})
	}

	resp, err := c.client.Route(ctx, &callprocessingv1.RouteRequest{
		CallId:   callID,
		Segments: pbSegments,
	})
	if err != nil {
		return nil, fmt.Errorf("routing rpc: %w", err)
	}

	if resp.GetRouting() == nil {
		return nil, fmt.Errorf("routing rpc: empty response")
	}

	return &RoutingResponse{
		IntentID:         resp.GetRouting().GetIntentId(),
		IntentConfidence: resp.GetRouting().GetIntentConfidence(),
		Priority:         resp.GetRouting().GetPriority(),
		SuggestedGroup:   resp.GetRouting().GetSuggestedGroup(),
	}, nil
}

func (c *RoutingClient) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}
