// internal/clients/transcription_client.go
package clients

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	callprocessingv1 "orchestrator/internal/gen"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type TranscriptionClient struct {
	conn   *grpc.ClientConn
	client callprocessingv1.TranscriptionServiceClient
}

func NewTranscriptionClient(addr string) (*TranscriptionClient, error) {
	conn, err := grpc.NewClient(
		addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("dial transcription grpc: %w", err)
	}

	return &TranscriptionClient{
		conn:   conn,
		client: callprocessingv1.NewTranscriptionServiceClient(conn),
	}, nil
}

type Segment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker string  `json:"speaker"`
	Role    string  `json:"role"`
	Text    string  `json:"text"`
}

type TranscriptionResponse struct {
	CallID      string                 `json:"call_id"`
	Segments    []Segment              `json:"segments"`
	RoleMapping map[string]string      `json:"role_mapping"`
	Metadata    map[string]interface{} `json:"metadata"`
}

func (c *TranscriptionClient) Transcribe(audioPath string) (*TranscriptionResponse, error) {
	audioData, err := os.ReadFile(audioPath)
	if err != nil {
		return nil, fmt.Errorf("read audio file: %w", err)
	}

	timeoutSec := 9999
	if raw := os.Getenv("TRANSCRIPTION_RPC_TIMEOUT_SECONDS"); raw != "" {
		if parsed, parseErr := strconv.Atoi(raw); parseErr == nil && parsed > 0 {
			timeoutSec = parsed
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutSec)*time.Second)
	defer cancel()

	resp, err := c.client.Transcribe(ctx, &callprocessingv1.TranscribeRequest{
		Audio:    audioData,
		Filename: filepath.Base(audioPath),
	})
	if err != nil {
		return nil, fmt.Errorf("transcription rpc: %w", err)
	}

	result := &TranscriptionResponse{
		CallID:      resp.GetTranscript().GetCallId(),
		RoleMapping: resp.GetTranscript().GetRoleMapping(),
	}

	if meta := resp.GetTranscript().GetMetadata(); meta != nil {
		result.Metadata = meta.AsMap()
	} else {
		result.Metadata = map[string]interface{}{}
	}

	for _, seg := range resp.GetTranscript().GetSegments() {
		result.Segments = append(result.Segments, Segment{
			Start:   seg.GetStart(),
			End:     seg.GetEnd(),
			Speaker: seg.GetSpeaker(),
			Role:    seg.GetRole(),
			Text:    seg.GetText(),
		})
	}

	return result, nil
}

func (c *TranscriptionClient) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}
