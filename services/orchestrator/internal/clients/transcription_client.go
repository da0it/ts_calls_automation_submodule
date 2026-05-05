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
)

// TranscriptionClient хранит gRPC-подключение и protobuf-клиент для transcription-service
type TranscriptionClient struct {
	conn   *grpc.ClientConn
	client callprocessingv1.TranscriptionServiceClient
}

// Функция создает gRPC-клиент transcription-service
func NewTranscriptionClient(addr string) (*TranscriptionClient, error) {
	conn, err := grpcConnForService(addr, "TRANSCRIPTION_GRPC")
	if err != nil {
		return nil, fmt.Errorf("dial transcription grpc: %w", err)
	}

	// Создаётся объект TranscriptionClient
	return &TranscriptionClient{
		conn:   conn,
		client: callprocessingv1.NewTranscriptionServiceClient(conn),
	}, nil
}

// Segment описывает один фрагмент распознанного диалога
type Segment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker string  `json:"speaker"`
	Role    string  `json:"role"`
	Text    string  `json:"text"`
}

// Результат транскрибации, который transcription client возвращает orchestrator
type TranscriptionResponse struct {
	CallID      string                 `json:"call_id"`
	Segments    []Segment              `json:"segments"`
	RoleMapping map[string]string      `json:"role_mapping"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// Метод транскрибации. Принимает путь к аудиофайлу, отправляет файл в transcription-service и возвращает распознанную транскрипцию
func (c *TranscriptionClient) Transcribe(audioPath string) (*TranscriptionResponse, error) {
	audioData, err := os.ReadFile(audioPath)
	if err != nil {
		return nil, fmt.Errorf("read audio file: %w", err)
	}

	callID := filepath.Base(audioPath)
	if ext := filepath.Ext(callID); ext != "" {
		callID = callID[:len(callID)-len(ext)]
	}
	if callID == "" {
		callID = "unknown-call"
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
		CallId:   callID,
	})
	if err != nil {
		return nil, fmt.Errorf("transcription rpc: %w", err)
	}

	result := &TranscriptionResponse{
		CallID:      resp.GetTranscript().GetCallId(),
		RoleMapping: resp.GetTranscript().GetRoleMapping(),
	}
	if result.CallID == "" {
		result.CallID = callID
	}

	if meta := resp.GetTranscript().GetMetadata(); meta != nil {
		result.Metadata = meta.AsMap()
	} else {
		result.Metadata = map[string]interface{}{}
	}

	// protobuf-сегменты преобразуются во внутренние Segment
	for _, seg := range resp.GetTranscript().GetSegments() {
		result.Segments = append(result.Segments, Segment{
			Start:   seg.GetStart(),
			End:     seg.GetEnd(),
			Speaker: seg.GetSpeaker(),
			Role:    seg.GetRole(),
			Text:    seg.GetText(),
		})
	}
	// Возвращается готовый TranscriptionResponse
	return result, nil
}

// Метод закрывает gRPC-соединение с transcription-service
func (c *TranscriptionClient) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}
