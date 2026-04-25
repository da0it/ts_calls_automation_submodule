package services

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var validSpamReviewDecisions = map[string]struct{}{
	"spam":     {},
	"not_spam": {},
}

type SpamFeedbackMeta struct {
	Status         string  `json:"status,omitempty"`
	PredictedLabel string  `json:"predicted_label,omitempty"`
	Confidence     float64 `json:"confidence,omitempty"`
	ThresholdLow   float64 `json:"threshold_low,omitempty"`
	ThresholdHigh  float64 `json:"threshold_high,omitempty"`
	Reason         string  `json:"reason,omitempty"`
	Backend        string  `json:"backend,omitempty"`
}

type SpamFeedbackRequest struct {
	CallID             string                      `json:"call_id"`
	SourceFilename     string                      `json:"source_filename,omitempty"`
	Decision           string                      `json:"decision"`
	TranscriptText     string                      `json:"transcript_text,omitempty"`
	TranscriptSegments []FeedbackTranscriptSegment `json:"transcript_segments,omitempty"`
	TrainingSample     string                      `json:"training_sample,omitempty"`
	SpamCheck          SpamFeedbackMeta            `json:"spam_check"`
}

type SpamFeedbackRecord struct {
	ID                 string                      `json:"id"`
	CreatedAt          string                      `json:"created_at"`
	CallID             string                      `json:"call_id"`
	SourceFilename     string                      `json:"source_filename,omitempty"`
	Decision           string                      `json:"decision"`
	Label              string                      `json:"label"`
	TranscriptText     string                      `json:"transcript_text,omitempty"`
	TranscriptSegments []FeedbackTranscriptSegment `json:"transcript_segments,omitempty"`
	TrainingSample     string                      `json:"training_sample,omitempty"`
	SpamCheck          SpamFeedbackMeta            `json:"spam_check"`
}

type SpamFeedbackService struct {
	feedbackPath  string
	positiveLabel string
	negativeLabel string
	mu            sync.Mutex
}

func NewSpamFeedbackService(feedbackPath, positiveLabel, negativeLabel string) *SpamFeedbackService {
	pos := strings.TrimSpace(positiveLabel)
	if pos == "" {
		pos = "spam"
	}
	neg := strings.TrimSpace(negativeLabel)
	if neg == "" {
		neg = "not_spam"
	}
	return &SpamFeedbackService{
		feedbackPath:  feedbackPath,
		positiveLabel: pos,
		negativeLabel: neg,
	}
}

func (s *SpamFeedbackService) SaveDecision(input SpamFeedbackRequest) (*SpamFeedbackRecord, error) {
	decision := strings.ToLower(strings.TrimSpace(input.Decision))
	if _, ok := validSpamReviewDecisions[decision]; !ok {
		return nil, fmt.Errorf("decision must be spam or not_spam")
	}

	callID := strings.TrimSpace(input.CallID)
	if callID == "" {
		callID = "unknown-call"
	}

	transcriptText := normalizeLongText(input.TranscriptText, 8000)
	segments := normalizeTranscriptSegments(input.TranscriptSegments, 160)
	trainingSample := normalizePlainText(input.TrainingSample)
	if trainingSample == "" {
		trainingSample = buildSampleFromTranscript(transcriptText)
	}
	if trainingSample != "" {
		trainingSample = normalizeLongText(trainingSample, 280)
	}

	label := s.negativeLabel
	if decision == "spam" {
		label = s.positiveLabel
	}

	record := &SpamFeedbackRecord{
		ID:                 fmt.Sprintf("spam-feedback-%d", time.Now().UnixNano()),
		CreatedAt:          time.Now().UTC().Format(time.RFC3339Nano),
		CallID:             callID,
		SourceFilename:     strings.TrimSpace(input.SourceFilename),
		Decision:           decision,
		Label:              label,
		TranscriptText:     transcriptText,
		TranscriptSegments: segments,
		TrainingSample:     trainingSample,
		SpamCheck: SpamFeedbackMeta{
			Status:         strings.TrimSpace(input.SpamCheck.Status),
			PredictedLabel: strings.TrimSpace(input.SpamCheck.PredictedLabel),
			Confidence:     input.SpamCheck.Confidence,
			ThresholdLow:   input.SpamCheck.ThresholdLow,
			ThresholdHigh:  input.SpamCheck.ThresholdHigh,
			Reason:         normalizeLongText(input.SpamCheck.Reason, 240),
			Backend:        strings.TrimSpace(input.SpamCheck.Backend),
		},
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if err := os.MkdirAll(filepath.Dir(s.feedbackPath), 0750); err != nil {
		return nil, fmt.Errorf("create spam feedback directory: %w", err)
	}

	file, err := os.OpenFile(s.feedbackPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return nil, fmt.Errorf("open spam feedback file: %w", err)
	}
	defer file.Close()

	body, err := json.Marshal(record)
	if err != nil {
		return nil, fmt.Errorf("marshal spam feedback: %w", err)
	}

	if _, err := file.Write(append(body, '\n')); err != nil {
		return nil, fmt.Errorf("append spam feedback: %w", err)
	}

	return record, nil
}
