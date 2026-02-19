package services

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var validReviewDecisions = map[string]struct{}{
	"accepted": {},
	"rejected": {},
}

var validErrorTypes = map[string]struct{}{
	"none":               {},
	"wrong_intent":       {},
	"wrong_group":        {},
	"wrong_priority":     {},
	"partial_transcript": {},
	"low_confidence":     {},
	"missing_context":    {},
	"other":              {},
}

type FeedbackAISuggestion struct {
	IntentID   string  `json:"intent_id"`
	Confidence float64 `json:"confidence,omitempty"`
	Priority   string  `json:"priority,omitempty"`
	Group      string  `json:"group,omitempty"`
}

type FeedbackFinalRouting struct {
	IntentID string `json:"intent_id"`
	Priority string `json:"priority,omitempty"`
	Group    string `json:"group"`
}

type FeedbackTranscriptSegment struct {
	Start   float64 `json:"start,omitempty"`
	End     float64 `json:"end,omitempty"`
	Speaker string  `json:"speaker,omitempty"`
	Role    string  `json:"role,omitempty"`
	Text    string  `json:"text,omitempty"`
}

type RoutingFeedbackRequest struct {
	CallID             string                      `json:"call_id"`
	SourceFilename     string                      `json:"source_filename,omitempty"`
	Decision           string                      `json:"decision"`
	ErrorType          string                      `json:"error_type,omitempty"`
	Comment            string                      `json:"comment,omitempty"`
	TranscriptText     string                      `json:"transcript_text,omitempty"`
	TranscriptSegments []FeedbackTranscriptSegment `json:"transcript_segments,omitempty"`
	TrainingSample     string                      `json:"training_sample,omitempty"`
	AI                 FeedbackAISuggestion        `json:"ai"`
	Final              FeedbackFinalRouting        `json:"final"`
}

type RoutingFeedbackRecord struct {
	ID                 string                      `json:"id"`
	CreatedAt          string                      `json:"created_at"`
	CallID             string                      `json:"call_id"`
	SourceFilename     string                      `json:"source_filename,omitempty"`
	Decision           string                      `json:"decision"`
	ErrorType          string                      `json:"error_type,omitempty"`
	Comment            string                      `json:"comment,omitempty"`
	TranscriptText     string                      `json:"transcript_text,omitempty"`
	TranscriptSegments []FeedbackTranscriptSegment `json:"transcript_segments,omitempty"`
	AI                 FeedbackAISuggestion        `json:"ai"`
	Final              FeedbackFinalRouting        `json:"final"`
	AutoLearnApplied   bool                        `json:"auto_learn_applied"`
	AutoLearnMessage   string                      `json:"auto_learn_message,omitempty"`
}

type RoutingFeedbackService struct {
	feedbackPath   string
	autoLearn      bool
	autoLearnLimit int
	configService  *RoutingConfigService
	mu             sync.Mutex
}

func NewRoutingFeedbackService(
	feedbackPath string,
	autoLearn bool,
	autoLearnLimit int,
	configService *RoutingConfigService,
) *RoutingFeedbackService {
	if autoLearnLimit <= 0 {
		autoLearnLimit = 50
	}
	return &RoutingFeedbackService{
		feedbackPath:   feedbackPath,
		autoLearn:      autoLearn,
		autoLearnLimit: autoLearnLimit,
		configService:  configService,
	}
}

func (s *RoutingFeedbackService) SaveFeedback(input RoutingFeedbackRequest) (*RoutingFeedbackRecord, error) {
	record, err := s.normalizeAndValidate(input)
	if err != nil {
		return nil, err
	}

	if s.autoLearn && s.configService != nil {
		shouldLearn := record.Decision == "rejected" ||
			record.AI.IntentID != record.Final.IntentID ||
			record.AI.Group != record.Final.Group ||
			record.AI.Priority != record.Final.Priority

		if shouldLearn {
			if strings.HasPrefix(record.Final.IntentID, "misc.") {
				record.AutoLearnApplied = false
				record.AutoLearnMessage = "auto-learn skipped for misc intents"
			} else {
				sample := normalizePlainText(input.TrainingSample)
				if sample == "" {
					sample = buildSampleFromTranscript(record.TranscriptText)
				}
				if !isTrainingSampleUsable(sample) {
					record.AutoLearnApplied = false
					record.AutoLearnMessage = "auto-learn skipped: sample is too short/noisy"
				} else {
					added, addErr := s.configService.AddExampleToIntent(
						record.Final.IntentID,
						sample,
						s.autoLearnLimit,
					)
					if addErr != nil {
						record.AutoLearnApplied = false
						record.AutoLearnMessage = fmt.Sprintf("auto-learn failed: %v", addErr)
					} else if added {
						record.AutoLearnApplied = true
						record.AutoLearnMessage = "training sample added to intent examples"
					} else {
						record.AutoLearnApplied = false
						record.AutoLearnMessage = "sample already exists for target intent"
					}
				}
			}
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if err := os.MkdirAll(filepath.Dir(s.feedbackPath), 0755); err != nil {
		return nil, fmt.Errorf("create feedback directory: %w", err)
	}

	file, err := os.OpenFile(s.feedbackPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("open feedback file: %w", err)
	}
	defer file.Close()

	body, err := json.Marshal(record)
	if err != nil {
		return nil, fmt.Errorf("marshal feedback: %w", err)
	}

	if _, err = file.Write(append(body, '\n')); err != nil {
		return nil, fmt.Errorf("append feedback: %w", err)
	}
	return record, nil
}

func (s *RoutingFeedbackService) normalizeAndValidate(input RoutingFeedbackRequest) (*RoutingFeedbackRecord, error) {
	decision := strings.ToLower(strings.TrimSpace(input.Decision))
	if _, ok := validReviewDecisions[decision]; !ok {
		return nil, errors.New("decision must be accepted or rejected")
	}

	errorType := strings.ToLower(strings.TrimSpace(input.ErrorType))
	if errorType == "" {
		if decision == "accepted" {
			errorType = "none"
		} else {
			errorType = "other"
		}
	}
	if _, ok := validErrorTypes[errorType]; !ok {
		return nil, fmt.Errorf("invalid error_type %q", errorType)
	}

	finalIntent := strings.TrimSpace(input.Final.IntentID)
	finalGroup := strings.TrimSpace(input.Final.Group)
	finalPriority := normalizePriorityValue(input.Final.Priority)
	if finalIntent == "" {
		return nil, errors.New("final.intent_id is required")
	}
	if finalGroup == "" {
		return nil, errors.New("final.group is required")
	}

	callID := strings.TrimSpace(input.CallID)
	if callID == "" {
		callID = "unknown-call"
	}

	record := &RoutingFeedbackRecord{
		ID:             fmt.Sprintf("feedback-%d", time.Now().UnixNano()),
		CreatedAt:      time.Now().UTC().Format(time.RFC3339Nano),
		CallID:         callID,
		SourceFilename: strings.TrimSpace(input.SourceFilename),
		Decision:       decision,
		ErrorType:      errorType,
		Comment:        normalizePlainText(input.Comment),
		TranscriptText: normalizeLongText(input.TranscriptText, 8000),
		TranscriptSegments: normalizeTranscriptSegments(
			input.TranscriptSegments,
			160,
		),
		AI: FeedbackAISuggestion{
			IntentID:   strings.TrimSpace(input.AI.IntentID),
			Confidence: input.AI.Confidence,
			Priority:   normalizePriorityValue(input.AI.Priority),
			Group:      strings.TrimSpace(input.AI.Group),
		},
		Final: FeedbackFinalRouting{
			IntentID: finalIntent,
			Priority: finalPriority,
			Group:    finalGroup,
		},
	}

	return record, nil
}

func normalizePriorityValue(value string) string {
	p := strings.ToLower(strings.TrimSpace(value))
	if p == "normal" {
		p = "medium"
	}
	if p == "" {
		return "medium"
	}
	if _, ok := validPriorities[p]; ok {
		return p
	}
	return "medium"
}

func normalizePlainText(value string) string {
	if value == "" {
		return ""
	}
	parts := strings.Fields(strings.TrimSpace(value))
	return strings.Join(parts, " ")
}

func normalizeLongText(value string, limit int) string {
	text := normalizePlainText(value)
	if limit <= 0 || len(text) <= limit {
		return text
	}
	return strings.TrimSpace(text[:limit])
}

func normalizeTranscriptSegments(items []FeedbackTranscriptSegment, maxItems int) []FeedbackTranscriptSegment {
	if len(items) == 0 {
		return nil
	}
	if maxItems <= 0 {
		maxItems = 160
	}

	out := make([]FeedbackTranscriptSegment, 0, min(len(items), maxItems))
	for _, item := range items {
		text := normalizeLongText(item.Text, 420)
		if text == "" {
			continue
		}

		start := item.Start
		if start < 0 {
			start = 0
		}
		end := item.End
		if end < 0 {
			end = 0
		}
		if end < start {
			end = start
		}

		out = append(out, FeedbackTranscriptSegment{
			Start:   start,
			End:     end,
			Speaker: normalizeLongText(item.Speaker, 64),
			Role:    normalizeLongText(item.Role, 32),
			Text:    text,
		})
		if len(out) >= maxItems {
			break
		}
	}

	if len(out) == 0 {
		return nil
	}
	return out
}

func buildSampleFromTranscript(transcript string) string {
	text := normalizeLongText(transcript, 420)
	if len(text) < 24 {
		return ""
	}
	parts := strings.FieldsFunc(text, func(r rune) bool {
		return r == '.' || r == '!' || r == '?' || r == ';'
	})
	for _, part := range parts {
		part = normalizeLongText(part, 180)
		if len(part) >= 24 {
			return part
		}
	}
	return normalizeLongText(text, 180)
}

func isTrainingSampleUsable(sample string) bool {
	sample = normalizePlainText(sample)
	if len(sample) < 24 || len(sample) > 220 {
		return false
	}
	digits := 0
	for _, r := range sample {
		if r >= '0' && r <= '9' {
			digits++
		}
	}
	if digits > len(sample)/3 {
		return false
	}
	return true
}
