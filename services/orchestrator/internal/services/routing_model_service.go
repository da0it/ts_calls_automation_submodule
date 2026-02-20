package services

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

type RoutingModelTrainRequest struct {
	Epochs       *int     `json:"epochs,omitempty"`
	BatchSize    *int     `json:"batch_size,omitempty"`
	LearningRate *float64 `json:"learning_rate,omitempty"`
	ValRatio     *float64 `json:"val_ratio,omitempty"`
	RandomSeed   *int     `json:"random_seed,omitempty"`
	FeedbackPath *string  `json:"feedback_path,omitempty"`
}

type RoutingModelService struct {
	baseURL    string
	adminToken string
	client     *http.Client
	datasetDir string
}

type CSVImportStats struct {
	SourceFile     string `json:"source_file"`
	OutputPath     string `json:"output_path"`
	TotalRows      int    `json:"total_rows"`
	ImportedRows   int    `json:"imported_rows"`
	SkippedRows    int    `json:"skipped_rows"`
	IntentColumn   string `json:"intent_column"`
	SampleColumn   string `json:"sample_column"`
	SegmentsColumn string `json:"segments_column,omitempty"`
}

func NewRoutingModelService(baseURL, adminToken string, timeout time.Duration, datasetDir string) *RoutingModelService {
	url := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if url == "" {
		return nil
	}
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	return &RoutingModelService{
		baseURL:    url,
		adminToken: strings.TrimSpace(adminToken),
		client: &http.Client{
			Timeout: timeout,
		},
		datasetDir: strings.TrimSpace(datasetDir),
	}
}

func (s *RoutingModelService) GetStatus() (map[string]any, error) {
	return s.requestJSON(http.MethodGet, "/admin/model/status", nil)
}

func (s *RoutingModelService) Reload() (map[string]any, error) {
	return s.requestJSON(http.MethodPost, "/admin/model/reload", map[string]any{})
}

func (s *RoutingModelService) Train(req RoutingModelTrainRequest) (map[string]any, error) {
	return s.requestJSON(http.MethodPost, "/admin/model/train", req)
}

func (s *RoutingModelService) TrainFromCSV(csvReader io.Reader, sourceFilename string, req RoutingModelTrainRequest) (map[string]any, error) {
	if s == nil {
		return nil, fmt.Errorf("routing model service is not configured")
	}
	feedbackPath, stats, err := s.convertCSVToFeedbackJSONL(csvReader, sourceFilename)
	if err != nil {
		return nil, err
	}
	req.FeedbackPath = &feedbackPath

	result, err := s.Train(req)
	if err != nil {
		return nil, err
	}
	result["csv_import"] = stats
	return result, nil
}

func (s *RoutingModelService) convertCSVToFeedbackJSONL(csvReader io.Reader, sourceFilename string) (string, CSVImportStats, error) {
	stats := CSVImportStats{SourceFile: strings.TrimSpace(sourceFilename)}
	if csvReader == nil {
		return "", stats, fmt.Errorf("csv file is required")
	}

	baseDir := s.datasetDir
	if baseDir == "" {
		baseDir = os.TempDir()
	}
	uploadDir := filepath.Join(baseDir, "routing_training_uploads")
	if err := os.MkdirAll(uploadDir, 0o755); err != nil {
		return "", stats, fmt.Errorf("create upload dir: %w", err)
	}

	safeName := sanitizeFilename(sourceFilename)
	if safeName == "" {
		safeName = "dataset"
	}
	outputPath := filepath.Join(
		uploadDir,
		fmt.Sprintf("routing_dataset_%d_%s.jsonl", time.Now().Unix(), safeName),
	)
	stats.OutputPath = outputPath

	outFile, err := os.Create(outputPath)
	if err != nil {
		return "", stats, fmt.Errorf("create dataset file: %w", err)
	}
	defer outFile.Close()

	writer := bufio.NewWriterSize(outFile, 1<<20)
	defer writer.Flush()

	reader := csv.NewReader(bufio.NewReaderSize(csvReader, 1<<20))
	reader.FieldsPerRecord = -1
	reader.LazyQuotes = true
	reader.ReuseRecord = true

	header, err := reader.Read()
	if err != nil {
		if err == io.EOF {
			return "", stats, fmt.Errorf("csv file is empty")
		}
		return "", stats, fmt.Errorf("read csv header: %w", err)
	}
	headerMap := mapHeaders(header)

	intentIdx, intentCol := findColumn(headerMap, "final_intent_id", "intent_id", "intent", "label", "target_intent")
	if intentIdx < 0 {
		return "", stats, fmt.Errorf("csv must contain column final_intent_id or intent_id")
	}

	sampleIdx, sampleCol := findColumn(headerMap, "training_sample", "text", "sample", "utterance", "phrase")
	transcriptIdx, _ := findColumn(headerMap, "transcript_text", "transcript")
	if sampleIdx < 0 && transcriptIdx < 0 {
		return "", stats, fmt.Errorf("csv must contain training_sample/text or transcript_text")
	}

	segmentsIdx, segmentsCol := findColumn(headerMap, "transcript_segments", "segments_json", "turns_json")
	stats.IntentColumn = intentCol
	if sampleIdx >= 0 {
		stats.SampleColumn = sampleCol
	} else {
		stats.SampleColumn = "transcript_text"
	}
	if segmentsIdx >= 0 {
		stats.SegmentsColumn = segmentsCol
	}

	for {
		row, readErr := reader.Read()
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return "", stats, fmt.Errorf("read csv row: %w", readErr)
		}
		stats.TotalRows++

		intentID := cell(row, intentIdx)
		trainingSample := ""
		if sampleIdx >= 0 {
			trainingSample = cell(row, sampleIdx)
		}
		transcriptText := ""
		if transcriptIdx >= 0 {
			transcriptText = cell(row, transcriptIdx)
		}
		if trainingSample == "" {
			trainingSample = transcriptText
		}

		if intentID == "" || trainingSample == "" {
			stats.SkippedRows++
			continue
		}

		record := map[string]any{
			"final": map[string]any{
				"intent_id": intentID,
			},
			"training_sample": trainingSample,
		}
		if transcriptText != "" {
			record["transcript_text"] = transcriptText
		}

		if segmentsIdx >= 0 {
			segmentsRaw := cell(row, segmentsIdx)
			if segmentsRaw != "" {
				var parsed []map[string]any
				if err := json.Unmarshal([]byte(segmentsRaw), &parsed); err == nil && len(parsed) > 0 {
					record["transcript_segments"] = parsed
				}
			}
		}

		raw, err := json.Marshal(record)
		if err != nil {
			stats.SkippedRows++
			continue
		}
		if _, err := writer.Write(raw); err != nil {
			return "", stats, fmt.Errorf("write jsonl row: %w", err)
		}
		if err := writer.WriteByte('\n'); err != nil {
			return "", stats, fmt.Errorf("write jsonl newline: %w", err)
		}
		stats.ImportedRows++
	}

	if stats.ImportedRows == 0 {
		return "", stats, fmt.Errorf("no valid training rows found in csv")
	}
	if err := writer.Flush(); err != nil {
		return "", stats, fmt.Errorf("flush dataset file: %w", err)
	}
	return outputPath, stats, nil
}

func mapHeaders(header []string) map[string]int {
	out := make(map[string]int, len(header))
	for i, raw := range header {
		key := normalizeHeader(raw)
		if key == "" {
			continue
		}
		if _, exists := out[key]; !exists {
			out[key] = i
		}
	}
	return out
}

func normalizeHeader(value string) string {
	value = strings.TrimSpace(strings.TrimPrefix(value, "\uFEFF"))
	value = strings.ToLower(value)
	value = strings.ReplaceAll(value, " ", "_")
	value = strings.ReplaceAll(value, "-", "_")
	value = strings.ReplaceAll(value, ".", "_")
	return value
}

func findColumn(columns map[string]int, names ...string) (int, string) {
	for _, name := range names {
		if idx, ok := columns[name]; ok {
			return idx, name
		}
	}
	return -1, ""
}

func cell(row []string, idx int) string {
	if idx < 0 || idx >= len(row) {
		return ""
	}
	return strings.TrimSpace(row[idx])
}

var nonFilenameChars = regexp.MustCompile(`[^a-zA-Z0-9._-]+`)

func sanitizeFilename(name string) string {
	base := filepath.Base(strings.TrimSpace(name))
	base = strings.TrimSuffix(base, filepath.Ext(base))
	base = nonFilenameChars.ReplaceAllString(base, "_")
	base = strings.Trim(base, "._-")
	if len(base) > 80 {
		base = base[:80]
	}
	return base
}

func (s *RoutingModelService) requestJSON(method, path string, payload any) (map[string]any, error) {
	if s == nil {
		return nil, fmt.Errorf("routing model service is not configured")
	}

	var body io.Reader
	if payload != nil {
		raw, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal request body: %w", err)
		}
		body = bytes.NewReader(raw)
	}

	req, err := http.NewRequest(method, s.baseURL+path, body)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if s.adminToken != "" {
		req.Header.Set("Authorization", "Bearer "+s.adminToken)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request router admin: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	var parsed map[string]any
	if len(respBody) > 0 {
		if err := json.Unmarshal(respBody, &parsed); err != nil {
			return nil, fmt.Errorf("decode response json: %w", err)
		}
	} else {
		parsed = map[string]any{}
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		if msg, ok := parsed["error"].(string); ok && strings.TrimSpace(msg) != "" {
			return nil, fmt.Errorf("%s", msg)
		}
		return nil, fmt.Errorf("router admin http %d", resp.StatusCode)
	}

	return parsed, nil
}
