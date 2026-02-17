package services

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

var validPriorities = map[string]struct{}{
	"low":      {},
	"medium":   {},
	"high":     {},
	"critical": {},
}

type RoutingGroup struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Description string `json:"description,omitempty"`
}

type RoutingIntent struct {
	ID           string   `json:"id"`
	Title        string   `json:"title"`
	Description  string   `json:"description,omitempty"`
	Examples     []string `json:"examples"`
	DefaultGroup string   `json:"default_group"`
	Priority     string   `json:"priority"`
	Tags         []string `json:"tags,omitempty"`
	Keywords     []string `json:"keywords,omitempty"`
}

type RoutingCatalog struct {
	Groups    []RoutingGroup  `json:"groups"`
	Intents   []RoutingIntent `json:"intents"`
	UpdatedAt string          `json:"updated_at,omitempty"`
}

type routingGroupFileEntry struct {
	Title       string `json:"title"`
	Description string `json:"description,omitempty"`
}

type routingIntentFileEntry struct {
	Title        string   `json:"title"`
	Description  string   `json:"description,omitempty"`
	Examples     []string `json:"examples"`
	DefaultGroup string   `json:"default_group"`
	Priority     string   `json:"priority"`
	Tags         []string `json:"tags,omitempty"`
	Keywords     []string `json:"keywords,omitempty"`
}

type RoutingConfigService struct {
	intentsPath string
	groupsPath  string
	mu          sync.Mutex
}

func NewRoutingConfigService(intentsPath, groupsPath string) *RoutingConfigService {
	return &RoutingConfigService{
		intentsPath: intentsPath,
		groupsPath:  groupsPath,
	}
}

func (s *RoutingConfigService) GetCatalog() (*RoutingCatalog, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.loadCatalogLocked()
}

func (s *RoutingConfigService) ReplaceCatalog(catalog *RoutingCatalog) (*RoutingCatalog, error) {
	if catalog == nil {
		return nil, errors.New("catalog is required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := validateCatalog(catalog); err != nil {
		return nil, err
	}
	if err := s.saveCatalogLocked(catalog); err != nil {
		return nil, err
	}
	return s.loadCatalogLocked()
}

func (s *RoutingConfigService) AddGroup(group RoutingGroup) (*RoutingCatalog, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	catalog, err := s.loadCatalogLocked()
	if err != nil {
		return nil, err
	}

	group = normalizeGroup(group)
	if group.ID == "" || group.Title == "" {
		return nil, errors.New("group id and title are required")
	}
	for _, existing := range catalog.Groups {
		if existing.ID == group.ID {
			return nil, fmt.Errorf("group %q already exists", group.ID)
		}
	}
	catalog.Groups = append(catalog.Groups, group)
	if err := validateCatalog(catalog); err != nil {
		return nil, err
	}
	if err := s.saveCatalogLocked(catalog); err != nil {
		return nil, err
	}
	return s.loadCatalogLocked()
}

func (s *RoutingConfigService) DeleteGroup(groupID string) (*RoutingCatalog, error) {
	groupID = strings.TrimSpace(groupID)
	if groupID == "" {
		return nil, errors.New("group id is required")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	catalog, err := s.loadCatalogLocked()
	if err != nil {
		return nil, err
	}

	found := false
	filtered := make([]RoutingGroup, 0, len(catalog.Groups))
	for _, group := range catalog.Groups {
		if group.ID == groupID {
			found = true
			continue
		}
		filtered = append(filtered, group)
	}
	if !found {
		return nil, fmt.Errorf("group %q not found", groupID)
	}

	for _, intent := range catalog.Intents {
		if intent.DefaultGroup == groupID {
			return nil, fmt.Errorf(
				"cannot delete group %q: it is used by intent %q",
				groupID,
				intent.ID,
			)
		}
	}

	catalog.Groups = filtered
	if err := s.saveCatalogLocked(catalog); err != nil {
		return nil, err
	}
	return s.loadCatalogLocked()
}

func (s *RoutingConfigService) AddIntent(intent RoutingIntent) (*RoutingCatalog, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	catalog, err := s.loadCatalogLocked()
	if err != nil {
		return nil, err
	}

	intent = normalizeIntent(intent)
	if intent.ID == "" || intent.Title == "" {
		return nil, errors.New("intent id and title are required")
	}
	for _, existing := range catalog.Intents {
		if existing.ID == intent.ID {
			return nil, fmt.Errorf("intent %q already exists", intent.ID)
		}
	}

	catalog.Intents = append(catalog.Intents, intent)
	if err := validateCatalog(catalog); err != nil {
		return nil, err
	}
	if err := s.saveCatalogLocked(catalog); err != nil {
		return nil, err
	}
	return s.loadCatalogLocked()
}

func (s *RoutingConfigService) DeleteIntent(intentID string) (*RoutingCatalog, error) {
	intentID = strings.TrimSpace(intentID)
	if intentID == "" {
		return nil, errors.New("intent id is required")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	catalog, err := s.loadCatalogLocked()
	if err != nil {
		return nil, err
	}

	found := false
	filtered := make([]RoutingIntent, 0, len(catalog.Intents))
	for _, intent := range catalog.Intents {
		if intent.ID == intentID {
			found = true
			continue
		}
		filtered = append(filtered, intent)
	}
	if !found {
		return nil, fmt.Errorf("intent %q not found", intentID)
	}

	catalog.Intents = filtered
	if err := validateCatalog(catalog); err != nil {
		return nil, err
	}
	if err := s.saveCatalogLocked(catalog); err != nil {
		return nil, err
	}
	return s.loadCatalogLocked()
}

func (s *RoutingConfigService) AddExampleToIntent(intentID, example string, maxExamples int) (bool, error) {
	intentID = strings.TrimSpace(intentID)
	example = normalizeExampleText(example)
	if intentID == "" {
		return false, errors.New("intent id is required")
	}
	if example == "" {
		return false, errors.New("example text is required")
	}
	if maxExamples <= 0 {
		maxExamples = 50
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	catalog, err := s.loadCatalogLocked()
	if err != nil {
		return false, err
	}

	intentIndex := -1
	for i, intent := range catalog.Intents {
		if intent.ID == intentID {
			intentIndex = i
			break
		}
	}
	if intentIndex == -1 {
		return false, fmt.Errorf("intent %q not found", intentID)
	}

	intent := normalizeIntent(catalog.Intents[intentIndex])
	exampleFold := strings.ToLower(example)
	for _, existing := range intent.Examples {
		if strings.ToLower(normalizeExampleText(existing)) == exampleFold {
			return false, nil
		}
	}

	intent.Examples = append([]string{example}, intent.Examples...)
	if len(intent.Examples) > maxExamples {
		intent.Examples = intent.Examples[:maxExamples]
	}
	catalog.Intents[intentIndex] = intent

	if err := validateCatalog(catalog); err != nil {
		return false, err
	}
	if err := s.saveCatalogLocked(catalog); err != nil {
		return false, err
	}
	return true, nil
}

func (s *RoutingConfigService) loadCatalogLocked() (*RoutingCatalog, error) {
	groupsPayload := map[string]routingGroupFileEntry{}
	if err := readJSONFile(s.groupsPath, &groupsPayload); err != nil {
		return nil, fmt.Errorf("load groups: %w", err)
	}

	intentsPayload := map[string]routingIntentFileEntry{}
	if err := readJSONFile(s.intentsPath, &intentsPayload); err != nil {
		return nil, fmt.Errorf("load intents: %w", err)
	}

	catalog := &RoutingCatalog{
		Groups:  make([]RoutingGroup, 0, len(groupsPayload)),
		Intents: make([]RoutingIntent, 0, len(intentsPayload)),
	}

	for id, group := range groupsPayload {
		catalog.Groups = append(catalog.Groups, normalizeGroup(RoutingGroup{
			ID:          id,
			Title:       group.Title,
			Description: group.Description,
		}))
	}

	for id, intent := range intentsPayload {
		catalog.Intents = append(catalog.Intents, normalizeIntent(RoutingIntent{
			ID:           id,
			Title:        intent.Title,
			Description:  intent.Description,
			Examples:     intent.Examples,
			DefaultGroup: intent.DefaultGroup,
			Priority:     intent.Priority,
			Tags:         intent.Tags,
			Keywords:     intent.Keywords,
		}))
	}

	sort.Slice(catalog.Groups, func(i, j int) bool {
		return catalog.Groups[i].ID < catalog.Groups[j].ID
	})
	sort.Slice(catalog.Intents, func(i, j int) bool {
		return catalog.Intents[i].ID < catalog.Intents[j].ID
	})

	if err := validateCatalog(catalog); err != nil {
		return nil, err
	}

	if stat, err := os.Stat(s.intentsPath); err == nil {
		catalog.UpdatedAt = stat.ModTime().UTC().Format(time.RFC3339)
	}
	return catalog, nil
}

func (s *RoutingConfigService) saveCatalogLocked(catalog *RoutingCatalog) error {
	groupsPayload := make(map[string]routingGroupFileEntry, len(catalog.Groups))
	for _, group := range catalog.Groups {
		normalized := normalizeGroup(group)
		groupsPayload[normalized.ID] = routingGroupFileEntry{
			Title:       normalized.Title,
			Description: normalized.Description,
		}
	}

	intentsPayload := make(map[string]routingIntentFileEntry, len(catalog.Intents))
	for _, intent := range catalog.Intents {
		normalized := normalizeIntent(intent)
		intentsPayload[normalized.ID] = routingIntentFileEntry{
			Title:        normalized.Title,
			Description:  normalized.Description,
			Examples:     normalized.Examples,
			DefaultGroup: normalized.DefaultGroup,
			Priority:     normalized.Priority,
			Tags:         normalized.Tags,
			Keywords:     normalized.Keywords,
		}
	}

	if err := writeJSONFileAtomic(s.groupsPath, groupsPayload); err != nil {
		return fmt.Errorf("save groups: %w", err)
	}
	if err := writeJSONFileAtomic(s.intentsPath, intentsPayload); err != nil {
		return fmt.Errorf("save intents: %w", err)
	}
	return nil
}

func validateCatalog(catalog *RoutingCatalog) error {
	if catalog == nil {
		return errors.New("catalog is nil")
	}

	groupSet := map[string]struct{}{}
	for _, rawGroup := range catalog.Groups {
		group := normalizeGroup(rawGroup)
		if group.ID == "" {
			return errors.New("group id is required")
		}
		if group.Title == "" {
			return fmt.Errorf("group %q title is required", group.ID)
		}
		if _, exists := groupSet[group.ID]; exists {
			return fmt.Errorf("duplicate group id %q", group.ID)
		}
		groupSet[group.ID] = struct{}{}
	}

	intentSet := map[string]struct{}{}
	for _, rawIntent := range catalog.Intents {
		intent := normalizeIntent(rawIntent)
		if intent.ID == "" {
			return errors.New("intent id is required")
		}
		if intent.Title == "" {
			return fmt.Errorf("intent %q title is required", intent.ID)
		}
		if intent.DefaultGroup == "" {
			return fmt.Errorf("intent %q default_group is required", intent.ID)
		}
		if _, ok := groupSet[intent.DefaultGroup]; !ok {
			return fmt.Errorf(
				"intent %q references unknown default_group %q",
				intent.ID,
				intent.DefaultGroup,
			)
		}
		if intent.Priority == "" {
			return fmt.Errorf("intent %q priority is required", intent.ID)
		}
		if _, ok := validPriorities[intent.Priority]; !ok {
			return fmt.Errorf("intent %q has invalid priority %q", intent.ID, intent.Priority)
		}
		if len(intent.Examples) == 0 {
			return fmt.Errorf("intent %q requires at least one example", intent.ID)
		}
		if _, exists := intentSet[intent.ID]; exists {
			return fmt.Errorf("duplicate intent id %q", intent.ID)
		}
		intentSet[intent.ID] = struct{}{}
	}
	return nil
}

func normalizeGroup(group RoutingGroup) RoutingGroup {
	group.ID = strings.TrimSpace(group.ID)
	group.Title = strings.TrimSpace(group.Title)
	group.Description = strings.TrimSpace(group.Description)
	return group
}

func normalizeIntent(intent RoutingIntent) RoutingIntent {
	intent.ID = strings.TrimSpace(intent.ID)
	intent.Title = strings.TrimSpace(intent.Title)
	intent.Description = strings.TrimSpace(intent.Description)
	intent.DefaultGroup = strings.TrimSpace(intent.DefaultGroup)
	intent.Priority = strings.ToLower(strings.TrimSpace(intent.Priority))
	if intent.Priority == "normal" {
		intent.Priority = "medium"
	}

	examples := make([]string, 0, len(intent.Examples))
	for _, sample := range intent.Examples {
		sample = strings.TrimSpace(sample)
		if sample != "" {
			examples = append(examples, sample)
		}
	}
	intent.Examples = examples

	tags := make([]string, 0, len(intent.Tags))
	seenTags := map[string]struct{}{}
	for _, tag := range intent.Tags {
		tag = strings.TrimSpace(strings.ToLower(tag))
		if tag == "" {
			continue
		}
		if _, exists := seenTags[tag]; exists {
			continue
		}
		seenTags[tag] = struct{}{}
		tags = append(tags, tag)
	}
	sort.Strings(tags)
	intent.Tags = tags

	keywords := make([]string, 0, len(intent.Keywords))
	seenKeywords := map[string]struct{}{}
	for _, keyword := range intent.Keywords {
		keyword = strings.TrimSpace(strings.ToLower(keyword))
		if keyword == "" {
			continue
		}
		if _, exists := seenKeywords[keyword]; exists {
			continue
		}
		seenKeywords[keyword] = struct{}{}
		keywords = append(keywords, keyword)
	}
	sort.Strings(keywords)
	intent.Keywords = keywords

	return intent
}

func normalizeExampleText(value string) string {
	parts := strings.Fields(strings.TrimSpace(value))
	return strings.Join(parts, " ")
}

func readJSONFile(path string, out any) error {
	payload, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(payload, out); err != nil {
		return err
	}
	return nil
}

func writeJSONFileAtomic(path string, payload any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	body, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	body = append(body, '\n')

	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, body, 0644); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}
