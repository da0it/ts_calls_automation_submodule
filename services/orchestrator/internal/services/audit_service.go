package services

import (
	"database/sql"
	"encoding/json"
	"fmt"
)

type AuditEvent struct {
	RequestID     string
	ActorUserID   *int64
	ActorUsername string
	ActorRole     string
	EventType     string
	ResourceType  string
	ResourceID    string
	Outcome       string
	Details       map[string]interface{}
	IPAddress     string
	UserAgent     string
}

type AuditService struct {
	db *sql.DB
}

func NewAuditService(db *sql.DB) *AuditService {
	return &AuditService{db: db}
}

func (s *AuditService) Migrate() error {
	query := `
	CREATE TABLE IF NOT EXISTS audit_events (
		id BIGSERIAL PRIMARY KEY,
		created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
		request_id VARCHAR(128) NOT NULL DEFAULT '',
		actor_user_id BIGINT NULL,
		actor_username VARCHAR(128) NOT NULL DEFAULT '',
		actor_role VARCHAR(32) NOT NULL DEFAULT '',
		event_type VARCHAR(128) NOT NULL,
		resource_type VARCHAR(64) NOT NULL DEFAULT '',
		resource_id VARCHAR(256) NOT NULL DEFAULT '',
		outcome VARCHAR(32) NOT NULL DEFAULT 'success',
		details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
		ip_address VARCHAR(64) NOT NULL DEFAULT '',
		user_agent TEXT NOT NULL DEFAULT ''
	);
	CREATE INDEX IF NOT EXISTS idx_audit_events_created_at ON audit_events (created_at DESC);
	CREATE INDEX IF NOT EXISTS idx_audit_events_event_type ON audit_events (event_type);
	CREATE INDEX IF NOT EXISTS idx_audit_events_actor_user_id ON audit_events (actor_user_id);
	CREATE INDEX IF NOT EXISTS idx_audit_events_request_id ON audit_events (request_id);
	`
	_, err := s.db.Exec(query)
	return err
}

func (s *AuditService) LogEvent(event AuditEvent) error {
	if event.EventType == "" {
		return fmt.Errorf("event_type is required")
	}
	if event.Outcome == "" {
		event.Outcome = "success"
	}
	if event.Details == nil {
		event.Details = map[string]interface{}{}
	}

	detailsRaw, err := json.Marshal(event.Details)
	if err != nil {
		return fmt.Errorf("marshal audit details: %w", err)
	}

	_, err = s.db.Exec(
		`INSERT INTO audit_events (
			request_id,
			actor_user_id,
			actor_username,
			actor_role,
			event_type,
			resource_type,
			resource_id,
			outcome,
			details_json,
			ip_address,
			user_agent
		) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11)`,
		event.RequestID,
		event.ActorUserID,
		event.ActorUsername,
		event.ActorRole,
		event.EventType,
		event.ResourceType,
		event.ResourceID,
		event.Outcome,
		string(detailsRaw),
		event.IPAddress,
		event.UserAgent,
	)
	if err != nil {
		return fmt.Errorf("insert audit event: %w", err)
	}
	return nil
}
