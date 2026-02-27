package services

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"
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

type AuditEventRecord struct {
	ID            int64                  `json:"id"`
	CreatedAt     time.Time              `json:"created_at"`
	RequestID     string                 `json:"request_id"`
	ActorUserID   *int64                 `json:"actor_user_id,omitempty"`
	ActorUsername string                 `json:"actor_username"`
	ActorRole     string                 `json:"actor_role"`
	EventType     string                 `json:"event_type"`
	ResourceType  string                 `json:"resource_type"`
	ResourceID    string                 `json:"resource_id"`
	Outcome       string                 `json:"outcome"`
	Details       map[string]interface{} `json:"details"`
	IPAddress     string                 `json:"ip_address"`
	UserAgent     string                 `json:"user_agent"`
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

func (s *AuditService) ListEvents(limit, offset int, eventType, outcome, resourceType string) ([]AuditEventRecord, error) {
	if limit <= 0 {
		limit = 100
	}
	if limit > 500 {
		limit = 500
	}
	if offset < 0 {
		offset = 0
	}

	query := `
		SELECT
			id,
			created_at,
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
		FROM audit_events
		WHERE ($1 = '' OR event_type = $1)
		  AND ($2 = '' OR outcome = $2)
		  AND ($3 = '' OR resource_type = $3)
		ORDER BY id DESC
		LIMIT $4 OFFSET $5
	`

	rows, err := s.db.Query(query, eventType, outcome, resourceType, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("query audit events: %w", err)
	}
	defer rows.Close()

	events := make([]AuditEventRecord, 0, limit)
	for rows.Next() {
		var rec AuditEventRecord
		var actorID sql.NullInt64
		var rawDetails []byte
		if err := rows.Scan(
			&rec.ID,
			&rec.CreatedAt,
			&rec.RequestID,
			&actorID,
			&rec.ActorUsername,
			&rec.ActorRole,
			&rec.EventType,
			&rec.ResourceType,
			&rec.ResourceID,
			&rec.Outcome,
			&rawDetails,
			&rec.IPAddress,
			&rec.UserAgent,
		); err != nil {
			return nil, fmt.Errorf("scan audit event: %w", err)
		}
		if actorID.Valid {
			value := actorID.Int64
			rec.ActorUserID = &value
		}
		rec.Details = map[string]interface{}{}
		if len(rawDetails) > 0 {
			_ = json.Unmarshal(rawDetails, &rec.Details)
		}
		events = append(events, rec)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate audit events: %w", err)
	}

	return events, nil
}
