-- infra/postgres/init.sql
-- Инициализация базы данных при первом запуске контейнера

CREATE TABLE IF NOT EXISTS tickets (
    id              BIGSERIAL PRIMARY KEY,
    ticket_id       VARCHAR(64)  NOT NULL UNIQUE,
    external_id     VARCHAR(128) NOT NULL,
    call_id         VARCHAR(128) NOT NULL,
    title           TEXT         NOT NULL,
    description     TEXT         NOT NULL DEFAULT '',
    priority        VARCHAR(32)  NOT NULL DEFAULT 'medium',
    status          VARCHAR(32)  NOT NULL DEFAULT 'open',
    assignee_type   VARCHAR(32)  NOT NULL DEFAULT 'group',
    assignee_id     VARCHAR(128) NOT NULL DEFAULT 'default_support',
    intent_id       VARCHAR(128),
    intent_confidence DOUBLE PRECISION,
    entities_json   JSONB        NOT NULL DEFAULT '{}',
    url             VARCHAR(512),
    system          VARCHAR(32)  NOT NULL DEFAULT 'mock',
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Индексы для частых запросов
CREATE INDEX IF NOT EXISTS idx_tickets_call_id    ON tickets (call_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status     ON tickets (status);
CREATE INDEX IF NOT EXISTS idx_tickets_priority   ON tickets (priority);
CREATE INDEX IF NOT EXISTS idx_tickets_assignee   ON tickets (assignee_id);
CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tickets_intent_id  ON tickets (intent_id);
