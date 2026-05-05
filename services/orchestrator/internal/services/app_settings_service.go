package services

import (
	"database/sql"
	"fmt"
	"time"
)

// Предопределенные константы для соглашения об уровне сервиса (SLA)
const (
	defaultSLAMinutes = 15
	minSLAMinutes     = 1
	maxSLAMinutes     = 24 * 60
)

// Модель настроек приложения
type AppSettings struct {

	// Количество минут SLA
	SLAMinutes int `json:"sla_minutes"`

	// Дата и время последнего изменения настройки
	UpdatedAt time.Time `json:"updated_at"`
}

// Сервис хранит подключение к БД
type AppSettingsService struct {
	db *sql.DB
}

// Функция-конструктор. Создает сервис настроек приложения
func NewAppSettingsService(db *sql.DB) *AppSettingsService {
	return &AppSettingsService{db: db}
}

// Метод создаёт таблицу app_settings и начальную запись с настройками.
func (s *AppSettingsService) Migrate() error {
	query := `
	CREATE TABLE IF NOT EXISTS app_settings (
		id          BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id = TRUE),
		sla_minutes INTEGER NOT NULL DEFAULT 15,
		updated_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
	);
	INSERT INTO app_settings (id, sla_minutes, updated_at)
	VALUES (TRUE, 15, NOW())
	ON CONFLICT (id) DO NOTHING;
	UPDATE app_settings
	SET sla_minutes = 15
	WHERE sla_minutes IS NULL OR sla_minutes < 1;
	`
	_, err := s.db.Exec(query)
	return err
}

// Метод Get возвращает текущие настройки приложения
func (s *AppSettingsService) Get() (*AppSettings, error) {
	var settings AppSettings
	err := s.db.QueryRow(
		`SELECT sla_minutes, updated_at FROM app_settings WHERE id = TRUE`,
	).Scan(&settings.SLAMinutes, &settings.UpdatedAt)
	if err == sql.ErrNoRows {
		if _, insertErr := s.db.Exec(
			`INSERT INTO app_settings (id, sla_minutes, updated_at) VALUES (TRUE, $1, NOW())
			 ON CONFLICT (id) DO NOTHING`,
			defaultSLAMinutes,
		); insertErr != nil {
			return nil, insertErr
		}
		return s.Get()
	}
	if err != nil {
		return nil, err
	}
	if settings.SLAMinutes < minSLAMinutes {
		settings.SLAMinutes = defaultSLAMinutes
	}
	return &settings, nil
}

// Метод UpdateSLA обновляет значение SLA
func (s *AppSettingsService) UpdateSLA(minutes int) (*AppSettings, error) {
	if minutes < minSLAMinutes || minutes > maxSLAMinutes {
		return nil, fmt.Errorf("sla_minutes must be between %d and %d", minSLAMinutes, maxSLAMinutes)
	}

	var settings AppSettings
	err := s.db.QueryRow(
		`UPDATE app_settings
		 SET sla_minutes = $1, updated_at = NOW()
		 WHERE id = TRUE
		 RETURNING sla_minutes, updated_at`,
		minutes,
	).Scan(&settings.SLAMinutes, &settings.UpdatedAt)
	if err == sql.ErrNoRows {
		if _, insertErr := s.db.Exec(
			`INSERT INTO app_settings (id, sla_minutes, updated_at) VALUES (TRUE, $1, NOW())
			 ON CONFLICT (id) DO UPDATE SET sla_minutes = EXCLUDED.sla_minutes, updated_at = EXCLUDED.updated_at`,
			minutes,
		); insertErr != nil {
			return nil, insertErr
		}
		return s.Get()
	}
	if err != nil {
		return nil, err
	}
	return &settings, nil
}
