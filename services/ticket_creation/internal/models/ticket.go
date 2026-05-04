// internal/models/ticket.go
package models

import "time"

// Segment представляет сегмент диалога из транскрипции
type Segment struct {

	// Время начала сегмента в секундах
	Start   float64 `json:"start"`

	// Время окончания сегмента
	End     float64 `json:"end"`

	// Техническая метка говорящего, например Speaker 1.
	Speaker string  `json:"speaker"`

	// Роль участника, например client или operator, если она определена.
	Role    string  `json:"role"`

	// Текст сегмента
	Text    string  `json:"text"`
}

// TranscriptData данные транскрипции
type TranscriptData struct {

	// Идентификатор звонка
	CallID      string                 `json:"call_id"`

	// Список речевых сегментов
	Segments    []Segment              `json:"segments"`

	// Сопоставление говорящих с ролями
	RoleMapping map[string]string      `json:"role_mapping"`

	// Служебные метаданные транскрибации: режим ASR, время обработки, backend и так далее.
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// RoutingData данные маршрутизации
type RoutingData struct {

	// Идентификатор цели звонка
	IntentID         string  `json:"intent_id"`

	// Уверенность классификатора
	IntentConfidence float64 `json:"intent_confidence"`

	// Приоритет обращения 
	Priority         string  `json:"priority"`

	// Рекомендуемая группа исполнителей
	SuggestedGroup   string  `json:"suggested_group,omitempty"`
}

// ExtractedEntity извлеченная сущность
type ExtractedEntity struct {

	// тип сущности: person, phone, email, order_id, date и так далее.
	Type       string  `json:"type"`

	// Найденное значение
	Value      string  `json:"value"`

	// Уверенность извлечения
	Confidence float64 `json:"confidence"`

	// Фрагмент текста, где была найдена сущность
	Context    string  `json:"context"`
}

// Entities все извлеченные сущности
type Entities struct {

	// Персоны
	Persons      []ExtractedEntity `json:"persons"`

	// Телефоны
	Phones       []ExtractedEntity `json:"phones"`

	// Адреса email
	Emails       []ExtractedEntity `json:"emails"`

	// Номера заказов
	OrderIDs     []ExtractedEntity `json:"order_ids"`

	// Идентификаторы пользователей
	AccountIDs   []ExtractedEntity `json:"account_ids"`

	// Денежные суммы
	MoneyAmounts []ExtractedEntity `json:"money_amounts"`

	// Даты
	Dates        []ExtractedEntity `json:"dates"`
}

// TicketSummary сгенерированное описание тикета
type TicketSummary struct {

	// Краткий заголовок заявки
	Title             string   `json:"title"`

	// Полное описание проблемы
	Description       string   `json:"description"`

	// Ключевые пункты обращения
	KeyPoints         []string `json:"key_points"`

	// Предложенное решение, если оно сформировано
	SuggestedSolution string   `json:"suggested_solution,omitempty"`

	// Обоснование срочности, если оно есть
	UrgencyReason     string   `json:"urgency_reason,omitempty"`
}

type TicketServiceMetadata struct {

	// Источник запроса, например call-processing
	Source        string    `json:"source"`

	// Компонент, который отправил запрос
	Component     string    `json:"component"`

	// Версия схемы payload 
	SchemaVersion string    `json:"schema_version"`

	// Время отправки запроса
	SentAt        time.Time `json:"sent_at"`

	// Идентификатор звонка
	CallID        string    `json:"call_id"`

	// Цель обращения
	IntentID      string    `json:"intent_id,omitempty"`

	// Приоритет обращения
	Priority      string    `json:"priority,omitempty"`
}

type TicketSystemPayload struct {

	// Служебные метаданные: источник, компонент, версия схемы, время отправки, call_id, интент и приоритет.
	Service TicketServiceMetadata `json:"service"`

	// Исходный запрос на создание тикета. В нём хранятся транскрипция, маршрутизация, сущности и ссылка на аудио.
	Request CreateTicketRequest   `json:"request"`

	// Сгенерированное LLM-описание заявки
	Summary *TicketSummary        `json:"summary,omitempty"`

	// Готовый черновик тикета
	Draft   *TicketDraft          `json:"draft"`
}

// TicketDraft черновик тикета
type TicketDraft struct {

	// Заголовок заявки
	Title        string                 `json:"title"`

	// Основное описание заявки
	Description  string                 `json:"description"`

	// Приоритет заявки
	Priority     string                 `json:"priority"`

	// Тип назначения заявки
	AssigneeType string                 `json:"assignee_type"`

	// Идентификатор исполнителя, группы или очереди
	AssigneeID   string                 `json:"assignee_id"`

	// Теги заявки
	Tags         []string               `json:"tags"`

	// Дополнительные поля для внешней тикет-системы
	CustomFields map[string]interface{} `json:"custom_fields,omitempty"`

	// Связи
	// Идентификатор исходного звонка
	CallID        string `json:"call_id,omitempty"`

	// Ссылка на аудиозапись звонка
	AudioURL      string `json:"audio_url,omitempty"`

	// Ссылка на сохраненную транскрипцию
	TranscriptURL string `json:"transcript_url,omitempty"`

	// Метаданные
	// Цель обращения, определенная маршрутизатором
	IntentID         string    `json:"intent_id,omitempty"`

	// Уверенность модели в выбранной цели
	IntentConfidence float64   `json:"intent_confidence,omitempty"`

	// Извлеченные сущности
	Entities         *Entities `json:"entities,omitempty"`
}

// TicketCreated результат создания тикета
type TicketCreated struct {

	// Внутренний идентификатор тикета в текущей подсистеме
	TicketID   string    `json:"ticket_id"`

	// Внутренний идентификатор тикета во внешней тикет-системе
	ExternalID string    `json:"external_id"`

	// Ссылка на созданную заявку
	URL        string    `json:"url"`

	// Название системы, где создан тикет
	System     string    `json:"system"`

	// Время создания тикета
	CreatedAt  time.Time `json:"created_at"`
}

// TicketRecord запись в БД
type TicketRecord struct {

	// Внутренний числовой ID записи в базе
	ID               int64     `db:"id"`

	// Внутренний идентификатор тикета
	TicketID         string    `db:"ticket_id"`

	// ID во внешней тикет-системе
	ExternalID       string    `db:"external_id"`

	// Идентификатор звонка, по которому был создан тикет
	CallID           string    `db:"call_id"`

	// Основные поля заявки: заголовок, описание, приоритет и статус
	Title            string    `db:"title"`
	Description      string    `db:"description"`
	Priority         string    `db:"priority"`
	Status           string    `db:"status"`

	// Данные о назначении заявки
	AssigneeType     string    `db:"assignee_type"`
	AssigneeID       string    `db:"assignee_id"`

	// Цель звонка и уверенность модели
	IntentID         *string   `db:"intent_id"`
	IntentConfidence *float64  `db:"intent_confidence"`

	// Извлечённые сущности, сохранённые в виде JSON-строки
	EntitiesJSON     string    `db:"entities_json"`

	// Ссылка на тикет
	URL              *string   `db:"url"`

	// Система, в которой создан тикет
	System           string    `db:"system"`

	// Время создания и последнего обновления записи
	CreatedAt        time.Time `db:"created_at"`
	UpdatedAt        time.Time `db:"updated_at"`
}

// CreateTicketRequest запрос на создание тикета
type CreateTicketRequest struct {

	// Результат транскрибации: call_id, сегменты, роли, метаданные
	Transcript TranscriptData `json:"transcript"`

	// Результат маршрутизации: цель обращения, уверенность, приоритет, группа назначения
	Routing    RoutingData    `json:"routing"`

	// Извлечённые сущности
	Entities   *Entities      `json:"entities,omitempty"`

	// Ссылка на аудиозапись
	AudioURL   string         `json:"audio_url,omitempty"`
}

// CreateTicketResponse ответ на создание тикета
type CreateTicketResponse struct {

	// Признак успешного создания
	Success bool           `json:"success"`

	// Данные созданного тикета
	Ticket  *TicketCreated `json:"ticket,omitempty"`

	// Текст ошибки, если создание не удалось
	Error   string         `json:"error,omitempty"`
}
