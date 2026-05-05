package handlers

import (
	"context"

	callprocessingv1 "ticket_module/internal/gen"
	"ticket_module/internal/models"
	"ticket_module/internal/services"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// Структура gRPC-хандлера
type TicketGRPCHandler struct {

	// служебная встраиваемая структура, которую рекомендует добавлять gRPC, чтобы хандлер корректно реализовывал
	//  серверный интерфейс protobuf-сервиса.
	callprocessingv1.UnimplementedTicketServiceServer

	// Ссылка на сервисный слой.
	service *services.TicketCreatorService
}

// Функция-конструктор. Принимает сервис и возвращает новый gRPC-хандлер
func NewTicketGRPCHandler(service *services.TicketCreatorService) *TicketGRPCHandler {
	return &TicketGRPCHandler{service: service}
}

// Основной метод файла. Обрабатывает gRPC-запрос на создание тикета. Вызывается у объекта хендлера. Принимает входной protobuf-запрос,
// Возвращает protobuf-ответ CreateTicketResponse/ошибку error.
func (h *TicketGRPCHandler) CreateTicket(ctx context.Context, req *callprocessingv1.CreateTicketRequest) (*callprocessingv1.CreateTicketResponse, error) {
	if req.GetTranscript() == nil {
		return nil, status.Error(codes.InvalidArgument, "transcript is required")
	}
	if len(req.GetTranscript().GetSegments()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "segments are required")
	}

	// Создание среза внутренних моделей сегментов
	segments := make([]models.Segment, 0, len(req.GetTranscript().GetSegments()))

	// Преобразование сегмента из protobuf в model
	for _, seg := range req.GetTranscript().GetSegments() {
		segments = append(segments, models.Segment{
			Start:   seg.GetStart(),
			End:     seg.GetEnd(),
			Speaker: seg.GetSpeaker(),
			Role:    seg.GetRole(),
			Text:    seg.GetText(),
		})
	}

	// Обработка метаданных
	var metadata map[string]interface{}
	if req.GetTranscript().GetMetadata() != nil {
		metadata = req.GetTranscript().GetMetadata().AsMap()
	}

	// Сборка внутреннего сервиса CreateTicketRequest
	createReq := &models.CreateTicketRequest{

		// Заполнение транскрипции
		Transcript: models.TranscriptData{
			CallID:      req.GetTranscript().GetCallId(),
			Segments:    segments,
			RoleMapping: req.GetTranscript().GetRoleMapping(),
			Metadata:    metadata,
		},

		// Заполнение результата маршрутизации
		Routing: models.RoutingData{
			IntentID:         req.GetRouting().GetIntentId(),
			IntentConfidence: req.GetRouting().GetIntentConfidence(),
			Priority:         req.GetRouting().GetPriority(),
			SuggestedGroup:   req.GetRouting().GetSuggestedGroup(),
		},

		// Protobuf-сущности преобразуются во внутреннюю модель models.Entities
		Entities: entitiesFromProto(req.GetEntities()),
		AudioURL: req.GetAudioUrl(),
	}

	// Хандлер передаёт подготовленный запрос в сервисный слой
	created, err := h.service.CreateTicket(createReq)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "create ticket: %v", err)
	}

	// Если тикет успешно создан, хандлер возвращает gRPC-ответ.
	return &callprocessingv1.CreateTicketResponse{
		Ticket: &callprocessingv1.TicketCreated{
			TicketId:   created.TicketID,
			ExternalId: created.ExternalID,
			Url:        created.URL,
			System:     created.System,
			CreatedAt:  timestamppb.New(created.CreatedAt),
		},
	}, nil
}

// Функция преобразует protobuf-структуру сущностей во внутреннюю структуру *models.Entities
func entitiesFromProto(src *callprocessingv1.Entities) *models.Entities {
	if src == nil {
		return nil
	}

	return &models.Entities{
		Persons:      entityListFromProto(src.GetPersons()),
		Phones:       entityListFromProto(src.GetPhones()),
		Emails:       entityListFromProto(src.GetEmails()),
		OrderIDs:     entityListFromProto(src.GetOrderIds()),
		AccountIDs:   entityListFromProto(src.GetAccountIds()),
		MoneyAmounts: entityListFromProto(src.GetMoneyAmounts()),
		Dates:        entityListFromProto(src.GetDates()),
	}
}

// Функция преобразует список protobuf-сущностей в список внутренних сущностей
func entityListFromProto(src []*callprocessingv1.ExtractedEntity) []models.ExtractedEntity {
	out := make([]models.ExtractedEntity, 0, len(src))
	for _, item := range src {
		out = append(out, models.ExtractedEntity{
			Type:       item.GetType(),
			Value:      item.GetValue(),
			Confidence: item.GetConfidence(),
			Context:    item.GetContext(),
		})
	}
	return out
}
