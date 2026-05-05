package handlers

import (
	"errors"
	"log"
	"net/http"
	"strconv"

	"orchestrator/internal/middleware"
	"orchestrator/internal/models"
	"orchestrator/internal/services"

	"github.com/gin-gonic/gin"
)

// Структура хранит зависимости для работы с пользователями
type AuthHandler struct {

	// Сервис пользователей.
	userService *services.UserService

	// Секретный ключ для подписи JWT-токена
	jwtSecret string

	// Время жизни JWT-токена
	jwtExpiry int

	// Сервис аудита
	auditService *services.AuditService
}

// Функция-конструктор. Создает новый хендлер аутентификации
func NewAuthHandler(userService *services.UserService, jwtSecret string, jwtExpiry int, auditService *services.AuditService) *AuthHandler {
	return &AuthHandler{
		userService:  userService,
		jwtSecret:    jwtSecret,
		jwtExpiry:    jwtExpiry,
		auditService: auditService,
	}
}

// Структура тела запроса для входа пользователя
type loginRequest struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

// Структура тела запроса для регистрации пользователя
type registerRequest struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

// Вспомогательный метод для записи событий аудита
func (h *AuthHandler) writeAudit(
	c *gin.Context,
	actor *models.User,
	eventType string,
	resourceType string,
	resourceID string,
	outcome string,
	details map[string]interface{},
) {
	if h.auditService == nil {
		return
	}
	var actorUserID *int64
	actorUsername := ""
	actorRole := ""
	if actor != nil {
		actorUserID = &actor.ID
		actorUsername = actor.Username
		actorRole = string(actor.Role)
	}
	if err := h.auditService.LogEvent(services.AuditEvent{
		RequestID:     c.GetString("request_id"),
		ActorUserID:   actorUserID,
		ActorUsername: actorUsername,
		ActorRole:     actorRole,
		EventType:     eventType,
		ResourceType:  resourceType,
		ResourceID:    resourceID,
		Outcome:       outcome,
		Details:       details,
		IPAddress:     c.ClientIP(),
		UserAgent:     c.GetHeader("User-Agent"),
	}); err != nil {
		log.Printf("Failed to write audit event (%s): %v", eventType, err)
	}
}

// Функция Login выполняет аутентификацию пользователя и возвращает JWT-токен.
func (h *AuthHandler) Login(c *gin.Context) {
	var req loginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.writeAudit(c, nil, "auth.login", "auth", "", "failed", map[string]interface{}{
			"reason": "invalid_payload",
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": "username and password are required"})
		return
	}

	// Непосредственный метод аутентификации
	user, err := h.userService.Authenticate(req.Username, req.Password)
	if err != nil {
		h.writeAudit(c, nil, "auth.login", "user", req.Username, "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		if errors.Is(err, services.ErrAccountPending) {
			c.JSON(http.StatusForbidden, gin.H{"error": "account pending admin approval"})
			return
		}
		if errors.Is(err, services.ErrAccountInactive) {
			c.JSON(http.StatusForbidden, gin.H{"error": "account is inactive, contact admin"})
			return
		}
		if errors.Is(err, services.ErrInvalidCredentials) {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "invalid credentials"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to authenticate"})
		return
	}

	// Генерация JWT
	token, err := middleware.GenerateToken(user, h.jwtSecret, h.jwtExpiry)
	if err != nil {
		h.writeAudit(c, user, "auth.login", "user", user.Username, "failed", map[string]interface{}{
			"reason": "token_generation_failed",
		})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to generate token"})
		return
	}
	h.writeAudit(c, user, "auth.login", "user", user.Username, "success", map[string]interface{}{})

	c.JSON(http.StatusOK, gin.H{
		"token": token,
		"user":  user,
	})
}

// Функция register регистрирует нового оператора.
// То есть создается заявка на регистрацию, которая ожидает подтверждения администратора.
func (h *AuthHandler) Register(c *gin.Context) {
	var req registerRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.writeAudit(c, nil, "auth.register", "user", "", "failed", map[string]interface{}{
			"reason": "invalid_payload",
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": "username and password are required"})
		return
	}

	// Вызов метода регистрации сервиса по управлению пользователями
	user, err := h.userService.RegisterOperator(req.Username, req.Password)
	if err != nil {
		h.writeAudit(c, nil, "auth.register", "user", req.Username, "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	h.writeAudit(c, user, "auth.register", "user", user.Username, "success", map[string]interface{}{
		"is_approved": user.IsApproved,
		"is_active":   user.IsActive,
	})

	c.JSON(http.StatusCreated, gin.H{
		"message": "registration submitted, wait for admin approval",
		"user":    user,
	})
}

// Функция Me возвращает текущего авторизованного пользователя.
func (h *AuthHandler) Me(c *gin.Context) {
	user := c.MustGet("user").(*models.User)
	c.JSON(http.StatusOK, user)
}

// Структура для создания пользователя администратором.
type createUserRequest struct {
	Username string      `json:"username" binding:"required"`
	Password string      `json:"password" binding:"required"`
	Role     models.Role `json:"role" binding:"required"`
}

// Структура ответа для операций подтверждения и деактивации пользователя
type approveUserResponse struct {
	Message string       `json:"message"`
	User    *models.User `json:"user"`
}

// Метод возвращает список всех пользователей. Админский метод
func (h *AuthHandler) ListUsers(c *gin.Context) {
	users, err := h.userService.List()
	if err != nil {
		actor := c.MustGet("user").(*models.User)
		h.writeAudit(c, actor, "admin.users.list", "user", "", "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	actor := c.MustGet("user").(*models.User)
	h.writeAudit(c, actor, "admin.users.list", "user", "", "success", map[string]interface{}{
		"users_count": len(users),
	})
	c.JSON(http.StatusOK, gin.H{"users": users})
}

// Метод создает нового пользователя от имени администратора
func (h *AuthHandler) CreateUser(c *gin.Context) {
	var req createUserRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "username, password, and role are required"})
		return
	}

	user, err := h.userService.Create(req.Username, req.Password, req.Role)
	if err != nil {
		actor := c.MustGet("user").(*models.User)
		h.writeAudit(c, actor, "admin.users.create", "user", req.Username, "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	actor := c.MustGet("user").(*models.User)
	h.writeAudit(c, actor, "admin.users.create", "user", user.Username, "success", map[string]interface{}{
		"role": user.Role,
	})

	c.JSON(http.StatusCreated, user)
}

// Метод подтверждает пользователя, который был зарегистрирован и ожидает одобрения администратора.
func (h *AuthHandler) ApproveUser(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid user id"})
		return
	}

	user, err := h.userService.ApproveOperator(id)
	if err != nil {
		actor := c.MustGet("user").(*models.User)
		h.writeAudit(c, actor, "admin.users.approve", "user", idStr, "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	actor := c.MustGet("user").(*models.User)
	h.writeAudit(c, actor, "admin.users.approve", "user", idStr, "success", map[string]interface{}{
		"username": user.Username,
	})

	c.JSON(http.StatusOK, approveUserResponse{
		Message: "user approved",
		User:    user,
	})
}

// Метод деактивирует пользователя.
func (h *AuthHandler) DeactivateUser(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid user id"})
		return
	}

	user, err := h.userService.DeactivateOperator(id)
	if err != nil {
		actor := c.MustGet("user").(*models.User)
		h.writeAudit(c, actor, "admin.users.deactivate", "user", idStr, "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	actor := c.MustGet("user").(*models.User)
	h.writeAudit(c, actor, "admin.users.deactivate", "user", idStr, "success", map[string]interface{}{
		"username": user.Username,
	})

	c.JSON(http.StatusOK, approveUserResponse{
		Message: "user deactivated",
		User:    user,
	})
}

// Метод удаляет пользователя по ID.
func (h *AuthHandler) DeleteUser(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid user id"})
		return
	}

	currentUser := c.MustGet("user").(*models.User)
	if err := h.userService.Delete(id, currentUser.ID); err != nil {
		h.writeAudit(c, currentUser, "admin.users.delete", "user", idStr, "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	h.writeAudit(c, currentUser, "admin.users.delete", "user", idStr, "success", map[string]interface{}{})

	c.JSON(http.StatusOK, gin.H{"message": "user deleted"})
}
