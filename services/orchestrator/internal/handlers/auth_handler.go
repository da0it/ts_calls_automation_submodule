package handlers

import (
	"errors"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"orchestrator/internal/middleware"
	"orchestrator/internal/models"
	"orchestrator/internal/services"
)

type AuthHandler struct {
	userService *services.UserService
	jwtSecret   string
	jwtExpiry   int
}

func NewAuthHandler(userService *services.UserService, jwtSecret string, jwtExpiry int) *AuthHandler {
	return &AuthHandler{
		userService: userService,
		jwtSecret:   jwtSecret,
		jwtExpiry:   jwtExpiry,
	}
}

type loginRequest struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

type registerRequest struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

// Login authenticates a user and returns a JWT token.
func (h *AuthHandler) Login(c *gin.Context) {
	var req loginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "username and password are required"})
		return
	}

	user, err := h.userService.Authenticate(req.Username, req.Password)
	if err != nil {
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

	token, err := middleware.GenerateToken(user, h.jwtSecret, h.jwtExpiry)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to generate token"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"token": token,
		"user":  user,
	})
}

// Register creates a new operator account in pending state.
func (h *AuthHandler) Register(c *gin.Context) {
	var req registerRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "username and password are required"})
		return
	}

	user, err := h.userService.RegisterOperator(req.Username, req.Password)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"message": "registration submitted, wait for admin approval",
		"user":    user,
	})
}

// Me returns the currently authenticated user.
func (h *AuthHandler) Me(c *gin.Context) {
	user := c.MustGet("user").(*models.User)
	c.JSON(http.StatusOK, user)
}

type createUserRequest struct {
	Username string      `json:"username" binding:"required"`
	Password string      `json:"password" binding:"required"`
	Role     models.Role `json:"role" binding:"required"`
}

type approveUserResponse struct {
	Message string       `json:"message"`
	User    *models.User `json:"user"`
}

// ListUsers returns all users (admin only).
func (h *AuthHandler) ListUsers(c *gin.Context) {
	users, err := h.userService.List()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"users": users})
}

// CreateUser creates a new user (admin only).
func (h *AuthHandler) CreateUser(c *gin.Context) {
	var req createUserRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "username, password, and role are required"})
		return
	}

	user, err := h.userService.Create(req.Username, req.Password, req.Role)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, user)
}

// ApproveUser activates a pending operator account (admin only).
func (h *AuthHandler) ApproveUser(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid user id"})
		return
	}

	user, err := h.userService.ApproveOperator(id)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, approveUserResponse{
		Message: "user approved",
		User:    user,
	})
}

// DeactivateUser marks an approved operator account as inactive (admin only).
func (h *AuthHandler) DeactivateUser(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid user id"})
		return
	}

	user, err := h.userService.DeactivateOperator(id)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, approveUserResponse{
		Message: "user deactivated",
		User:    user,
	})
}

// DeleteUser deletes a user by ID (admin only).
func (h *AuthHandler) DeleteUser(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid user id"})
		return
	}

	currentUser := c.MustGet("user").(*models.User)
	if err := h.userService.Delete(id, currentUser.ID); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "user deleted"})
}
