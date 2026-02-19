package models

import "time"

type Role string

const (
	RoleOperator Role = "operator"
	RoleAdmin    Role = "admin"
)

type User struct {
	ID         int64     `json:"id"`
	Username   string    `json:"username"`
	Password   string    `json:"-"`
	Role       Role      `json:"role"`
	IsActive   bool      `json:"is_active"`
	IsApproved bool      `json:"is_approved"`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
}
