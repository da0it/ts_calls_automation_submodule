package services

import (
	"database/sql"
	"errors"
	"log"
	"time"

	"golang.org/x/crypto/bcrypt"
	"orchestrator/internal/models"
)

type UserService struct {
	db *sql.DB
}

var (
	ErrInvalidCredentials = errors.New("invalid credentials")
	ErrAccountPending     = errors.New("account pending approval")
	ErrAccountInactive    = errors.New("account inactive")
)

func NewUserService(db *sql.DB) *UserService {
	return &UserService{db: db}
}

// Migrate creates the users table if it does not exist.
func (s *UserService) Migrate() error {
	query := `
	CREATE TABLE IF NOT EXISTS users (
		id          BIGSERIAL    PRIMARY KEY,
		username    VARCHAR(64)  NOT NULL UNIQUE,
		password    VARCHAR(128) NOT NULL,
		role        VARCHAR(16)  NOT NULL DEFAULT 'operator'
		            CHECK (role IN ('operator', 'admin')),
		is_active   BOOLEAN      NOT NULL DEFAULT TRUE,
		is_approved BOOLEAN      NOT NULL DEFAULT TRUE,
		created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
		updated_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
	);
	ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;
	ALTER TABLE users ADD COLUMN IF NOT EXISTS is_approved BOOLEAN;
	UPDATE users
	SET is_approved = CASE
		WHEN role = 'operator' AND is_active = FALSE THEN FALSE
		ELSE TRUE
	END
	WHERE is_approved IS NULL;
	ALTER TABLE users ALTER COLUMN is_approved SET NOT NULL;
	ALTER TABLE users ALTER COLUMN is_approved SET DEFAULT TRUE;
	CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
	`
	_, err := s.db.Exec(query)
	return err
}

// SeedAdmin creates the admin user if it does not already exist.
func (s *UserService) SeedAdmin(username, password string) error {
	if password == "" {
		return nil
	}
	existing, err := s.GetByUsername(username)
	if err != nil {
		return err
	}
	if existing != nil {
		log.Printf("Admin user '%s' already exists, skipping seed", username)
		return nil
	}
	_, err = s.create(username, password, models.RoleAdmin, true, true)
	if err != nil {
		return err
	}
	log.Printf("✓ Admin user '%s' seeded", username)
	return nil
}

// Authenticate verifies credentials and returns the user.
func (s *UserService) Authenticate(username, password string) (*models.User, error) {
	user, err := s.GetByUsername(username)
	if err != nil {
		return nil, err
	}
	if user == nil {
		return nil, ErrInvalidCredentials
	}
	if err := bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(password)); err != nil {
		return nil, ErrInvalidCredentials
	}
	if !user.IsApproved {
		return nil, ErrAccountPending
	}
	if !user.IsActive {
		return nil, ErrAccountInactive
	}
	return user, nil
}

// GetByID returns a user by ID, or nil if not found.
func (s *UserService) GetByID(id int64) (*models.User, error) {
	var u models.User
	err := s.db.QueryRow(
		`SELECT id, username, password, role, is_active, is_approved, created_at, updated_at FROM users WHERE id = $1`,
		id,
	).Scan(&u.ID, &u.Username, &u.Password, &u.Role, &u.IsActive, &u.IsApproved, &u.CreatedAt, &u.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &u, nil
}

// GetByUsername returns a user by username, or nil if not found.
func (s *UserService) GetByUsername(username string) (*models.User, error) {
	var u models.User
	err := s.db.QueryRow(
		`SELECT id, username, password, role, is_active, is_approved, created_at, updated_at FROM users WHERE username = $1`,
		username,
	).Scan(&u.ID, &u.Username, &u.Password, &u.Role, &u.IsActive, &u.IsApproved, &u.CreatedAt, &u.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &u, nil
}

// Create inserts a new user with a bcrypt-hashed password.
func (s *UserService) Create(username, password string, role models.Role) (*models.User, error) {
	return s.create(username, password, role, true, true)
}

// RegisterOperator creates a new operator account in inactive state.
func (s *UserService) RegisterOperator(username, password string) (*models.User, error) {
	return s.create(username, password, models.RoleOperator, false, false)
}

func (s *UserService) create(username, password string, role models.Role, isActive bool, isApproved bool) (*models.User, error) {
	if len(username) < 3 || len(username) > 64 {
		return nil, errors.New("username must be 3-64 characters")
	}
	if len(password) < 6 {
		return nil, errors.New("password must be at least 6 characters")
	}
	if role != models.RoleOperator && role != models.RoleAdmin {
		return nil, errors.New("role must be 'operator' or 'admin'")
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, err
	}

	var u models.User
	err = s.db.QueryRow(
		`INSERT INTO users (username, password, role, is_active, is_approved) VALUES ($1, $2, $3, $4, $5)
		 RETURNING id, username, password, role, is_active, is_approved, created_at, updated_at`,
		username, string(hash), string(role), isActive, isApproved,
	).Scan(&u.ID, &u.Username, &u.Password, &u.Role, &u.IsActive, &u.IsApproved, &u.CreatedAt, &u.UpdatedAt)
	if err != nil {
		return nil, err
	}
	return &u, nil
}

// List returns all users.
func (s *UserService) List() ([]models.User, error) {
	rows, err := s.db.Query(
		`SELECT id, username, role, is_active, is_approved, created_at, updated_at FROM users ORDER BY id`,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var users []models.User
	for rows.Next() {
		var u models.User
		if err := rows.Scan(&u.ID, &u.Username, &u.Role, &u.IsActive, &u.IsApproved, &u.CreatedAt, &u.UpdatedAt); err != nil {
			return nil, err
		}
		users = append(users, u)
	}
	return users, rows.Err()
}

// ApproveOperator activates a pending operator account.
func (s *UserService) ApproveOperator(id int64) (*models.User, error) {
	user, err := s.GetByID(id)
	if err != nil {
		return nil, err
	}
	if user == nil {
		return nil, errors.New("user not found")
	}
	if user.Role != models.RoleOperator {
		return nil, errors.New("only operator accounts can be approved")
	}
	if user.IsActive && user.IsApproved {
		return user, nil
	}

	var updated models.User
	err = s.db.QueryRow(
		`UPDATE users
		 SET is_active = TRUE, is_approved = TRUE, updated_at = $2
		 WHERE id = $1
		 RETURNING id, username, password, role, is_active, is_approved, created_at, updated_at`,
		id, time.Now(),
	).Scan(
		&updated.ID,
		&updated.Username,
		&updated.Password,
		&updated.Role,
		&updated.IsActive,
		&updated.IsApproved,
		&updated.CreatedAt,
		&updated.UpdatedAt,
	)
	if err != nil {
		return nil, err
	}
	return &updated, nil
}

// DeactivateOperator marks an approved operator account as inactive.
func (s *UserService) DeactivateOperator(id int64) (*models.User, error) {
	user, err := s.GetByID(id)
	if err != nil {
		return nil, err
	}
	if user == nil {
		return nil, errors.New("user not found")
	}
	if user.Role != models.RoleOperator {
		return nil, errors.New("only operator accounts can be deactivated")
	}
	if !user.IsApproved {
		return nil, errors.New("pending operator cannot be deactivated")
	}
	if !user.IsActive {
		return user, nil
	}

	var updated models.User
	err = s.db.QueryRow(
		`UPDATE users
		 SET is_active = FALSE, updated_at = $2
		 WHERE id = $1
		 RETURNING id, username, password, role, is_active, is_approved, created_at, updated_at`,
		id, time.Now(),
	).Scan(
		&updated.ID,
		&updated.Username,
		&updated.Password,
		&updated.Role,
		&updated.IsActive,
		&updated.IsApproved,
		&updated.CreatedAt,
		&updated.UpdatedAt,
	)
	if err != nil {
		return nil, err
	}
	return &updated, nil
}

// Delete removes a user by ID. Prevents deleting the last admin.
func (s *UserService) Delete(id int64, currentUserID int64) error {
	if id == currentUserID {
		return errors.New("cannot delete yourself")
	}

	user, err := s.GetByID(id)
	if err != nil {
		return err
	}
	if user == nil {
		return errors.New("user not found")
	}

	if user.Role == models.RoleAdmin {
		var count int
		err := s.db.QueryRow(`SELECT COUNT(*) FROM users WHERE role = 'admin'`).Scan(&count)
		if err != nil {
			return err
		}
		if count <= 1 {
			return errors.New("cannot delete the last admin")
		}
	}

	_, err = s.db.Exec(`DELETE FROM users WHERE id = $1`, id)
	return err
}

// UpdatePassword changes a user's password.
func (s *UserService) UpdatePassword(id int64, newPassword string) error {
	if len(newPassword) < 6 {
		return errors.New("password must be at least 6 characters")
	}
	hash, err := bcrypt.GenerateFromPassword([]byte(newPassword), bcrypt.DefaultCost)
	if err != nil {
		return err
	}
	_, err = s.db.Exec(
		`UPDATE users SET password = $1, updated_at = $2 WHERE id = $3`,
		string(hash), time.Now(), id,
	)
	return err
}
