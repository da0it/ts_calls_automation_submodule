package services

import (
	"database/sql"
	"errors"
	"log"
	"time"

	"orchestrator/internal/models"

	"golang.org/x/crypto/bcrypt"
)

// Структура, отвечающая за хранение подключения к базе данных
type UserService struct {
	db *sql.DB
}

// Ошибки сервиса
var (
	ErrInvalidCredentials = errors.New("invalid credentials")
	ErrAccountPending     = errors.New("account pending approval")
	ErrAccountInactive    = errors.New("account inactive")
)

// Конструктор сервиса пользователей. Принимает подключение и возвращает готовый сервис
func NewUserService(db *sql.DB) *UserService {
	return &UserService{db: db}
}

// Миграция создает таблицу пользователей, если она еще не создана
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

	// Выполнение SQL-запроса
	_, err := s.db.Exec(query)
	return err
}

// Функция создает пользователя-админа, если он не существует в таблице
func (s *UserService) SeedAdmin(username, password string) error {
	if password == "" {
		return nil
	}

	// Поиск пользователя по юзернейму.
	existing, err := s.GetByUsername(username)
	if err != nil {
		return err
	}
	if existing != nil {
		log.Printf("Admin user '%s' already exists, skipping seed", username)
		return nil
	}

	// Если пользователь не найден, он создается с переданным юзернеймом и паролем.
	_, err = s.create(username, password, models.RoleAdmin, true, true)
	if err != nil {
		return err
	}
	log.Printf("✓ Admin user '%s' seeded", username)
	return nil
}

// Функция проверяет введенные юзернейм и пароль, и возвращает пользователя
func (s *UserService) Authenticate(username, password string) (*models.User, error) {
	// Поиск пользователя в таблице
	user, err := s.GetByUsername(username)
	if err != nil {
		return nil, err
	}
	if user == nil {
		return nil, ErrInvalidCredentials
	}

	// Сравнение хэшей введенного пароля и сохраненного хэша пароля в базе данных
	if err := bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(password)); err != nil {
		return nil, ErrInvalidCredentials
	}

	// Проверка, одобрен ли пользователь администратором (без этого оператор не получит доступ к системе)
	if !user.IsApproved {
		return nil, ErrAccountPending
	}

	// Проверка активирован ли пользовательский аккаунт. Не активные аккаунты не имеют доступа к системе.
	if !user.IsActive {
		return nil, ErrAccountInactive
	}
	return user, nil
}

// Функция возвращает пользователя по введенному идентификатору, или nil если он не найден
func (s *UserService) GetByID(id int64) (*models.User, error) {
	var u models.User

	// Выполнение SQL-запроса
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

// Функция возвращает пользователя по введенному юзернейму, или nil если пользователь не найден.
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
// Вставка нового пользователя в базу данных сразу с активным и подтвержденным статусом
func (s *UserService) Create(username, password string, role models.Role) (*models.User, error) {
	return s.create(username, password, role, true, true)
}

// Метод используется для самостоятельной регистрации оператора
func (s *UserService) RegisterOperator(username, password string) (*models.User, error) {
	return s.create(username, password, models.RoleOperator, false, false)
}

// Общий внутренний метод создания пользователя.
func (s *UserService) create(username, password string, role models.Role, isActive bool, isApproved bool) (*models.User, error) {

	// Блок валидации имени пользователя, пароля, и роли в системе
	if len(username) < 3 || len(username) > 64 {
		return nil, errors.New("username must be 3-64 characters")
	}
	if len(password) < 6 {
		return nil, errors.New("password must be at least 6 characters")
	}
	if role != models.RoleOperator && role != models.RoleAdmin {
		return nil, errors.New("role must be 'operator' or 'admin'")
	}

	// Преобразование пароля в bcrypt хэш. В базу данных далее сохраняется именно хэш
	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, err
	}

	// Вставка пользователя
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

// Функция возвращает список всех пользователей
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

// Функция активирует оператора из статуса ожидания
func (s *UserService) ApproveOperator(id int64) (*models.User, error) {

	// Поиск пользователя по идентификатору
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

	// Обновление статуса пользователя
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

// Функция для изменения статуса аккаунта подтвержденного оператора в неактивное состояние
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

// Удаляет пользователя. Принимает идентификатор пользователя и ID текущего пользователя
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

// Функция позволяет изменить пароль пользователя
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
