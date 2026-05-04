// internal/database/postgres.go
package database

import (
    "fmt"
    "time"

    // sqlx - библиотека-надстройка над стандартным database/sql
    "github.com/jmoiron/sqlx"
    _ "github.com/lib/pq"
)

// Database — это обёртка над объектом подключения к базе.
type Database struct {

    // Хранит подключение/пул соединений через sqlx 
    DB *sqlx.DB
}

// Конструктор подключения к базе данных. Принимает строку подключения и возвращает либо готовый объект Database либо ошибку.
func NewDatabase(databaseURL string) (*Database, error) {

    // Создание подключения к PostgreSQL
    db, err := sqlx.Connect("postgres", databaseURL)
    if err != nil {
        return nil, fmt.Errorf("connect to database: %w", err)
    }
    
    // Настройки пула соединений
    // Максимальное количество открытых соединений с БД
    db.SetMaxOpenConns(25)

    // Максимальное количество простаивающих соединений
    db.SetMaxIdleConns(5)

    // Максимальное время жизни соединения (5 минут)
    db.SetConnMaxLifetime(5 * time.Minute)
    
    // Проверка соединения
    if err := db.Ping(); err != nil {
        return nil, fmt.Errorf("ping database: %w", err)
    }
    
    return &Database{DB: db}, nil
}

// Метод закрытия подключения к базе
func (d *Database) Close() error {
    return d.DB.Close()
}

// Метод проверяет доступность базы данных
func (d *Database) Health() error {
    return d.DB.Ping()
}