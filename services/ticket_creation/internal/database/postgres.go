// internal/database/postgres.go
package database

import (
    "fmt"
    "time"
    
    "github.com/jmoiron/sqlx"
    _ "github.com/lib/pq"
)

type Database struct {
    DB *sqlx.DB
}

func NewDatabase(databaseURL string) (*Database, error) {
    db, err := sqlx.Connect("postgres", databaseURL)
    if err != nil {
        return nil, fmt.Errorf("connect to database: %w", err)
    }
    
    // Настройки пула соединений
    db.SetMaxOpenConns(25)
    db.SetMaxIdleConns(5)
    db.SetConnMaxLifetime(5 * time.Minute)
    
    // Проверка соединения
    if err := db.Ping(); err != nil {
        return nil, fmt.Errorf("ping database: %w", err)
    }
    
    return &Database{DB: db}, nil
}

func (d *Database) Close() error {
    return d.DB.Close()
}

func (d *Database) Health() error {
    return d.DB.Ping()
}