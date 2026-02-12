package db

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"time"

	_ "github.com/lib/pq"
)

// DB wraps the database connection pool.
type DB struct {
	*sql.DB
}

// New creates a new database connection pool.
func New(connStr string) (*DB, error) {
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	// Verify connection
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &DB{db}, nil
}

// Migrate runs database migrations.
func (db *DB) Migrate() error {
	log.Println("[db] Running migrations...")

	migrations := []string{
		// Events table - raw webhook events from scanner
		`CREATE TABLE IF NOT EXISTS events (
			id SERIAL PRIMARY KEY,
			chain_id INTEGER NOT NULL,
			chain VARCHAR(10) NOT NULL,
			rpc_url TEXT NOT NULL,
			wss_url TEXT,
			sender VARCHAR(128) NOT NULL,
			pattern VARCHAR(64) NOT NULL,
			match_type VARCHAR(16) NOT NULL DEFAULT 'prefix',
			payload JSONB,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)`,

		// Jobs table - mining jobs
		`CREATE TABLE IF NOT EXISTS jobs (
			id VARCHAR(64) PRIMARY KEY,
			event_id INTEGER REFERENCES events(id),
			chain VARCHAR(10) NOT NULL,
			pattern VARCHAR(64) NOT NULL,
			match_type VARCHAR(16) NOT NULL DEFAULT 'prefix',
			min_score INTEGER NOT NULL DEFAULT 1,
			full_keypair_mode BOOLEAN NOT NULL DEFAULT FALSE,
			status VARCHAR(16) NOT NULL DEFAULT 'pending',
			assigned_worker VARCHAR(64),
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)`,

		// Workers table - GPU worker registry
		`CREATE TABLE IF NOT EXISTS workers (
			id VARCHAR(64) PRIMARY KEY,
			hostname VARCHAR(256) NOT NULL,
			gpu_count INTEGER NOT NULL DEFAULT 0,
			gpu_names TEXT[],
			version VARCHAR(32),
			token VARCHAR(128) NOT NULL,
			current_job VARCHAR(64),
			hashrate_mhs DOUBLE PRECISION NOT NULL DEFAULT 0,
			total_checked BIGINT NOT NULL DEFAULT 0,
			best_score INTEGER NOT NULL DEFAULT 0,
			online BOOLEAN NOT NULL DEFAULT TRUE,
			registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)`,

		// Results table - derived vanity addresses
		`CREATE TABLE IF NOT EXISTS results (
			id SERIAL PRIMARY KEY,
			job_id VARCHAR(64) REFERENCES jobs(id),
			chain VARCHAR(10) NOT NULL,
			address VARCHAR(128) NOT NULL,
			score INTEGER NOT NULL,
			private_key BYTEA NOT NULL,
			public_key BYTEA,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)`,

		// Indexes for common queries
		`CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)`,
		`CREATE INDEX IF NOT EXISTS idx_jobs_chain ON jobs(chain)`,
		`CREATE INDEX IF NOT EXISTS idx_workers_online ON workers(online)`,
		`CREATE INDEX IF NOT EXISTS idx_results_chain ON results(chain)`,
		`CREATE INDEX IF NOT EXISTS idx_results_address ON results(address)`,
	}

	for _, m := range migrations {
		if _, err := db.Exec(m); err != nil {
			return fmt.Errorf("migration failed: %w\nSQL: %s", err, m)
		}
	}

	log.Println("[db] Migrations complete")
	return nil
}

// Close closes the database connection pool.
func (db *DB) Close() error {
	return db.DB.Close()
}
