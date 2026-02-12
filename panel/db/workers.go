package db

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"time"

	"github.com/lib/pq"
)

// Worker represents a GPU worker node.
type Worker struct {
	ID            string    `json:"id"`
	Hostname      string    `json:"hostname"`
	GPUCount      int       `json:"gpu_count"`
	GPUNames      []string  `json:"gpu_names"`
	Version       string    `json:"version"`
	Token         string    `json:"-"` // Never expose token in JSON
	CurrentJob    string    `json:"current_job,omitempty"`
	HashrateMHS   float64   `json:"hashrate_mhs"`
	TotalChecked  int64     `json:"total_checked"`
	BestScore     int       `json:"best_score"`
	Online        bool      `json:"online"`
	RegisteredAt  time.Time `json:"registered_at"`
	LastHeartbeat time.Time `json:"last_heartbeat"`
}

func generateToken() string {
	b := make([]byte, 32)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// RegisterWorker creates a new worker.
func (db *DB) RegisterWorker(ctx context.Context, hostname string, gpuCount int, gpuNames []string, version string) (*Worker, error) {
	w := &Worker{
		ID:            generateID(),
		Hostname:      hostname,
		GPUCount:      gpuCount,
		GPUNames:      gpuNames,
		Version:       version,
		Token:         generateToken(),
		Online:        true,
		RegisteredAt:  time.Now(),
		LastHeartbeat: time.Now(),
	}

	_, err := db.ExecContext(ctx, `
		INSERT INTO workers (id, hostname, gpu_count, gpu_names, version, token, online, registered_at, last_heartbeat)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`, w.ID, w.Hostname, w.GPUCount, pq.Array(w.GPUNames), w.Version, w.Token, w.Online, w.RegisteredAt, w.LastHeartbeat)
	if err != nil {
		return nil, err
	}
	return w, nil
}

// AuthWorker validates worker credentials.
func (db *DB) AuthWorker(ctx context.Context, workerID, token string) (*Worker, error) {
	w := &Worker{}
	var currentJob sql.NullString
	err := db.QueryRowContext(ctx, `
		SELECT id, hostname, gpu_count, gpu_names, version, token, current_job, hashrate_mhs, total_checked, best_score, online, registered_at, last_heartbeat
		FROM workers WHERE id = $1
	`, workerID).Scan(&w.ID, &w.Hostname, &w.GPUCount, pq.Array(&w.GPUNames), &w.Version, &w.Token, &currentJob, &w.HashrateMHS, &w.TotalChecked, &w.BestScore, &w.Online, &w.RegisteredAt, &w.LastHeartbeat)
	if err != nil {
		return nil, err
	}
	if w.Token != token {
		return nil, sql.ErrNoRows
	}
	if currentJob.Valid {
		w.CurrentJob = currentJob.String
	}
	return w, nil
}

// UpdateWorkerProgress updates a worker's mining progress.
func (db *DB) UpdateWorkerProgress(ctx context.Context, workerID string, hashrate float64, totalChecked int64, bestScore int) error {
	_, err := db.ExecContext(ctx, `
		UPDATE workers SET hashrate_mhs = $1, total_checked = $2, best_score = $3, last_heartbeat = NOW()
		WHERE id = $4
	`, hashrate, totalChecked, bestScore, workerID)
	return err
}

// WorkerHeartbeat updates a worker's last heartbeat time.
func (db *DB) WorkerHeartbeat(ctx context.Context, workerID string) error {
	_, err := db.ExecContext(ctx, `
		UPDATE workers SET last_heartbeat = NOW(), online = TRUE WHERE id = $1
	`, workerID)
	return err
}

// ClearWorkerJob clears a worker's current job assignment.
func (db *DB) ClearWorkerJob(ctx context.Context, workerID string) error {
	_, err := db.ExecContext(ctx, `
		UPDATE workers SET current_job = NULL WHERE id = $1
	`, workerID)
	return err
}

// ListWorkers returns all workers.
func (db *DB) ListWorkers(ctx context.Context) ([]*Worker, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT id, hostname, gpu_count, gpu_names, version, current_job, hashrate_mhs, total_checked, best_score, online, registered_at, last_heartbeat
		FROM workers ORDER BY registered_at DESC
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var workers []*Worker
	for rows.Next() {
		w := &Worker{}
		var currentJob sql.NullString
		if err := rows.Scan(&w.ID, &w.Hostname, &w.GPUCount, pq.Array(&w.GPUNames), &w.Version, &currentJob, &w.HashrateMHS, &w.TotalChecked, &w.BestScore, &w.Online, &w.RegisteredAt, &w.LastHeartbeat); err != nil {
			return nil, err
		}
		if currentJob.Valid {
			w.CurrentJob = currentJob.String
		}
		workers = append(workers, w)
	}
	return workers, rows.Err()
}

// ReapStaleWorkers marks workers as offline if they haven't sent a heartbeat.
func (db *DB) ReapStaleWorkers(ctx context.Context, timeout time.Duration) error {
	cutoff := time.Now().Add(-timeout)

	// Mark workers offline and requeue their jobs
	_, err := db.ExecContext(ctx, `
		WITH stale AS (
			UPDATE workers SET online = FALSE, current_job = NULL
			WHERE online = TRUE AND last_heartbeat < $1
			RETURNING id, current_job
		)
		UPDATE jobs SET status = 'pending', assigned_worker = NULL, updated_at = NOW()
		WHERE id IN (SELECT current_job FROM stale WHERE current_job IS NOT NULL)
	`, cutoff)
	return err
}

// WorkerStats returns aggregate worker statistics.
type WorkerStats struct {
	Total         int     `json:"total"`
	Online        int     `json:"online"`
	TotalHashrate float64 `json:"total_hashrate"`
}

func (db *DB) GetWorkerStats(ctx context.Context) (*WorkerStats, error) {
	s := &WorkerStats{}
	err := db.QueryRowContext(ctx, `
		SELECT 
			COUNT(*) as total,
			COUNT(*) FILTER (WHERE online = TRUE) as online,
			COALESCE(SUM(hashrate_mhs) FILTER (WHERE online = TRUE), 0) as total_hashrate
		FROM workers
	`).Scan(&s.Total, &s.Online, &s.TotalHashrate)
	return s, err
}
