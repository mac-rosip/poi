package db

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"time"

	"github.com/lib/pq"
)

// JobStatus represents the state of a mining job.
type JobStatus string

const (
	JobStatusPending  JobStatus = "pending"
	JobStatusActive   JobStatus = "active"
	JobStatusComplete JobStatus = "complete"
)

// Job represents a vanity address mining job.
type Job struct {
	ID              string    `json:"id"`
	EventID         *int64    `json:"event_id,omitempty"`
	StrategyID      *int64    `json:"strategy_id,omitempty"`
	Chain           string    `json:"chain"`
	Pattern         string    `json:"pattern"`
	PrefixChars     int       `json:"prefix_chars"`
	SuffixChars     int       `json:"suffix_chars"`
	MinScore        int       `json:"min_score"`
	FullKeypairMode bool      `json:"full_keypair_mode"`
	Status          JobStatus `json:"status"`
	AssignedWorker  string    `json:"assigned_worker,omitempty"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

func generateID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// CreateJob creates a new mining job.
func (db *DB) CreateJob(ctx context.Context, j *Job) error {
	if j.ID == "" {
		j.ID = generateID()
	}
	if j.Status == "" {
		j.Status = JobStatusPending
	}
	if j.PrefixChars == 0 && j.SuffixChars == 0 {
		j.PrefixChars = 4 // Default
	}
	if j.MinScore == 0 {
		j.MinScore = j.PrefixChars + j.SuffixChars
	}

	_, err := db.ExecContext(ctx, `
		INSERT INTO jobs (id, event_id, strategy_id, chain, pattern, prefix_chars, suffix_chars, min_score, full_keypair_mode, status)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	`, j.ID, j.EventID, j.StrategyID, j.Chain, j.Pattern, j.PrefixChars, j.SuffixChars, j.MinScore, j.FullKeypairMode, j.Status)
	return err
}

// GetJob retrieves a job by ID.
func (db *DB) GetJob(ctx context.Context, id string) (*Job, error) {
	j := &Job{}
	var eventID, strategyID sql.NullInt64
	var assignedWorker sql.NullString
	err := db.QueryRowContext(ctx, `
		SELECT id, event_id, strategy_id, chain, pattern, prefix_chars, suffix_chars, min_score, full_keypair_mode, status, assigned_worker, created_at, updated_at
		FROM jobs WHERE id = $1
	`, id).Scan(&j.ID, &eventID, &strategyID, &j.Chain, &j.Pattern, &j.PrefixChars, &j.SuffixChars, &j.MinScore, &j.FullKeypairMode, &j.Status, &assignedWorker, &j.CreatedAt, &j.UpdatedAt)
	if err != nil {
		return nil, err
	}
	if eventID.Valid {
		j.EventID = &eventID.Int64
	}
	if strategyID.Valid {
		j.StrategyID = &strategyID.Int64
	}
	if assignedWorker.Valid {
		j.AssignedWorker = assignedWorker.String
	}
	return j, nil
}

// GetNextPendingJob finds and assigns the next pending job to a worker.
func (db *DB) GetNextPendingJob(ctx context.Context, workerID string, supportedChains []string) (*Job, error) {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	// Build query with optional chain filter
	query := `
		SELECT id, event_id, strategy_id, chain, pattern, prefix_chars, suffix_chars, min_score, full_keypair_mode, status, created_at, updated_at
		FROM jobs
		WHERE status = 'pending'
	`
	args := []interface{}{}

	if len(supportedChains) > 0 {
		query += ` AND chain = ANY($1)`
		args = append(args, pq.Array(supportedChains))
	}

	query += ` ORDER BY created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED`

	j := &Job{}
	var eventID, strategyID sql.NullInt64
	err = tx.QueryRowContext(ctx, query, args...).Scan(
		&j.ID, &eventID, &strategyID, &j.Chain, &j.Pattern, &j.PrefixChars, &j.SuffixChars, &j.MinScore, &j.FullKeypairMode, &j.Status, &j.CreatedAt, &j.UpdatedAt,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if eventID.Valid {
		j.EventID = &eventID.Int64
	}
	if strategyID.Valid {
		j.StrategyID = &strategyID.Int64
	}

	// Assign to worker
	_, err = tx.ExecContext(ctx, `
		UPDATE jobs SET status = 'active', assigned_worker = $1, updated_at = NOW()
		WHERE id = $2
	`, workerID, j.ID)
	if err != nil {
		return nil, err
	}

	// Update worker's current job
	_, err = tx.ExecContext(ctx, `
		UPDATE workers SET current_job = $1 WHERE id = $2
	`, j.ID, workerID)
	if err != nil {
		return nil, err
	}

	if err := tx.Commit(); err != nil {
		return nil, err
	}

	j.Status = JobStatusActive
	j.AssignedWorker = workerID
	return j, nil
}

// CompleteJob marks a job as complete.
func (db *DB) CompleteJob(ctx context.Context, jobID string) error {
	_, err := db.ExecContext(ctx, `
		UPDATE jobs SET status = 'complete', updated_at = NOW() WHERE id = $1
	`, jobID)
	return err
}

// RequeueJob returns a job to pending status (e.g., when worker disconnects).
func (db *DB) RequeueJob(ctx context.Context, jobID string) error {
	_, err := db.ExecContext(ctx, `
		UPDATE jobs SET status = 'pending', assigned_worker = NULL, updated_at = NOW() WHERE id = $1
	`, jobID)
	return err
}

// DeleteJob removes a job.
func (db *DB) DeleteJob(ctx context.Context, id string) error {
	_, err := db.ExecContext(ctx, `DELETE FROM jobs WHERE id = $1`, id)
	return err
}

// ListJobs returns recent jobs ordered by creation time (limited to 200).
func (db *DB) ListJobs(ctx context.Context) ([]*Job, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT id, event_id, strategy_id, chain, pattern, prefix_chars, suffix_chars, min_score, full_keypair_mode, status, assigned_worker, created_at, updated_at
		FROM jobs ORDER BY created_at DESC LIMIT 200
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var jobs []*Job
	for rows.Next() {
		j := &Job{}
		var eventID, strategyID sql.NullInt64
		var assignedWorker sql.NullString
		if err := rows.Scan(&j.ID, &eventID, &strategyID, &j.Chain, &j.Pattern, &j.PrefixChars, &j.SuffixChars, &j.MinScore, &j.FullKeypairMode, &j.Status, &assignedWorker, &j.CreatedAt, &j.UpdatedAt); err != nil {
			return nil, err
		}
		if eventID.Valid {
			j.EventID = &eventID.Int64
		}
		if strategyID.Valid {
			j.StrategyID = &strategyID.Int64
		}
		if assignedWorker.Valid {
			j.AssignedWorker = assignedWorker.String
		}
		jobs = append(jobs, j)
	}
	return jobs, rows.Err()
}

// JobStats returns aggregate job statistics.
type JobStats struct {
	Total     int `json:"total"`
	Pending   int `json:"pending"`
	Active    int `json:"active"`
	Completed int `json:"completed"`
}

func (db *DB) GetJobStats(ctx context.Context) (*JobStats, error) {
	s := &JobStats{}
	err := db.QueryRowContext(ctx, `
		SELECT 
			COUNT(*) as total,
			COUNT(*) FILTER (WHERE status = 'pending') as pending,
			COUNT(*) FILTER (WHERE status = 'active') as active,
			COUNT(*) FILTER (WHERE status = 'complete') as completed
		FROM jobs
	`).Scan(&s.Total, &s.Pending, &s.Active, &s.Completed)
	return s, err
}
