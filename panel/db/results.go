package db

import (
	"context"
	"time"
)

// Result represents a derived vanity address.
type Result struct {
	ID         int64     `json:"id"`
	JobID      string    `json:"job_id"`
	Chain      string    `json:"chain"`
	Address    string    `json:"address"`
	Score      int       `json:"score"`
	PrivateKey []byte    `json:"-"` // Never expose private key in JSON by default
	PublicKey  []byte    `json:"public_key,omitempty"`
	CreatedAt  time.Time `json:"created_at"`
}

// CreateResult stores a new derivation result.
func (db *DB) CreateResult(ctx context.Context, r *Result) (int64, error) {
	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO results (job_id, chain, address, score, private_key, public_key)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING id
	`, r.JobID, r.Chain, r.Address, r.Score, r.PrivateKey, r.PublicKey).Scan(&id)
	if err != nil {
		return 0, err
	}
	return id, nil
}

// GetResult retrieves a result by ID (includes private key).
func (db *DB) GetResult(ctx context.Context, id int64) (*Result, error) {
	r := &Result{}
	err := db.QueryRowContext(ctx, `
		SELECT id, job_id, chain, address, score, private_key, public_key, created_at
		FROM results WHERE id = $1
	`, id).Scan(&r.ID, &r.JobID, &r.Chain, &r.Address, &r.Score, &r.PrivateKey, &r.PublicKey, &r.CreatedAt)
	if err != nil {
		return nil, err
	}
	return r, nil
}

// GetResultByJobID retrieves a result by job ID.
func (db *DB) GetResultByJobID(ctx context.Context, jobID string) (*Result, error) {
	r := &Result{}
	err := db.QueryRowContext(ctx, `
		SELECT id, job_id, chain, address, score, private_key, public_key, created_at
		FROM results WHERE job_id = $1
	`, jobID).Scan(&r.ID, &r.JobID, &r.Chain, &r.Address, &r.Score, &r.PrivateKey, &r.PublicKey, &r.CreatedAt)
	if err != nil {
		return nil, err
	}
	return r, nil
}

// ListResults returns all results.
func (db *DB) ListResults(ctx context.Context) ([]*Result, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT id, job_id, chain, address, score, created_at
		FROM results ORDER BY created_at DESC
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []*Result
	for rows.Next() {
		r := &Result{}
		if err := rows.Scan(&r.ID, &r.JobID, &r.Chain, &r.Address, &r.Score, &r.CreatedAt); err != nil {
			return nil, err
		}
		results = append(results, r)
	}
	return results, rows.Err()
}

// ListResultsByChain returns all results for a specific chain.
func (db *DB) ListResultsByChain(ctx context.Context, chain string) ([]*Result, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT id, job_id, chain, address, score, created_at
		FROM results WHERE chain = $1 ORDER BY created_at DESC
	`, chain)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []*Result
	for rows.Next() {
		r := &Result{}
		if err := rows.Scan(&r.ID, &r.JobID, &r.Chain, &r.Address, &r.Score, &r.CreatedAt); err != nil {
			return nil, err
		}
		results = append(results, r)
	}
	return results, rows.Err()
}

// AddressForBalanceCheck is a minimal struct for balance checking.
type AddressForBalanceCheck struct {
	Chain   string `json:"chain"`
	Address string `json:"address"`
}

// GetAllAddressesForBalanceCheck returns all derived addresses grouped by chain.
func (db *DB) GetAllAddressesForBalanceCheck(ctx context.Context) (map[string][]string, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT DISTINCT chain, address FROM results ORDER BY chain
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result := make(map[string][]string)
	for rows.Next() {
		var chain, address string
		if err := rows.Scan(&chain, &address); err != nil {
			return nil, err
		}
		result[chain] = append(result[chain], address)
	}
	return result, rows.Err()
}
