package db

import (
	"context"
	"database/sql"
	"time"
)

// Result represents a derived vanity address.
type Result struct {
	ID               int64      `json:"id"`
	JobID            string     `json:"job_id"`
	Chain            string     `json:"chain"`
	Address          string     `json:"address"`
	Score            int        `json:"score"`
	PrivateKey       []byte     `json:"-"` // Never expose private key in JSON by default
	PublicKey        []byte     `json:"public_key,omitempty"`
	FundingStatus    string     `json:"funding_status"`
	BundleID         string     `json:"bundle_id,omitempty"`
	FundingError     string     `json:"funding_error,omitempty"`
	BalanceLamports  int64      `json:"balance_lamports"`
	LastBalanceCheck *time.Time `json:"last_balance_check,omitempty"`
	CreatedAt        time.Time  `json:"created_at"`
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
		SELECT id, job_id, chain, address, score, funding_status, bundle_id, balance_lamports, last_balance_check, created_at
		FROM results WHERE chain = $1 ORDER BY created_at DESC
	`, chain)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []*Result
	for rows.Next() {
		r := &Result{}
		var bundleID, fundingStatus sql.NullString
		var lastCheck sql.NullTime
		if err := rows.Scan(&r.ID, &r.JobID, &r.Chain, &r.Address, &r.Score, 
			&fundingStatus, &bundleID, &r.BalanceLamports, &lastCheck, &r.CreatedAt); err != nil {
			return nil, err
		}
		r.FundingStatus = fundingStatus.String
		r.BundleID = bundleID.String
		if lastCheck.Valid {
			r.LastBalanceCheck = &lastCheck.Time
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

// UpdateResultFundingStatus updates the funding status of a result.
func (db *DB) UpdateResultFundingStatus(ctx context.Context, id int64, status, bundleID, errorMsg string) error {
	_, err := db.ExecContext(ctx, `
		UPDATE results SET funding_status = $2, bundle_id = $3, funding_error = $4
		WHERE id = $1
	`, id, status, nullStr(bundleID), nullStr(errorMsg))
	return err
}

func nullStr(s string) interface{} {
	if s == "" {
		return nil
	}
	return s
}

// UpdateResultBalance updates the balance of a result.
func (db *DB) UpdateResultBalance(ctx context.Context, id int64, balanceLamports int64) error {
	_, err := db.ExecContext(ctx, `
		UPDATE results SET balance_lamports = $2, last_balance_check = NOW()
		WHERE id = $1
	`, id, balanceLamports)
	return err
}

// GetSolanaResultsForBalanceCheck returns all Solana results with their IDs for balance checking.
func (db *DB) GetSolanaResultsForBalanceCheck(ctx context.Context) ([]*Result, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT id, job_id, chain, address, score, funding_status, bundle_id, balance_lamports, last_balance_check, created_at
		FROM results WHERE chain = 'sol' ORDER BY created_at DESC
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []*Result
	for rows.Next() {
		r := &Result{}
		var bundleID, fundingStatus sql.NullString
		var lastCheck sql.NullTime
		if err := rows.Scan(&r.ID, &r.JobID, &r.Chain, &r.Address, &r.Score,
			&fundingStatus, &bundleID, &r.BalanceLamports, &lastCheck, &r.CreatedAt); err != nil {
			return nil, err
		}
		r.FundingStatus = fundingStatus.String
		r.BundleID = bundleID.String
		if lastCheck.Valid {
			r.LastBalanceCheck = &lastCheck.Time
		}
		results = append(results, r)
	}
	return results, rows.Err()
}
