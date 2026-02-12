package db

import (
	"context"
	"database/sql"
	"time"
)

// Strategy defines derivation parameters based on USD value conditions.
type Strategy struct {
	ID          int64     `json:"id"`
	Name        string    `json:"name"`
	IsActive    bool      `json:"is_active"`
	PrefixChars int       `json:"prefix_chars"`
	SuffixChars int       `json:"suffix_chars"`
	MinUSDValue *float64  `json:"min_usd_value,omitempty"`
	MaxUSDValue *float64  `json:"max_usd_value,omitempty"`
	Priority    int       `json:"priority"`
	CreatedAt   time.Time `json:"created_at"`
}

// CreateStrategy creates a new derivation strategy.
func (db *DB) CreateStrategy(ctx context.Context, s *Strategy) (int64, error) {
	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO strategies (name, is_active, prefix_chars, suffix_chars, min_usd_value, max_usd_value, priority)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		RETURNING id
	`, s.Name, s.IsActive, s.PrefixChars, s.SuffixChars,
		nullFloat64Ptr(s.MinUSDValue), nullFloat64Ptr(s.MaxUSDValue), s.Priority).Scan(&id)
	if err != nil {
		return 0, err
	}
	return id, nil
}

func nullFloat64Ptr(f *float64) sql.NullFloat64 {
	if f == nil {
		return sql.NullFloat64{}
	}
	return sql.NullFloat64{Float64: *f, Valid: true}
}

// GetStrategy retrieves a strategy by ID.
func (db *DB) GetStrategy(ctx context.Context, id int64) (*Strategy, error) {
	s := &Strategy{}
	var minUSD, maxUSD sql.NullFloat64
	err := db.QueryRowContext(ctx, `
		SELECT id, name, is_active, prefix_chars, suffix_chars, min_usd_value, max_usd_value, priority, created_at
		FROM strategies WHERE id = $1
	`, id).Scan(&s.ID, &s.Name, &s.IsActive, &s.PrefixChars, &s.SuffixChars, &minUSD, &maxUSD, &s.Priority, &s.CreatedAt)
	if err != nil {
		return nil, err
	}
	if minUSD.Valid {
		s.MinUSDValue = &minUSD.Float64
	}
	if maxUSD.Valid {
		s.MaxUSDValue = &maxUSD.Float64
	}
	return s, nil
}

// GetMatchingStrategy finds the best matching active strategy for a given USD value.
// Returns nil if no strategy matches.
func (db *DB) GetMatchingStrategy(ctx context.Context, usdValue float64) (*Strategy, error) {
	s := &Strategy{}
	var minUSD, maxUSD sql.NullFloat64
	err := db.QueryRowContext(ctx, `
		SELECT id, name, is_active, prefix_chars, suffix_chars, min_usd_value, max_usd_value, priority, created_at
		FROM strategies
		WHERE is_active = TRUE
		  AND (min_usd_value IS NULL OR min_usd_value <= $1)
		  AND (max_usd_value IS NULL OR max_usd_value >= $1)
		ORDER BY priority DESC, id ASC
		LIMIT 1
	`, usdValue).Scan(&s.ID, &s.Name, &s.IsActive, &s.PrefixChars, &s.SuffixChars, &minUSD, &maxUSD, &s.Priority, &s.CreatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if minUSD.Valid {
		s.MinUSDValue = &minUSD.Float64
	}
	if maxUSD.Valid {
		s.MaxUSDValue = &maxUSD.Float64
	}
	return s, nil
}

// UpdateStrategy updates an existing strategy.
func (db *DB) UpdateStrategy(ctx context.Context, s *Strategy) error {
	_, err := db.ExecContext(ctx, `
		UPDATE strategies SET
			name = $1, is_active = $2, prefix_chars = $3, suffix_chars = $4,
			min_usd_value = $5, max_usd_value = $6, priority = $7
		WHERE id = $8
	`, s.Name, s.IsActive, s.PrefixChars, s.SuffixChars,
		nullFloat64Ptr(s.MinUSDValue), nullFloat64Ptr(s.MaxUSDValue), s.Priority, s.ID)
	return err
}

// DeleteStrategy removes a strategy.
func (db *DB) DeleteStrategy(ctx context.Context, id int64) error {
	_, err := db.ExecContext(ctx, `DELETE FROM strategies WHERE id = $1`, id)
	return err
}

// DeleteOverlappingStrategies removes active strategies whose USD ranges overlap with the given range.
// Returns the IDs of deleted strategies.
func (db *DB) DeleteOverlappingStrategies(ctx context.Context, minUSD, maxUSD *float64) ([]int64, error) {
	// Two ranges [a,b] and [c,d] overlap if: a <= d AND b >= c
	// For NULLs: NULL min = -infinity, NULL max = +infinity
	// new_min <= existing_max: true if new_min is NULL OR existing_max is NULL OR new_min <= existing_max
	// new_max >= existing_min: true if new_max is NULL OR existing_min is NULL OR new_max >= existing_min
	query := `
		DELETE FROM strategies
		WHERE is_active = TRUE
		  AND ($1::float8 IS NULL OR max_usd_value IS NULL OR $1 <= max_usd_value)
		  AND ($2::float8 IS NULL OR min_usd_value IS NULL OR $2 >= min_usd_value)
		RETURNING id
	`

	rows, err := db.QueryContext(ctx, query, nullFloat64Ptr(minUSD), nullFloat64Ptr(maxUSD))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var deleted []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		deleted = append(deleted, id)
	}
	return deleted, rows.Err()
}

// ListStrategies returns all strategies ordered by priority.
func (db *DB) ListStrategies(ctx context.Context) ([]*Strategy, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT id, name, is_active, prefix_chars, suffix_chars, min_usd_value, max_usd_value, priority, created_at
		FROM strategies ORDER BY priority DESC, id ASC
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var strategies []*Strategy
	for rows.Next() {
		s := &Strategy{}
		var minUSD, maxUSD sql.NullFloat64
		if err := rows.Scan(&s.ID, &s.Name, &s.IsActive, &s.PrefixChars, &s.SuffixChars, &minUSD, &maxUSD, &s.Priority, &s.CreatedAt); err != nil {
			return nil, err
		}
		if minUSD.Valid {
			s.MinUSDValue = &minUSD.Float64
		}
		if maxUSD.Valid {
			s.MaxUSDValue = &maxUSD.Float64
		}
		strategies = append(strategies, s)
	}
	return strategies, rows.Err()
}

// ListActiveStrategies returns only active strategies.
func (db *DB) ListActiveStrategies(ctx context.Context) ([]*Strategy, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT id, name, is_active, prefix_chars, suffix_chars, min_usd_value, max_usd_value, priority, created_at
		FROM strategies WHERE is_active = TRUE ORDER BY priority DESC, id ASC
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var strategies []*Strategy
	for rows.Next() {
		s := &Strategy{}
		var minUSD, maxUSD sql.NullFloat64
		if err := rows.Scan(&s.ID, &s.Name, &s.IsActive, &s.PrefixChars, &s.SuffixChars, &minUSD, &maxUSD, &s.Priority, &s.CreatedAt); err != nil {
			return nil, err
		}
		if minUSD.Valid {
			s.MinUSDValue = &minUSD.Float64
		}
		if maxUSD.Valid {
			s.MaxUSDValue = &maxUSD.Float64
		}
		strategies = append(strategies, s)
	}
	return strategies, rows.Err()
}
