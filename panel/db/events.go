package db

import (
	"context"
	"encoding/json"
	"time"
)

// Event represents a webhook event from the scanner.
type Event struct {
	ID        int64           `json:"id"`
	ChainID   int             `json:"chain_id"`
	Chain     string          `json:"chain"`
	RPCUrl    string          `json:"rpc_url"`
	WSSUrl    string          `json:"wss_url,omitempty"`
	Sender    string          `json:"sender"`
	Pattern   string          `json:"pattern"`
	MatchType string          `json:"match_type"`
	Payload   json.RawMessage `json:"payload,omitempty"`
	CreatedAt time.Time       `json:"created_at"`
}

// CreateEvent inserts a new event and returns its ID.
func (db *DB) CreateEvent(ctx context.Context, e *Event) (int64, error) {
	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO events (chain_id, chain, rpc_url, wss_url, sender, pattern, match_type, payload)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		RETURNING id
	`, e.ChainID, e.Chain, e.RPCUrl, e.WSSUrl, e.Sender, e.Pattern, e.MatchType, e.Payload).Scan(&id)
	if err != nil {
		return 0, err
	}
	return id, nil
}

// GetEvent retrieves an event by ID.
func (db *DB) GetEvent(ctx context.Context, id int64) (*Event, error) {
	e := &Event{}
	err := db.QueryRowContext(ctx, `
		SELECT id, chain_id, chain, rpc_url, wss_url, sender, pattern, match_type, payload, created_at
		FROM events WHERE id = $1
	`, id).Scan(&e.ID, &e.ChainID, &e.Chain, &e.RPCUrl, &e.WSSUrl, &e.Sender, &e.Pattern, &e.MatchType, &e.Payload, &e.CreatedAt)
	if err != nil {
		return nil, err
	}
	return e, nil
}

// ListRecentEvents returns the most recent events.
func (db *DB) ListRecentEvents(ctx context.Context, limit int) ([]*Event, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT id, chain_id, chain, rpc_url, wss_url, sender, pattern, match_type, payload, created_at
		FROM events ORDER BY created_at DESC LIMIT $1
	`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var events []*Event
	for rows.Next() {
		e := &Event{}
		if err := rows.Scan(&e.ID, &e.ChainID, &e.Chain, &e.RPCUrl, &e.WSSUrl, &e.Sender, &e.Pattern, &e.MatchType, &e.Payload, &e.CreatedAt); err != nil {
			return nil, err
		}
		events = append(events, e)
	}
	return events, rows.Err()
}
