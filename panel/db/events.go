package db

import (
	"context"
	"database/sql"
	"time"
)

// Event represents a webhook event from the scanner.
type Event struct {
	ID         int64     `json:"id"`
	ChainID    int       `json:"chain_id"`
	Chain      string    `json:"chain"`
	Blockchain string    `json:"blockchain,omitempty"`
	RPCUrl     string    `json:"rpc_url"`
	WSSUrl     string    `json:"wss_url,omitempty"`
	Sender     string    `json:"sender"`
	Recipient  string    `json:"recipient"`
	Contract   string    `json:"contract,omitempty"`
	TokenName  string    `json:"token_name,omitempty"`
	USDValue   float64   `json:"usd_value,omitempty"`
	Pattern    string    `json:"pattern"`
	CreatedAt  time.Time `json:"created_at"`
}

// CreateEvent inserts a new event and returns its ID.
func (db *DB) CreateEvent(ctx context.Context, e *Event) (int64, error) {
	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO events (chain_id, chain, blockchain, rpc_url, wss_url, sender, recipient, contract, token_name, usd_value, pattern)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
		RETURNING id
	`, e.ChainID, e.Chain, nullString(e.Blockchain), e.RPCUrl, nullString(e.WSSUrl),
		e.Sender, e.Recipient, nullString(e.Contract), nullString(e.TokenName),
		nullFloat(e.USDValue), e.Pattern).Scan(&id)
	if err != nil {
		return 0, err
	}
	return id, nil
}

func nullString(s string) sql.NullString {
	if s == "" {
		return sql.NullString{}
	}
	return sql.NullString{String: s, Valid: true}
}

func nullFloat(f float64) sql.NullFloat64 {
	if f == 0 {
		return sql.NullFloat64{}
	}
	return sql.NullFloat64{Float64: f, Valid: true}
}

// GetEvent retrieves an event by ID.
func (db *DB) GetEvent(ctx context.Context, id int64) (*Event, error) {
	e := &Event{}
	var blockchain, wssUrl, contract, tokenName sql.NullString
	var usdValue sql.NullFloat64
	err := db.QueryRowContext(ctx, `
		SELECT id, chain_id, chain, blockchain, rpc_url, wss_url, sender, recipient, contract, token_name, usd_value, pattern, created_at
		FROM events WHERE id = $1
	`, id).Scan(&e.ID, &e.ChainID, &e.Chain, &blockchain, &e.RPCUrl, &wssUrl,
		&e.Sender, &e.Recipient, &contract, &tokenName, &usdValue, &e.Pattern, &e.CreatedAt)
	if err != nil {
		return nil, err
	}
	e.Blockchain = blockchain.String
	e.WSSUrl = wssUrl.String
	e.Contract = contract.String
	e.TokenName = tokenName.String
	e.USDValue = usdValue.Float64
	return e, nil
}

// ListRecentEvents returns the most recent events.
func (db *DB) ListRecentEvents(ctx context.Context, limit int) ([]*Event, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT id, chain_id, chain, blockchain, rpc_url, wss_url, sender, recipient, contract, token_name, usd_value, pattern, created_at
		FROM events ORDER BY created_at DESC LIMIT $1
	`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var events []*Event
	for rows.Next() {
		e := &Event{}
		var blockchain, wssUrl, contract, tokenName sql.NullString
		var usdValue sql.NullFloat64
		if err := rows.Scan(&e.ID, &e.ChainID, &e.Chain, &blockchain, &e.RPCUrl, &wssUrl,
			&e.Sender, &e.Recipient, &contract, &tokenName, &usdValue, &e.Pattern, &e.CreatedAt); err != nil {
			return nil, err
		}
		e.Blockchain = blockchain.String
		e.WSSUrl = wssUrl.String
		e.Contract = contract.String
		e.TokenName = tokenName.String
		e.USDValue = usdValue.Float64
		events = append(events, e)
	}
	return events, rows.Err()
}
