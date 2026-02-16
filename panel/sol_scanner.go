package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/user/hyperfanity/panel/db"
)

const (
	// MinTransferLamports is the minimum SOL transfer to trigger a job (50 SOL).
	MinTransferLamports = 50_000_000_000
	// SolanaSystemProgram is the System Program address.
	SolanaSystemProgram = "11111111111111111111111111111111"
)

// SolScanner monitors Solana native SOL transfers via Helius enhanced WebSocket.
type SolScanner struct {
	db     *db.DB
	wssURL string
	rpcURL string
	mu     sync.Mutex
	conn   *websocket.Conn
}

// NewSolScanner creates a new scanner. Returns nil if HELIUS_WSS_URL is not set.
func NewSolScanner(database *db.DB) *SolScanner {
	wssURL := os.Getenv("HELIUS_WSS_URL")
	if wssURL == "" {
		return nil
	}

	rpcURL := os.Getenv("SOLANA_RPC_URL")
	if rpcURL == "" {
		rpcURL = "https://api.mainnet-beta.solana.com"
	}

	return &SolScanner{
		db:     database,
		wssURL: wssURL,
		rpcURL: rpcURL,
	}
}

// Start begins the scanner with automatic reconnection.
func (s *SolScanner) Start(ctx context.Context) {
	log.Printf("[sol-scanner] Starting (min: %.0f SOL, endpoint: %s...)", float64(MinTransferLamports)/1e9, s.wssURL[:min(60, len(s.wssURL))])

	for {
		select {
		case <-ctx.Done():
			log.Println("[sol-scanner] Stopped")
			return
		default:
			if err := s.run(ctx); err != nil {
				log.Printf("[sol-scanner] Error: %v, reconnecting in 5s...", err)
			}
			time.Sleep(5 * time.Second)
		}
	}
}

func (s *SolScanner) run(ctx context.Context) error {
	conn, _, err := websocket.DefaultDialer.DialContext(ctx, s.wssURL, nil)
	if err != nil {
		return err
	}
	defer conn.Close()

	s.mu.Lock()
	s.conn = conn
	s.mu.Unlock()

	log.Println("[sol-scanner] Connected to Helius WebSocket")

	// Subscribe to all System Program transactions (native SOL transfers)
	sub := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      420,
		"method":  "transactionSubscribe",
		"params": []interface{}{
			map[string]interface{}{
				"failed":         false,
				"accountInclude": []string{SolanaSystemProgram},
			},
			map[string]interface{}{
				"commitment":                     "confirmed",
				"encoding":                       "jsonParsed",
				"transactionDetails":             "full",
				"maxSupportedTransactionVersion": 0,
			},
		},
	}

	if err := conn.WriteJSON(sub); err != nil {
		return err
	}

	log.Println("[sol-scanner] Subscribed to System Program transactions")

	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		_, message, err := conn.ReadMessage()
		if err != nil {
			return err
		}

		s.handleMessage(ctx, message)
	}
}

// Helius transactionNotification response types.
type txNotification struct {
	Method string `json:"method"`
	Params struct {
		Result struct {
			Signature   string `json:"signature"`
			Transaction struct {
				Meta struct {
					Err interface{} `json:"err"`
				} `json:"meta"`
				Transaction struct {
					Message struct {
						Instructions []json.RawMessage `json:"instructions"`
					} `json:"message"`
				} `json:"transaction"`
			} `json:"transaction"`
		} `json:"result"`
	} `json:"params"`
}

type parsedSystemIx struct {
	Program   string `json:"program"`
	ProgramID string `json:"programId"`
	Parsed    struct {
		Type string `json:"type"`
		Info struct {
			Source      string `json:"source"`
			Destination string `json:"destination"`
			Lamports    uint64 `json:"lamports"`
		} `json:"info"`
	} `json:"parsed"`
}

func (s *SolScanner) handleMessage(ctx context.Context, message []byte) {
	// Quick check: skip subscription confirmations and non-notifications
	var peek struct {
		Method string `json:"method"`
		ID     *int64 `json:"id"`
	}
	if json.Unmarshal(message, &peek) != nil {
		return
	}
	if peek.ID != nil {
		// Subscription confirmation
		log.Printf("[sol-scanner] Subscription confirmed (id: %d)", *peek.ID)
		return
	}
	if peek.Method != "transactionNotification" {
		return
	}

	var notif txNotification
	if err := json.Unmarshal(message, &notif); err != nil {
		return
	}

	// Skip failed transactions
	if notif.Params.Result.Transaction.Meta.Err != nil {
		return
	}

	sig := notif.Params.Result.Signature

	// Scan each instruction for native SOL transfers
	for _, rawIx := range notif.Params.Result.Transaction.Transaction.Message.Instructions {
		var ix parsedSystemIx
		if err := json.Unmarshal(rawIx, &ix); err != nil {
			continue
		}
		if ix.Program != "system" || ix.Parsed.Type != "transfer" {
			continue
		}
		if ix.Parsed.Info.Lamports < MinTransferLamports {
			continue
		}

		info := ix.Parsed.Info
		solAmount := float64(info.Lamports) / 1e9

		log.Printf("[sol-scanner] %.2f SOL: %s -> %s (tx: %s)",
			solAmount, info.Source[:8], info.Destination[:8], sig[:12])

		// Create event + job directly in DB
		_, jobID, _, _, err := createJobFromTransfer(
			ctx,
			s.db,
			"sol",
			"SOL",
			s.rpcURL,
			s.wssURL,
			info.Source,
			info.Destination,
			"",    // no contract (native SOL)
			"SOL", // token name
			0,     // chain_id not applicable
			solAmount,
		)
		if err != nil {
			log.Printf("[sol-scanner] Error creating job: %v", err)
			continue
		}

		log.Printf("[sol-scanner] Job created: %s", jobID[:12])
	}
}
