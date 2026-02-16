package main

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/btcsuite/btcutil/base58"
	"github.com/user/hyperfanity/panel/db"
)

const (
	// Transfer amount: 0.0001 SOL = 100,000 lamports
	TransferAmountLamports = 100_000
	// Jito tip: 1000 lamports (0.000001 SOL)
	JitoTipLamports = 1_000
	// Rent exemption minimum for accounts
	RentExemptMinimum = 890_880
	// Total funding needed: transfer + rent + jito tip + some buffer for fees
	TotalFundingLamports = TransferAmountLamports + RentExemptMinimum + JitoTipLamports + 5_000

	// Jito block engine endpoint
	JitoBlockEngineURL = "https://mainnet.block-engine.jito.wtf/api/v1/bundles"
	// Jito tip account (one of many, this is the first)
	JitoTipAccount = "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5"

	// System program ID
	SystemProgramID = "11111111111111111111111111111111"
)

// SolanaClient handles Solana RPC calls and Jito bundle submission.
type SolanaClient struct {
	rpcURL         string
	fundingPrivKey ed25519.PrivateKey
	fundingPubKey  ed25519.PublicKey
}

// NewSolanaClient creates a new Solana client from environment variables.
func NewSolanaClient() (*SolanaClient, error) {
	rpcURL := os.Getenv("SOLANA_RPC_URL")
	if rpcURL == "" {
		return nil, fmt.Errorf("SOLANA_RPC_URL environment variable not set")
	}

	fundingKeyBase58 := os.Getenv("SOLANA_FUNDING_PRIVKEY")
	if fundingKeyBase58 == "" {
		return nil, fmt.Errorf("SOLANA_FUNDING_PRIVKEY environment variable not set")
	}

	// Decode base58 private key (64 bytes: 32 seed + 32 pubkey)
	privKeyBytes := base58.Decode(fundingKeyBase58)
	if len(privKeyBytes) != 64 {
		return nil, fmt.Errorf("invalid funding private key length: expected 64, got %d", len(privKeyBytes))
	}

	privKey := ed25519.PrivateKey(privKeyBytes)
	pubKey := privKey.Public().(ed25519.PublicKey)

	log.Printf("[solana] Initialized with funding wallet: %s", base58.Encode(pubKey))

	return &SolanaClient{
		rpcURL:         rpcURL,
		fundingPrivKey: privKey,
		fundingPubKey:  pubKey,
	}, nil
}

// GetFundingAddress returns the base58 address of the funding wallet.
func (sc *SolanaClient) GetFundingAddress() string {
	return base58.Encode(sc.fundingPubKey)
}

// FundAndTransfer creates and submits a Jito bundle that:
// 1. Transfers SOL from funding wallet to derived wallet
// 2. Transfers 0.00001 SOL from derived wallet to recipient
func (sc *SolanaClient) FundAndTransfer(ctx context.Context, derivedPrivKey []byte, recipientAddr string) (string, error) {
	// Decode derived wallet keys
	if len(derivedPrivKey) != 32 && len(derivedPrivKey) != 64 {
		return "", fmt.Errorf("invalid derived private key length: %d", len(derivedPrivKey))
	}

	var derivedPriv ed25519.PrivateKey
	if len(derivedPrivKey) == 32 {
		// Seed only - derive full keypair
		derivedPriv = ed25519.NewKeyFromSeed(derivedPrivKey)
	} else {
		derivedPriv = ed25519.PrivateKey(derivedPrivKey)
	}
	derivedPub := derivedPriv.Public().(ed25519.PublicKey)
	derivedAddr := base58.Encode(derivedPub)

	log.Printf("[solana] Creating Jito bundle: funding -> %s -> %s", derivedAddr[:12], recipientAddr[:12])

	// Get recent blockhash
	blockhash, err := sc.getRecentBlockhash(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to get blockhash: %w", err)
	}

	// Decode addresses
	recipientPubKey := base58.Decode(recipientAddr)
	if len(recipientPubKey) != 32 {
		return "", fmt.Errorf("invalid recipient address")
	}
	jitoTipPubKey := base58.Decode(JitoTipAccount)

	// Build transaction 1: Funding wallet -> Derived wallet (amount + rent + fees)
	tx1 := sc.buildTransferTx(
		sc.fundingPubKey,
		derivedPub,
		TotalFundingLamports,
		blockhash,
	)
	tx1Signed := sc.signTransaction(tx1, sc.fundingPrivKey)

	// Build transaction 2: Derived wallet -> Recipient (0.00001 SOL)
	tx2 := sc.buildTransferTx(
		derivedPub,
		recipientPubKey,
		TransferAmountLamports,
		blockhash,
	)
	tx2Signed := signTransaction(tx2, derivedPriv)

	// Build transaction 3: Jito tip from derived wallet
	tx3 := sc.buildTransferTx(
		derivedPub,
		jitoTipPubKey,
		JitoTipLamports,
		blockhash,
	)
	tx3Signed := signTransaction(tx3, derivedPriv)

	// Submit bundle to Jito
	bundleSig, err := sc.submitJitoBundle(ctx, [][]byte{tx1Signed, tx2Signed, tx3Signed})
	if err != nil {
		return "", fmt.Errorf("failed to submit Jito bundle: %w", err)
	}

	log.Printf("[solana] Jito bundle submitted: %s", bundleSig)
	return bundleSig, nil
}

// buildTransferTx creates a SOL transfer transaction (unsigned).
func (sc *SolanaClient) buildTransferTx(from, to ed25519.PublicKey, lamports uint64, blockhash []byte) []byte {
	// Transaction format (simplified):
	// - 1 byte: number of signatures (1)
	// - 64 bytes: signature placeholder
	// - Message:
	//   - 1 byte: number of required signatures (1)
	//   - 1 byte: number of read-only signed accounts (0)
	//   - 1 byte: number of read-only unsigned accounts (1) - system program
	//   - Compact array of account keys (3): from, to, system program
	//   - 32 bytes: recent blockhash
	//   - Compact array of instructions (1)

	var buf bytes.Buffer

	// Number of signatures
	buf.WriteByte(1)
	// Signature placeholder (64 zeros, will be filled later)
	buf.Write(make([]byte, 64))

	// Message header
	buf.WriteByte(1) // num_required_signatures
	buf.WriteByte(0) // num_readonly_signed_accounts
	buf.WriteByte(1) // num_readonly_unsigned_accounts (system program)

	// Account keys (compact array: length + keys)
	buf.WriteByte(3) // 3 accounts
	buf.Write(from)
	buf.Write(to)
	buf.Write(base58.Decode(SystemProgramID))

	// Recent blockhash
	buf.Write(blockhash)

	// Instructions (compact array)
	buf.WriteByte(1) // 1 instruction

	// System program transfer instruction
	buf.WriteByte(2)    // program_id_index (system program is at index 2)
	buf.WriteByte(2)    // accounts length
	buf.WriteByte(0)    // from account index
	buf.WriteByte(1)    // to account index
	buf.WriteByte(12)   // data length (4 + 8 bytes)
	binary.LittleEndian.PutUint32(buf.Next(4)[:0], 2) // Transfer instruction (index 2)
	buf.Write(make([]byte, 4))
	// Overwrite last 4 bytes with instruction index
	data := buf.Bytes()
	binary.LittleEndian.PutUint32(data[len(data)-4:], 2)

	// Actually let me rebuild this properly
	return buildSolanaTransferTx(from, to, lamports, blockhash)
}

// buildSolanaTransferTx properly builds a Solana transfer transaction.
func buildSolanaTransferTx(from, to []byte, lamports uint64, blockhash []byte) []byte {
	systemProgramID := base58.Decode(SystemProgramID)

	// Message
	msg := []byte{
		1, // num_required_signatures
		0, // num_readonly_signed
		1, // num_readonly_unsigned (system program)
	}

	// Account keys (compact-u16 length prefix)
	msg = append(msg, 3) // 3 accounts
	msg = append(msg, from...)
	msg = append(msg, to...)
	msg = append(msg, systemProgramID...)

	// Recent blockhash
	msg = append(msg, blockhash...)

	// Instructions
	msg = append(msg, 1) // 1 instruction

	// Transfer instruction
	msg = append(msg, 2)    // program_id_index
	msg = append(msg, 2)    // num accounts
	msg = append(msg, 0)    // from (index 0, signer+writable)
	msg = append(msg, 1)    // to (index 1, writable)
	msg = append(msg, 12)   // data length

	// Instruction data: transfer = index 2, then u64 lamports
	instrData := make([]byte, 12)
	binary.LittleEndian.PutUint32(instrData[0:4], 2) // Transfer instruction
	binary.LittleEndian.PutUint64(instrData[4:12], lamports)
	msg = append(msg, instrData...)

	// Full transaction: num_signatures + signature_placeholder + message
	tx := []byte{1} // 1 signature
	tx = append(tx, make([]byte, 64)...) // placeholder
	tx = append(tx, msg...)

	return tx
}

// signTransaction signs a transaction with the given private key.
func (sc *SolanaClient) signTransaction(tx []byte, privKey ed25519.PrivateKey) []byte {
	return signTransaction(tx, privKey)
}

func signTransaction(tx []byte, privKey ed25519.PrivateKey) []byte {
	// Sign the message part (everything after num_signatures + signature)
	message := tx[65:] // skip 1 byte num_sigs + 64 byte signature placeholder
	sig := ed25519.Sign(privKey, message)

	// Copy signature into transaction
	result := make([]byte, len(tx))
	copy(result, tx)
	copy(result[1:65], sig)

	return result
}

// getRecentBlockhash fetches a recent blockhash from the RPC.
func (sc *SolanaClient) getRecentBlockhash(ctx context.Context) ([]byte, error) {
	payload := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "getLatestBlockhash",
		"params":  []interface{}{map[string]string{"commitment": "finalized"}},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", sc.rpcURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Result struct {
			Value struct {
				Blockhash string `json:"blockhash"`
			} `json:"value"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if result.Error != nil {
		return nil, fmt.Errorf("RPC error: %s", result.Error.Message)
	}

	return base58.Decode(result.Result.Value.Blockhash), nil
}

// GetBalance returns the balance in lamports for an address.
func (sc *SolanaClient) GetBalance(ctx context.Context, address string) (uint64, error) {
	payload := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "getBalance",
		"params":  []interface{}{address},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return 0, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", sc.rpcURL, bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	var result struct {
		Result struct {
			Value uint64 `json:"value"`
		} `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, err
	}

	return result.Result.Value, nil
}

// submitJitoBundle submits a bundle of transactions to Jito block engine.
func (sc *SolanaClient) submitJitoBundle(ctx context.Context, txs [][]byte) (string, error) {
	// Encode transactions as base64
	encodedTxs := make([]string, len(txs))
	for i, tx := range txs {
		encodedTxs[i] = base64.StdEncoding.EncodeToString(tx)
	}

	payload := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "sendBundle",
		"params":  []interface{}{encodedTxs},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", JitoBlockEngineURL, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	var result struct {
		Result string `json:"result"`
		Error  *struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("failed to parse response: %s", string(respBody))
	}

	if result.Error != nil {
		return "", fmt.Errorf("Jito error: %s", result.Error.Message)
	}

	return result.Result, nil
}

// PostResultHandler processes a completed derivation and initiates the funding flow.
type PostResultHandler struct {
	db     *db.DB
	solana *SolanaClient
}

// NewPostResultHandler creates a new post-result handler.
func NewPostResultHandler(database *db.DB) (*PostResultHandler, error) {
	solana, err := NewSolanaClient()
	if err != nil {
		log.Printf("[solana] Warning: Solana client not initialized: %v", err)
		return &PostResultHandler{db: database, solana: nil}, nil
	}

	return &PostResultHandler{
		db:     database,
		solana: solana,
	}, nil
}

// HandleResult processes a new derivation result.
func (h *PostResultHandler) HandleResult(ctx context.Context, result *db.Result, event *db.Event) error {
	if h.solana == nil {
		log.Printf("[solana] Skipping post-result handling (client not initialized)")
		return nil
	}

	// Only handle Solana results
	if result.Chain != "sol" {
		log.Printf("[solana] Skipping non-Solana result (chain: %s)", result.Chain)
		return nil
	}

	if event == nil {
		log.Printf("[solana] No event associated with result, skipping funding")
		return nil
	}

	// Submit Jito bundle
	bundleSig, err := h.solana.FundAndTransfer(ctx, result.PrivateKey, event.Recipient)
	if err != nil {
		log.Printf("[solana] Failed to submit Jito bundle: %v", err)
		// Update result status to failed
		if updateErr := h.db.UpdateResultFundingStatus(ctx, result.ID, "failed", "", err.Error()); updateErr != nil {
			log.Printf("[solana] Failed to update funding status: %v", updateErr)
		}
		return err
	}

	// Update result with bundle signature
	if err := h.db.UpdateResultFundingStatus(ctx, result.ID, "submitted", bundleSig, ""); err != nil {
		log.Printf("[solana] Failed to update funding status: %v", err)
	}

	log.Printf("[solana] Funding initiated for %s, bundle: %s", result.Address[:12], bundleSig)
	return nil
}

// CheckBundleStatus checks if a submitted bundle has landed.
func (h *PostResultHandler) CheckBundleStatus(ctx context.Context, bundleID string) (string, error) {
	if h.solana == nil {
		return "", fmt.Errorf("solana client not initialized")
	}

	payload := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "getBundleStatuses",
		"params":  []interface{}{[]string{bundleID}},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", JitoBlockEngineURL, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		Result struct {
			Value []struct {
				BundleID           string `json:"bundle_id"`
				ConfirmationStatus string `json:"confirmation_status"`
			} `json:"value"`
		} `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	if len(result.Result.Value) > 0 {
		return result.Result.Value[0].ConfirmationStatus, nil
	}

	return "unknown", nil
}
