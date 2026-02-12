package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/big"
	"net/http"
	"strings"
	"time"

	"github.com/user/hyperfanity/panel/db"
)

// BalanceChecker periodically checks balances of all derived addresses.
type BalanceChecker struct {
	db           *db.DB
	telegramBot  string
	telegramChat string
	interval     time.Duration
	rpcURLs      map[string]string // chain -> default RPC URL
}

// NewBalanceChecker creates a new balance checker.
func NewBalanceChecker(database *db.DB, botToken, chatID string) *BalanceChecker {
	return &BalanceChecker{
		db:           database,
		telegramBot:  botToken,
		telegramChat: chatID,
		interval:     1 * time.Hour,
		rpcURLs: map[string]string{
			"eth": "https://eth.llamarpc.com",
			"trx": "https://api.trongrid.io",
			"sol": "https://api.mainnet-beta.solana.com",
			"btc": "", // BTC requires different API
		},
	}
}

// Start begins the periodic balance checking.
func (bc *BalanceChecker) Start(ctx context.Context) {
	if bc.telegramBot == "" || bc.telegramChat == "" {
		log.Println("[balance] Telegram not configured, balance alerts disabled")
		return
	}

	log.Printf("[balance] Starting hourly balance checker (Telegram chat: %s)", bc.telegramChat)

	// Run immediately on start, then every interval
	bc.checkAllBalances(ctx)

	ticker := time.NewTicker(bc.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("[balance] Balance checker stopped")
			return
		case <-ticker.C:
			bc.checkAllBalances(ctx)
		}
	}
}

func (bc *BalanceChecker) checkAllBalances(ctx context.Context) {
	log.Println("[balance] Running balance check...")

	addressesByChain, err := bc.db.GetAllAddressesForBalanceCheck(ctx)
	if err != nil {
		log.Printf("[balance] Error fetching addresses: %v", err)
		return
	}

	totalChecked := 0
	fundedWallets := []string{}

	for chain, addresses := range addressesByChain {
		for _, addr := range addresses {
			balance, err := bc.getBalance(chain, addr)
			if err != nil {
				log.Printf("[balance] Error checking %s %s: %v", chain, addr[:12], err)
				continue
			}
			totalChecked++

			if balance.Cmp(big.NewInt(0)) > 0 {
				fundedWallets = append(fundedWallets, fmt.Sprintf("%s: %s (%s)", chain, addr, formatBalance(chain, balance)))
			}
		}
	}

	log.Printf("[balance] Checked %d addresses, %d with balance", totalChecked, len(fundedWallets))

	// Send Telegram alert if any wallets have balance
	if len(fundedWallets) > 0 {
		message := fmt.Sprintf("🚨 *Funded Wallets Detected!*\n\n%s", strings.Join(fundedWallets, "\n"))
		if err := bc.sendTelegram(message); err != nil {
			log.Printf("[balance] Error sending Telegram alert: %v", err)
		}
	}
}

func (bc *BalanceChecker) getBalance(chain, address string) (*big.Int, error) {
	switch chain {
	case "eth":
		return bc.getETHBalance(address)
	case "trx":
		return bc.getTRXBalance(address)
	case "sol":
		return bc.getSOLBalance(address)
	default:
		return big.NewInt(0), nil
	}
}

func (bc *BalanceChecker) getETHBalance(address string) (*big.Int, error) {
	rpcURL := bc.rpcURLs["eth"]
	payload := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "eth_getBalance",
		"params":  []interface{}{address, "latest"},
		"id":      1,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(rpcURL, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Result string `json:"result"`
		Error  *struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if result.Error != nil {
		return nil, fmt.Errorf("RPC error: %s", result.Error.Message)
	}

	balance := new(big.Int)
	if strings.HasPrefix(result.Result, "0x") {
		balance.SetString(result.Result[2:], 16)
	}
	return balance, nil
}

func (bc *BalanceChecker) getTRXBalance(address string) (*big.Int, error) {
	url := fmt.Sprintf("https://api.trongrid.io/v1/accounts/%s", address)

	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var result struct {
		Data []struct {
			Balance int64 `json:"balance"`
		} `json:"data"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	if len(result.Data) == 0 {
		return big.NewInt(0), nil
	}

	return big.NewInt(result.Data[0].Balance), nil
}

func (bc *BalanceChecker) getSOLBalance(address string) (*big.Int, error) {
	rpcURL := bc.rpcURLs["sol"]
	payload := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "getBalance",
		"params":  []interface{}{address},
		"id":      1,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(rpcURL, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Result struct {
			Value int64 `json:"value"`
		} `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return big.NewInt(result.Result.Value), nil
}

func formatBalance(chain string, balance *big.Int) string {
	switch chain {
	case "eth":
		// Convert wei to ETH (18 decimals)
		eth := new(big.Float).Quo(new(big.Float).SetInt(balance), big.NewFloat(1e18))
		return fmt.Sprintf("%.6f ETH", eth)
	case "trx":
		// Convert sun to TRX (6 decimals)
		trx := new(big.Float).Quo(new(big.Float).SetInt(balance), big.NewFloat(1e6))
		return fmt.Sprintf("%.2f TRX", trx)
	case "sol":
		// Convert lamports to SOL (9 decimals)
		sol := new(big.Float).Quo(new(big.Float).SetInt(balance), big.NewFloat(1e9))
		return fmt.Sprintf("%.4f SOL", sol)
	default:
		return balance.String()
	}
}

func (bc *BalanceChecker) sendTelegram(message string) error {
	url := fmt.Sprintf("https://api.telegram.org/bot%s/sendMessage", bc.telegramBot)

	payload := map[string]interface{}{
		"chat_id":    bc.telegramChat,
		"text":       message,
		"parse_mode": "Markdown",
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	resp, err := http.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("telegram API error: %s", string(respBody))
	}

	log.Printf("[balance] Telegram alert sent to %s", bc.telegramChat)
	return nil
}
