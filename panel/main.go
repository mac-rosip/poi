package main

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/user/hyperfanity/panel/db"
)

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lmsgprefix)
	log.SetPrefix("[panel] ")

	grpcAddr := envOr("PANEL_PORT", "50051")
	webAddr := envOr("WEB_PORT", "8080")
	dbURL := envOr("DATABASE_URL", "")
	telegramBot := envOr("TELEGRAM_BOT_TOKEN", "")
	telegramChat := envOr("TELEGRAM_CHAT_ID", "")

	// Initialize database (required)
	if dbURL == "" {
		log.Fatal("DATABASE_URL environment variable is required")
	}
	database, err := db.New(dbURL)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer database.Close()

	if err := database.Migrate(); err != nil {
		log.Fatalf("Failed to run migrations: %v", err)
	}

	// Initialize post-result handler (for Solana Jito bundles)
	postResultHandler, err := NewPostResultHandler(database)
	if err != nil {
		log.Printf("Warning: Post-result handler not initialized: %v", err)
	}

	// Initialize balance checker
	balanceChecker := NewBalanceChecker(database, telegramBot, telegramChat)

	// Initialize Solana transfer scanner
	solScanner := NewSolScanner(database)

	server := NewPanelServer(database, postResultHandler, balanceChecker)

	// Background: reap stale workers every 30s
	go func() {
		ctx := context.Background()
		for {
			time.Sleep(30 * time.Second)
			if err := database.ReapStaleWorkers(ctx, 90*time.Second); err != nil {
				log.Printf("Error reaping stale workers: %v", err)
			}
		}
	}()

	// Background: periodic balance checker
	go balanceChecker.Start(context.Background())

	// Background: Solana transfer scanner
	if solScanner != nil {
		go solScanner.Start(context.Background())
	}

	// Start HTTP in background
	go func() {
		if err := server.StartHTTP(":" + webAddr); err != nil {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	log.Println("Hyperfanity Panel starting")
	log.Printf("  gRPC: :%s", grpcAddr)
	log.Printf("  Web:  :%s", webAddr)

	// Start gRPC (blocks)
	if err := server.StartGRPC(":" + grpcAddr); err != nil {
		log.Fatalf("gRPC server error: %v", err)
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
