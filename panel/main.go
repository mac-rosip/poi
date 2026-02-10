package main

import (
	"log"
	"os"
	"time"
)

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lmsgprefix)
	log.SetPrefix("[panel] ")

	grpcAddr := envOr("PANEL_PORT", "50051")
	webAddr := envOr("WEB_PORT", "8080")

	jm := NewJobManager()
	server := NewPanelServer(jm)

	// Background: reap stale workers every 30s
	go func() {
		for {
			time.Sleep(30 * time.Second)
			jm.ReapStaleWorkers(90 * time.Second)
		}
	}()

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
