package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/user/hyperfanity/panel/db"
)

// RunPodTrigger monitors the job queue and triggers RunPod serverless workers
// when there are pending jobs but no online workers.
type RunPodTrigger struct {
	db         *db.DB
	apiKey     string
	endpointID string
	maxRuntime int // seconds per RunPod invocation

	mu          sync.Mutex
	inflightID  string    // RunPod job ID currently in-flight
	inflightAt  time.Time // when we last triggered
}

// NewRunPodTrigger creates a new trigger. Returns nil if RUNPOD_API_KEY is not set.
func NewRunPodTrigger(database *db.DB) *RunPodTrigger {
	apiKey := os.Getenv("RUNPOD_API_KEY")
	endpointID := os.Getenv("RUNPOD_ENDPOINT_ID")
	if apiKey == "" || endpointID == "" {
		return nil
	}

	maxRuntime := 3600 // 1 hour default
	return &RunPodTrigger{
		db:         database,
		apiKey:     apiKey,
		endpointID: endpointID,
		maxRuntime: maxRuntime,
	}
}

// Start begins the periodic check loop.
func (rt *RunPodTrigger) Start(ctx context.Context) {
	log.Printf("[runpod] Auto-trigger enabled (endpoint: %s)", rt.endpointID)

	// Check immediately on start
	rt.check(ctx)

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("[runpod] Auto-trigger stopped")
			return
		case <-ticker.C:
			rt.check(ctx)
		}
	}
}

func (rt *RunPodTrigger) check(ctx context.Context) {
	// Get job stats
	stats, err := rt.db.GetJobStats(ctx)
	if err != nil {
		log.Printf("[runpod] Error getting job stats: %v", err)
		return
	}

	if stats.Pending == 0 {
		return
	}

	// Get worker stats
	wStats, err := rt.db.GetWorkerStats(ctx)
	if err != nil {
		log.Printf("[runpod] Error getting worker stats: %v", err)
		return
	}

	if wStats.Online > 0 {
		return // Workers are active, no need to trigger
	}

	rt.mu.Lock()
	defer rt.mu.Unlock()

	// Don't spam — wait for the previous invocation's max_runtime to expire
	// plus a small buffer before re-triggering
	if rt.inflightID != "" && time.Since(rt.inflightAt) < time.Duration(rt.maxRuntime+60)*time.Second {
		return
	}

	// Trigger RunPod
	log.Printf("[runpod] %d pending jobs, 0 online workers — triggering RunPod", stats.Pending)

	jobID, err := rt.trigger()
	if err != nil {
		log.Printf("[runpod] Error triggering: %v", err)
		return
	}

	rt.inflightID = jobID
	rt.inflightAt = time.Now()
	log.Printf("[runpod] Triggered job %s (max_runtime: %ds)", jobID, rt.maxRuntime)
}

func (rt *RunPodTrigger) trigger() (string, error) {
	url := fmt.Sprintf("https://api.runpod.ai/v2/%s/run", rt.endpointID)

	payload := map[string]interface{}{
		"input": map[string]interface{}{
			"max_runtime": rt.maxRuntime,
		},
	}
	body, _ := json.Marshal(payload)

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+rt.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		ID     string `json:"id"`
		Status string `json:"status"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("RunPod returned status %d: %s", resp.StatusCode, result.Status)
	}

	return result.ID, nil
}
