package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strings"

	"github.com/user/hyperfanity/panel/db"
	pb "github.com/user/hyperfanity/panel/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// PanelServer implements the gRPC HyperfanityService and the HTTP web UI.
type PanelServer struct {
	pb.UnimplementedHyperfanityServiceServer
	db *db.DB
}

// NewPanelServer creates a new panel server.
func NewPanelServer(database *db.DB) *PanelServer {
	return &PanelServer{db: database}
}

// =============================================================================
// gRPC service implementation
// =============================================================================

func (s *PanelServer) Register(ctx context.Context, req *pb.RegisterRequest) (*pb.RegisterResponse, error) {
	w, err := s.db.RegisterWorker(ctx, req.Hostname, int(req.GpuCount), req.GpuNames, req.Version)
	if err != nil {
		log.Printf("[grpc] Failed to register worker: %v", err)
		return nil, status.Errorf(codes.Internal, "failed to register worker")
	}
	log.Printf("[grpc] Worker registered: %s (%s, %d GPUs)", w.ID[:12], w.Hostname, w.GPUCount)
	return &pb.RegisterResponse{
		WorkerId: w.ID,
		Token:    w.Token,
	}, nil
}

func (s *PanelServer) GetJob(ctx context.Context, req *pb.GetJobRequest) (*pb.Job, error) {
	_, err := s.db.AuthWorker(ctx, req.WorkerId, req.Token)
	if err != nil {
		return nil, status.Errorf(codes.Unauthenticated, "invalid worker credentials")
	}

	job, err := s.db.GetNextPendingJob(ctx, req.WorkerId, req.SupportedChains)
	if err != nil {
		log.Printf("[grpc] Error getting job: %v", err)
		return nil, status.Errorf(codes.Internal, "failed to get job")
	}
	if job == nil {
		return nil, status.Errorf(codes.NotFound, "no jobs available")
	}

	log.Printf("[grpc] Job %s assigned to worker %s (%s %s %s)",
		job.ID[:12], req.WorkerId[:12], job.Chain, job.MatchType, job.Pattern)

	return &pb.Job{
		JobId:           job.ID,
		Chain:           job.Chain,
		Pattern:         job.Pattern,
		MatchType:       job.MatchType,
		MinScore:        uint32(job.MinScore),
		FullKeypairMode: job.FullKeypairMode,
	}, nil
}

func (s *PanelServer) ReportProgress(stream pb.HyperfanityService_ReportProgressServer) error {
	ctx := stream.Context()
	for {
		update, err := stream.Recv()
		if err == io.EOF {
			return stream.SendAndClose(&pb.Ack{Success: true, Message: "progress stream closed"})
		}
		if err != nil {
			return err
		}
		if err := s.db.UpdateWorkerProgress(ctx, update.WorkerId, update.HashrateMhs, int64(update.TotalChecked), int(update.BestScore)); err != nil {
			log.Printf("[grpc] Error updating progress: %v", err)
		}
	}
}

func (s *PanelServer) ReportResult(ctx context.Context, req *pb.VanityResult) (*pb.Ack, error) {
	// Store the result
	result := &db.Result{
		JobID:      req.JobId,
		Chain:      req.Chain,
		Address:    req.Address,
		Score:      int(req.Score),
		PrivateKey: req.PrivateKey,
		PublicKey:  req.PublicKey,
	}
	if _, err := s.db.CreateResult(ctx, result); err != nil {
		log.Printf("[grpc] Error storing result: %v", err)
		return &pb.Ack{Success: false, Message: err.Error()}, nil
	}

	// Mark job complete
	if err := s.db.CompleteJob(ctx, req.JobId); err != nil {
		log.Printf("[grpc] Error completing job: %v", err)
		return &pb.Ack{Success: false, Message: err.Error()}, nil
	}

	// Clear worker's current job
	if err := s.db.ClearWorkerJob(ctx, req.WorkerId); err != nil {
		log.Printf("[grpc] Error clearing worker job: %v", err)
	}

	log.Printf("[grpc] Result for job %s: %s (score %d)", req.JobId[:12], req.Address, req.Score)
	return &pb.Ack{Success: true, Message: "result accepted"}, nil
}

func (s *PanelServer) Heartbeat(ctx context.Context, req *pb.HeartbeatRequest) (*pb.HeartbeatResponse, error) {
	_, err := s.db.AuthWorker(ctx, req.WorkerId, req.Token)
	if err != nil {
		return nil, status.Errorf(codes.Unauthenticated, "invalid worker credentials")
	}
	if err := s.db.WorkerHeartbeat(ctx, req.WorkerId); err != nil {
		log.Printf("[grpc] Error updating heartbeat: %v", err)
	}
	return &pb.HeartbeatResponse{ShouldContinue: true}, nil
}

// StartGRPC starts the gRPC server on the given address.
func (s *PanelServer) StartGRPC(addr string) error {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", addr, err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterHyperfanityServiceServer(grpcServer, s)

	log.Printf("[grpc] Listening on %s", addr)
	return grpcServer.Serve(lis)
}

// =============================================================================
// HTTP Web UI + REST API
// =============================================================================

func (s *PanelServer) StartHTTP(addr string) error {
	mux := http.NewServeMux()

	// Dashboard page
	mux.HandleFunc("/", s.handleDashboard)

	// Webhook for scanner events
	mux.HandleFunc("/webhook", s.handleWebhook)

	// REST API for the web UI
	mux.HandleFunc("/api/stats", s.handleAPIStats)
	mux.HandleFunc("/api/jobs", s.handleAPIJobs)
	mux.HandleFunc("/api/workers", s.handleAPIWorkers)

	log.Printf("[http] Listening on %s", addr)
	return http.ListenAndServe(addr, mux)
}

func (s *PanelServer) handleDashboard(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(dashboardHTML))
}

func (s *PanelServer) handleAPIStats(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	jobStats, _ := s.db.GetJobStats(ctx)
	workerStats, _ := s.db.GetWorkerStats(ctx)

	stats := map[string]interface{}{
		"TotalJobs":     jobStats.Total,
		"PendingJobs":   jobStats.Pending,
		"ActiveJobs":    jobStats.Active,
		"CompletedJobs": jobStats.Completed,
		"TotalWorkers":  workerStats.Total,
		"OnlineWorkers": workerStats.Online,
		"TotalHashrate": workerStats.TotalHashrate,
	}
	writeJSON(w, stats)
}

// jobJSON is the JSON-safe representation of a Job.
type jobJSON struct {
	ID             string `json:"id"`
	Chain          string `json:"chain"`
	Pattern        string `json:"pattern"`
	MatchType      string `json:"match_type"`
	MinScore       int    `json:"min_score"`
	Status         string `json:"status"`
	AssignedWorker string `json:"assigned_worker,omitempty"`
	CreatedAt      string `json:"created_at"`
	ResultAddress  string `json:"result_address,omitempty"`
	ResultScore    int    `json:"result_score,omitempty"`
	ResultKey      string `json:"result_key,omitempty"`
}

func jobToJSON(j *db.Job, result *db.Result) jobJSON {
	jj := jobJSON{
		ID:             j.ID,
		Chain:          j.Chain,
		Pattern:        j.Pattern,
		MatchType:      j.MatchType,
		MinScore:       j.MinScore,
		Status:         string(j.Status),
		AssignedWorker: j.AssignedWorker,
		CreatedAt:      j.CreatedAt.Format("2006-01-02 15:04:05"),
	}
	if result != nil {
		jj.ResultAddress = result.Address
		jj.ResultScore = result.Score
		if len(result.PrivateKey) > 0 {
			jj.ResultKey = hex.EncodeToString(result.PrivateKey)
		}
	}
	return jj
}

func (s *PanelServer) handleAPIJobs(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	switch r.Method {
	case http.MethodGet:
		jobs, err := s.db.ListJobs(ctx)
		if err != nil {
			http.Error(w, "failed to list jobs", http.StatusInternalServerError)
			return
		}
		result := make([]jobJSON, len(jobs))
		for i, j := range jobs {
			var res *db.Result
			if j.Status == db.JobStatusComplete {
				res, _ = s.db.GetResultByJobID(ctx, j.ID)
			}
			result[i] = jobToJSON(j, res)
		}
		writeJSON(w, result)

	case http.MethodPost:
		var req struct {
			Chain     string `json:"chain"`
			Pattern   string `json:"pattern"`
			MatchType string `json:"match_type"`
			MinScore  int    `json:"min_score"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
		req.Chain = strings.ToLower(strings.TrimSpace(req.Chain))
		req.Pattern = strings.TrimSpace(req.Pattern)
		if req.MatchType == "" {
			req.MatchType = "prefix"
		}
		if req.Chain == "" || req.Pattern == "" {
			http.Error(w, "chain and pattern are required", http.StatusBadRequest)
			return
		}
		validChains := map[string]bool{"trx": true, "eth": true, "sol": true, "btc": true}
		if !validChains[req.Chain] {
			http.Error(w, "invalid chain (use trx, eth, sol, btc)", http.StatusBadRequest)
			return
		}
		if req.MinScore == 0 {
			req.MinScore = len(req.Pattern)
		}
		job := &db.Job{
			Chain:     req.Chain,
			Pattern:   req.Pattern,
			MatchType: req.MatchType,
			MinScore:  req.MinScore,
		}
		if err := s.db.CreateJob(ctx, job); err != nil {
			http.Error(w, "failed to create job", http.StatusInternalServerError)
			return
		}
		log.Printf("[http] Job created: %s (%s %s %s)", job.ID[:12], job.Chain, job.MatchType, job.Pattern)
		w.WriteHeader(http.StatusCreated)
		writeJSON(w, jobToJSON(job, nil))

	case http.MethodDelete:
		jobID := r.URL.Query().Get("id")
		if jobID == "" {
			http.Error(w, "id parameter required", http.StatusBadRequest)
			return
		}
		s.db.DeleteJob(ctx, jobID)
		writeJSON(w, map[string]bool{"deleted": true})

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// workerJSON is the JSON-safe representation of a Worker.
type workerJSON struct {
	ID           string   `json:"id"`
	Hostname     string   `json:"hostname"`
	GPUCount     int      `json:"gpu_count"`
	GPUNames     []string `json:"gpu_names"`
	Version      string   `json:"version"`
	CurrentJob   string   `json:"current_job,omitempty"`
	HashrateMHS  float64  `json:"hashrate_mhs"`
	TotalChecked int64    `json:"total_checked"`
	BestScore    int      `json:"best_score"`
	Online       bool     `json:"online"`
	LastSeen     string   `json:"last_seen"`
}

func (s *PanelServer) handleAPIWorkers(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	workers, err := s.db.ListWorkers(ctx)
	if err != nil {
		http.Error(w, "failed to list workers", http.StatusInternalServerError)
		return
	}
	result := make([]workerJSON, len(workers))
	for i, wk := range workers {
		result[i] = workerJSON{
			ID:           wk.ID,
			Hostname:     wk.Hostname,
			GPUCount:     wk.GPUCount,
			GPUNames:     wk.GPUNames,
			Version:      wk.Version,
			CurrentJob:   wk.CurrentJob,
			HashrateMHS:  wk.HashrateMHS,
			TotalChecked: wk.TotalChecked,
			BestScore:    wk.BestScore,
			Online:       wk.Online,
			LastSeen:     wk.LastHeartbeat.Format("2006-01-02 15:04:05"),
		}
	}
	writeJSON(w, result)
}

func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// =============================================================================
// Webhook Handler
// =============================================================================

// WebhookRequest is the expected payload from the scanner.
type WebhookRequest struct {
	ChainID   int             `json:"chain_id"`
	RPCUrl    string          `json:"rpc_url"`
	WSSUrl    string          `json:"wss_url,omitempty"`
	Sender    string          `json:"sender"`
	Pattern   string          `json:"pattern"`
	MatchType string          `json:"match_type,omitempty"`
	Payload   json.RawMessage `json:"payload,omitempty"`
}

func chainIDToName(chainID int) string {
	switch chainID {
	case 1:
		return "eth"
	case 56:
		return "eth" // BSC uses same address format
	case 137:
		return "eth" // Polygon uses same address format
	case 43114:
		return "eth" // Avalanche C-Chain
	case 728126428: // Tron mainnet
		return "trx"
	default:
		return "eth" // Default to ETH format for EVM chains
	}
}

func (s *PanelServer) handleWebhook(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()

	var req WebhookRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Validate required fields
	if req.RPCUrl == "" || req.Sender == "" || req.Pattern == "" {
		http.Error(w, "rpc_url, sender, and pattern are required", http.StatusBadRequest)
		return
	}

	if req.MatchType == "" {
		req.MatchType = "prefix"
	}

	chain := chainIDToName(req.ChainID)

	// Store the event
	event := &db.Event{
		ChainID:   req.ChainID,
		Chain:     chain,
		RPCUrl:    req.RPCUrl,
		WSSUrl:    req.WSSUrl,
		Sender:    req.Sender,
		Pattern:   req.Pattern,
		MatchType: req.MatchType,
		Payload:   req.Payload,
	}
	eventID, err := s.db.CreateEvent(ctx, event)
	if err != nil {
		log.Printf("[webhook] Error storing event: %v", err)
		http.Error(w, "failed to store event", http.StatusInternalServerError)
		return
	}

	// Create a mining job from the event
	job := &db.Job{
		EventID:   &eventID,
		Chain:     chain,
		Pattern:   req.Pattern,
		MatchType: req.MatchType,
		MinScore:  len(req.Pattern),
	}
	if err := s.db.CreateJob(ctx, job); err != nil {
		log.Printf("[webhook] Error creating job: %v", err)
		http.Error(w, "failed to create job", http.StatusInternalServerError)
		return
	}

	log.Printf("[webhook] Event %d -> Job %s (%s %s %s from %s)",
		eventID, job.ID[:12], chain, req.MatchType, req.Pattern, req.Sender[:12])

	w.WriteHeader(http.StatusCreated)
	writeJSON(w, map[string]interface{}{
		"event_id": eventID,
		"job_id":   job.ID,
		"chain":    chain,
		"pattern":  req.Pattern,
		"status":   "pending",
	})
}
