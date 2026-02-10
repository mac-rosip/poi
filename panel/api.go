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

	pb "github.com/user/hyperfanity/panel/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// PanelServer implements the gRPC HyperfanityService and the HTTP web UI.
type PanelServer struct {
	pb.UnimplementedHyperfanityServiceServer
	jm *JobManager
}

// NewPanelServer creates a new panel server.
func NewPanelServer(jm *JobManager) *PanelServer {
	return &PanelServer{jm: jm}
}

// =============================================================================
// gRPC service implementation
// =============================================================================

func (s *PanelServer) Register(ctx context.Context, req *pb.RegisterRequest) (*pb.RegisterResponse, error) {
	w := s.jm.RegisterWorker(req.Hostname, req.GpuCount, req.GpuNames, req.Version)
	log.Printf("[grpc] Worker registered: %s (%s, %d GPUs)", w.ID[:12], w.Hostname, w.GPUCount)
	return &pb.RegisterResponse{
		WorkerId: w.ID,
		Token:    w.Token,
	}, nil
}

func (s *PanelServer) GetJob(ctx context.Context, req *pb.GetJobRequest) (*pb.Job, error) {
	if s.jm.AuthWorker(req.WorkerId, req.Token) == nil {
		return nil, status.Errorf(codes.Unauthenticated, "invalid worker credentials")
	}

	job := s.jm.GetNextJob(req.WorkerId, req.SupportedChains)
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
		MinScore:        job.MinScore,
		FullKeypairMode: job.FullKeypairMode,
	}, nil
}

func (s *PanelServer) ReportProgress(stream pb.HyperfanityService_ReportProgressServer) error {
	for {
		update, err := stream.Recv()
		if err == io.EOF {
			return stream.SendAndClose(&pb.Ack{Success: true, Message: "progress stream closed"})
		}
		if err != nil {
			return err
		}
		s.jm.UpdateProgress(update.WorkerId, update.HashrateMhs, update.TotalChecked, update.BestScore)
	}
}

func (s *PanelServer) ReportResult(ctx context.Context, req *pb.VanityResult) (*pb.Ack, error) {
	err := s.jm.CompleteJob(req.JobId, req.Address, req.Score, req.PrivateKey, req.PublicKey)
	if err != nil {
		return &pb.Ack{Success: false, Message: err.Error()}, nil
	}
	log.Printf("[grpc] Result for job %s: %s (score %d)", req.JobId[:12], req.Address, req.Score)
	return &pb.Ack{Success: true, Message: "result accepted"}, nil
}

func (s *PanelServer) Heartbeat(ctx context.Context, req *pb.HeartbeatRequest) (*pb.HeartbeatResponse, error) {
	if s.jm.AuthWorker(req.WorkerId, req.Token) == nil {
		return nil, status.Errorf(codes.Unauthenticated, "invalid worker credentials")
	}
	s.jm.Heartbeat(req.WorkerId)
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
	writeJSON(w, s.jm.Stats())
}

// jobJSON is the JSON-safe representation of a MiningJob.
type jobJSON struct {
	ID             string `json:"id"`
	Chain          string `json:"chain"`
	Pattern        string `json:"pattern"`
	MatchType      string `json:"match_type"`
	MinScore       uint32 `json:"min_score"`
	Status         string `json:"status"`
	AssignedWorker string `json:"assigned_worker,omitempty"`
	CreatedAt      string `json:"created_at"`
	ResultAddress  string `json:"result_address,omitempty"`
	ResultScore    uint32 `json:"result_score,omitempty"`
	ResultKey      string `json:"result_key,omitempty"`
}

func jobToJSON(j *MiningJob) jobJSON {
	statusStr := "pending"
	switch j.Status {
	case JobActive:
		statusStr = "active"
	case JobComplete:
		statusStr = "complete"
	}
	jj := jobJSON{
		ID:             j.ID,
		Chain:          j.Chain,
		Pattern:        j.Pattern,
		MatchType:      j.MatchType,
		MinScore:       j.MinScore,
		Status:         statusStr,
		AssignedWorker: j.AssignedWorker,
		CreatedAt:      j.CreatedAt.Format("2006-01-02 15:04:05"),
		ResultAddress:  j.ResultAddress,
		ResultScore:    j.ResultScore,
	}
	if len(j.ResultPrivateKey) > 0 {
		jj.ResultKey = hex.EncodeToString(j.ResultPrivateKey)
	}
	return jj
}

func (s *PanelServer) handleAPIJobs(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		jobs := s.jm.ListJobs()
		result := make([]jobJSON, len(jobs))
		for i, j := range jobs {
			result[i] = jobToJSON(j)
		}
		writeJSON(w, result)

	case http.MethodPost:
		var req struct {
			Chain     string `json:"chain"`
			Pattern   string `json:"pattern"`
			MatchType string `json:"match_type"`
			MinScore  uint32 `json:"min_score"`
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
			req.MinScore = uint32(len(req.Pattern))
		}
		job := s.jm.CreateJob(req.Chain, req.Pattern, req.MatchType, req.MinScore)
		log.Printf("[http] Job created: %s (%s %s %s)", job.ID[:12], job.Chain, job.MatchType, job.Pattern)
		w.WriteHeader(http.StatusCreated)
		writeJSON(w, jobToJSON(job))

	case http.MethodDelete:
		jobID := r.URL.Query().Get("id")
		if jobID == "" {
			http.Error(w, "id parameter required", http.StatusBadRequest)
			return
		}
		s.jm.DeleteJob(jobID)
		writeJSON(w, map[string]bool{"deleted": true})

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// workerJSON is the JSON-safe representation of a WorkerInfo.
type workerJSON struct {
	ID           string   `json:"id"`
	Hostname     string   `json:"hostname"`
	GPUCount     int32    `json:"gpu_count"`
	GPUNames     []string `json:"gpu_names"`
	Version      string   `json:"version"`
	CurrentJob   string   `json:"current_job,omitempty"`
	HashrateMHS  float64  `json:"hashrate_mhs"`
	TotalChecked uint64   `json:"total_checked"`
	BestScore    uint32   `json:"best_score"`
	Online       bool     `json:"online"`
	LastSeen     string   `json:"last_seen"`
}

func (s *PanelServer) handleAPIWorkers(w http.ResponseWriter, r *http.Request) {
	workers := s.jm.ListWorkers()
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
