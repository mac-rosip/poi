package main

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// JobStatus represents the state of a mining job.
type JobStatus int

const (
	JobPending  JobStatus = iota
	JobActive             // assigned to a worker
	JobComplete           // result found
)

// MiningJob is a vanity address mining request queued in the panel.
type MiningJob struct {
	ID              string
	Chain           string
	Pattern         string
	MatchType       string // "prefix", "suffix", "contains"
	MinScore        uint32
	FullKeypairMode bool
	Status          JobStatus
	AssignedWorker  string
	CreatedAt       time.Time
	UpdatedAt       time.Time
	// Result fields (populated when complete)
	ResultAddress    string
	ResultScore      uint32
	ResultPrivateKey []byte
	ResultPublicKey  []byte
}

// WorkerInfo describes a connected GPU worker.
type WorkerInfo struct {
	ID            string
	Hostname      string
	GPUCount      int32
	GPUNames      []string
	Version       string
	Token         string
	CurrentJob    string
	HashrateMHS   float64
	TotalChecked  uint64
	BestScore     uint32
	RegisteredAt  time.Time
	LastHeartbeat time.Time
	Online        bool
}

// JobManager manages the job queue and worker registry.
type JobManager struct {
	mu      sync.RWMutex
	jobs    map[string]*MiningJob
	workers map[string]*WorkerInfo

	// Ordered lists for display
	jobOrder    []string
	workerOrder []string
}

// NewJobManager creates a new job manager.
func NewJobManager() *JobManager {
	return &JobManager{
		jobs:    make(map[string]*MiningJob),
		workers: make(map[string]*WorkerInfo),
	}
}

// generateID creates a random hex ID.
func generateID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// generateToken creates a random auth token.
func generateToken() string {
	b := make([]byte, 32)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// --- Job operations ---

// CreateJob adds a new mining job to the queue.
func (jm *JobManager) CreateJob(chain, pattern, matchType string, minScore uint32) *MiningJob {
	jm.mu.Lock()
	defer jm.mu.Unlock()

	job := &MiningJob{
		ID:        generateID(),
		Chain:     chain,
		Pattern:   pattern,
		MatchType: matchType,
		MinScore:  minScore,
		Status:    JobPending,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	jm.jobs[job.ID] = job
	jm.jobOrder = append(jm.jobOrder, job.ID)
	return job
}

// GetNextJob finds the next pending job for a worker.
// Returns nil if no jobs are available.
func (jm *JobManager) GetNextJob(workerID string, supportedChains []string) *MiningJob {
	jm.mu.Lock()
	defer jm.mu.Unlock()

	chainSet := make(map[string]bool, len(supportedChains))
	for _, c := range supportedChains {
		chainSet[c] = true
	}

	for _, id := range jm.jobOrder {
		job := jm.jobs[id]
		if job.Status != JobPending {
			continue
		}
		// Check chain support (empty = all chains)
		if len(chainSet) > 0 && !chainSet[job.Chain] {
			continue
		}
		job.Status = JobActive
		job.AssignedWorker = workerID
		job.UpdatedAt = time.Now()

		// Update worker's current job
		if w, ok := jm.workers[workerID]; ok {
			w.CurrentJob = job.ID
		}
		return job
	}
	return nil
}

// CompleteJob marks a job as complete with the vanity result.
func (jm *JobManager) CompleteJob(jobID string, address string, score uint32, privKey, pubKey []byte) error {
	jm.mu.Lock()
	defer jm.mu.Unlock()

	job, ok := jm.jobs[jobID]
	if !ok {
		return fmt.Errorf("job %s not found", jobID)
	}
	job.Status = JobComplete
	job.ResultAddress = address
	job.ResultScore = score
	job.ResultPrivateKey = privKey
	job.ResultPublicKey = pubKey
	job.UpdatedAt = time.Now()

	// Clear worker's current job
	if w, ok := jm.workers[job.AssignedWorker]; ok {
		if w.CurrentJob == jobID {
			w.CurrentJob = ""
		}
	}
	return nil
}

// DeleteJob removes a job from the queue.
func (jm *JobManager) DeleteJob(jobID string) {
	jm.mu.Lock()
	defer jm.mu.Unlock()

	delete(jm.jobs, jobID)
	for i, id := range jm.jobOrder {
		if id == jobID {
			jm.jobOrder = append(jm.jobOrder[:i], jm.jobOrder[i+1:]...)
			break
		}
	}
}

// ListJobs returns all jobs in creation order.
func (jm *JobManager) ListJobs() []*MiningJob {
	jm.mu.RLock()
	defer jm.mu.RUnlock()

	jobs := make([]*MiningJob, 0, len(jm.jobOrder))
	for _, id := range jm.jobOrder {
		if j, ok := jm.jobs[id]; ok {
			jobs = append(jobs, j)
		}
	}
	return jobs
}

// GetJob returns a single job by ID.
func (jm *JobManager) GetJob(jobID string) *MiningJob {
	jm.mu.RLock()
	defer jm.mu.RUnlock()
	return jm.jobs[jobID]
}

// --- Worker operations ---

// RegisterWorker adds a new worker to the registry.
func (jm *JobManager) RegisterWorker(hostname string, gpuCount int32, gpuNames []string, version string) *WorkerInfo {
	jm.mu.Lock()
	defer jm.mu.Unlock()

	w := &WorkerInfo{
		ID:            generateID(),
		Hostname:      hostname,
		GPUCount:      gpuCount,
		GPUNames:      gpuNames,
		Version:       version,
		Token:         generateToken(),
		RegisteredAt:  time.Now(),
		LastHeartbeat: time.Now(),
		Online:        true,
	}
	jm.workers[w.ID] = w
	jm.workerOrder = append(jm.workerOrder, w.ID)
	return w
}

// AuthWorker checks worker ID + token. Returns the worker or nil.
func (jm *JobManager) AuthWorker(workerID, token string) *WorkerInfo {
	jm.mu.RLock()
	defer jm.mu.RUnlock()

	w, ok := jm.workers[workerID]
	if !ok || w.Token != token {
		return nil
	}
	return w
}

// UpdateProgress records a worker's progress update.
func (jm *JobManager) UpdateProgress(workerID string, hashrate float64, totalChecked uint64, bestScore uint32) {
	jm.mu.Lock()
	defer jm.mu.Unlock()

	if w, ok := jm.workers[workerID]; ok {
		w.HashrateMHS = hashrate
		w.TotalChecked = totalChecked
		w.BestScore = bestScore
		w.LastHeartbeat = time.Now()
	}
}

// Heartbeat refreshes a worker's last-seen timestamp.
func (jm *JobManager) Heartbeat(workerID string) {
	jm.mu.Lock()
	defer jm.mu.Unlock()

	if w, ok := jm.workers[workerID]; ok {
		w.LastHeartbeat = time.Now()
		w.Online = true
	}
}

// ListWorkers returns all workers in registration order.
func (jm *JobManager) ListWorkers() []*WorkerInfo {
	jm.mu.RLock()
	defer jm.mu.RUnlock()

	workers := make([]*WorkerInfo, 0, len(jm.workerOrder))
	for _, id := range jm.workerOrder {
		if w, ok := jm.workers[id]; ok {
			workers = append(workers, w)
		}
	}
	return workers
}

// ReapStaleWorkers marks workers that haven't sent a heartbeat as offline.
func (jm *JobManager) ReapStaleWorkers(timeout time.Duration) {
	jm.mu.Lock()
	defer jm.mu.Unlock()

	cutoff := time.Now().Add(-timeout)
	for _, w := range jm.workers {
		if w.Online && w.LastHeartbeat.Before(cutoff) {
			w.Online = false
			// Re-queue any job the worker was running
			if w.CurrentJob != "" {
				if j, ok := jm.jobs[w.CurrentJob]; ok && j.Status == JobActive {
					j.Status = JobPending
					j.AssignedWorker = ""
				}
				w.CurrentJob = ""
			}
		}
	}
}

// Stats returns aggregate statistics.
type Stats struct {
	TotalJobs     int
	PendingJobs   int
	ActiveJobs    int
	CompletedJobs int
	OnlineWorkers int
	TotalWorkers  int
	TotalHashrate float64
}

func (jm *JobManager) Stats() Stats {
	jm.mu.RLock()
	defer jm.mu.RUnlock()

	var s Stats
	s.TotalJobs = len(jm.jobs)
	s.TotalWorkers = len(jm.workers)

	for _, j := range jm.jobs {
		switch j.Status {
		case JobPending:
			s.PendingJobs++
		case JobActive:
			s.ActiveJobs++
		case JobComplete:
			s.CompletedJobs++
		}
	}
	for _, w := range jm.workers {
		if w.Online {
			s.OnlineWorkers++
			s.TotalHashrate += w.HashrateMHS
		}
	}
	return s
}
