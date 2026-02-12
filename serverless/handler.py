#!/usr/bin/env python3
"""
RunPod Serverless Handler for Hyperfanity

This handler:
1. Connects to the panel via gRPC
2. Registers as a worker
3. Fetches jobs and runs the hyperfanity binary
4. Reports results back to the panel
"""

import os
import re
import socket
import subprocess
import sys
import time
from typing import Optional

import grpc
import runpod

# Import generated protobuf stubs
sys.path.insert(0, os.path.dirname(__file__))
import hyperfanity_pb2 as pb
import hyperfanity_pb2_grpc as pb_grpc

# Configuration
PANEL_ADDR = os.environ.get("PANEL_ADDR", "178.128.157.147:50051")
WORKER_BINARY = os.environ.get("WORKER_BINARY", "/usr/local/bin/hyperfanity")
WORKER_VERSION = "1.0.0"


def get_gpu_info():
    """Get GPU information from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpus = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            return gpus
    except Exception as e:
        print(f"[!] Error getting GPU info: {e}")
    return ["Unknown GPU"]


def register_worker(stub: pb_grpc.HyperfanityServiceStub) -> tuple[str, str]:
    """Register this worker with the panel."""
    gpus = get_gpu_info()
    hostname = socket.gethostname()
    
    req = pb.RegisterRequest(
        hostname=hostname,
        gpu_count=len(gpus),
        gpu_names=gpus,
        version=WORKER_VERSION,
    )
    
    resp = stub.Register(req)
    print(f"[+] Registered as worker: {resp.worker_id}")
    return resp.worker_id, resp.token


def get_job(stub: pb_grpc.HyperfanityServiceStub, worker_id: str, token: str) -> Optional[pb.Job]:
    """Fetch a job from the panel."""
    req = pb.GetJobRequest(
        worker_id=worker_id,
        token=token,
        supported_chains=["evm", "trx", "sol", "btc"],
    )
    
    try:
        job = stub.GetJob(req)
        if job.job_id:
            return job
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.NOT_FOUND:
            return None
        raise
    return None


def run_hyperfanity(job: pb.Job) -> dict:
    """Run the hyperfanity binary and parse results."""
    # Map chain names
    chain_map = {"evm": "eth", "ethereum": "eth", "tron": "trx"}
    chain = chain_map.get(job.chain.lower(), job.chain.lower())
    
    # Build command
    cmd = [WORKER_BINARY, "--chain", chain]
    
    # Use prefix_chars/suffix_chars from job if set
    if job.prefix_chars > 0 and job.pattern:
        prefix = job.pattern[:job.prefix_chars]
        cmd.extend(["--prefix", prefix])
    elif job.suffix_chars > 0 and job.pattern:
        suffix = job.pattern[-job.suffix_chars:]
        cmd.extend(["--suffix", suffix])
    elif job.pattern:
        # Fall back to match_type
        if job.match_type == "prefix":
            cmd.extend(["--prefix", job.pattern])
        elif job.match_type == "suffix":
            cmd.extend(["--suffix", job.pattern])
        else:
            cmd.extend(["--contains", job.pattern])
    
    if job.min_score > 0:
        cmd.extend(["--min-score", str(job.min_score)])
    
    print(f"[*] Running: {' '.join(cmd)}")
    
    # Run the worker
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    result = {
        "found": False,
        "address": "",
        "private_key": "",
        "score": 0,
        "chain": chain,
    }
    
    # Parse output in real-time
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        
        # Print progress
        if line.startswith("Speed:") or "H/s" in line:
            print(f"\r  {line}", end="", flush=True)
            continue
        
        print(line)
        
        # Parse result
        if "MATCH FOUND" in line:
            result["found"] = True
        elif line.startswith("Address:"):
            result["address"] = line.split(":", 1)[1].strip()
        elif line.startswith("Private Key:"):
            result["private_key"] = line.split(":", 1)[1].strip()
        elif line.startswith("Score:"):
            try:
                result["score"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    
    process.wait()
    print()  # Newline after progress
    
    return result


def report_result(stub: pb_grpc.HyperfanityServiceStub, worker_id: str, job_id: str, result: dict):
    """Report result back to the panel."""
    private_key_bytes = bytes.fromhex(result.get("private_key", "")) if result.get("private_key") else b""
    
    req = pb.VanityResult(
        worker_id=worker_id,
        job_id=job_id,
        private_key=private_key_bytes,
        public_key=b"",  # Not captured from CLI output
        address=result.get("address", ""),
        score=result.get("score", 0),
        chain=result.get("chain", ""),
    )
    
    resp = stub.ReportResult(req)
    if resp.success:
        print(f"[+] Result reported successfully: {resp.message}")
    else:
        print(f"[!] Failed to report result: {resp.message}")


def handler(event):
    """
    RunPod serverless handler.
    
    Event can contain:
    - job_id: specific job to work on (optional)
    - max_runtime: maximum seconds to run (default: 3600)
    """
    print(f"[*] Starting Hyperfanity worker")
    print(f"[*] Panel: {PANEL_ADDR}")
    
    max_runtime = event.get("input", {}).get("max_runtime", 3600)
    start_time = time.time()
    
    # Connect to panel
    channel = grpc.insecure_channel(PANEL_ADDR)
    stub = pb_grpc.HyperfanityServiceStub(channel)
    
    # Register
    try:
        worker_id, token = register_worker(stub)
    except grpc.RpcError as e:
        return {"error": f"Failed to register: {e}"}
    
    jobs_completed = 0
    results = []
    
    # Main loop
    while time.time() - start_time < max_runtime:
        # Get a job
        job = get_job(stub, worker_id, token)
        if not job:
            print("[*] No jobs available, waiting...")
            time.sleep(10)
            continue
        
        print(f"[+] Got job: {job.job_id}")
        print(f"    Chain: {job.chain}")
        print(f"    Pattern: {job.pattern}")
        print(f"    Prefix chars: {job.prefix_chars}, Suffix chars: {job.suffix_chars}")
        
        # Run the miner
        result = run_hyperfanity(job)
        
        if result["found"]:
            print(f"[+] Found match: {result['address']}")
            report_result(stub, worker_id, job.job_id, result)
            results.append({
                "job_id": job.job_id,
                "address": result["address"],
                "score": result["score"],
            })
        
        jobs_completed += 1
    
    channel.close()
    
    return {
        "worker_id": worker_id,
        "jobs_completed": jobs_completed,
        "results": results,
        "runtime_seconds": int(time.time() - start_time),
    }


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
