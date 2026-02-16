#!/usr/bin/env python3
"""
RunPod Handler for Hyperfanity

Supports two modes:
1. Serverless mode (default): Triggered by RunPod serverless jobs
2. Pod mode (WORKER_MODE=pod): Persistent worker that runs continuously

Both modes:
- Connect to the panel via gRPC
- Register as a worker
- Fetch jobs and run the hyperfanity binary
- Report results back to the panel
- Send heartbeats to stay online
"""

import os
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

import grpc

# Import generated protobuf stubs
sys.path.insert(0, os.path.dirname(__file__))
import hyperfanity_pb2 as pb
import hyperfanity_pb2_grpc as pb_grpc

# Configuration
PANEL_ADDR = os.environ.get("PANEL_ADDR", "167.99.234.241:50051")
WORKER_BINARY = os.environ.get("WORKER_BINARY", "/usr/local/bin/hyperfanity")
WORKER_VERSION = "1.0.0"
WORKER_MODE = os.environ.get("WORKER_MODE", "serverless")  # "serverless" or "pod"


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


def start_heartbeat(stub, worker_id, token, stop_event):
    """Send heartbeats every 30 seconds in a background thread."""
    while not stop_event.is_set():
        try:
            stub.Heartbeat(pb.HeartbeatRequest(worker_id=worker_id, token=token))
        except Exception as e:
            print(f"[!] Heartbeat error: {e}")
        stop_event.wait(30)


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

    # The binary only supports one mode at a time (prefix OR suffix OR contains).
    # When both prefix_chars and suffix_chars are set, use prefix — it's the
    # primary match mode. Set min_score to prefix_chars only.
    effective_score = 0

    if job.prefix_chars > 0 and job.pattern:
        prefix = job.pattern[:job.prefix_chars]
        cmd.extend(["--prefix", prefix])
        effective_score = job.prefix_chars
    elif job.suffix_chars > 0 and job.pattern:
        suffix = job.pattern[-job.suffix_chars:]
        cmd.extend(["--suffix", suffix])
        effective_score = job.suffix_chars
    elif job.pattern:
        if job.match_type == "suffix":
            cmd.extend(["--suffix", job.pattern])
        elif job.match_type == "contains":
            cmd.extend(["--contains", job.pattern])
        else:
            cmd.extend(["--prefix", job.pattern])
        effective_score = len(job.pattern)

    # Use effective score (prefix OR suffix length), not combined
    if effective_score > 0:
        cmd.extend(["--min-score", str(effective_score)])

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
        public_key=b"",
        address=result.get("address", ""),
        score=result.get("score", 0),
        chain=result.get("chain", ""),
    )

    resp = stub.ReportResult(req)
    if resp.success:
        print(f"[+] Result reported successfully: {resp.message}")
    else:
        print(f"[!] Failed to report result: {resp.message}")


def worker_loop(max_runtime=None):
    """
    Main worker loop. Connects to panel, registers, and processes jobs.
    If max_runtime is set, stops after that many seconds.
    """
    print(f"[*] Starting Hyperfanity worker")
    print(f"[*] Panel: {PANEL_ADDR}")

    start_time = time.time()

    # Connect to panel
    channel = grpc.insecure_channel(PANEL_ADDR)
    stub = pb_grpc.HyperfanityServiceStub(channel)

    # Register
    worker_id, token = register_worker(stub)

    # Start heartbeat thread
    stop_heartbeat = threading.Event()
    hb_thread = threading.Thread(target=start_heartbeat, args=(stub, worker_id, token, stop_heartbeat), daemon=True)
    hb_thread.start()

    jobs_completed = 0
    results = []

    try:
        while True:
            # Check runtime limit
            if max_runtime and (time.time() - start_time) >= max_runtime:
                print(f"[*] Max runtime {max_runtime}s reached, stopping")
                break

            # Get a job
            job = get_job(stub, worker_id, token)
            if not job:
                print("[*] No jobs available, waiting 5s...")
                time.sleep(5)
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
    finally:
        stop_heartbeat.set()
        channel.close()

    return {
        "worker_id": worker_id,
        "jobs_completed": jobs_completed,
        "results": results,
        "runtime_seconds": int(time.time() - start_time),
    }


# --- Entry point ---

if WORKER_MODE == "pod":
    # Persistent pod mode: run forever
    print("[*] Running in persistent pod mode")
    while True:
        try:
            worker_loop(max_runtime=None)
        except Exception as e:
            print(f"[!] Worker error: {e}, reconnecting in 10s...")
            time.sleep(10)
else:
    # Serverless mode: RunPod handler
    import runpod

    def handler(event):
        max_runtime = event.get("input", {}).get("max_runtime", 3600)
        try:
            return worker_loop(max_runtime=max_runtime)
        except grpc.RpcError as e:
            return {"error": f"gRPC error: {e}"}

    runpod.serverless.start({"handler": handler})
