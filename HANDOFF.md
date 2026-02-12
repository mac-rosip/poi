# Handoff: Deploy Hyperfanity Panel to Digital Ocean Droplet

## Goal
Deploy the panel service (Go gRPC server + web UI + PostgreSQL) to a DO droplet. GPU workers will run on RunPod serverless (separate future task).

## What's Done

### Infrastructure
- SSH key `hyperfanity-mac` added to DO (ID `53987226`)
- Droplet created: **`hyperfanity-panel`** at **192.241.151.164** (s-1vcpu-1gb, nyc1, `docker-20-04` image)
- DO container registry exists: `registry.digitalocean.com/alphabetregistry` (empty, logged in locally)
- Git remote: `origin` → `git@github.com:mac-rosip/poi.git` (pushed, master up to date)
- gh profile: `mac-rosip` (exists but not active — run `gh auth switch -u mac-rosip` if needed)

### Panel Code (complete)
- `panel/db/` — PostgreSQL persistence layer (events, jobs, workers, results)
- `panel/balance_checker.go` — Hourly balance checks with Telegram alerts
- `panel/api.go` — gRPC + HTTP handlers using db layer
- `Dockerfile.panel` — Go 1.23, simplified build (pb stubs pre-generated)

## Deployment Steps

### 1. Set up PostgreSQL on droplet
```bash
ssh root@192.241.151.164

# Run PostgreSQL container
docker run -d --name postgres --restart unless-stopped \
  -e POSTGRES_USER=hyperfanity \
  -e POSTGRES_PASSWORD=changeme \
  -e POSTGRES_DB=hyperfanity \
  -v pgdata:/var/lib/postgresql/data \
  -p 127.0.0.1:5432:5432 \
  postgres:16-alpine
```

### 2. Build and deploy panel (on droplet)
```bash
# From local machine: rsync source to droplet
rsync -avz --exclude build/ --exclude .git/ \
  /Users/dddd/hype/hyperfanity/ root@192.241.151.164:/opt/hyperfanity/

ssh root@192.241.151.164
cd /opt/hyperfanity
docker build -f Dockerfile.panel -t hyperfanity-panel .

# Run panel with DATABASE_URL
docker run -d --name panel --restart unless-stopped \
  --link postgres:postgres \
  -e DATABASE_URL="postgres://hyperfanity:changeme@postgres:5432/hyperfanity?sslmode=disable" \
  -e TELEGRAM_BOT_TOKEN="8067092718:AAEM6M5WECKLB2VGYfjKKJEJSB0x460Hsko" \
  -e TELEGRAM_CHAT_ID="-5250190218" \
  -p 50051:50051 -p 8080:8080 \
  hyperfanity-panel
```

### 3. Verify
- Web UI: `http://192.241.151.164:8080`
- REST API: `curl http://192.241.151.164:8080/api/stats`
- gRPC endpoint: `192.241.151.164:50051` (workers connect here)
- Create test job: `curl -X POST http://192.241.151.164:8080/api/jobs -d '{"chain":"eth","pattern":"dead"}'`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `PANEL_PORT` | No | gRPC port (default: 50051) |
| `WEB_PORT` | No | HTTP port (default: 8080) |
| `TELEGRAM_BOT_TOKEN` | No | For balance alerts |
| `TELEGRAM_CHAT_ID` | No | Telegram chat for alerts |

## Reference

| Item | Value |
|------|-------|
| Droplet IP | `192.241.151.164` |
| Droplet name | `hyperfanity-panel` |
| Droplet size | s-1vcpu-1gb (~$6/mo) |
| DO registry | `registry.digitalocean.com/alphabetregistry` |
| gRPC port | 50051 |
| Web UI port | 8080 |
| Go version | 1.23 |
| Dockerfile | `Dockerfile.panel` (repo root) |
| Panel source | `panel/` (main.go, api.go, balance_checker.go, db/, pb/) |
