# Deployment Guide

## Architecture

```
GitHub Pages (frontend)          Railway (backend API)
  frontend/index.html    →  →  →   api.py reads results.jsonl
       ↑ fetch()                         ↑
  User sees live dashboard         Agent runs on your Mac
```

## Step 1 — Push to GitHub

```bash
cd uncertainty_agent

# Initialize git
git init
git add .
git commit -m "initial commit"

# Create repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/uncertainty-agent.git
git branch -M main
git push -u origin main
```

## Step 2 — Enable GitHub Pages

1. Go to your repo on GitHub
2. Settings → Pages
3. Source: GitHub Actions
4. The workflow at .github/workflows/deploy.yml runs automatically
5. Your dashboard will be live at: https://YOUR_USERNAME.github.io/uncertainty-agent

## Step 3 — Deploy API to Railway (free tier)

Railway gives you a free server that runs your FastAPI backend 24/7.

1. Go to https://railway.app and sign up with GitHub
2. New Project → Deploy from GitHub repo → select uncertainty-agent
3. Railway auto-detects Procfile and deploys
4. In Railway dashboard → Variables, add:
   ```
   RESULTS_FILE=results.jsonl
   CALIB_FILE=calibration.json
   VALIDATION_FILE=prospective_validation.json
   ```
5. Copy your Railway URL (looks like: https://uncertainty-agent-production.railway.app)

## Step 4 — Connect frontend to API

Open `frontend/index.html`, find this line:

```javascript
: 'https://YOUR-RAILWAY-APP.railway.app';
```

Replace with your actual Railway URL:

```javascript
: 'https://uncertainty-agent-production.railway.app';
```

Commit and push — GitHub Pages auto-deploys:

```bash
git add frontend/index.html
git commit -m "connect to live API"
git push
```

## Step 5 — Sync agent data to Railway

Railway needs your results.jsonl to serve live data.
The cleanest way: sync from your Mac automatically.

Add to your crontab (runs every 10 min):
```bash
*/10 * * * * railway run --service uncertainty-agent rsync -avz \
  /path/to/uncertainty_agent/results.jsonl \
  railway:/app/results.jsonl
```

Or simpler: use Railway's volume mount and point your agent to write there directly.

## Alternative: run everything locally for presentations

If presenting in person, just run both locally:

```bash
# Terminal 1 — agent
python agent.py --run

# Terminal 2 — API
pip install -r requirements_api.txt
uvicorn api:app --reload --port 8000

# Open browser
open frontend/index.html
```

The dashboard auto-detects localhost and connects to http://localhost:8000.
