# VPS Training Architecture Design
**Date:** 2026-04-26
**Status:** Approved for implementation

## Problem

All ML training currently runs in the browser. This causes crashes on heavy cycles, limits training data depth, and provides no path to 24/7 autonomous operation (Coinbase trading, future phase). Model weights live in `localStorage` — ephemeral, single-device, not production-grade.

## Goals

1. Serve the React dashboard from the VPS — accessible from any browser
2. Move all training to a persistent Node.js server process on the VPS
3. Schedule nightly retraining on fresh Polygon data with a validation gate
4. Enable online learning from reviewed live decisions with a performance gate
5. Frontend becomes a read-only display — no compute, no localStorage model state

## Out of Scope

- Coinbase autonomous trading execution (future phase, after model is validated)
- Database (weights stored as JSON files — sufficient for current scale)
- Docker / containerisation (premature for current phase)

## Architecture

```
VPS
├── Nginx (port 80/443)
│   ├── /        → serves /root/Trading/dist  (vite build, static)
│   └── /api/*   → proxy_pass http://localhost:3001
│
├── Node.js Trading Service  (PM2, always-on, auto-restart)
│   ├── Express API          — weights, status, outcomes, manual trigger
│   ├── node-cron            — nightly retraining at 2am
│   ├── training/            — GBM, NN, LR, Regime, Bagging modules
│   └── data/
│       ├── weights/         — JSON files, one per model+universe
│       └── outcomes/        — log.jsonl, rolling live decisions
│
└── server/.env
    ├── POLYGON_KEY
    ├── PORT=3001
    └── TRAINING_CRON="0 2 * * *"
```

## Data Sources (unchanged from current)

Live quote fetching stays client-side. Server only handles training data.

| Data point | Source | Rationale |
|---|---|---|
| Historical OHLCV (training) | Polygon | Years of depth; 5 calls/min fine for nightly batch |
| Live quotes | Finnhub free (60/min) → Yahoo fallback | Polygon Starter too slow for 15-symbol live refresh |
| Today's intraday 5m bars | Yahoo | No rate limit concern, free |
| Fundamentals / metrics | Finnhub | Dedicated fundamentals API, not available on Polygon Starter |
| Earnings data | Finnhub | Calendar, surprise data |

Note: Finnhub is currently on free tier. Quote fetch priority should be Finnhub-first (more reliable), Yahoo fallback — currently reversed in `fetchQuote`, small fix needed.

## File & Module Structure

```
server/
├── index.js                 — Express app, mounts routes, starts cron
├── ecosystem.config.js      — PM2 config (auto-restart, boot startup)
├── .env                     — POLYGON_KEY, PORT, TRAINING_CRON
├── config/
│   └── watchlist.json       — canonical symbol lists per universe (equities/crypto/btc)
│                              read by both training pipeline and served to frontend
├── routes/
│   ├── weights.js           — GET /api/weights
│   ├── train.js             — POST /api/train/trigger, GET /api/train/log
│   └── outcome.js           — POST /api/outcome
├── training/
│   ├── pipeline.js          — orchestrates nightly job
│   ├── fetch.js             — Polygon data fetching (ported from src/polygon.js)
│   ├── gbm.js               — GBM training (ported from src/gbm.js)
│   ├── nn.js                — NN training (ported from src/nn.js)
│   ├── lr.js                — LR + adaptWeights (ported from src/model.js)
│   ├── regime.js            — Regime training (ported from src/regime.js)
│   └── bagging.js           — Bagging (ported from src/bagging.js)
└── data/
    ├── weights/             — {model}_{universe}.json + last_run.json
    └── outcomes/            — log.jsonl
```

Training modules are a near-direct port — all training logic is pure JS with no browser APIs. The only required change is replacing `localStorage.getItem/setItem` with `fs.readFileSync/writeFileSync`.

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/weights` | All current weights + metadata (last trained, scores) |
| GET | `/api/status` | Last training run report, gate status, uptime |
| POST | `/api/outcome` | Log a reviewed decision outcome, trigger online update |
| POST | `/api/train/trigger` | Manual retrain (same pipeline as nightly) |
| GET | `/api/train/log` | Last N lines of training output (for UI status panel) |

## Training Pipeline (Nightly, 2am)

```
For each symbol in watchlist:
  1. Fetch Polygon daily bars  — rolling 12-month window
  2. Fetch Polygon 5m bars     — rolling 90-day window

For each universe (equities, crypto, btc):
  1. Run walk-forward training: GBM → NN → LR → Regime → Bagging
  2. Score new model on held-out recent 30 days
  3. Compare against current deployed model on same window
  4. If new model wins or ties  → promote, save to data/weights/
  5. If new model loses         → keep old weights, log failure

Write training report to data/weights/last_run.json:
  — timestamp, per-model scores, promoted/rejected status
```

## Online Learning Pipeline (On Reviewed Outcome)

```
POST /api/outcome receives: { symbol, verdict, outcome, features, universe }

1. Append to data/outcomes/log.jsonl

2. Performance gate check:
   — rolling 20-trade win rate < 40%  → freeze updates, flag in /api/status
   — fewer than 10 outcomes in log    → skip (insufficient signal)
   — last regime model output changed cluster in past 3 bars → skip (noisy labels during transitions)

3. If gate passes:
   — LR:      gradient descent update (adaptWeights logic from src/model.js)
   — GBM:     warm-start with new sample (existing warm-start in src/gbm.js)
   — NN:      backprop on new sample (existing logic in src/nn.js)

4. Save updated weights to data/weights/ immediately
```

Key constraint: only update on outcomes that have been **reviewed and confirmed** — not on raw predictions. The existing `reviewedLog` mechanism in the frontend is the correct gate for this.

## Frontend Changes

Three targeted changes only — no structural changes to the React app:

### 1. Weight loading (App.jsx mount)
```js
// Replace: loadWeights() from localStorage
// With:
const res = await fetch('/api/weights');
const weights = await res.json();
// Falls back to hardcoded defaults if server unreachable
```

### 2. Outcome logging (trade review handler)
```js
// Add alongside existing local state update:
fetch('/api/outcome', {
  method: 'POST',
  body: JSON.stringify({ symbol, verdict, outcome, features, universe })
});
```

### 3. Status panel (Model tab addition)
Small display addition showing:
- Last training run: timestamp + which models promoted/rejected
- Online learning gate: open / frozen (with reason)
- Next scheduled run: countdown

## Deployment

### PM2 (process management)
```js
// server/ecosystem.config.js
module.exports = {
  apps: [{
    name: 'trading-server',
    script: 'server/index.js',
    restart_delay: 5000,
    max_restarts: 10,
  }]
};
```

Start on VPS boot: `pm2 startup && pm2 save`

### Nginx config
```nginx
server {
  listen 80;
  server_name _;

  location /api/ {
    proxy_pass http://localhost:3001;
    proxy_http_version 1.1;
  }

  location / {
    root /root/Trading/dist;
    try_files $uri $uri/ /index.html;
  }
}
```

### Build & deploy workflow
```bash
# In /root/Trading:
npm run build          # produces dist/
pm2 restart trading-server
```

## Error Handling

- **Polygon fetch fails** during nightly job: skip that symbol, log warning, continue with remaining symbols. Don't abort the whole run.
- **Model validation fails** (new model worse than current): keep old weights, log in `last_run.json`. Next nightly run tries again.
- **Online learning gate frozen**: POST /api/outcome still logs the outcome (for future analysis) but skips the weight update. Gate resets when rolling 20-trade win rate recovers above 40%.
- **Server unreachable** (frontend): fall back to hardcoded default weights — same behaviour as today's localStorage miss.
- **PM2 crash + restart**: weights are on disk, persisted between restarts. No state loss.

## Open Tasks (tracked separately)

1. Audit data sources — verify Finnhub free coverage vs paid; confirm Yahoo fallback still works
2. Swap quote fetch priority — Finnhub first, Yahoo fallback (currently reversed)
3. Port training modules from browser to Node.js — replace localStorage with file I/O
4. Build nightly pipeline with walk-forward validation gate
5. Build online learning endpoint with performance gate
6. Update frontend: weight hydration from API, outcome POST, status panel
7. Configure Nginx + PM2 on VPS
8. Future: Coinbase execution module (after model validated)
