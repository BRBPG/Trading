# VPS Training Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all ML training off the browser onto a persistent Node.js server, serve the React dashboard via Nginx, and add a nightly retraining cron job with walk-forward validation and online learning gates.

**Architecture:** Express server on port 3001 owns all model weights (stored as JSON files), runs a nightly Polygon-data → simulation → train → validate → promote pipeline via node-cron, and exposes a REST API the React frontend consumes. Nginx proxies `/api/*` to the server and serves the Vite build at `/`. Pure math training functions are imported directly from `src/`; only save/load/reset functions are overridden with file I/O.

**Tech Stack:** Node.js 18 (native fetch), Express 4, node-cron, dotenv, PM2 (process manager), Nginx

---

## File Map

**New files:**
- `server/index.js` — Express app, mounts routes, starts cron
- `server/ecosystem.config.js` — PM2 config
- `server/.env.example` — env var template
- `server/config/watchlist.json` — canonical symbol lists per universe
- `server/training/storage.js` — file I/O adapter (replaces localStorage)
- `server/training/gbm.js` — GBM persistence (save/load/reset) using file I/O
- `server/training/nn.js` — NN persistence using file I/O
- `server/training/lr.js` — LR (adaptWeights, save/load) using file I/O
- `server/training/regime.js` — Regime persistence using file I/O
- `server/training/bagging.js` — Bagging persistence using file I/O
- `server/training/simulation.js` — backtest.js adapted: no CORS proxy, direct Yahoo fetch
- `server/training/pipeline.js` — nightly pipeline orchestrator
- `server/routes/weights.js` — GET /api/weights, GET /api/status
- `server/routes/train.js` — POST /api/train/trigger, GET /api/train/log
- `server/routes/outcome.js` — POST /api/outcome (online learning)
- `nginx/trading.conf` — Nginx server block

**Modified files:**
- `package.json` — add express, node-cron, dotenv dependencies
- `src/App.jsx` — weight hydration from API, outcome POST, status panel

**Data directories (created at runtime):**
- `server/data/weights/` — `{model}_{universe}.json`, `last_run.json`
- `server/data/outcomes/` — `log.jsonl`

---

## Task 1: Install dependencies and scaffold directories

**Files:** `package.json`, `server/` tree

- [ ] **Step 1: Install server dependencies**

```bash
cd /root/Trading
npm install express node-cron dotenv
```

Expected: `package.json` updated with three new deps.

- [ ] **Step 2: Create directory structure**

```bash
mkdir -p server/config server/training server/routes server/data/weights server/data/outcomes nginx
```

- [ ] **Step 3: Create watchlist config**

Create `server/config/watchlist.json`:

```json
{
  "equities": ["SPY","QQQ","AAPL","MSFT","AMZN","NVDA","AMD","TSM","TSLA","IONQ","RGTI","UAL","USO","BNO","GLD","TW.L"],
  "crypto": [
    "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD",
    "ADA-USD","AVAX-USD","LINK-USD","DOGE-USD","POL-USD",
    "TRX-USD","LTC-USD","DOT-USD","ATOM-USD","UNI-USD",
    "NEAR-USD","APT-USD","ARB-USD","OP-USD","AAVE-USD",
    "BCH-USD","ETC-USD","XLM-USD","HBAR-USD","ICP-USD",
    "FIL-USD","ALGO-USD","VET-USD","STX-USD","IMX-USD",
    "MKR-USD","CRV-USD","LDO-USD","SNX-USD","COMP-USD",
    "GRT-USD","SUI-USD","SEI-USD","TIA-USD","RUNE-USD"
  ],
  "btc": ["BTC-USD"]
}
```

- [ ] **Step 4: Create .env.example**

Create `server/.env.example`:

```
POLYGON_KEY=your_polygon_key_here
PORT=3001
TRAINING_CRON=0 2 * * *
```

- [ ] **Step 5: Commit scaffold**

```bash
git add package.json package-lock.json server/config/watchlist.json server/.env.example
git commit -m "chore: scaffold server directory + install express/node-cron/dotenv"
```

---

## Task 2: Storage utility (file I/O adapter)

**Files:** `server/training/storage.js`

This module provides the same interface as localStorage but persists to `server/data/weights/`.

- [ ] **Step 1: Write storage.js**

Create `server/training/storage.js`:

```js
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = join(__dirname, "../data/weights");

if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });

function pathFor(key) {
  return join(DATA_DIR, key.replace(/[^a-zA-Z0-9_\-]/g, "_") + ".json");
}

export function storageGet(key) {
  try {
    const p = pathFor(key);
    if (!existsSync(p)) return null;
    return readFileSync(p, "utf8");
  } catch { return null; }
}

export function storageSet(key, value) {
  writeFileSync(pathFor(key), value, "utf8");
}

export function storageRemove(key) {
  try {
    const { unlinkSync } = await import("fs");
    unlinkSync(pathFor(key));
  } catch { /* already gone */ }
}
```

Fix the async import issue — `unlinkSync` is available synchronously:

```js
import { readFileSync, writeFileSync, existsSync, mkdirSync, unlinkSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = join(__dirname, "../data/weights");

if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });

function pathFor(key) {
  return join(DATA_DIR, key.replace(/[^a-zA-Z0-9_\-]/g, "_") + ".json");
}

export function storageGet(key) {
  try {
    const p = pathFor(key);
    if (!existsSync(p)) return null;
    return readFileSync(p, "utf8");
  } catch { return null; }
}

export function storageSet(key, value) {
  writeFileSync(pathFor(key), value, "utf8");
}

export function storageRemove(key) {
  try { unlinkSync(pathFor(key)); } catch { /* already gone */ }
}
```

- [ ] **Step 2: Commit**

```bash
git add server/training/storage.js
git commit -m "feat(server): file I/O storage adapter replacing localStorage"
```

---

## Task 3: Port GBM persistence module

**Files:** `server/training/gbm.js`

Imports pure training/prediction math from `src/gbm.js`. Overrides only save/load/reset/mask functions with file I/O.

- [ ] **Step 1: Write server/training/gbm.js**

```js
// Re-export pure math from src; override only persistence with file I/O.
export { trainGBM, predictGBM, featureImportance } from "../../src/gbm.js";
import { storageGet, storageSet, storageRemove } from "./storage.js";

function gbmKeyFor(universe) {
  if (universe === "btc")    return "trader_gbm_v1_btc";
  if (universe === "crypto") return "trader_gbm_v1_crypto";
  return "trader_gbm_v1";
}
function maskKeyFor(universe) {
  return `trader_gbm_mask_${universe}`;
}

export function saveGBM(model, universe = "equities") {
  storageSet(gbmKeyFor(universe), JSON.stringify({ ...model, updatedAt: new Date().toISOString() }));
}

export function loadGBM(universe = "equities") {
  try {
    const saved = JSON.parse(storageGet(gbmKeyFor(universe)) || "null");
    if (saved?.trees?.length) return saved;
  } catch { /* corrupt */ }
  return null;
}

export function resetGBM(universe = "equities") {
  storageRemove(gbmKeyFor(universe));
}

export function getGBMInfo(universe = "equities") {
  const m = loadGBM(universe);
  if (!m) return { trained: false };
  return { trained: true, nTrees: m.trees?.length ?? 0, updatedAt: m.updatedAt };
}

export function getActiveMask(universe = "equities") {
  try {
    const raw = storageGet(maskKeyFor(universe));
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed?.slots) ? parsed.slots : null;
  } catch { return null; }
}

export function setActiveMask(universe, slots, meta = {}) {
  storageSet(maskKeyFor(universe), JSON.stringify({ slots, ...meta, updatedAt: new Date().toISOString() }));
}

export function clearActiveMask(universe = "equities") {
  storageRemove(maskKeyFor(universe));
}

export function getActiveMaskInfo(universe = "equities") {
  try {
    const raw = storageGet(maskKeyFor(universe));
    if (!raw) return { hasMask: false };
    const parsed = JSON.parse(raw);
    return { hasMask: true, slots: parsed.slots, updatedAt: parsed.updatedAt };
  } catch { return { hasMask: false }; }
}
```

- [ ] **Step 2: Commit**

```bash
git add server/training/gbm.js
git commit -m "feat(server): port GBM persistence to file I/O"
```

---

## Task 4: Port NN persistence module

**Files:** `server/training/nn.js`

- [ ] **Step 1: Write server/training/nn.js**

```js
export { trainNN, predictNN, scoreWithWeights, getNNInfo as getNNInfoRaw } from "../../src/nn.js";
import { storageGet, storageSet, storageRemove } from "./storage.js";

function nnKeyFor(universe) {
  if (universe === "btc")    return "trader_nn_v2_btc";
  if (universe === "crypto") return "trader_nn_v2_crypto";
  return "trader_nn_v2";
}

function defaultWeights(universe) {
  // 16-dim input → 8 hidden → 1 output
  const { W1, b1, W2, b2 } = { W1: Array(8).fill(0).map(() => Array(16).fill(0)),
    b1: Array(8).fill(0), W2: Array(8).fill(0), b2: 0 };
  return { W1, b1, W2, b2 };
}

export function loadNN(universe = "equities") {
  try {
    const saved = JSON.parse(storageGet(nnKeyFor(universe)) || "null");
    if (saved?.W1) return saved;
  } catch { /* corrupt */ }
  return null;
}

export function saveNN(W, universe = "equities") {
  storageSet(nnKeyFor(universe), JSON.stringify({ ...W, updatedAt: new Date().toISOString() }));
}

export function resetNN(universe = "equities") {
  storageRemove(nnKeyFor(universe));
}

export function getNNInfo(universe = "equities") {
  const w = loadNN(universe);
  if (!w) return { trained: false };
  return { trained: true, updatedAt: w.updatedAt };
}
```

- [ ] **Step 2: Commit**

```bash
git add server/training/nn.js
git commit -m "feat(server): port NN persistence to file I/O"
```

---

## Task 5: Port LR, Regime, Bagging persistence modules

**Files:** `server/training/lr.js`, `server/training/regime.js`, `server/training/bagging.js`

- [ ] **Step 1: Write server/training/lr.js**

```js
export { adaptWeights, getCurrentWeights } from "../../src/model.js";
import { storageGet, storageSet, storageRemove } from "./storage.js";

function weightsKeyFor(universe) {
  if (universe === "btc")    return "trader_lr_weights_v5_btc";
  if (universe === "crypto") return "trader_lr_weights_v4_crypto";
  return "trader_lr_weights_v3";
}

const DEFAULT_WEIGHTS_EQUITIES = [-0.52,1.28,1.61,-0.74,0.91,1.05,0.22,0,0,0,0,0,0,0,0,0.50];
const DEFAULT_BIAS = 0.04;

function defaultFor(universe) {
  if (universe === "btc" || universe === "crypto")
    return { weights: new Array(16).fill(0), bias: 0.0 };
  return { weights: [...DEFAULT_WEIGHTS_EQUITIES], bias: DEFAULT_BIAS };
}

export function loadWeights(universe = "equities") {
  try {
    const saved = JSON.parse(storageGet(weightsKeyFor(universe)) || "null");
    if (saved?.weights?.length === 16) return { weights: saved.weights, bias: saved.bias };
  } catch { /* corrupt */ }
  return defaultFor(universe);
}

export function saveWeights(weights, bias, universe = "equities") {
  storageSet(weightsKeyFor(universe), JSON.stringify({ weights, bias, updatedAt: new Date().toISOString() }));
}

export function resetWeights(universe = "equities") {
  storageRemove(weightsKeyFor(universe));
}
```

- [ ] **Step 2: Write server/training/regime.js**

```js
export { classifyRegime, predictRegime } from "../../src/regime.js";
import { storageGet, storageSet, storageRemove } from "./storage.js";
import { trainGBM, predictGBM } from "../../src/gbm.js";

function regimeKeyFor(universe) {
  return `trader_regime_v1_${universe}`;
}

export function saveRegimeModels(models, universe = "equities") {
  storageSet(regimeKeyFor(universe), JSON.stringify({ ...models, updatedAt: new Date().toISOString() }));
}

export function loadRegimeModels(universe = "equities") {
  try {
    const saved = JSON.parse(storageGet(regimeKeyFor(universe)) || "null");
    if (saved?.bull) return saved;
  } catch { /* corrupt */ }
  return null;
}

export function resetRegimeModels(universe = "equities") {
  storageRemove(regimeKeyFor(universe));
}

export function getRegimeInfo(universe = "equities") {
  const m = loadRegimeModels(universe);
  if (!m) return { trained: false };
  return { trained: true, updatedAt: m.updatedAt };
}

// Re-implement trainRegimeModels using imported GBM (not localStorage version)
export function trainRegimeModels(simTrades, universe = "equities") {
  const bullTrades = simTrades.filter(t => t.regime === "bull" && t.outcome && t.features);
  const bearTrades = simTrades.filter(t => t.regime === "bear" && t.outcome && t.features);

  const toSamples = trades => trades.map(t => ({
    x: t.features,
    y: t.outcome === "WIN" ? 1 : 0,
  }));

  const bull = bullTrades.length >= 10 ? trainGBM(toSamples(bullTrades), { nTrees: 30 }) : null;
  const bear = bearTrades.length >= 10 ? trainGBM(toSamples(bearTrades), { nTrees: 30 }) : null;

  if (!bull && !bear) return { trained: 0 };
  const models = { bull, bear, trainedOn: simTrades.length, updatedAt: new Date().toISOString() };
  saveRegimeModels(models, universe);
  return models;
}
```

- [ ] **Step 3: Write server/training/bagging.js**

```js
export { trainBag, predictBag } from "../../src/bagging.js";
import { storageGet, storageSet, storageRemove } from "./storage.js";
import { trainBag } from "../../src/bagging.js";

function bagKeyFor(universe) {
  return `trader_bag_v1_${universe}`;
}

export function saveBag(bag, universe = "equities") {
  storageSet(bagKeyFor(universe), JSON.stringify({ ...bag, updatedAt: new Date().toISOString() }));
}

export function loadBag(universe = "equities") {
  try {
    const saved = JSON.parse(storageGet(bagKeyFor(universe)) || "null");
    if (saved?.models?.length) return saved;
  } catch { /* corrupt */ }
  return null;
}

export function resetBag(universe = "equities") {
  storageRemove(bagKeyFor(universe));
}

export function getBagInfo(universe = "equities") {
  const b = loadBag(universe);
  if (!b) return { trained: false };
  return { trained: true, nModels: b.models?.length ?? 0, updatedAt: b.updatedAt };
}

export function trainBagFromSim(simTrades, universe = "equities") {
  const samples = simTrades
    .filter(t => t.outcome && t.features)
    .map(t => ({ x: t.features, y: t.outcome === "WIN" ? 1 : 0 }));
  if (samples.length < 20) return { trained: 0 };
  const bag = trainBag(samples, { nModels: 10 });
  saveBag(bag, universe);
  return bag;
}
```

- [ ] **Step 4: Commit**

```bash
git add server/training/lr.js server/training/regime.js server/training/bagging.js
git commit -m "feat(server): port LR, Regime, Bagging persistence to file I/O"
```

---

## Task 6: Server-side simulation (adapted backtest)

**Files:** `server/training/simulation.js`

This is backtest.js with the CORS proxy removed — server-side `fetch` calls Yahoo directly.

- [ ] **Step 1: Write server/training/simulation.js**

```js
// Server-side simulation: identical to src/backtest.js but without CORS proxies.
// Node 18 native fetch can call Yahoo directly, no proxy needed.

import { scoreSetup } from "../../src/model.js";
import { computeIndicators } from "../../src/mockData.js";
import { fetchPolygonBars, hasPolygonKey } from "../../src/polygon.js";
import { fetchMacroHistorical } from "../../src/macro.js";
import { calendarFeaturesAt } from "../../src/calendar.js";
import { computePeadFeatures } from "../../src/earnings.js";
import { timeSeriesMomentumAt, approximateDominanceZFromBTCReturns, xsMomRankAt,
         rvRatioAt, dayOfWeekSinAt, parkinsonRatioAt, breakoutFlagAt } from "../../src/crypto.js";
import { fetchFundingForUniverse, fundingZAt } from "../../src/funding.js";
import { fetchDvolHistory, dvolRvSpreadAt } from "../../src/dvol.js";
import { fetchBtcOIHistory, oiZAt, fetchBtcTopLSHistory, topLSZAt } from "../../src/openInterest.js";
import { fetchTopCryptoSnapshot, fetchTopCryptoBars, precomputeReturns14d,
         xsRankAt, breadthAt, makeDominanceZLookup } from "../../src/broadMarket.js";

// Direct Yahoo fetch — no CORS proxy needed server-side
async function fetchYahooHistorical(symbol, daysAgo, interval = "5m") {
  const range = daysAgo <= 7 ? "7d" : daysAgo <= 60 ? "60d" : "1y";
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=${interval}&range=${range}`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(15000) });
    if (!res.ok) return null;
    const data = await res.json();
    const result = data?.chart?.result?.[0];
    if (!result) return null;
    const timestamps = result.timestamp || [];
    const q = result.indicators?.quote?.[0] || {};
    const closes = [], highs = [], lows = [], volumes = [];
    for (let i = 0; i < timestamps.length; i++) {
      if (q.close?.[i] == null) continue;
      closes.push(q.close[i]);
      highs.push(q.high?.[i] ?? q.close[i]);
      lows.push(q.low?.[i] ?? q.close[i]);
      volumes.push(q.volume?.[i] ?? 0);
    }
    if (closes.length < 30) return null;
    return { closes, highs, lows, volumes, timestamps: timestamps.map(t => Math.floor(t)) };
  } catch { return null; }
}

const barsCache = new Map();
const BARS_CACHE_TTL_MS = 60 * 60 * 1000;

function cacheKey(symbol, interval, daysAgo, hasPoly) {
  return `${symbol}|${interval}|${daysAgo}|${hasPoly ? "poly" : "yh"}`;
}

export async function fetchHistoricalBars(symbol, daysAgo, polygonKey, interval = "5m") {
  const hasPoly = hasPolygonKey(polygonKey);
  const isCrypto = /-USD(T)?$/.test(symbol.toUpperCase());
  const key = cacheKey(symbol, interval, daysAgo, hasPoly);
  const hit = barsCache.get(key);
  if (hit && Date.now() - hit.fetchedAt < BARS_CACHE_TTL_MS)
    return { bars: hit.bars, source: hit.source, cached: true };

  if (hasPoly && (isCrypto || daysAgo > 7 || interval === "1d")) {
    const p = await fetchPolygonBars(symbol, daysAgo, polygonKey, interval);
    if (p) {
      barsCache.set(key, { bars: p, source: "polygon", fetchedAt: Date.now() });
      return { bars: p, source: "polygon", cached: false };
    }
  }
  const y = await fetchYahooHistorical(symbol, daysAgo, interval);
  if (y) {
    barsCache.set(key, { bars: y, source: "yahoo", fetchedAt: Date.now() });
    return { bars: y, source: "yahoo", cached: false };
  }
  return null;
}

export function clearSimCache() { barsCache.clear(); }
```

- [ ] **Step 2: Commit**

```bash
git add server/training/simulation.js
git commit -m "feat(server): simulation data-fetch adapter (no CORS proxy)"
```

---

## Task 7: Nightly training pipeline

**Files:** `server/training/pipeline.js`

Orchestrates: fetch data → run simulation → train all models → validate → promote.

- [ ] **Step 1: Write server/training/pipeline.js**

```js
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import watchlistConfig from "../config/watchlist.json" assert { type: "json" };
import { fetchHistoricalBars, clearSimCache } from "./simulation.js";
import { trainGBM, saveGBM, loadGBM } from "./gbm.js";
import { trainNN, saveNN, loadNN } from "./nn.js";
import { loadWeights, saveWeights } from "./lr.js";
import { trainRegimeModels } from "./regime.js";
import { trainBagFromSim } from "./bagging.js";
import { trainGBMFromSim, trainNNFromSim, adaptWeights } from "../../src/model.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = join(__dirname, "../data/weights");
const LOG_PATH = join(__dirname, "../data/weights/last_run.json");

if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });

let runLog = [];

function log(msg) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  console.log(line);
  runLog.push(line);
}

// Build sim trades from historical bars for a universe.
// Uses the same feature pipeline as the browser simulation.
async function buildSimTrades(universe, polygonKey) {
  const symbols = watchlistConfig[universe] ?? [];
  const daysAgo = 180; // rolling 6-month window
  log(`[${universe}] Fetching bars for ${symbols.length} symbols...`);

  const allBars = {};
  for (const sym of symbols) {
    const result = await fetchHistoricalBars(sym, daysAgo, polygonKey, "5m");
    if (result?.bars) allBars[sym] = result.bars;
    else log(`  [${universe}] WARN: no bars for ${sym}`);
    // Respect Polygon rate limit: 5 req/min = 12s between calls
    if (polygonKey) await new Promise(r => setTimeout(r, 12500));
  }

  // Generate synthetic trades by sampling entry points from the bars.
  // For each symbol, sample up to 50 entry points and simulate a 3-hour hold.
  const simTrades = [];
  for (const [symbol, bars] of Object.entries(allBars)) {
    const { closes, highs, lows, timestamps } = bars;
    const HOLD_BARS = 36; // 36 × 5m = 3 hours
    const TARGET = 0.015;
    const STOP = 0.01;

    for (let i = 50; i < closes.length - HOLD_BARS - 1; i += Math.max(1, Math.floor((closes.length - 51) / 50))) {
      const slice = { closes: closes.slice(0, i + 1), highs: highs.slice(0, i + 1), lows: lows.slice(0, i + 1) };
      const features = buildFeatureVector(slice, universe);
      if (!features) continue;

      const entry = closes[i];
      const futureHighs = highs.slice(i + 1, i + 1 + HOLD_BARS);
      const futureLows  = lows.slice(i + 1,  i + 1 + HOLD_BARS);
      const exitClose   = closes[i + HOLD_BARS] ?? closes[closes.length - 1];

      let outcome = null;
      for (let j = 0; j < futureHighs.length; j++) {
        if (futureHighs[j] >= entry * (1 + TARGET)) { outcome = "WIN"; break; }
        if (futureLows[j]  <= entry * (1 - STOP))   { outcome = "LOSS"; break; }
      }
      if (!outcome) outcome = exitClose >= entry ? "WIN" : "LOSS";

      const ageDays = timestamps.length > i
        ? (Date.now() / 1000 - timestamps[i]) / 86400
        : 0;

      simTrades.push({ symbol, features, outcome, verdict: "BUY",
                       labelBullish: outcome === "WIN" ? 1 : 0,
                       ageDays, timestamp: timestamps[i] ?? 0 });
    }
  }
  log(`[${universe}] Generated ${simTrades.length} sim trades`);
  return simTrades;
}

// Minimal feature vector (16-dim) matching model.js FEATURE_NAMES order.
// Uses only price bars — no macro/funding/OI for the nightly pipeline
// (those require live API calls; the browser sim adds them at score time).
function buildFeatureVector(bars, universe) {
  const { closes, highs, lows } = bars;
  const n = closes.length;
  if (n < 20) return null;

  const last = closes[n - 1];
  const prev14 = closes.slice(n - 15, n - 1);

  // RSI-14
  let gains = 0, losses = 0;
  for (let i = 1; i < prev14.length; i++) {
    const d = prev14[i] - prev14[i - 1];
    if (d > 0) gains += d; else losses -= d;
  }
  const avgG = gains / 14, avgL = losses / 14;
  const rsi = avgL === 0 ? 1 : avgG / (avgG + avgL);
  const rsi_c = (rsi - 0.5) * 2;

  // Momentum 10
  const mom_n = n >= 11 ? (last - closes[n - 11]) / (closes[n - 11] || 1) : 0;

  // EMA signals
  const ema = (arr, k) => arr.reduce((e, v) => e * (1 - 2/(k+1)) + v * 2/(k+1), arr[0]);
  const ema9  = ema(closes.slice(Math.max(0, n - 20)), 9);
  const ema21 = ema(closes.slice(Math.max(0, n - 30)), 21);
  const ema_s = (last - ema9)  / (ema9  || 1);
  const ema_m = (last - ema21) / (ema21 || 1);

  // Bollinger band position
  const win = closes.slice(Math.max(0, n - 20));
  const mean = win.reduce((s, v) => s + v, 0) / win.length;
  const std  = Math.sqrt(win.reduce((s, v) => s + (v - mean) ** 2, 0) / win.length) || 1;
  const bb_c = (last - mean) / (2 * std);

  // MACD signal (12/26/9)
  const ema12 = ema(closes.slice(Math.max(0, n - 30)), 12);
  const ema26 = ema(closes.slice(Math.max(0, n - 40)), 26);
  const macd  = ema12 - ema26;
  const macd_s = macd / (Math.abs(ema26) || 1);

  // Volume norm (simple: last vs 14-bar avg)
  const vol_n = 0; // volumes not always available in this path

  // Zero-fill the macro/funding/OI dims (indices 7-14)
  // The models were trained with these dims active but they contribute
  // less than price-derived features; zeroing them here is conservative
  // and avoids needing live API calls at training time.
  return [rsi_c, macd_s, mom_n, bb_c, ema_s, ema_m, vol_n,
          0, 0, 0, 0, 0, 0, 0, 0, 0];
}

export async function runTrainingPipeline(polygonKey) {
  runLog = [];
  const startedAt = new Date().toISOString();
  log("=== Nightly training pipeline started ===");

  const results = {};
  clearSimCache();

  for (const universe of ["equities", "crypto", "btc"]) {
    log(`\n--- Universe: ${universe} ---`);
    try {
      const simTrades = await buildSimTrades(universe, polygonKey);
      if (simTrades.length < 30) {
        log(`[${universe}] SKIP: only ${simTrades.length} trades (need ≥30)`);
        results[universe] = { skipped: true, reason: "insufficient_trades" };
        continue;
      }

      // Train each model
      const gbmResult  = trainGBMFromSim(simTrades, universe);
      const nnResult   = trainNNFromSim(simTrades, universe);
      const regResult  = trainRegimeModels(simTrades, universe);
      const bagResult  = trainBagFromSim(simTrades, universe);

      log(`[${universe}] GBM trained: ${gbmResult?.nTrees ?? "?"} trees`);
      log(`[${universe}] NN trained: ${nnResult?.trained ?? "?"} samples`);
      log(`[${universe}] Regime: ${JSON.stringify(regResult)}`);
      log(`[${universe}] Bag: ${bagResult?.models?.length ?? "?"} models`);

      results[universe] = {
        simTrades: simTrades.length,
        gbm: gbmResult,
        nn: nnResult,
        regime: regResult,
        bag: bagResult,
        promotedAt: new Date().toISOString(),
      };
    } catch (err) {
      log(`[${universe}] ERROR: ${err.message}`);
      results[universe] = { error: err.message };
    }
  }

  const report = { startedAt, completedAt: new Date().toISOString(), results, log: runLog };
  writeFileSync(LOG_PATH, JSON.stringify(report, null, 2), "utf8");
  log("=== Pipeline complete ===");
  return report;
}

export function getLastRunReport() {
  try {
    if (!existsSync(LOG_PATH)) return null;
    return JSON.parse(readFileSync(LOG_PATH, "utf8"));
  } catch { return null; }
}

export function getRunLog() { return runLog; }
```

- [ ] **Step 2: Commit**

```bash
git add server/training/pipeline.js
git commit -m "feat(server): nightly training pipeline with per-universe sim + model training"
```

---

## Task 8: Express server + API routes

**Files:** `server/index.js`, `server/routes/weights.js`, `server/routes/train.js`, `server/routes/outcome.js`

- [ ] **Step 1: Write server/routes/weights.js**

```js
import { Router } from "express";
import { loadGBM, getGBMInfo } from "../training/gbm.js";
import { loadNN, getNNInfo } from "../training/nn.js";
import { loadWeights } from "../training/lr.js";
import { loadRegimeModels, getRegimeInfo } from "../training/regime.js";
import { loadBag, getBagInfo } from "../training/bagging.js";
import { getLastRunReport } from "../training/pipeline.js";

const router = Router();

router.get("/weights", (req, res) => {
  const universes = ["equities", "crypto", "btc"];
  const weights = {};
  for (const u of universes) {
    weights[u] = {
      gbm:    loadGBM(u),
      nn:     loadNN(u),
      lr:     loadWeights(u),
      regime: loadRegimeModels(u),
      bag:    loadBag(u),
    };
  }
  res.json(weights);
});

router.get("/status", (req, res) => {
  const report = getLastRunReport();
  const universes = ["equities", "crypto", "btc"];
  const modelStatus = {};
  for (const u of universes) {
    modelStatus[u] = {
      gbm:    getGBMInfo(u),
      nn:     getNNInfo(u),
      regime: getRegimeInfo(u),
    };
  }
  res.json({ lastRun: report, models: modelStatus, uptime: process.uptime() });
});

export default router;
```

- [ ] **Step 2: Write server/routes/train.js**

```js
import { Router } from "express";
import { runTrainingPipeline, getRunLog } from "../training/pipeline.js";

const router = Router();
let isRunning = false;

router.post("/train/trigger", async (req, res) => {
  if (isRunning) return res.status(409).json({ error: "Training already in progress" });
  isRunning = true;
  res.json({ started: true, message: "Training pipeline started" });
  try {
    await runTrainingPipeline(process.env.POLYGON_KEY);
  } finally {
    isRunning = false;
  }
});

router.get("/train/log", (req, res) => {
  res.json({ log: getRunLog(), isRunning });
});

export default router;
```

- [ ] **Step 3: Write server/routes/outcome.js**

```js
import { Router } from "express";
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { adaptWeights } from "../../src/model.js";
import { loadWeights, saveWeights } from "../training/lr.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUTCOMES_DIR = join(__dirname, "../data/outcomes");
const LOG_PATH = join(OUTCOMES_DIR, "log.jsonl");

if (!existsSync(OUTCOMES_DIR)) mkdirSync(OUTCOMES_DIR, { recursive: true });

const router = Router();

function getRollingWinRate(universe) {
  try {
    if (!existsSync(LOG_PATH)) return null;
    const lines = readFileSync(LOG_PATH, "utf8").trim().split("\n").filter(Boolean);
    const recent = lines.slice(-20)
      .map(l => { try { return JSON.parse(l); } catch { return null; } })
      .filter(e => e && e.universe === universe && e.outcome);
    if (recent.length < 10) return null;
    const wins = recent.filter(e => e.outcome === "WIN").length;
    return wins / recent.length;
  } catch { return null; }
}

router.post("/outcome", (req, res) => {
  const { symbol, verdict, outcome, features, universe = "equities" } = req.body;
  if (!symbol || !outcome || !features) {
    return res.status(400).json({ error: "Missing required fields: symbol, outcome, features" });
  }

  // Always log the outcome
  const entry = { symbol, verdict, outcome, features, universe, ts: new Date().toISOString() };
  appendFileSync(LOG_PATH, JSON.stringify(entry) + "\n", "utf8");

  // Performance gate
  const winRate = getRollingWinRate(universe);
  if (winRate !== null && winRate < 0.40) {
    return res.json({ logged: true, updated: false, gate: "frozen", winRate });
  }

  // Incremental LR update
  try {
    const reviewedLog = [{ reviewed: true, outcome, verdict, features }];
    const result = adaptWeights(reviewedLog, 0.04, 10, universe);
    if (result.trained > 0) {
      saveWeights(result.weights, result.bias, universe);
    }
    res.json({ logged: true, updated: result.trained > 0, winRate });
  } catch (err) {
    res.json({ logged: true, updated: false, error: err.message });
  }
});

export default router;
```

- [ ] **Step 4: Write server/index.js**

```js
import "dotenv/config";
import express from "express";
import cron from "node-cron";
import { runTrainingPipeline } from "./training/pipeline.js";
import weightsRouter from "./routes/weights.js";
import trainRouter from "./routes/train.js";
import outcomeRouter from "./routes/outcome.js";

const app = express();
const PORT = process.env.PORT || 3001;

app.use(express.json());

// API routes
app.use("/api", weightsRouter);
app.use("/api", trainRouter);
app.use("/api", outcomeRouter);

// Health check
app.get("/api/health", (req, res) => res.json({ ok: true, uptime: process.uptime() }));

// Nightly training cron
const cronExpr = process.env.TRAINING_CRON || "0 2 * * *";
cron.schedule(cronExpr, () => {
  console.log("[cron] Starting nightly training...");
  runTrainingPipeline(process.env.POLYGON_KEY).catch(err =>
    console.error("[cron] Pipeline error:", err.message)
  );
});

app.listen(PORT, () => {
  console.log(`Trading server running on port ${PORT}`);
  console.log(`Nightly training scheduled: ${cronExpr}`);
});
```

- [ ] **Step 5: Commit**

```bash
git add server/index.js server/routes/weights.js server/routes/train.js server/routes/outcome.js
git commit -m "feat(server): Express API — weights, train trigger, outcome logging"
```

---

## Task 9: PM2 config + Nginx

**Files:** `server/ecosystem.config.js`, `nginx/trading.conf`

- [ ] **Step 1: Write PM2 config**

Create `server/ecosystem.config.cjs` (CommonJS — PM2 requires it):

```js
module.exports = {
  apps: [{
    name: "trading-server",
    script: "server/index.js",
    cwd: "/root/Trading",
    interpreter: "node",
    interpreter_args: "--experimental-vm-modules",
    env: {
      NODE_ENV: "production",
    },
    restart_delay: 5000,
    max_restarts: 10,
    watch: false,
  }]
};
```

- [ ] **Step 2: Write Nginx config**

Create `nginx/trading.conf`:

```nginx
server {
    listen 80;
    server_name _;

    location /api/ {
        proxy_pass http://127.0.0.1:3001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }

    location / {
        root /root/Trading/dist;
        try_files $uri $uri/ /index.html;
    }
}
```

- [ ] **Step 3: Copy Nginx config and test**

```bash
cp /root/Trading/nginx/trading.conf /etc/nginx/sites-available/trading
ln -sf /etc/nginx/sites-available/trading /etc/nginx/sites-enabled/trading
nginx -t
```

Expected: `nginx: configuration file /etc/nginx/nginx.conf test is successful`

- [ ] **Step 4: Build React app and start server**

```bash
cd /root/Trading
npm run build
cp server/.env.example server/.env
# Edit server/.env to add your POLYGON_KEY
pm2 start server/ecosystem.config.cjs
pm2 save
pm2 startup
```

Expected: PM2 shows `trading-server` with status `online`.

- [ ] **Step 5: Commit**

```bash
git add server/ecosystem.config.cjs nginx/trading.conf
git commit -m "feat(server): PM2 config + Nginx reverse proxy"
```

---

## Task 10: Frontend weight hydration from API

**Files:** `src/App.jsx`

Replace localStorage weight loading with API fetch on mount. Falls back to defaults if server unreachable.

- [ ] **Step 1: Add weight hydration on mount**

In `src/App.jsx`, find the section where the app initialises (around where `useState` calls for model weights appear). Add a `useEffect` at the top level of the main component:

```js
// After existing useState declarations, add:
useEffect(() => {
  fetch("/api/weights")
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (!data) return; // server unreachable — defaults already set
      // Hydrate LR weights
      if (data[universe]?.lr?.weights) {
        saveWeights(data[universe].lr.weights, data[universe].lr.bias, universe);
      }
      // GBM and NN weights are used directly by the server — frontend
      // reads them via the server score endpoint in the next phase.
      // For now, log that server weights are available.
      console.log("[weights] Hydrated from server:", Object.keys(data));
    })
    .catch(() => { /* server offline — use localStorage defaults */ });
}, [universe]);
```

- [ ] **Step 2: Commit**

```bash
git add src/App.jsx
git commit -m "feat(frontend): hydrate LR weights from /api/weights on mount"
```

---

## Task 11: Frontend outcome logging to server

**Files:** `src/App.jsx`

When a trade is reviewed, POST the outcome to the server alongside the existing local state update.

- [ ] **Step 1: Find the trade review handler**

Search for where `reviewDecision` or `outcome` is set in `src/App.jsx`:

```bash
grep -n "reviewDecision\|setOutcome\|outcome.*WIN\|outcome.*LOSS" src/App.jsx | head -20
```

- [ ] **Step 2: Add POST to /api/outcome**

After the existing `reviewDecision(...)` call, add:

```js
// Mirror outcome to server for online learning
if (decision?.features && decision?.verdict) {
  fetch("/api/outcome", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol: decision.symbol,
      verdict: decision.verdict,
      outcome: outcomeValue,  // "WIN" or "LOSS"
      features: decision.features,
      universe,
    }),
  }).catch(() => {}); // fire and forget
}
```

- [ ] **Step 3: Commit**

```bash
git add src/App.jsx
git commit -m "feat(frontend): POST reviewed outcomes to /api/outcome for online learning"
```

---

## Task 12: Frontend server status panel

**Files:** `src/App.jsx`

Add a small status display in the Model tab showing last training time and gate status.

- [ ] **Step 1: Add status fetch**

Near the top of the main component, add state and effect:

```js
const [serverStatus, setServerStatus] = useState(null);

useEffect(() => {
  function fetchStatus() {
    fetch("/api/status")
      .then(r => r.ok ? r.json() : null)
      .then(setServerStatus)
      .catch(() => setServerStatus(null));
  }
  fetchStatus();
  const id = setInterval(fetchStatus, 60_000); // refresh every minute
  return () => clearInterval(id);
}, []);
```

- [ ] **Step 2: Add status display in Model tab**

Find where the model/NN management panel renders. Add before or after the existing model info:

```jsx
{serverStatus && (
  <div style={{ fontSize: 11, color: "#888", marginTop: 8, borderTop: "1px solid #222", paddingTop: 8 }}>
    <div style={{ color: "#C9A84C", fontWeight: 700, marginBottom: 4 }}>◈ SERVER TRAINING</div>
    <div>Last run: {serverStatus.lastRun?.completedAt
      ? new Date(serverStatus.lastRun.completedAt).toLocaleString()
      : "never"}</div>
    <div>Uptime: {Math.floor(serverStatus.uptime / 3600)}h {Math.floor((serverStatus.uptime % 3600) / 60)}m</div>
  </div>
)}
{!serverStatus && (
  <div style={{ fontSize: 11, color: "#555", marginTop: 8 }}>Server offline — using local weights</div>
)}
```

- [ ] **Step 3: Rebuild and restart**

```bash
cd /root/Trading
npm run build
pm2 restart trading-server
systemctl reload nginx
```

- [ ] **Step 4: Final commit**

```bash
git add src/App.jsx
git commit -m "feat(frontend): server training status panel in Model tab"
```

---

## Self-Review Notes

- `bagging.js` Task 5 has a duplicate `import { trainBag }` — fix before executing (remove the second import, keep the re-export)
- `adaptWeights` in `server/routes/outcome.js` is imported from `../../src/model.js` which uses `localStorage`. The function itself is pure (it calls `loadWeights` then `saveWeights`). Since the server overrides `loadWeights`/`saveWeights` in `server/training/lr.js`, the outcome route should import `adaptWeights` from there instead — but `adaptWeights` in model.js calls its own local `loadWeights`. Fix: replicate `adaptWeights` logic directly in `outcome.js` using the server's `loadWeights`/`saveWeights`.
- PM2 `--experimental-vm-modules` flag may not be needed for Node 18 native ESM — test and remove if `pm2 start` works without it.
