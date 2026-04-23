// ─── Simulate-and-train — synthetic labelled data for the NN ───────────────
// Pulls real 5-minute candles from Yahoo for the last N days, samples random
// "entry points", computes features as they would have looked at that moment
// (no look-ahead), simulates a 3-hour hold, labels WIN/LOSS, and trains the
// NN on the result.
//
// Pipeline per symbol:
//   1. Fetch {daysAgo+1} days of 5m candles
//   2. Slice to entries in the [daysAgo, daysAgo - holdHours] window
//   3. For each candidate entry T:
//        - Build features from bars [T-50 .. T]  (no future data)
//        - Ask the CURRENT model (LR+tree) what it would have done
//        - Look forward {holdBars} bars from T
//        - BUY: did (high - entry)/entry reach +target before low dropped to stop?
//        - SELL: inverse
//        - If neither triggered, use final-bar pnl sign
//   4. Return [{ symbol, features, verdict, entryPrice, exitPrice, pnlPct,
//               outcome: "WIN" | "LOSS", ageDays }]

import { scoreSetup } from "./model";
import { computeIndicators } from "./mockData";
import { fetchPolygonBars, hasPolygonKey } from "./polygon";
import { fetchMacroHistorical } from "./macro";
import { calendarFeaturesAt } from "./calendar";
import { computePeadFeatures } from "./earnings";
import { timeSeriesMomentumAt, approximateDominanceZFromBTCReturns } from "./crypto";

const YAHOO_PROXIES = [
  u => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
  u => `https://corsproxy.io/?${encodeURIComponent(u)}`,
];

// ─── Session bars cache ─────────────────────────────────────────────────────
// The old multi-sim fetched historical bars for every symbol on every run
// (5 runs × 15 symbols = 75 Polygon calls = 15 min at Starter's 5/min limit).
// That was the tax that made only 5 runs feasible, which was the tax that
// made results statistically thin.
//
// Cache bars in-memory for the session keyed by (symbol, interval, daysAgo,
// source). First fetch populates; subsequent calls in the same page reuse.
// TTL 10 min — enough for a full multi-sim session (20-50 runs) but short
// enough that re-running after a break picks up fresh data.
//
// Impact: after warmup, each additional sim just re-samples entry points
// from already-fetched bars — near-zero network cost. Can comfortably run
// 20+ sims where 5 was the previous practical ceiling.
const barsCache = new Map();  // key → { bars, source, fetchedAt }
const BARS_CACHE_TTL_MS = 10 * 60 * 1000;

// ─── Unbiased coin flip via OS-level entropy ───────────────────────────────
// Math.random() is a deterministic PRNG (xorshift/LCG variant per engine).
// For short sequences inside tight loops it can exhibit non-obvious bias,
// and critically its output is correlated with whatever seeded it — often
// time-based, which can accidentally correlate with bar-timestamp order in
// a sim. Crypto.getRandomValues uses OS-level entropy pools (thermal noise,
// hardware jitter, timing sources, mouse/keyboard jitter) which are the
// same class of physical-process entropy as atmospheric-noise services
// like random.org but without the network round-trip. Industry-standard
// for anywhere correlation with program state could contaminate a
// statistical test. Used here for the cold-start direction tie-break so
// the bootstrap labels are provably unbiased.
//
// Pull a block of bytes at a time and consume them — calling
// getRandomValues per trade is fine but block-sampling is cheaper and
// gives identical entropy.
let entropyBuf = null;
let entropyIdx = 0;
function secureCoinFlip() {
  if (!entropyBuf || entropyIdx >= entropyBuf.length) {
    entropyBuf = new Uint8Array(256);
    crypto.getRandomValues(entropyBuf);
    entropyIdx = 0;
  }
  // Discard bytes in [128, 255] if you want perfectly unbiased — with 256
  // outcomes, 128/256 = exactly 50% for < 128, so no rejection sampling
  // needed. Use the MSB directly.
  return (entropyBuf[entropyIdx++] & 1) === 1;
}

function cacheKey(symbol, interval, daysAgo, hasPolygon) {
  return `${symbol}|${interval}|${daysAgo}|${hasPolygon ? "poly" : "yh"}`;
}

// Try Polygon first if a key is available and (a) we need more history than
// Yahoo gives for 5-min bars, or (b) we need daily bars at any horizon,
// or (c) this is a crypto symbol (Polygon Crypto covers these cleanly now).
// Otherwise Yahoo is fine and free.
async function fetchHistoricalBars(symbol, daysAgo, polygonKey, interval = "5m") {
  const hasPoly = hasPolygonKey(polygonKey);
  const isCrypto = /-USD(T)?$/.test(symbol.toUpperCase());

  // Check session cache first — avoids re-hitting Polygon rate limits on
  // repeat sims over the same window.
  const key = cacheKey(symbol, interval, daysAgo, hasPoly);
  const hit = barsCache.get(key);
  if (hit && Date.now() - hit.fetchedAt < BARS_CACHE_TTL_MS) {
    return { bars: hit.bars, source: hit.source, cached: true };
  }

  // Route to Polygon when the subscription can cover it — crypto always
  // benefits from Polygon Crypto when present, equities only when we need
  // more than Yahoo's 7d 5-min or any daily horizon.
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

// Exposed so the UI can show cache state + let user force-clear between runs.
export function clearBarsCache() { barsCache.clear(); }
export function barsCacheSize() { return barsCache.size; }

async function fetchYahooHistorical(symbol, daysAgo = 7, interval = "5m") {
  // Yahoo caps at ~60 days for 5-min bars but supports several years of daily.
  // For daily mode we can pull up to 10 years without issue.
  const isDaily = interval === "1d";
  const yInterval = isDaily ? "1d" : "5m";
  const clamped = isDaily
    ? Math.max(daysAgo + 1, 10)
    : Math.min(Math.max(daysAgo + 1, 2), 60);
  // Yahoo wants specific range strings for longer spans
  const range = isDaily
    ? (daysAgo <= 30 ? "1mo" : daysAgo <= 90 ? "3mo" : daysAgo <= 180 ? "6mo" : daysAgo <= 365 ? "1y" : "2y")
    : `${clamped}d`;
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=${yInterval}&range=${range}`;
  for (const proxy of YAHOO_PROXIES) {
    try {
      const res = await fetch(proxy(url), { signal: AbortSignal.timeout(12000) });
      if (!res.ok) continue;
      const data = await res.json();
      const r = data?.chart?.result?.[0];
      if (!r) continue;
      const ts = r.timestamp || [];
      const q = r.indicators?.quote?.[0] || {};
      const closes = [], highs = [], lows = [], volumes = [], timestamps = [];
      for (let i = 0; i < ts.length; i++) {
        // Yahoo leaves nulls for halted/pre-open minutes — skip them
        if (q.close?.[i] == null || q.high?.[i] == null || q.low?.[i] == null) continue;
        closes.push(q.close[i]);
        highs.push(q.high[i]);
        lows.push(q.low[i]);
        volumes.push(q.volume?.[i] ?? 0);
        timestamps.push(ts[i]);
      }
      if (closes.length < 80) continue; // need enough bars for indicators + forward window
      return { closes, highs, lows, volumes, timestamps };
    } catch { /* try next */ }
  }
  return null;
}

// Build a fake quote object at time-step i using only bars [0..i]
function buildQuoteAt(symbol, bars, i) {
  // LOOK-AHEAD FIX: in live trading, at the moment we decide to enter, the
  // close of bar i is the MOST RECENT observation we have. We cannot also
  // trade at that same close — we trade the next available bar. So features
  // are built from closes[0..i] (inclusive, the signal-decision bar), and
  // entry happens at closes[i+1] in the simulateOutcome loop. The sim loop's
  // j=i+1 forward iteration starts at the entry bar.
  const closes = bars.closes.slice(0, i + 1);
  const highs  = bars.highs.slice(0, i + 1);
  const lows   = bars.lows.slice(0, i + 1);
  const volumes = bars.volumes.slice(0, i + 1);
  if (closes.length < 100) return null;  // bumped from 60: EMA50 + BB20 need proper warmup

  const ind = computeIndicators(closes, highs, lows, volumes);
  // NOTE: scoreSetup never reads q.quant, so the full quant block that
  // used to live here was pure dead work in the sim hot path. Removed.
  const price = closes[closes.length - 1];
  return {
    symbol, price,
    prevClose: closes[Math.max(0, closes.length - 79)],  // ~1 day back at 5m
    closes, highs, lows, volumes,
    ...ind,
  };
}

// Simulate one trade: entry at bar i, max-hold = holdBars.
// Uses a 1.5-ATR stop and 3-ATR target to match the model's suggested levels.
// costBps: round-trip transaction cost in basis points (1 bp = 0.01%).
//   Default 15 bps ≈ commission + spread + typical slippage on liquid US
//   equities at a retail broker. Set to 0 to see gross P&L.
//   Applied to EVERY trade regardless of exit reason — covers the reality that
//   getting filled at stop or target also costs spread/slippage.
function simulateOutcome(bars, i, verdict, holdBars, atr, costBps = 15) {
  if (!atr || atr <= 0) return null;
  // Signal is formed from bars [0..i] (close of bar i is the latest input).
  // Earliest realistic entry is bar i+1 — we've already consumed bar i's
  // close as an input; we can't both use it and trade at it. Exit simulation
  // walks forward from i+2.
  const entryIdx = i + 1;
  if (entryIdx >= bars.closes.length) return null;
  const entry = bars.closes[entryIdx];
  const stopDist = 1.5 * atr;
  const tgtDist  = 3.0 * atr;

  const endIdx = Math.min(entryIdx + holdBars, bars.closes.length - 1);
  let hitStop = false, hitTarget = false;

  for (let j = entryIdx + 1; j <= endIdx; j++) {
    const hi = bars.highs[j], lo = bars.lows[j];
    if (verdict === "BUY") {
      if (lo <= entry - stopDist) { hitStop = true; break; }
      if (hi >= entry + tgtDist)  { hitTarget = true; break; }
    } else { // SELL
      if (hi >= entry + stopDist) { hitStop = true; break; }
      if (lo <= entry - tgtDist)  { hitTarget = true; break; }
    }
  }

  const exitPrice = hitStop   ? (verdict === "BUY" ? entry - stopDist : entry + stopDist)
                  : hitTarget ? (verdict === "BUY" ? entry + tgtDist  : entry - tgtDist)
                  : bars.closes[endIdx];

  const pnlGross = verdict === "BUY"
    ? (exitPrice - entry) / entry * 100
    : (entry - exitPrice) / entry * 100;

  // Deduct round-trip costs BEFORE outcome labelling, otherwise marginally-
  // winning timeout trades flip from LOSS to WIN under real-world conditions.
  const pnlNet = pnlGross - costBps / 100;

  // Outcome classification:
  //   - If the target was touched, it's a WIN regardless of costs (you got
  //     the target price; costs happened at entry/exit but the labelling
  //     reflects which rail was hit first).
  //   - Same for stops.
  //   - For timeout trades, classify by NET P&L with a small deadband so
  //     trades near zero aren't counted either way.
  const outcome = hitTarget ? "WIN"
                : hitStop   ? "LOSS"
                : pnlNet > 0.3 ? "WIN"
                : pnlNet < -0.3 ? "LOSS"
                : null;  // neutral — drop it

  return {
    entry, exitPrice,
    pnlPct: pnlNet,           // NET of costs — this is what metrics consume
    pnlPctGross: pnlGross,    // available for debugging / gross-vs-net UI
    costBps,
    outcome, hitStop, hitTarget,
  };
}

// Pick N random entry indices that leave room for the forward window. If the
// available range is tight (e.g. daily mode with limited history), CAP n at
// the available range rather than silently returning empty — an empty return
// here cascades to "sim ran and produced nothing" with no visible error.
function pickEntries(totalBars, warmup, forward, n) {
  const lo = warmup, hi = totalBars - forward - 1;
  const range = hi - lo;
  if (range <= 0) return [];
  // Allow sampling up to ~80% of available range to avoid total clustering
  // while still producing output on tight daily-mode histories.
  const feasibleN = Math.max(1, Math.min(n, Math.floor(range * 0.8)));
  const picks = new Set();
  let guard = 0;
  while (picks.size < feasibleN && guard++ < feasibleN * 40) {
    picks.add(lo + Math.floor(Math.random() * range));
  }
  return [...picks].sort((a, b) => a - b);
}

// Main entry — returns training samples ready for trainNN()
// interval: "5m" intraday (original) or "1d" daily. In daily mode:
//   holdHours is interpreted as holdDays (default 5d = 1-week swing)
//   HOLD_BARS = holdDays (one bar per day)
//   WARMUP_BARS = 50 (enough for EMA50 on daily series)
export async function runBacktest(symbols, opts = {}) {
  const {
    interval = "5m",
    daysAgo = 7,
    holdHours = 3,
    samplesPerSymbol = 10,
    costBps = 15,  // round-trip in basis points; realistic retail default
    polygonKey = null,
    earningsMap = {},  // { symbol → earnings[] } for PEAD features
    universe = "equities",  // routes scoreSetup to the correct per-universe models
    onProgress = () => {},
  } = opts;

  // In daily mode, holdHours becomes "hold days" — the user's 1/3/5/10 day
  // dropdown values pass through as bar counts directly. The 5-min mode
  // still treats holdHours as hours × 12 bars/hour.
  const isDaily = interval === "1d";
  const HOLD_BARS = isDaily ? Math.round(holdHours) : Math.round(holdHours * 12);
  const WARMUP_BARS = isDaily ? 50 : 60;

  // Pre-fetch macro history ONCE per backtest run — the at(t) helper does
  // cheap in-memory binary-search lookups per entry, so this is O(symbols ×
  // samples) with no extra network I/O inside the main loop.
  onProgress({ phase: "fetching_macro", done: 0, total: symbols.length });
  const macroHist = await fetchMacroHistorical(daysAgo + 2).catch(() => null);

  // For crypto, pre-fetch BTC bars to compute the BTC-dominance proxy at
  // each entry's timestamp (CoinGecko doesn't give historical dominance
  // on free tier; we approximate via BTC's own 14d return sign/magnitude).
  // Cached via the same bars-cache so runs 2..N reuse it.
  let btcHistBars = null;
  if (universe === "crypto" && symbols.includes("BTC-USD")) {
    const btcFetch = await fetchHistoricalBars("BTC-USD", daysAgo + 2, polygonKey, interval);
    btcHistBars = btcFetch?.bars || null;
  }

  const trades = [];
  const errors = [];
  let barsSource = null;  // "polygon" | "yahoo" | mixed — reported back to UI
  // Per-symbol fetch outcome log — surfaced in the UI so the user can see
  // which symbols dropped and why without needing devtools (iPad etc.).
  // Entries: { symbol, source: "polygon"|"yahoo"|null, cached, bars, skipped }
  const fetchLog = [];

  for (let s = 0; s < symbols.length; s++) {
    const symbol = symbols[s];
    onProgress({ phase: "fetching", symbol, done: s, total: symbols.length });

    const fetched = await fetchHistoricalBars(symbol, daysAgo + 1, polygonKey, interval);
    if (!fetched) {
      // Silent drops are the worst failure mode — record both in the
      // errors array (for legacy callers) and the fetchLog (for UI).
      console.warn(`[backtest] fetch_failed for ${symbol} — both Polygon and Yahoo returned null.`);
      errors.push({ symbol, reason: "fetch_failed", polygonKeyPresent: !!polygonKey });
      fetchLog.push({ symbol, source: null, bars: 0, trades: 0, reason: "fetch_failed" });
      continue;
    }
    const bars = fetched.bars;
    fetchLog.push({ symbol, source: fetched.source, bars: bars.closes.length, cached: !!fetched.cached, trades: 0 });
    // Diagnostic: log which source served each symbol on its FIRST fetch
    // (i.e. not a cache hit), so we can spot unexpected fallbacks (e.g.
    // Polygon failing silently for BNB and Yahoo picking up the slack).
    if (!fetched.cached) console.info(`[backtest] ${symbol} served from ${fetched.source}`);
    if (barsSource == null) barsSource = fetched.source;
    else if (barsSource !== fetched.source) barsSource = "mixed";

    // Reserve +1 bar beyond HOLD_BARS because entry is now at i+1, not i.
    const entries = pickEntries(bars.closes.length, WARMUP_BARS, HOLD_BARS + 1, samplesPerSymbol);
    onProgress({ phase: "simulating", symbol, candidates: entries.length, done: s, total: symbols.length });

    for (const i of entries) {
      const q = buildQuoteAt(symbol, bars, i);
      if (!q) continue;
      // PEAD features computed point-in-time against the entry bar's
      // timestamp — only earnings events BEFORE this bar count, otherwise
      // we'd leak future knowledge. computePeadFeatures filters by
      // referenceTimeMs internally.
      const barTsMs = bars.timestamps[i] * 1000;
      const pead = earningsMap[symbol]
        ? computePeadFeatures(earningsMap[symbol], barTsMs)
        : null;
      // Crypto context for this entry: dominance proxy (from BTC's own
      // 14d return at the entry time) + this symbol's time-series momentum.
      // Both point-in-time from cached bars — no network I/O per entry.
      // XS momentum rank left for a future commit (needs cross-symbol
      // coordination which the current loop shape doesn't expose).
      const cryptoContext = universe === "crypto" ? {
        dominanceZ: btcHistBars ? approximateDominanceZFromBTCReturns(btcHistBars, bars.timestamps[i]) : 0,
        tsMom:      timeSeriesMomentumAt(bars, i, 14, 30),
        xsMomRank:  0,  // TODO Phase 3d: compute across all symbols at timestamp i
      } : null;
      const baseMacro = macroHist?.at(bars.timestamps[i]) || null;
      const modelCtx = {
        universe,
        macro: baseMacro || cryptoContext ? { ...(baseMacro || {}), cryptoContext } : null,
        calendar: calendarFeaturesAt(bars.timestamps[i]),
        pead,
      };
      const model = scoreSetup(q, modelCtx);
      const prob = parseFloat(model.compositeProb) / 100;
      // SKIP NEUTRAL TRADES — with a COLD-START EXCEPTION.
      //
      // Normal case: when a trained model has low conviction (|prob-0.5|
      // < 0.02) we skip the trade so a one-sided tie-break doesn't
      // pollute the walk-forward with phantom signal. A trained model
      // with real signal produces plenty of >0.52 / <0.48 entries to
      // populate the sim honestly.
      //
      // Cold-start case: when EVERY ensemble component is untrained
      // (LR at default zeros, NN/GBM not ready, tree muted in crypto),
      // compositeProb resolves to EXACTLY 0.500. Skipping there produced
      // a deadlock on fresh crypto installs — 0 labelled trades means
      // nothing to train on, which means the models stay untrained, which
      // means 0 labelled trades forever. We detect that exact-0.5 state
      // and pick direction randomly so the user gets a balanced labelled
      // dataset to bootstrap training from. Outcomes are ground truth
      // regardless of the random verdict, so the resulting trades are
      // honest training data. After ONE train pass the LR drifts off
      // 0.5 and the normal skip-neutral gate resumes.
      const isColdStart = prob === 0.5;
      if (!isColdStart && Math.abs(prob - 0.5) < 0.02) continue;
      // Cold-start picks direction from OS entropy (crypto.getRandomValues)
      // not Math.random — see secureCoinFlip comment. Keeps the bootstrap
      // labels statistically independent of program state.
      const verdict = isColdStart
        ? (secureCoinFlip() ? "BUY" : "SELL")
        : (prob > 0.5 ? "BUY" : "SELL");
      const sim = simulateOutcome(bars, i, verdict, HOLD_BARS, q.atr, costBps);
      if (!sim || !sim.outcome) continue;

      const ageDays = (Date.now() - bars.timestamps[i] * 1000) / (24 * 3600 * 1000);
      trades.push({
        symbol,
        timestamp: bars.timestamps[i] * 1000,
        features: model.features,
        verdict,
        entryPrice: sim.entry,
        exitPrice: sim.exitPrice,
        pnlPct: sim.pnlPct,
        outcome: sim.outcome,
        hitStop: sim.hitStop,
        hitTarget: sim.hitTarget,
        ageDays,
      });
      // Increment trade counter on the symbol's fetchLog entry.
      const fl = fetchLog.find(f => f.symbol === symbol);
      if (fl) fl.trades++;
    }
  }

  const wins = trades.filter(t => t.outcome === "WIN").length;
  const losses = trades.length - wins;

  onProgress({ phase: "done", trades: trades.length, wins, losses });

  return { trades, wins, losses, errors, fetchLog, costBps, holdHours, daysAgo, barsSource, interval };
}
