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

const YAHOO_PROXIES = [
  u => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
  u => `https://corsproxy.io/?${encodeURIComponent(u)}`,
];

// Try Polygon first if a key is available and (a) we need more history than
// Yahoo gives for 5-min bars, or (b) we need daily bars at any horizon.
// Otherwise Yahoo is fine and free.
async function fetchHistoricalBars(symbol, daysAgo, polygonKey, interval = "5m") {
  if (hasPolygonKey(polygonKey) && (daysAgo > 7 || interval === "1d")) {
    const p = await fetchPolygonBars(symbol, daysAgo, polygonKey, interval);
    if (p) return { bars: p, source: "polygon" };
  }
  const y = await fetchYahooHistorical(symbol, daysAgo, interval);
  if (y) return { bars: y, source: "yahoo" };
  return null;
}

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

  const trades = [];
  const errors = [];
  let barsSource = null;  // "polygon" | "yahoo" | mixed — reported back to UI

  for (let s = 0; s < symbols.length; s++) {
    const symbol = symbols[s];
    onProgress({ phase: "fetching", symbol, done: s, total: symbols.length });

    const fetched = await fetchHistoricalBars(symbol, daysAgo + 1, polygonKey, interval);
    if (!fetched) {
      errors.push({ symbol, reason: "fetch_failed" });
      continue;
    }
    const bars = fetched.bars;
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
      const modelCtx = {
        universe,
        macro: macroHist?.at(bars.timestamps[i]) || null,
        calendar: calendarFeaturesAt(bars.timestamps[i]),
        pead,
      };
      const model = scoreSetup(q, modelCtx);
      const verdict = parseFloat(model.compositeProb) > 50 ? "BUY" : "SELL";
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
    }
  }

  const wins = trades.filter(t => t.outcome === "WIN").length;
  const losses = trades.length - wins;

  onProgress({ phase: "done", trades: trades.length, wins, losses });

  return { trades, wins, losses, errors, costBps, holdHours, daysAgo, barsSource, interval };
}
