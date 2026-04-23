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
import { calcADX, calcWilliamsR, calcStochastic, calcROC, calcZScore,
         calcCMF, calcMaxDrawdown, calcSharpe } from "./quant";

const YAHOO_PROXIES = [
  u => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
  u => `https://corsproxy.io/?${encodeURIComponent(u)}`,
];

async function fetchYahooHistorical(symbol, daysAgo = 7) {
  // 5m candles capped at 60 days range by Yahoo — 7 days fits easily.
  const range = `${Math.max(daysAgo + 1, 2)}d`;
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=5m&range=${range}`;
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
  const closes = bars.closes.slice(0, i + 1);
  const highs  = bars.highs.slice(0, i + 1);
  const lows   = bars.lows.slice(0, i + 1);
  const volumes = bars.volumes.slice(0, i + 1);
  if (closes.length < 60) return null;

  const ind = computeIndicators(closes, highs, lows, volumes);
  const quant = {
    adx:        calcADX(highs, lows, closes),
    williamsR:  calcWilliamsR(highs, lows, closes),
    stochastic: calcStochastic(highs, lows, closes),
    roc:        calcROC(closes),
    zScore:     calcZScore(closes),
    cmf:        calcCMF(highs, lows, closes, volumes),
    maxDrawdown:calcMaxDrawdown(closes),
    sharpe:     calcSharpe(closes),
  };
  const price = closes[closes.length - 1];
  return {
    symbol, price,
    prevClose: closes[Math.max(0, closes.length - 79)],  // ~1 day back at 5m
    closes, highs, lows, volumes, quant,
    ...ind,
  };
}

// Simulate one trade: entry at bar i, hold for holdBars bars.
// Uses a 1.5-ATR stop and 3-ATR target to match the model's suggested levels.
function simulateOutcome(bars, i, verdict, holdBars, atr) {
  if (!atr || atr <= 0) return null;
  const entry = bars.closes[i];
  const stopDist = 1.5 * atr;
  const tgtDist  = 3.0 * atr;

  const endIdx = Math.min(i + holdBars, bars.closes.length - 1);
  let hitStop = false, hitTarget = false;

  for (let j = i + 1; j <= endIdx; j++) {
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

  const pnl = verdict === "BUY"
    ? (exitPrice - entry) / entry * 100
    : (entry - exitPrice) / entry * 100;

  const outcome = hitTarget ? "WIN"
                : hitStop   ? "LOSS"
                : pnl > 0.3 ? "WIN"
                : pnl < -0.3 ? "LOSS"
                : null;  // neutral — drop it

  return { entry, exitPrice, pnlPct: pnl, outcome, hitStop, hitTarget };
}

// Pick N random entry indices that leave room for the forward window
function pickEntries(totalBars, warmup, forward, n) {
  const lo = warmup, hi = totalBars - forward - 1;
  if (hi - lo < n * 2) return []; // not enough room
  const picks = new Set();
  let guard = 0;
  while (picks.size < n && guard++ < n * 20) {
    picks.add(lo + Math.floor(Math.random() * (hi - lo)));
  }
  return [...picks].sort((a, b) => a - b);
}

// Main entry — returns training samples ready for trainNN()
export async function runBacktest(symbols, opts = {}) {
  const {
    daysAgo = 7,
    holdHours = 3,
    samplesPerSymbol = 10,
    onProgress = () => {},
  } = opts;

  const HOLD_BARS = Math.round(holdHours * 12); // 5-min bars per hour = 12
  const WARMUP_BARS = 60;

  const trades = [];
  const errors = [];

  for (let s = 0; s < symbols.length; s++) {
    const symbol = symbols[s];
    onProgress({ phase: "fetching", symbol, done: s, total: symbols.length });

    const bars = await fetchYahooHistorical(symbol, daysAgo + 1);
    if (!bars) {
      errors.push({ symbol, reason: "fetch_failed" });
      continue;
    }

    const entries = pickEntries(bars.closes.length, WARMUP_BARS, HOLD_BARS, samplesPerSymbol);
    onProgress({ phase: "simulating", symbol, candidates: entries.length, done: s, total: symbols.length });

    for (const i of entries) {
      const q = buildQuoteAt(symbol, bars, i);
      if (!q) continue;
      const model = scoreSetup(q);
      const verdict = parseFloat(model.compositeProb) > 50 ? "BUY" : "SELL";
      const sim = simulateOutcome(bars, i, verdict, HOLD_BARS, q.atr);
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

  return { trades, wins, losses, errors };
}
