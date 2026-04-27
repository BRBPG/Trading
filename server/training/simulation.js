// Server-side historical bar fetcher — no CORS proxy needed (Node 18 native fetch).
import { fetchPolygonBars, hasPolygonKey } from "../../src/polygon.js";

// Direct Yahoo fetch without proxy
async function fetchYahooHistorical(symbol, daysAgo, interval = "5m") {
  const range = daysAgo <= 7 ? "7d" : daysAgo <= 59 ? "60d" : "1y";
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=${interval}&range=${range}`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(15000),
      headers: { "User-Agent": "Mozilla/5.0" } });
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
      highs.push(q.high?.[i]  ?? q.close[i]);
      lows.push(q.low?.[i]   ?? q.close[i]);
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
    return { bars: hit.bars, source: hit.source };

  if (hasPoly && (isCrypto || daysAgo > 7 || interval === "1d")) {
    const p = await fetchPolygonBars(symbol, daysAgo, polygonKey, interval);
    if (p) {
      barsCache.set(key, { bars: p, source: "polygon", fetchedAt: Date.now() });
      return { bars: p, source: "polygon" };
    }
  }
  const y = await fetchYahooHistorical(symbol, daysAgo, interval);
  if (y) {
    barsCache.set(key, { bars: y, source: "yahoo", fetchedAt: Date.now() });
    return { bars: y, source: "yahoo" };
  }
  return null;
}

export function clearSimCache() { barsCache.clear(); }

// Minimal 16-dim feature vector matching FEATURE_NAMES order in model.js.
// Price-only — macro/funding/OI dims zeroed (those require live API calls
// not suitable for a batch training job; models still train on the 9 active
// price dims which carry the primary signal).
export function buildFeatureVector(bars) {
  const { closes, highs, lows } = bars;
  const n = closes.length;
  if (n < 26) return null;

  const last = closes[n - 1];

  // RSI-14
  let gains = 0, losses = 0;
  for (let i = n - 14; i < n; i++) {
    const d = closes[i] - closes[i - 1];
    if (d > 0) gains += d; else losses -= d;
  }
  const avgG = gains / 14, avgL = losses / 14;
  const rsi = avgL === 0 ? 1 : avgG / (avgG + avgL);
  const rsi_c = (rsi - 0.5) * 2;

  // Momentum-10
  const mom_n = (last - closes[n - 11]) / (closes[n - 11] || 1);

  // EMA signals (fast/slow)
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

  // MACD signal
  const ema12 = ema(closes.slice(Math.max(0, n - 26)), 12);
  const ema26 = ema(closes.slice(Math.max(0, n - 40)), 26);
  const macd  = ema12 - ema26;
  const macd_s = macd / (Math.abs(ema26) || 1);

  // [0] rsi_c  [1] macd_s  [2] mom_n  [3] bb_c  [4] ema_s  [5] ema_m  [6] vol_n
  // [7..14] macro/funding/OI — zeroed  [15] PEAD/surprise prior — zeroed
  return [rsi_c, macd_s, mom_n, bb_c, ema_s, ema_m, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0];
}

// Generate labeled sim trades from historical bars.
// Samples up to maxSamples entry points per symbol, simulates a 3-hour hold.
export function generateSimTrades(symbol, bars, maxSamples = 60) {
  const { closes, highs, lows, timestamps } = bars;
  const HOLD_BARS = 36; // 36 × 5m = 3 hours
  const TARGET = 0.015;
  const STOP   = 0.010;

  const simTrades = [];
  const total = closes.length - HOLD_BARS - 51;
  if (total <= 0) return simTrades;

  const step = Math.max(1, Math.floor(total / maxSamples));

  for (let i = 50; i < closes.length - HOLD_BARS - 1; i += step) {
    const slice = {
      closes: closes.slice(0, i + 1),
      highs:  highs.slice(0, i + 1),
      lows:   lows.slice(0, i + 1),
    };
    const features = buildFeatureVector(slice);
    if (!features) continue;

    const entry = closes[i];
    let outcome = null;
    for (let j = i + 1; j <= i + HOLD_BARS && j < closes.length; j++) {
      if (highs[j] >= entry * (1 + TARGET)) { outcome = "WIN";  break; }
      if (lows[j]  <= entry * (1 - STOP))   { outcome = "LOSS"; break; }
    }
    if (!outcome) {
      const exitClose = closes[Math.min(i + HOLD_BARS, closes.length - 1)];
      outcome = exitClose >= entry ? "WIN" : "LOSS";
    }

    const ageDays = timestamps?.[i]
      ? (Date.now() / 1000 - timestamps[i]) / 86400
      : 0;

    simTrades.push({
      symbol,
      features,
      outcome,
      verdict: "BUY",
      labelBullish: outcome === "WIN" ? 1 : 0,
      ageDays,
      timestamp: timestamps?.[i] ?? 0,
    });
  }
  return simTrades;
}
