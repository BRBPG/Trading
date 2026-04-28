// Server-side historical bar fetcher — no CORS proxy needed (Node 18 native fetch).
import { randomFillSync } from "node:crypto";
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

// OS-grade entropy coin flip — mirrors src/backtest.js secureCoinFlip via
// node:crypto.randomFillSync. Buffered 256 bytes at a time.
let entropyBuf = null;
let entropyIdx = 0;
function secureCoinFlip() {
  if (!entropyBuf || entropyIdx >= entropyBuf.length) {
    entropyBuf = Buffer.alloc(256);
    randomFillSync(entropyBuf);
    entropyIdx = 0;
  }
  return (entropyBuf[entropyIdx++] & 1) === 1;
}

// 16-dim feature vector matching FEATURE_NAMES_BTC order in src/model.js.
//   [0..6]   technicals from the bars themselves
//   [7..15]  crypto-context features supplied via lookups (each is a function
//            timestampSec -> number, returns 0 when underlying data missing).
//
// `timestampSec` is the entry-bar's timestamp in seconds (matches bars.timestamps[i]).
// `lookups` is the object returned by lookups.js buildBtcLookups; pass a default
// shape with no-op fns when lookups aren't available so the function still works.
export function buildFeatureVector(bars, timestampSec = 0, lookups = {}) {
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

  // Vol normalised — placeholder zero for slot 6 (the frontend's vol_n is also
  // computed off the raw quote object, not from bars; keeping at 0 matches
  // pre-existing behaviour).
  const vol_n = 0;

  // Slots 7-15 from the lookup object — each is (t) -> number-or-zero.
  // Order matches FEATURE_NAMES_BTC in src/model.js:
  //   [7] BTC_dom_z, [8] TS_mom_z, [9] XS_rank_150, [10] Fund_z,
  //   [11] DVOL-RV_z, [12] OI_z, [13] RV_ratio, [14] Park_ratio, [15] Breakout_20d
  const safe = (fn) => {
    try {
      const v = typeof fn === "function" ? fn(timestampSec) : 0;
      return Number.isFinite(v) ? v : 0;
    } catch { return 0; }
  };

  return [
    rsi_c, macd_s, mom_n, bb_c, ema_s, ema_m, vol_n,
    safe(lookups.dominanceZAt),
    safe(lookups.tsMomZAt),
    safe(lookups.xsRankAt),
    safe(lookups.fundingZAt),
    safe(lookups.dvolRvZAt),
    safe(lookups.oiZAt),
    safe(lookups.rvRatioAt),
    safe(lookups.parkinsonAt),
    safe(lookups.breakoutFlagAt),
  ];
}

// Bidirectional outcome simulator. Mirrors src/backtest.js simulateOutcome but
// trimmed for the server pipeline (no ATR-driven stops; uses fixed ±TARGET/STOP
// against entry close, same as the legacy server sim). For BUY: target above,
// stop below; for SELL: target below, stop above.
function simulateOutcomeBidi(bars, i, verdict, holdBars, target, stop) {
  const { closes, highs, lows } = bars;
  const entry = closes[i];
  const endIdx = Math.min(i + holdBars, closes.length - 1);
  let outcome = null;
  for (let j = i + 1; j <= endIdx; j++) {
    const hi = highs[j], lo = lows[j];
    if (verdict === "BUY") {
      if (hi >= entry * (1 + target)) { outcome = "WIN";  break; }
      if (lo <= entry * (1 - stop))   { outcome = "LOSS"; break; }
    } else {
      if (lo <= entry * (1 - target)) { outcome = "WIN";  break; }
      if (hi >= entry * (1 + stop))   { outcome = "LOSS"; break; }
    }
  }
  if (!outcome) {
    const exitClose = closes[endIdx];
    if (verdict === "BUY")  outcome = exitClose >= entry ? "WIN" : "LOSS";
    else                    outcome = exitClose <= entry ? "WIN" : "LOSS";
  }
  return { entry, outcome };
}

// Generate labeled sim trades from historical bars.
// Samples up to maxSamples entry points per symbol, simulates a 3-hour hold.
// `lookups` is the object from lookups.js buildBtcLookups; pass `{}` to fall
// back to all-zero slots 7-15 (matches the pre-fix behaviour).
export function generateSimTrades(symbol, bars, lookups = {}, maxSamples = 60) {
  const { closes, timestamps } = bars;
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
      highs:  bars.highs.slice(0, i + 1),
      lows:   bars.lows.slice(0, i + 1),
    };
    const ts = timestamps?.[i] ?? 0;
    const features = buildFeatureVector(slice, ts, lookups);
    if (!features) continue;

    const verdict = secureCoinFlip() ? "BUY" : "SELL";
    const { outcome } = simulateOutcomeBidi(bars, i, verdict, HOLD_BARS, TARGET, STOP);

    // labelBullish = "did the market go UP from entry to exit", which is what
    // compositeProb predicts. Mirrors src/backtest.js:660-661 exactly.
    const labelBullish = (verdict === "BUY"  && outcome === "WIN")  ||
                         (verdict === "SELL" && outcome === "LOSS") ? 1 : 0;

    const ageDays = ts ? (Date.now() / 1000 - ts) / 86400 : 0;

    simTrades.push({
      symbol,
      features,
      outcome,
      verdict,
      labelBullish,
      ageDays,
      timestamp: ts,
    });
  }
  return simTrades;
}
