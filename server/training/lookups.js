// Server-side fetchers for slots 7-15 of the 16-dim feature vector.
// Frontend equivalents live in src/funding.js, src/dvol.js, src/openInterest.js,
// src/crypto.js, src/broadMarket.js — but we can't import them: they go through
// /api/proxy and rely on browser caches. Server runs in Node 18+, hits the
// upstream APIs directly, and keeps in-memory session caches.
//
// IP CONSTRAINTS:
//   - Hetzner IP 5.161.246.161 is GEO-BLOCKED by Binance fapi (HTTP 451).
//     Funding history MUST go through Coinalyze; OI is fapi-first with
//     Coinalyze fallback.
//   - Deribit, CoinGecko, Coinalyze are all reachable.
//
// FAILURE POLICY: every fetcher returns null on error and the corresponding
// lookup function then returns 0. We never fabricate, never abort the pipeline
// — slot just stays zero and the per-run log records why.

const COINALYZE_BASE  = "https://api.coinalyze.net/v1";
const DERIBIT_BASE    = "https://www.deribit.com/api/v2/public";
const BINANCE_OI_BASE = "https://fapi.binance.com/futures/data/openInterestHist";
const COINGECKO_BASE  = "https://api.coingecko.com/api/v3";

// Free-tier rate limits we honour:
//   Coinalyze ~100/min   → 600ms between calls
//   CoinGecko free ~30/min → 2000ms between calls
//   Deribit ~20/sec      → no pacing needed for the handful we make
const COINALYZE_DELAY_MS = 600;
const COINGECKO_DELAY_MS = 2000;

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ─── Funding (slot 10) — Coinalyze ───────────────────────────────────────────
// Coinalyze free tier covers BTC perpetual funding. Endpoint returns aggregated
// funding-rate history; we want BTCUSDT_PERP.A (Binance perp aggregated).
export async function fetchFundingHistoryCoinalyze(daysAgo = 365) {
  const key = process.env.COINALYZE_KEY;
  if (!key) {
    return { records: null, reason: "COINALYZE_KEY not set" };
  }
  const to = Math.floor(Date.now() / 1000);
  const from = to - daysAgo * 86400;
  // BTCUSDT_PERP.A = Binance USDT-margined BTC perp on Coinalyze's symbol scheme.
  const url = `${COINALYZE_BASE}/funding-rate-history?symbols=BTCUSDT_PERP.A&interval=daily&from=${from}&to=${to}&api_key=${encodeURIComponent(key)}`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(15000) });
    if (!res.ok) return { records: null, reason: `coinalyze HTTP ${res.status}` };
    const data = await res.json();
    const block = Array.isArray(data) ? data[0] : null;
    const history = block?.history;
    if (!Array.isArray(history) || history.length < 10) {
      return { records: null, reason: "coinalyze empty history" };
    }
    const records = history
      .map(h => ({
        time: Math.floor(h.t || 0),
        rate: parseFloat(h.c ?? h.close ?? h.rate ?? 0),
      }))
      .filter(r => r.time > 0 && Number.isFinite(r.rate))
      .sort((a, b) => a.time - b.time);
    if (records.length < 10) return { records: null, reason: "coinalyze parse_short" };
    return { records, reason: `coinalyze ok (${records.length} pts)` };
  } catch (err) {
    return { records: null, reason: `coinalyze fetch error: ${err.message}` };
  }
}

// ─── DVOL (slot 11) — Deribit (works from Hetzner) ───────────────────────────
export async function fetchDvolHistory(daysAgo = 365) {
  const end = Date.now();
  const start = end - daysAgo * 86400 * 1000;
  const url = `${DERIBIT_BASE}/get_volatility_index_data?currency=BTC&start_timestamp=${start}&end_timestamp=${end}&resolution=1D`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(15000) });
    if (!res.ok) return { records: null, reason: `deribit HTTP ${res.status}` };
    const data = await res.json();
    const rows = data?.result?.data;
    if (!Array.isArray(rows) || rows.length < 10) {
      return { records: null, reason: "deribit empty" };
    }
    const records = rows
      .map(([tsMs, , , , close]) => ({
        time: Math.floor(tsMs / 1000),
        close: parseFloat(close),
      }))
      .filter(r => r.time > 0 && Number.isFinite(r.close) && r.close > 0)
      .sort((a, b) => a.time - b.time);
    if (records.length < 10) return { records: null, reason: "deribit parse_short" };
    return { records, reason: `deribit ok (${records.length} pts)` };
  } catch (err) {
    return { records: null, reason: `deribit fetch error: ${err.message}` };
  }
}

// ─── OI (slot 12) — Binance fapi first, Coinalyze fallback ───────────────────
async function fetchBtcOIBinance(period = "1d", limit = 30) {
  const url = `${BINANCE_OI_BASE}?symbol=BTCUSDT&period=${period}&limit=${limit}`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(15000) });
    if (!res.ok) return { records: null, reason: `binance HTTP ${res.status}` };
    const raw = await res.json();
    if (!Array.isArray(raw) || raw.length < 5) {
      return { records: null, reason: "binance empty" };
    }
    const records = raw
      .map(d => ({
        time:  Math.floor(d.timestamp / 1000),
        oi:    parseFloat(d.sumOpenInterest),
        oiUsd: parseFloat(d.sumOpenInterestValue),
      }))
      .filter(r => r.time > 0 && Number.isFinite(r.oi) && r.oi > 0)
      .sort((a, b) => a.time - b.time);
    if (records.length < 5) return { records: null, reason: "binance parse_short" };
    return { records, reason: `binance ok (${records.length} pts)` };
  } catch (err) {
    return { records: null, reason: `binance fetch error: ${err.message}` };
  }
}

async function fetchBtcOICoinalyze(daysAgo = 30) {
  const key = process.env.COINALYZE_KEY;
  if (!key) return { records: null, reason: "COINALYZE_KEY not set" };
  const to = Math.floor(Date.now() / 1000);
  const from = to - daysAgo * 86400;
  const url = `${COINALYZE_BASE}/open-interest-history?symbols=BTCUSDT_PERP.A&interval=daily&from=${from}&to=${to}&convert_to_usd=true&api_key=${encodeURIComponent(key)}`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(15000) });
    if (!res.ok) return { records: null, reason: `coinalyze OI HTTP ${res.status}` };
    const data = await res.json();
    const block = Array.isArray(data) ? data[0] : null;
    const history = block?.history;
    if (!Array.isArray(history) || history.length < 5) {
      return { records: null, reason: "coinalyze OI empty" };
    }
    const records = history
      .map(h => {
        const close = parseFloat(h.c ?? h.close ?? 0);
        return { time: Math.floor(h.t || 0), oi: close, oiUsd: close };
      })
      .filter(r => r.time > 0 && Number.isFinite(r.oi) && r.oi > 0)
      .sort((a, b) => a.time - b.time);
    if (records.length < 5) return { records: null, reason: "coinalyze OI parse_short" };
    return { records, reason: `coinalyze ok (${records.length} pts)` };
  } catch (err) {
    return { records: null, reason: `coinalyze OI fetch error: ${err.message}` };
  }
}

export async function fetchBtcOIHistory() {
  const bin = await fetchBtcOIBinance();
  if (bin.records) return bin;
  // Hetzner→fapi.binance.com normally returns 451 — try Coinalyze.
  await sleep(COINALYZE_DELAY_MS);
  const ca = await fetchBtcOICoinalyze();
  if (ca.records) return { ...ca, reason: `${bin.reason} → fallback ${ca.reason}` };
  return { records: null, reason: `${bin.reason} → ${ca.reason}` };
}

// ─── Top-150 crypto snapshot for dominance + XS rank (slots 7, 9) ────────────
// CoinGecko /coins/markets gives us symbols, marketCaps, supplies — we then
// need historical bars for the top coins to compute point-in-time dominance.
// On the server we don't want to spend Polygon calls on 150 alts (that would
// 150× the rate-limit budget) so we settle for a simpler approximation:
//   - Dominance z: rolling z of CURRENT BTC dominance vs trailing window of
//     CoinGecko-reported dominance. We can fetch the historical global series
//     via /global/market_cap_chart (free).
//   - XS rank: BTC's percentile in the snapshot's 14d returns. Same value at
//     every entry timestamp because we don't have per-coin history without
//     Polygon — better than zero but not point-in-time.
// If either fetch fails the slot stays zero.
export async function fetchBtcDominanceHistory(days = 365) {
  // Free CoinGecko endpoint gives [[ts_ms, dom_pct], ...]
  const url = `${COINGECKO_BASE}/global/market_cap_chart?days=${days}`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(15000) });
    if (!res.ok) return { records: null, reason: `coingecko dom HTTP ${res.status}` };
    const data = await res.json();
    // Response shape varies by tier; on free tier we get
    //   { market_cap_chart: { market_cap: [[ts, usd], ...] } }
    // and we'd need BTC cap separately. The pro endpoint /global/market_cap_chart
    // with btc-share is paid. Fall back to a simpler scheme: pull the BTC and
    // total market_cap_chart series and divide.
    const totalSeries = data?.market_cap_chart?.market_cap;
    if (!Array.isArray(totalSeries) || totalSeries.length < 30) {
      return { records: null, reason: "coingecko dom shape unexpected (likely needs pro tier)" };
    }
    // Without a BTC-specific series we can't compute dominance from this alone.
    // Mark unavailable rather than fabricate.
    return { records: null, reason: "coingecko free tier lacks BTC-share history" };
  } catch (err) {
    return { records: null, reason: `coingecko dom fetch error: ${err.message}` };
  }
}

export async function fetchTopCryptoSnapshot(n = 150) {
  const url = `${COINGECKO_BASE}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=${n}&page=1&price_change_percentage=14d`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(15000) });
    if (!res.ok) return { snapshot: null, reason: `coingecko markets HTTP ${res.status}` };
    const data = await res.json();
    if (!Array.isArray(data) || data.length < 20) {
      return { snapshot: null, reason: "coingecko markets short" };
    }
    const EXCLUDE = new Set([
      "usdt","usdc","dai","busd","fdusd","tusd","usdd","frax","gusd","usdp",
      "usdn","husd","sai","lusd","pyusd","usde","usdy","usd1",
      "wbtc","weth","steth","wsteth","reth","cbeth","tbtc","renbtc","meth",
      "lseth","rseth","ezeth","weeth","wbeth","btc",
    ]);
    const snapshot = data
      .filter(d => !EXCLUDE.has((d.symbol || "").toLowerCase()))
      .map(d => ({
        symbol: (d.symbol || "").toUpperCase(),
        marketCap: parseFloat(d.market_cap) || 0,
        ret14dPct: parseFloat(d.price_change_percentage_14d_in_currency),
      }))
      .filter(d => d.marketCap > 0);
    if (snapshot.length < 20) return { snapshot: null, reason: "coingecko markets parse_short" };
    return { snapshot, reason: `coingecko ok (${snapshot.length} coins)` };
  } catch (err) {
    return { snapshot: null, reason: `coingecko markets fetch error: ${err.message}` };
  }
}

// ─── Aggregation: 5-min bars → daily bars ────────────────────────────────────
// Many features (RV, Parkinson, breakout, dominance/XS via 14d returns) are
// daily-defined in the frontend (gated on isDaily). The server pipeline fetches
// 5-min bars for sampling; we aggregate into a daily series for those features.
// Returns { closes, highs, lows, timestamps } where timestamps[i] is the START
// of UTC day i (00:00:00).
export function aggregateDailyBars(bars) {
  if (!bars?.closes?.length) return null;
  const { closes, highs, lows, timestamps } = bars;
  const byDay = new Map();
  for (let i = 0; i < closes.length; i++) {
    const day = Math.floor(timestamps[i] / 86400);
    const slot = byDay.get(day);
    if (!slot) {
      byDay.set(day, { high: highs[i], low: lows[i], lastClose: closes[i], lastTs: timestamps[i] });
    } else {
      if (highs[i] > slot.high) slot.high = highs[i];
      if (lows[i] < slot.low)   slot.low = lows[i];
      if (timestamps[i] > slot.lastTs) {
        slot.lastTs = timestamps[i];
        slot.lastClose = closes[i];
      }
    }
  }
  const sorted = Array.from(byDay.entries()).sort((a, b) => a[0] - b[0]);
  const out = { closes: [], highs: [], lows: [], timestamps: [] };
  for (const [day, s] of sorted) {
    out.closes.push(s.lastClose);
    out.highs.push(s.high);
    out.lows.push(s.low);
    out.timestamps.push(day * 86400);
  }
  return out;
}

// Binary search: latest index with timestamps[idx] <= t. -1 if none.
function findIdxAtOrBefore(timestamps, t) {
  if (!timestamps?.length || t < timestamps[0]) return -1;
  let lo = 0, hi = timestamps.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (timestamps[mid] <= t) { ans = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  return ans;
}

function findRecordAtOrBefore(records, timestampSec) {
  if (!records?.length || timestampSec < records[0].time) return -1;
  let lo = 0, hi = records.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (records[mid].time <= timestampSec) { ans = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  return ans;
}

// ─── Lookup builders ─────────────────────────────────────────────────────────
// All return a function (timestampSec) -> number (the feature value clipped
// to [-1, 1]) or 0 when the underlying data is missing / insufficient.

export function makeFundingZAt(records, window = 21) {
  if (!records?.length) return () => 0;
  return function (timestampSec) {
    const idx = findRecordAtOrBefore(records, timestampSec);
    if (idx < window) return 0;
    const slice = records.slice(idx - window + 1, idx + 1).map(r => r.rate);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length;
    const sd = Math.sqrt(variance);
    if (sd === 0) return 0;
    const z = (records[idx].rate - mean) / sd;
    return Math.max(-1, Math.min(1, z / 2));
  };
}

// dvolRvSpread depends on the daily BTC bar series, not raw timestamp alone.
export function makeDvolRvZAt(dailyBars, dvolRecords) {
  if (!dailyBars?.closes?.length || !dvolRecords?.length) return () => 0;
  function rv30d(closes, idx) {
    if (idx < 30) return null;
    let sum = 0, sumSq = 0;
    for (let k = idx - 29; k <= idx; k++) {
      if (closes[k - 1] <= 0 || closes[k] <= 0) return null;
      const r = Math.log(closes[k] / closes[k - 1]);
      sum += r; sumSq += r * r;
    }
    const mean = sum / 30;
    const variance = sumSq / 30 - mean * mean;
    return Math.sqrt(Math.max(0, variance)) * Math.sqrt(365);
  }
  return function (timestampSec) {
    const barIdx = findIdxAtOrBefore(dailyBars.timestamps, timestampSec);
    if (barIdx < 30) return 0;
    const dvolIdx = findRecordAtOrBefore(dvolRecords, timestampSec);
    if (dvolIdx < 0) return 0;
    const rvNow = rv30d(dailyBars.closes, barIdx);
    if (rvNow == null) return 0;
    const spreadNow = dvolRecords[dvolIdx].close / 100 - rvNow;
    const history = [];
    for (let k = barIdx - 1; k >= Math.max(30, barIdx - 60); k--) {
      const tk = dailyBars.timestamps[k];
      const di = findRecordAtOrBefore(dvolRecords, tk);
      if (di < 0) continue;
      const rvK = rv30d(dailyBars.closes, k);
      if (rvK == null) continue;
      history.push(dvolRecords[di].close / 100 - rvK);
    }
    if (history.length < 10) return 0;
    const mean = history.reduce((a, b) => a + b, 0) / history.length;
    const variance = history.reduce((a, b) => a + (b - mean) ** 2, 0) / history.length;
    const sd = Math.sqrt(variance);
    if (sd < 1e-9) return 0;
    const z = (spreadNow - mean) / sd;
    return Math.max(-1, Math.min(1, z / 2));
  };
}

export function makeOiZAt(records, window = 21) {
  if (!records?.length) return () => 0;
  return function (timestampSec) {
    if (timestampSec <= records[0].time) return 0;
    let lo = 0, hi = records.length - 1, idx = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (records[mid].time < timestampSec) { idx = mid; lo = mid + 1; }
      else hi = mid - 1;
    }
    if (idx < 1) return 0;
    const prev = records[idx - 1], curr = records[idx];
    if (prev.oiUsd <= 0 || curr.oiUsd <= 0) return 0;
    const currChange = Math.log(curr.oiUsd / prev.oiUsd);
    const history = [];
    const loBound = Math.max(1, idx - window);
    for (let k = loBound; k < idx; k++) {
      const a = records[k - 1], b = records[k];
      if (a.oiUsd <= 0 || b.oiUsd <= 0) continue;
      history.push(Math.log(b.oiUsd / a.oiUsd));
    }
    if (history.length < Math.min(8, window - 2)) return 0;
    const mean = history.reduce((a, b) => a + b, 0) / history.length;
    const variance = history.reduce((a, b) => a + (b - mean) ** 2, 0) / history.length;
    const sd = Math.sqrt(variance);
    if (sd < 1e-9) return 0;
    const z = (currChange - mean) / sd;
    return Math.max(-1, Math.min(1, z / 2));
  };
}

// Approximate dominance z from BTC's OWN trailing 14d return. This mirrors the
// frontend fallback used on the multi-crypto universe (see crypto.js
// approximateDominanceZFromBTCReturns). When the server can't get a true
// historical dominance series the BTC-14d-return proxy is documented and
// directionally correct.
export function makeDominanceZAt(dailyBars) {
  if (!dailyBars?.closes?.length) return () => 0;
  return function (timestampSec) {
    const idx = findIdxAtOrBefore(dailyBars.timestamps, timestampSec);
    if (idx < 14) return 0;
    const prior = dailyBars.closes[idx - 14];
    if (!(prior > 0)) return 0;
    const ret14 = (dailyBars.closes[idx] - prior) / prior;
    return Math.max(-1, Math.min(1, ret14 * 10));
  };
}

// XS rank is the BTC percentile within the top-150 14d returns. We don't have
// per-coin point-in-time history server-side without spending Polygon calls on
// 150 alts; we use the snapshot's CURRENT 14d returns as a stationary peer
// distribution and rank BTC's own trailing 14d return at each timestamp
// against it. This is weaker than the frontend's full per-coin history but
// still informative — the peer distribution shape is roughly stable on multi-
// month timescales.
export function makeXsRankAt(dailyBars, snapshot) {
  if (!dailyBars?.closes?.length || !snapshot?.length) return () => 0;
  const peers = snapshot
    .map(s => s.ret14dPct)
    .filter(v => Number.isFinite(v))
    .map(v => v / 100);
  if (peers.length < 20) return () => 0;
  return function (timestampSec) {
    const idx = findIdxAtOrBefore(dailyBars.timestamps, timestampSec);
    if (idx < 14) return 0;
    const prior = dailyBars.closes[idx - 14];
    if (!(prior > 0)) return 0;
    const btcRet = (dailyBars.closes[idx] - prior) / prior;
    let below = 0;
    for (const r of peers) if (r < btcRet) below++;
    const pct = below / peers.length;
    return Math.max(-1, Math.min(1, 2 * (pct - 0.5)));
  };
}

// TS momentum z: BTC's own 14-bar return z-scored vs trailing 30-bar window
// of same-lookback returns. Uses the daily series.
export function makeTsMomZAt(dailyBars, lookback = 14, zWindow = 30) {
  if (!dailyBars?.closes?.length) return () => 0;
  return function (timestampSec) {
    const idx = findIdxAtOrBefore(dailyBars.timestamps, timestampSec);
    if (idx < lookback + zWindow) return 0;
    const closes = dailyBars.closes;
    const current = closes[idx];
    const prior = closes[idx - lookback];
    if (!(prior > 0)) return 0;
    const momNow = (current - prior) / prior;
    const history = [];
    for (let i = zWindow; i <= idx; i++) {
      const p = closes[i - lookback];
      const c = closes[i];
      if (p > 0) history.push((c - p) / p);
    }
    if (history.length < 10) return Math.max(-1, Math.min(1, momNow * 10));
    const mean = history.reduce((a, b) => a + b, 0) / history.length;
    const sd = Math.sqrt(history.reduce((a, b) => a + (b - mean) ** 2, 0) / history.length);
    const z = sd > 0 ? (momNow - mean) / sd : 0;
    return Math.max(-1, Math.min(1, z / 2));
  };
}

// RV ratio (close-to-close): rvShort / rvLong − 1, clipped ±1.
function realizedVolAt(closes, idx, window) {
  if (idx < window) return null;
  let sum = 0, sumSq = 0;
  for (let k = idx - window + 1; k <= idx; k++) {
    if (closes[k - 1] <= 0 || closes[k] <= 0) return null;
    const r = Math.log(closes[k] / closes[k - 1]);
    sum += r; sumSq += r * r;
  }
  const mean = sum / window;
  const variance = sumSq / window - mean * mean;
  return Math.sqrt(Math.max(0, variance));
}

export function makeRvRatioAt(dailyBars, shortW = 5, longW = 30) {
  if (!dailyBars?.closes?.length) return () => 0;
  return function (timestampSec) {
    const idx = findIdxAtOrBefore(dailyBars.timestamps, timestampSec);
    if (idx < longW + 1) return 0;
    const rvShort = realizedVolAt(dailyBars.closes, idx, shortW);
    const rvLong  = realizedVolAt(dailyBars.closes, idx, longW);
    if (rvShort == null || rvLong == null || rvLong < 1e-9) return 0;
    return Math.max(-1, Math.min(1, rvShort / rvLong - 1));
  };
}

// Parkinson ratio (range-based vol).
function parkinsonVolAt(highs, lows, idx, window) {
  if (idx < window) return null;
  const L2 = Math.log(2);
  let sumLogSq = 0, count = 0;
  for (let k = idx - window + 1; k <= idx; k++) {
    const h = highs[k], l = lows[k];
    if (h <= 0 || l <= 0 || h < l) continue;
    const lnHL = Math.log(h / l);
    sumLogSq += lnHL * lnHL;
    count++;
  }
  if (count < window - 2) return null;
  const variance = sumLogSq / (4 * L2 * count);
  return Math.sqrt(Math.max(0, variance));
}

export function makeParkinsonRatioAt(dailyBars, shortW = 5, longW = 30) {
  if (!dailyBars?.highs?.length) return () => 0;
  return function (timestampSec) {
    const idx = findIdxAtOrBefore(dailyBars.timestamps, timestampSec);
    if (idx < longW) return 0;
    const pShort = parkinsonVolAt(dailyBars.highs, dailyBars.lows, idx, shortW);
    const pLong  = parkinsonVolAt(dailyBars.highs, dailyBars.lows, idx, longW);
    if (pShort == null || pLong == null || pLong < 1e-9) return 0;
    return Math.max(-1, Math.min(1, pShort / pLong - 1));
  };
}

// Donchian 20d breakout flag ∈ {-1, 0, +1}.
export function makeBreakoutFlagAt(dailyBars, window = 20) {
  if (!dailyBars?.closes?.length) return () => 0;
  return function (timestampSec) {
    const idx = findIdxAtOrBefore(dailyBars.timestamps, timestampSec);
    if (idx < window) return 0;
    const close = dailyBars.closes[idx];
    if (!(close > 0)) return 0;
    let priorHigh = -Infinity, priorLow = Infinity;
    for (let k = idx - window; k < idx; k++) {
      if (dailyBars.highs[k] > priorHigh) priorHigh = dailyBars.highs[k];
      if (dailyBars.lows[k] > 0 && dailyBars.lows[k] < priorLow) priorLow = dailyBars.lows[k];
    }
    if (close > priorHigh) return 1;
    if (close < priorLow)  return -1;
    return 0;
  };
}

// ─── One-shot: fetch every external dataset for a btc-universe pipeline run.
// Returns { lookups, dailyBars, diagnostics } where lookups is the object
// passed to buildFeatureVector and diagnostics records what populated.
export async function buildBtcLookups(bars5m) {
  const dailyBars = aggregateDailyBars(bars5m);
  const diagnostics = {};

  // Funding (Coinalyze)
  const funding = await fetchFundingHistoryCoinalyze(365);
  diagnostics.funding = funding.reason;
  await sleep(COINALYZE_DELAY_MS);

  // DVOL (Deribit)
  const dvol = await fetchDvolHistory(365);
  diagnostics.dvol = dvol.reason;

  // OI (Binance fapi → Coinalyze)
  const oi = await fetchBtcOIHistory();
  diagnostics.oi = oi.reason;
  await sleep(COINGECKO_DELAY_MS);

  // Snapshot (CoinGecko) for XS rank
  const snap = await fetchTopCryptoSnapshot(150);
  diagnostics.snapshot = snap.reason;

  const lookups = {
    fundingZAt:      makeFundingZAt(funding.records),
    dvolRvZAt:       makeDvolRvZAt(dailyBars, dvol.records),
    oiZAt:           makeOiZAt(oi.records),
    dominanceZAt:    makeDominanceZAt(dailyBars),
    xsRankAt:        makeXsRankAt(dailyBars, snap.snapshot),
    tsMomZAt:        makeTsMomZAt(dailyBars),
    rvRatioAt:       makeRvRatioAt(dailyBars),
    parkinsonAt:     makeParkinsonRatioAt(dailyBars),
    breakoutFlagAt:  makeBreakoutFlagAt(dailyBars),
  };

  return { lookups, dailyBars, diagnostics };
}
