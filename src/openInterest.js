// ─── Binance perp Open Interest z-score (Phase 4 Commit 4) ─────────────────
// Open Interest on Binance BTCUSDT perpetual — divergences between OI
// changes and price changes have documented short-horizon directional
// content. Positive ΔOI + up-price = fresh longs entering (confirmatory);
// positive ΔOI + down-price = fresh shorts entering; negative ΔOI = either
// long or short liquidations unwinding.
//
// Research backing:
//   - Alexander, Deng, Feng, Wan (2023), "Net buying pressure and the
//     information in bitcoin option trades," J. Financial Markets 63.
//     Key result: OI changes (especially when divergent from price
//     direction) carry 1-5 day directional signal.
//   - Delphi Digital research (2022) — practitioner corroboration that
//     OI z-score of percent-change, not absolute level, is the signal.
//
// Feature construction: z-score of Δlog(OI) against a rolling 21-day
// baseline. Absolute OI is non-stationary (trends with market cap) so
// z-scoring the LEVEL is a non-starter. z-scoring the LOG-CHANGE makes
// the feature stationary and regime-relative.
//
// Hard constraint: Binance /futures/data/openInterestHist returns ~30
// days of history regardless of startTime param. For backtest entries
// older than 30 days, the feature is 0. That's honest — we don't have
// the data — and the GBM with val-loss truncation handles 0-padded
// features fine (one split separates the dead window from the live).

const BINANCE_OI_BASE = "https://fapi.binance.com/futures/data/openInterestHist";
const COINALYZE_BASE  = "https://api.coinalyze.net/v1";
// Coinalyze's aggregated BTC perp symbol — same one used in funding.js.
const COINALYZE_BTC_PERP = "BTCUSDT_PERP.A";

// 10-min session cache — OI history doesn't revise.
const oiCache = { records: null, fetchedAt: 0 };
const OI_TTL_MS = 60 * 60 * 1000;  // 60 min — matches bars cache

// Hetzner→Binance fapi is geo-blocked (HTTP 451); when that fires we fall
// back to Coinalyze which mirrors the same series. One-warning-per-mode
// to avoid log spam in normal-operation sessions.
const oiWarned = new Set();
async function fetchBtcOICoinalyze(daysAgo = 30) {
  const key = import.meta.env?.VITE_COINALYZE_KEY;
  if (!key) {
    if (!oiWarned.has("nokey")) {
      console.warn("[OI] VITE_COINALYZE_KEY not set — BTC OI feature will be 0");
      oiWarned.add("nokey");
    }
    return null;
  }
  const to = Math.floor(Date.now() / 1000);
  const from = to - daysAgo * 86400;
  const url = `${COINALYZE_BASE}/open-interest-history?symbols=${COINALYZE_BTC_PERP}&interval=daily&from=${from}&to=${to}&convert_to_usd=true&api_key=${encodeURIComponent(key)}`;
  try {
    const res = await fetch(`/api/proxy?url=${encodeURIComponent(url)}`, {
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) {
      console.warn(`[OI] coinalyze fallback returned ${res.status}`);
      return null;
    }
    const data = await res.json();
    const block = Array.isArray(data) ? data[0] : null;
    const history = block?.history;
    if (!Array.isArray(history) || history.length < 5) return null;
    const records = history
      .map(h => {
        const close = parseFloat(h.c ?? h.close ?? 0);
        return { time: Math.floor(h.t || 0), oi: close, oiUsd: close };
      })
      .filter(r => r.time > 0 && Number.isFinite(r.oi) && r.oi > 0)
      .sort((a, b) => a.time - b.time);
    return records.length >= 5 ? records : null;
  } catch {
    console.warn("[OI] coinalyze fallback errored");
    return null;
  }
}

// Fetch daily OI history for BTCUSDT. 30 records is the max daily depth
// Binance provides on this endpoint (documented soft-30d cap). Returns
// array of { time: seconds, oi: number, oiUsd: number } sorted ascending.
export async function fetchBtcOIHistory(period = "1d", limit = 30) {
  const now = Date.now();
  if (oiCache.records && now - oiCache.fetchedAt < OI_TTL_MS) {
    return oiCache.records;
  }
  // Primary: Binance fapi via /api/proxy.
  let records = null;
  try {
    const url = `${BINANCE_OI_BASE}?symbol=BTCUSDT&period=${period}&limit=${limit}`;
    const res = await fetch(`/api/proxy?url=${encodeURIComponent(url)}`, {
      signal: AbortSignal.timeout(15000),
    });
    if (res.ok) {
      const raw = await res.json();
      if (Array.isArray(raw) && raw.length >= 5) {
        // Binance values are stringified, timestamps in ms.
        records = raw
          .map(d => ({
            time: Math.floor(d.timestamp / 1000),
            oi:    parseFloat(d.sumOpenInterest),
            oiUsd: parseFloat(d.sumOpenInterestValue),
          }))
          .filter(r => r.time > 0 && Number.isFinite(r.oi) && r.oi > 0)
          .sort((a, b) => a.time - b.time);
        if (records.length < 5) records = null;
      }
    }
  } catch { /* fall through */ }

  // Fallback: Coinalyze (matches server-side lookups.js behavior).
  if (!records) records = await fetchBtcOICoinalyze(limit);
  if (!records) return null;
  oiCache.records = records;
  oiCache.fetchedAt = now;
  return records;
}

// Binary search: latest OI record strictly BEFORE `timestampSec`. Strict
// inequality because Binance timestamps the START of each bucket — a
// record with timestamp == entryTs would be the bucket that OPENED at
// our entry, whose forward interval leaks future information.
function findOIBefore(records, timestampSec) {
  if (!records?.length || timestampSec <= records[0].time) return -1;
  let lo = 0, hi = records.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (records[mid].time < timestampSec) { ans = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  return ans;
}

// Rolling z-score of Δlog(OI) at a point-in-time. Requires at least
// `window+1` prior records to compute the trailing stdev.
//
// records: output of fetchBtcOIHistory (sorted ascending).
// timestampSec: the entry bar's timestamp in seconds.
// window: rolling window for z-baseline. 21 matches funding-rate z
//   (Binance has 3 funding events/day × 7 days ≈ 21; for daily OI that's
//   3 weeks — about one market micro-regime).
//
// Returns ±1 clipped via z/2. Zero when:
//   - entry timestamp older than our 30-day OI window
//   - insufficient prior records for z-score baseline
//   - stdev is zero (OI flat)
export function oiZAt(records, timestampSec, window = 21) {
  if (!records?.length) return 0;
  const idx = findOIBefore(records, timestampSec);
  if (idx < 1) return 0;  // need at least one prior record for log-change

  // Current log-change (today's OI vs yesterday's)
  const prev = records[idx - 1];
  const curr = records[idx];
  if (prev.oiUsd <= 0 || curr.oiUsd <= 0) return 0;
  const currChange = Math.log(curr.oiUsd / prev.oiUsd);

  // Rolling history of log-changes within the window BEFORE idx.
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
}

// Live-path helper — latest record's log-change z-score against rolling
// baseline, assuming "now" is the latest record in the series.
export function oiZLive(records, window = 21) {
  if (!records?.length) return 0;
  // Use the most recent record's timestamp + a tiny offset so findOIBefore
  // treats it as "at or before". Actually we want the LATEST record itself,
  // so pass latest+1 sec:
  return oiZAt(records, records[records.length - 1].time + 1, window);
}

// ─── Top-trader long/short ratio (Phase 4 Commit 6) ────────────────────────
// Binance /futures/data/topLongShortPositionRatio — the aggregated
// long/short POSITION ratio of accounts in the top 20% by USD margin
// balance. This is the "sharp money" subset; the global long/short ratio
// (retail dominant) has been shown to be noise at best. Only the TOP
// version has contrarian-edge literature:
//   - Kakinaka & Umeno (2022), "Asymmetric volatility dynamics in
//     cryptocurrency markets," N. American J. Econ. Finance 62.
// Top-trader long/short is contrarian at extremes: top positioning
// skewed heavily long = crowded = reversal risk; heavily short = same
// in reverse. Middle of the range = no signal.
//
// Same 30-day soft cap as OI endpoint. Same CORS behavior.
// Feature: z-score of log(ratio) against rolling 21-day baseline.
// Log because ratio is naturally multiplicative (2:1 long vs 1:2 long
// are equally extreme); the log makes them symmetric around 0.

const BINANCE_LS_BASE = "https://fapi.binance.com/futures/data/topLongShortPositionRatio";
const lsCache = { records: null, fetchedAt: 0 };
const LS_TTL_MS = 60 * 60 * 1000;  // 60 min — matches bars cache (top L/S)

// Coinalyze's `long-short-ratio-history` is the closest analog when Binance
// is geo-blocked. It's the AGGREGATE retail/all-traders ratio rather than
// the top-trader subset; the contrarian framing in topLSZAt still holds at
// extremes (any extreme positioning crowd is contrarian) just with a noisier
// signal. Better than 0 when the key is set, equivalent to 0 when it isn't.
const lsWarned = new Set();
async function fetchBtcLSCoinalyze(daysAgo = 30) {
  const key = import.meta.env?.VITE_COINALYZE_KEY;
  if (!key) {
    if (!lsWarned.has("nokey")) {
      console.warn("[topLS] VITE_COINALYZE_KEY not set — top long/short feature will be 0");
      lsWarned.add("nokey");
    }
    return null;
  }
  const to = Math.floor(Date.now() / 1000);
  const from = to - daysAgo * 86400;
  const url = `${COINALYZE_BASE}/long-short-ratio-history?symbols=${COINALYZE_BTC_PERP}&interval=daily&from=${from}&to=${to}&api_key=${encodeURIComponent(key)}`;
  try {
    const res = await fetch(`/api/proxy?url=${encodeURIComponent(url)}`, {
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) {
      console.warn(`[topLS] coinalyze fallback returned ${res.status}`);
      return null;
    }
    const data = await res.json();
    const block = Array.isArray(data) ? data[0] : null;
    const history = block?.history;
    if (!Array.isArray(history) || history.length < 5) return null;
    const records = history
      .map(h => ({
        time:  Math.floor(h.t || 0),
        ratio: parseFloat(h.r ?? h.c ?? h.close ?? 0),
      }))
      .filter(r => r.time > 0 && Number.isFinite(r.ratio) && r.ratio > 0)
      .sort((a, b) => a.time - b.time);
    return records.length >= 5 ? records : null;
  } catch {
    console.warn("[topLS] coinalyze fallback errored");
    return null;
  }
}

export async function fetchBtcTopLSHistory(period = "1d", limit = 30) {
  const now = Date.now();
  if (lsCache.records && now - lsCache.fetchedAt < LS_TTL_MS) {
    return lsCache.records;
  }
  // Primary: Binance.
  let records = null;
  try {
    const url = `${BINANCE_LS_BASE}?symbol=BTCUSDT&period=${period}&limit=${limit}`;
    const res = await fetch(`/api/proxy?url=${encodeURIComponent(url)}`, {
      signal: AbortSignal.timeout(15000),
    });
    if (res.ok) {
      const raw = await res.json();
      if (Array.isArray(raw) && raw.length >= 5) {
        // Shape: [{ symbol, longAccount, shortAccount, longShortRatio, timestamp }]
        records = raw
          .map(d => ({
            time: Math.floor(d.timestamp / 1000),
            ratio: parseFloat(d.longShortRatio),
          }))
          .filter(r => r.time > 0 && Number.isFinite(r.ratio) && r.ratio > 0)
          .sort((a, b) => a.time - b.time);
        if (records.length < 5) records = null;
      }
    }
  } catch { /* fall through */ }

  if (!records) records = await fetchBtcLSCoinalyze(limit);
  if (!records) return null;
  lsCache.records = records;
  lsCache.fetchedAt = now;
  return records;
}

// Point-in-time z-score of log(ratio) against rolling baseline.
// Strict < timestamp for same reason as OI — bucket-start timestamps.
export function topLSZAt(records, timestampSec, window = 21) {
  if (!records?.length) return 0;
  // Reuse findOIBefore behavior inline — identical logic, different data.
  let idx = -1;
  {
    if (timestampSec <= records[0].time) return 0;
    let lo = 0, hi = records.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (records[mid].time < timestampSec) { idx = mid; lo = mid + 1; }
      else hi = mid - 1;
    }
  }
  if (idx < 0) return 0;

  const currLog = Math.log(records[idx].ratio);

  // Rolling history of log(ratio) up to but not including idx.
  const history = [];
  const loBound = Math.max(0, idx - window);
  for (let k = loBound; k < idx; k++) {
    if (records[k].ratio > 0) history.push(Math.log(records[k].ratio));
  }
  if (history.length < Math.min(8, window - 2)) return 0;

  const mean = history.reduce((a, b) => a + b, 0) / history.length;
  const variance = history.reduce((a, b) => a + (b - mean) ** 2, 0) / history.length;
  const sd = Math.sqrt(variance);
  if (sd < 1e-9) return 0;
  const z = (currLog - mean) / sd;
  // Contrarian framing: positive z (crowded long) → we expect REVERSAL
  // (negative directional signal). Flip sign so feature aligns with
  // P(bullish) — positive value of slot [15] suggests bullish.
  return Math.max(-1, Math.min(1, -z / 2));
}
