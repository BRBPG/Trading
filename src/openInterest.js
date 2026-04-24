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

// 10-min session cache — OI history doesn't revise.
const oiCache = { records: null, fetchedAt: 0 };
const OI_TTL_MS = 10 * 60 * 1000;

// Fetch daily OI history for BTCUSDT. 30 records is the max daily depth
// Binance provides on this endpoint (documented soft-30d cap). Returns
// array of { time: seconds, oi: number, oiUsd: number } sorted ascending.
export async function fetchBtcOIHistory(period = "1d", limit = 30) {
  const now = Date.now();
  if (oiCache.records && now - oiCache.fetchedAt < OI_TTL_MS) {
    return oiCache.records;
  }
  try {
    const url = `${BINANCE_OI_BASE}?symbol=BTCUSDT&period=${period}&limit=${limit}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!res.ok) return null;
    const raw = await res.json();
    if (!Array.isArray(raw) || raw.length < 5) return null;
    // Response gotcha: values returned as strings, timestamps in ms.
    // Sort ascending — Binance returns oldest-first most of the time
    // but docs don't guarantee it.
    const records = raw
      .map(d => ({
        time: Math.floor(d.timestamp / 1000),
        oi:    parseFloat(d.sumOpenInterest),
        oiUsd: parseFloat(d.sumOpenInterestValue),
      }))
      .filter(r => r.time > 0 && Number.isFinite(r.oi) && r.oi > 0)
      .sort((a, b) => a.time - b.time);
    if (records.length < 5) return null;
    oiCache.records = records;
    oiCache.fetchedAt = now;
    return records;
  } catch {
    return null;
  }
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
