// ─── Deribit DVOL − realized vol spread (Phase 4 Commit 3) ─────────────────
// Implied vs realized volatility spread — the "volatility risk premium".
// One of the most consistently documented edges in the BTC options
// literature at retail scale:
//   - Alexander & Imeraj (2023), "Inter-temporal hedging using
//     cryptocurrency derivatives," J. Futures Markets 43(11) — IV-RV
//     spread has predictive content for short-horizon BTC returns.
//   - Bollerslev, Tauchen & Zhou (2009), RFS — the variance risk premium
//     is a leading predictor of forward returns across asset classes.
//   - Bekaert & Hoerova (2014) — standard IV-RV construction uses
//     matched 30-day windows; Deribit DVOL is 30d constant-maturity IV,
//     so 30d realized is the correct match.
//
// Sign convention matters: positive spread (IV > RV) = market pricing
// future vol higher than recent realized ≈ fear premium ≈ historically
// bullish forward returns. Negative spread (RV > IV) = complacency about
// realized turbulence ≈ historically bearish.
//
// Feature output: z-score of the spread vs a rolling 60d window of past
// spreads, clipped to ±1 via z/2 to match the model's clip1 convention.

const DVOL_BASE = "https://www.deribit.com/api/v2/public";

// Session-scoped cache — DVOL history 10-min TTL. Historical values
// don't change retroactively so stale-but-recent is fine.
const dvolCache = { records: null, fetchedAt: 0 };
const DVOL_TTL_MS = 60 * 60 * 1000;  // 60 min — matches bars cache

// Fetch daily DVOL history from Deribit public REST. The index name
// changed to "btc_dvol_usdc" around 2023 after USDC-settled rollout —
// old "btcdvol" tickers error silently. Response shape is
//   { result: { data: [[ts_ms, o, h, l, c], ...], continuation: null } }
// Note the array-tuple form — NOT objects. ts is in milliseconds.
//
// Deribit DVOL launched 2021-03-24 (hard floor for historical lookback).
// Daily res caps around 5000 candles per call — easily covers ~13 years
// of daily data, so single call is enough.
export async function fetchDvolHistory(days = 365) {
  const now = Date.now();
  if (dvolCache.records && now - dvolCache.fetchedAt < DVOL_TTL_MS) {
    return dvolCache.records;
  }
  try {
    const end = now;
    const start = end - days * 86400 * 1000;
    const url = `${DVOL_BASE}/get_volatility_index_data?currency=BTC&start_timestamp=${start}&end_timestamp=${end}&resolution=1D`;
    const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!res.ok) return null;
    const data = await res.json();
    const rows = data?.result?.data;
    if (!Array.isArray(rows) || rows.length < 10) return null;
    // Normalise to { time: seconds, close: number }
    const records = rows
      .map(([tsMs, , , , close]) => ({
        time: Math.floor(tsMs / 1000),
        close: parseFloat(close),
      }))
      .filter(r => r.time > 0 && Number.isFinite(r.close) && r.close > 0)
      .sort((a, b) => a.time - b.time);
    if (records.length < 10) return null;
    dvolCache.records = records;
    dvolCache.fetchedAt = now;
    return records;
  } catch {
    return null;
  }
}

// Find the DVOL record at or before a given timestamp. Binary search.
function findDvolAtOrBefore(records, timestampSec) {
  if (!records?.length || timestampSec < records[0].time) return null;
  let lo = 0, hi = records.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (records[mid].time <= timestampSec) { ans = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  return ans >= 0 ? records[ans] : null;
}

// Annualized realized vol from daily log returns. Crypto convention is
// ×√365 because it trades 24/7 (equities use √252 because of weekends).
// Returns annualized stdev in decimal form (e.g. 0.72 = 72% annualized).
function realizedVol30d(closes, idx) {
  if (!closes || idx < 30) return null;
  let sum = 0, sumSq = 0;
  for (let k = idx - 29; k <= idx; k++) {
    if (closes[k - 1] <= 0 || closes[k] <= 0) return null;
    const r = Math.log(closes[k] / closes[k - 1]);
    sum += r;
    sumSq += r * r;
  }
  const mean = sum / 30;
  const variance = sumSq / 30 - mean * mean;
  const dailyStd = Math.sqrt(Math.max(0, variance));
  return dailyStd * Math.sqrt(365);
}

// Compute DVOL-RV spread at a given timestamp + the rolling z-score.
//
// bars: BTC daily bars (closes[] + timestamps[]) — used to compute RV
//   point-in-time. For intraday mode we need the daily closes equivalent
//   — caller should either use daily mode for this feature or pass the
//   aggregated daily bars. For intraday mode we degrade to 0 (calling
//   code checks and skips).
// barIdx: index in bars[] representing the entry bar.
// dvolRecords: output of fetchDvolHistory.
//
// Output: ±1 clipped z-score of spread = (DVOL_decimal - RV_decimal) vs
// the rolling 60d window of prior spreads. Zero if inputs missing.
export function dvolRvSpreadAt(bars, barIdx, dvolRecords) {
  if (!bars?.closes || !dvolRecords?.length) return 0;
  const tsSec = bars.timestamps[barIdx];

  // Current spread
  const dvolRec = findDvolAtOrBefore(dvolRecords, tsSec);
  if (!dvolRec) return 0;
  const rv = realizedVol30d(bars.closes, barIdx);
  if (rv == null) return 0;
  // DVOL is in percent (e.g. 52.34 = 52.34%). Convert to decimal.
  const dvolDec = dvolRec.close / 100;
  const spreadNow = dvolDec - rv;

  // Rolling history of spreads for z-score. Compute spread at each prior
  // daily bar that has both a DVOL record and enough history for RV_30d.
  // Use trailing 60 samples (≈60 days) for the baseline distribution —
  // this is the "volatility regime" the current spread is being measured
  // against. Shorter windows are too noisy, longer windows include
  // regime shifts that shouldn't anchor current extremity.
  const history = [];
  for (let k = barIdx - 1; k >= Math.max(30, barIdx - 60); k--) {
    const tk = bars.timestamps[k];
    const dRec = findDvolAtOrBefore(dvolRecords, tk);
    if (!dRec) continue;
    const rvK = realizedVol30d(bars.closes, k);
    if (rvK == null) continue;
    history.push(dRec.close / 100 - rvK);
  }
  if (history.length < 10) return 0;

  const mean = history.reduce((a, b) => a + b, 0) / history.length;
  const variance = history.reduce((a, b) => a + (b - mean) ** 2, 0) / history.length;
  const sd = Math.sqrt(variance);
  if (sd < 1e-9) return 0;
  const z = (spreadNow - mean) / sd;
  return Math.max(-1, Math.min(1, z / 2));
}

// Live-path sibling. Uses the most recent bar in the passed closes/
// timestamps as "now", and fetched DVOL record nearest-before.
export function dvolRvSpreadLive(bars, dvolRecords) {
  if (!bars?.closes?.length) return 0;
  return dvolRvSpreadAt(bars, bars.closes.length - 1, dvolRecords);
}
