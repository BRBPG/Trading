// ─── Crypto-native macro features ───────────────────────────────────────────
// Replaces the equity VIX / cross-asset / PEAD features (zeroed in crypto
// mode) with signals that actually have documented edge on crypto:
//
//   1. BTC dominance — ratio of BTC market cap to total crypto market cap.
//      High/rising dominance = risk-off, money rotating into BTC.
//      Low/falling dominance = alt-season, risk-on. Regime indicator.
//      Source: CoinGecko /global (free, no key, rate-limited ~10-30 req/min).
//
//   2. Symbol's own 14-day return z-score — time-series momentum. Moskowitz-
//      Ooi-Pedersen (2012 JFE) documented TS momentum across every major
//      asset class including crypto. Requires no external API; computed from
//      the bar series the backtest already has.
//
//   3. Cross-sectional momentum rank — the symbol's 14d return percentile
//      among the active watchlist. Liu-Tsyvinski (2021, 2022) documented
//      this as the strongest single crypto factor. Requires the backtest
//      to pass a per-timestamp universe-wide returns snapshot to scoreSetup.
//
// Phase 3c first pass adds #1 and #2 (both tractable with no new API
// dependencies beyond CoinGecko). #3 requires backtest refactor to pass
// universe-wide context per entry and will land in a later commit.
//
// The CoinGecko free tier has generous enough limits for our use case:
// one call per 2 minutes (macro cache TTL) is well under any rate limit.

const BTC_DOM_CACHE = { data: null, fetchedAt: 0 };
const BTC_DOM_TTL_MS = 2 * 60 * 1000;
// Keep a rolling window of the last N dominance readings so we can compute
// a z-score. CoinGecko doesn't give historical dominance in one call, so
// we bootstrap this during the session — each refresh adds one data point.
// After ~30 refreshes (~60 min session) the z-score is statistically meaningful.
const BTC_DOM_HISTORY = [];  // [{ t: ms, dom: 0-100 }]
const BTC_DOM_HISTORY_CAP = 120;

// CoinGecko's /global endpoint returns `market_cap_percentage.btc` as the
// BTC dominance percent. Also returns total market cap, fear/greed proxies.
// Free tier, no auth required.
export async function fetchBTCDominance() {
  const now = Date.now();
  if (BTC_DOM_CACHE.data && now - BTC_DOM_CACHE.fetchedAt < BTC_DOM_TTL_MS) {
    return BTC_DOM_CACHE.data;
  }
  try {
    const res = await fetch("https://api.coingecko.com/api/v3/global", {
      signal: AbortSignal.timeout(6000),
    });
    if (!res.ok) return null;
    const data = await res.json();
    const dom = data?.data?.market_cap_percentage?.btc;
    const totalCap = data?.data?.total_market_cap?.usd;
    if (dom == null) return null;

    BTC_DOM_HISTORY.push({ t: now, dom });
    if (BTC_DOM_HISTORY.length > BTC_DOM_HISTORY_CAP) BTC_DOM_HISTORY.shift();

    // Z-score vs the observed history. At session start history is tiny,
    // so z is unreliable — gate on ≥20 observations before returning a
    // non-zero z-score. Before that, feature slot stays at 0 (neutral).
    let z = 0;
    if (BTC_DOM_HISTORY.length >= 20) {
      const values = BTC_DOM_HISTORY.map(h => h.dom);
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
      const sd = Math.sqrt(variance);
      z = sd > 0 ? (dom - mean) / sd : 0;
    }

    const out = {
      dominance: dom,         // 0-100, current raw value
      dominanceZ: z,           // z-score vs session history
      totalMarketCap: totalCap,
      historySamples: BTC_DOM_HISTORY.length,
      fetchedAt: now,
    };
    BTC_DOM_CACHE.data = out;
    BTC_DOM_CACHE.fetchedAt = now;
    return out;
  } catch {
    return null;
  }
}

// Historical dominance for backtest point-in-time. We don't have true
// historical dominance from CoinGecko free (that's a paid endpoint), but
// for daily-horizon backtesting we can make a defensible approximation:
// use BTC's own trailing 14d return as a proxy for dominance movement.
// When BTC is outperforming (positive 14d return), dominance tends to
// rise. Not a perfect proxy but free and directionally correct.
export function approximateDominanceZFromBTCReturns(btcBars, timestampSec) {
  if (!btcBars?.closes || btcBars.closes.length < 30) return 0;
  // Find the bar nearest to (but not after) the reference timestamp
  const refIdx = findBarIndex(btcBars.timestamps, timestampSec);
  if (refIdx < 14) return 0;
  const currentClose = btcBars.closes[refIdx];
  const priorClose = btcBars.closes[refIdx - 14];
  if (!priorClose) return 0;
  const ret14 = (currentClose - priorClose) / priorClose;
  // Clip to [-1, 1] — a 10% 14d return maps to ~1.0
  return Math.max(-1, Math.min(1, ret14 * 10));
}

function findBarIndex(timestamps, t) {
  if (!timestamps?.length || t < timestamps[0]) return -1;
  let lo = 0, hi = timestamps.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (timestamps[mid] <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}

// Symbol's own 14-bar momentum, z-scored against its trailing 30-bar
// distribution. Moskowitz-Ooi-Pedersen style time-series momentum.
// Returns ±1 clipped.
export function timeSeriesMomentum(closes, lookback = 14, zWindow = 30) {
  if (!closes || closes.length < Math.max(lookback, zWindow) + 1) return 0;
  const n = closes.length;
  const current = closes[n - 1];
  const prior = closes[n - 1 - lookback];
  if (!prior) return 0;
  const momNow = (current - prior) / prior;

  // Historical distribution of the same lookback return
  const history = [];
  for (let i = zWindow; i < n; i++) {
    const p = closes[i - lookback];
    const c = closes[i];
    if (p > 0) history.push((c - p) / p);
  }
  if (history.length < 10) return Math.max(-1, Math.min(1, momNow * 10));
  const mean = history.reduce((a, b) => a + b, 0) / history.length;
  const sd = Math.sqrt(history.reduce((a, b) => a + (b - mean) ** 2, 0) / history.length);
  const z = sd > 0 ? (momNow - mean) / sd : 0;
  return Math.max(-1, Math.min(1, z / 2));  // z of ±2 maps to ±1
}

// Point-in-time version for backtest — compute TSM using only bars up to
// index i (no future leakage).
export function timeSeriesMomentumAt(bars, idx, lookback = 14, zWindow = 30) {
  if (!bars?.closes || idx < lookback + zWindow) return 0;
  const slice = bars.closes.slice(0, idx + 1);
  return timeSeriesMomentum(slice, lookback, zWindow);
}
