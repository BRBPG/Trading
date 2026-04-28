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
    const res = await fetch(`/api/proxy?url=${encodeURIComponent("https://api.coingecko.com/api/v3/global")}`, {
      signal: AbortSignal.timeout(15000),
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

// ─── Realized-volatility regime feature (Phase 4 Commit 2) ──────────────────
// Short-window RV ÷ long-window RV.
// Ratio >1 = vol expanding (regime shift, post-compression breakout often).
// Ratio <1 = vol compressing (coiling before move).
// Centred at 0 (ratio=1 → 0), clipped to [-1,+1] via (ratio-1), then bounded.
//
// Research backing:
//   - Catania & Sandholdt (2019), JRFM 12(1) — HAR-style RV features improve
//     both volatility AND directional models on BTC.
//   - Bergsli et al. (2022), Research in International Business and Finance
//     59 — multi-horizon RV ratios act as regime indicators that carry
//     forward-directional signal independent of OHLCV momentum features.
//
// Computed from log-returns (proper RV definition), not simple returns.
// GBMs handle untransformed RV fine but the ratio formulation lets us use
// a single clipped feature slot to carry a regime signal.
function realizedVolAt(closes, idx, window) {
  if (idx < window) return null;
  let sum = 0, sumSq = 0;
  for (let k = idx - window + 1; k <= idx; k++) {
    if (closes[k - 1] <= 0 || closes[k] <= 0) return null;
    const r = Math.log(closes[k] / closes[k - 1]);
    sum += r;
    sumSq += r * r;
  }
  const mean = sum / window;
  const variance = sumSq / window - mean * mean;
  return Math.sqrt(Math.max(0, variance));
}

// Backtest version — point-in-time using only bars[0..idx].
export function rvRatioAt(bars, idx, shortW = 5, longW = 30) {
  if (!bars?.closes || idx < longW + 1) return 0;
  const rvShort = realizedVolAt(bars.closes, idx, shortW);
  const rvLong  = realizedVolAt(bars.closes, idx, longW);
  if (rvShort == null || rvLong == null || rvLong < 1e-9) return 0;
  // Ratio centred at 1 (neutral regime). Map (ratio-1) into ±1 clip:
  //   ratio=1.0 → 0     (neutral)
  //   ratio=2.0 → +1    (vol doubled short-window vs long)
  //   ratio=0.5 → -0.5  (vol halved — compressing)
  return Math.max(-1, Math.min(1, rvShort / rvLong - 1));
}

// Live-path version — takes the latest bar series from a quote object.
export function rvRatioLive(closes, shortW = 5, longW = 30) {
  if (!closes || closes.length < longW + 1) return 0;
  return rvRatioAt({ closes }, closes.length - 1, shortW, longW);
}

// ─── Day-of-week cyclical encoding (Phase 4 Commit 5) ──────────────────────
// Caporale & Plastun (2019), "The day of the week effect in the
// cryptocurrency market," Finance Research Letters 31, document a
// persistent Monday/weekend effect on BTC returns. Classic result —
// weekend-closed-equities analog doesn't apply to 24/7 crypto but the
// effect emerges from retail-participant behavior patterns.
//
// Encoding: sin(2π·dow/7) — single cyclical value that preserves the
// week's periodicity in one slot. Range ≈ [-1, 1]. Monday (dow=1)
// ≈ 0.78, Wednesday (dow=3) ≈ 0.43, Friday (dow=5) ≈ -0.97, Sunday
// (dow=0) = 0.
//
// GBMs can ordinarily express day-of-week fine via a single feature
// with 7 values — the sin encoding gives comparable expressiveness
// (minus the exact Tuesday-vs-Thursday discrimination) in one slot
// instead of two (which would be sin+cos for full cyclic recovery).
// Given we only have 5 slots total and DOW is the LOWEST-priority of
// the four documented features, one-slot sin is the right trade.
export function dayOfWeekSinAt(timestampSec) {
  const d = new Date(timestampSec * 1000);
  const dow = d.getUTCDay();  // 0 = Sunday, 6 = Saturday
  return Math.sin(2 * Math.PI * dow / 7);
}

// ─── Parkinson volatility ratio (Phase 5 Commit C, slot [14] on btc) ───────
// Parkinson (1980) range-based estimator:
//   σ²_P = (1 / (4·ln(2))) · mean(ln(H/L)²)
// Uses the intraday high-low range instead of close-to-close returns.
// Statistically ~5× more efficient than close-to-close for capturing
// true volatility (Parkinson's original proof; confirmed on BTC daily
// by Petukhina et al. 2021 Q.Finance and Ardia/Bluteau/Rüede 2019 FRL).
// On BTC specifically Liu-Tsyvinski-Wu (RFS 2022) use range-based vol
// as a distinct factor from close-to-close RV.
//
// This feature is ORTHOGONAL to our existing RV 5d/30d slot [13] (which
// is close-to-close): it captures overnight / intra-bar volatility
// bursts that closing-price methods systematically miss.
//
// Ratio formulation: Parkinson_5d / Parkinson_30d, centred at 0 via
// (ratio − 1), clipped ±1.
function parkinsonVolAt(highs, lows, idx, window) {
  if (idx < window) return null;
  const L2 = Math.log(2);
  let sumLogSq = 0;
  let count = 0;
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

export function parkinsonRatioAt(bars, idx, shortW = 5, longW = 30) {
  if (!bars?.highs || !bars?.lows || idx < longW) return 0;
  const pShort = parkinsonVolAt(bars.highs, bars.lows, idx, shortW);
  const pLong  = parkinsonVolAt(bars.highs, bars.lows, idx, longW);
  if (pShort == null || pLong == null || pLong < 1e-9) return 0;
  return Math.max(-1, Math.min(1, pShort / pLong - 1));
}

// Live-path Parkinson ratio — takes a quote's full bar history.
export function parkinsonRatioLive(highs, lows, shortW = 5, longW = 30) {
  if (!highs || !lows || highs.length < longW + 1) return 0;
  return parkinsonRatioAt({ highs, lows }, highs.length - 1, shortW, longW);
}

// ─── Donchian 20-day breakout flag (Phase 5 Commit C, slot [15] on btc) ────
// Classic Turtle-trader signal. +1 if today's close > rolling 20-day
// high excluding today, -1 if < 20-day low, 0 otherwise.
//
// Documented BTC edge:
//   - Hudson & Urquhart (2021), European Journal of Finance — Donchian/
//     breakout rules survive transaction costs on BTC where most
//     technical rules do not.
//   - Detzel, Liu, Strauss, Zhou, Zhu (2021), Management Science —
//     "Learning and Predictability via Technical Analysis: Evidence
//     from Bitcoin." Moving-average / breakout signals carry OOS
//     predictive power on BTC returns.
//
// Why GBMs love this: axis-aligned splits. A tree can easily express
// "if breakout_flag >= 0.5 then branch A" which is exactly the signal
// structure the feature carries. Discrete ternary output ∈ {-1, 0, +1}.
//
// Uses previous N bars' high/low EXCLUDING current bar (point-in-time
// safe — we decide at bar T based on bars 0..T-1's structure).
export function breakoutFlagAt(bars, idx, window = 20) {
  if (!bars?.highs || !bars?.lows || !bars?.closes || idx < window) return 0;
  const close = bars.closes[idx];
  if (!(close > 0)) return 0;
  let priorHigh = -Infinity, priorLow = Infinity;
  for (let k = idx - window; k < idx; k++) {
    if (bars.highs[k] > priorHigh) priorHigh = bars.highs[k];
    if (bars.lows[k] > 0 && bars.lows[k] < priorLow) priorLow = bars.lows[k];
  }
  if (close > priorHigh) return 1;
  if (close < priorLow)  return -1;
  return 0;
}

export function breakoutFlagLive(highs, lows, closes, window = 20) {
  if (!highs || !lows || !closes || closes.length < window + 1) return 0;
  return breakoutFlagAt({ highs, lows, closes }, closes.length - 1, window);
}

// ─── Cross-sectional momentum rank (Liu-Tsyvinski 2022) ─────────────────────
// For the target symbol at a given timestamp, compute its 14-bar return's
// percentile rank within the active universe's 14-bar returns AT THE SAME
// TIMESTAMP (nearest bar ≤ t per symbol — handles minor timestamp drift
// across venues).
//
// Liu & Tsyvinski (2022, JFE): XS momentum is the strongest single factor
// in crypto after size. Sort today's universe by trailing N-day return,
// long top quintile / short bottom quintile → documented Sharpe > 1 on
// daily rebalance in the 2014-2020 sample.
//
// Returns a ±1-clipped value: +1 = best performer in the universe,
// -1 = worst. 0 = median. Feeds model slot [9] ("XS_mom_rank") in crypto
// mode; slot was zeroed placeholder prior to this commit.
//
// universeReturns: Map<symbol, { timestamps: number[], ret14: (number|null)[] }>
//   Caller precomputes once per backtest run — see backtest.js Phase A.
export function xsMomRankAt(targetSymbol, timestampSec, universeReturns) {
  if (!universeReturns || universeReturns.size < 3) return 0;

  // Get target's own return at this timestamp
  const self = universeReturns.get(targetSymbol);
  if (!self) return 0;
  const selfIdx = findBarIndex(self.timestamps, timestampSec);
  if (selfIdx < 14 || self.ret14[selfIdx] == null) return 0;
  const myRet = self.ret14[selfIdx];

  // Collect peer returns at the same (or nearest-prior) timestamp
  const peers = [];
  for (const [symbol, series] of universeReturns) {
    if (symbol === targetSymbol) continue;
    const idx = findBarIndex(series.timestamps, timestampSec);
    if (idx >= 14 && series.ret14[idx] != null) {
      peers.push(series.ret14[idx]);
    }
  }
  if (peers.length < 2) return 0;

  // Percentile rank of myRet among peers: fraction of peers strictly
  // below us. 1.0 = best in universe, 0.0 = worst.
  let below = 0;
  for (const r of peers) if (r < myRet) below++;
  const pct = below / peers.length;

  // Map [0,1] percentile to [-1,+1] centred score so positive = top half
  // of cross-section, negative = bottom. Matches the model's clip1 convention.
  return Math.max(-1, Math.min(1, 2 * (pct - 0.5)));
}

// Live-path version — no timestamps needed, just take each quote's most
// recent 14-bar return from its in-memory closes array.
// quotes: { [symbol]: { closes: number[] } }
export function xsMomRankLive(targetSymbol, quotes) {
  const selfQ = quotes?.[targetSymbol];
  if (!selfQ?.closes || selfQ.closes.length < 15) return 0;
  const selfRet = tailReturn(selfQ.closes, 14);
  if (selfRet == null) return 0;

  const peers = [];
  for (const [sym, q] of Object.entries(quotes || {})) {
    if (sym === targetSymbol) continue;
    if (!q?.closes || q.closes.length < 15) continue;
    const r = tailReturn(q.closes, 14);
    if (r != null) peers.push(r);
  }
  if (peers.length < 2) return 0;

  let below = 0;
  for (const r of peers) if (r < selfRet) below++;
  const pct = below / peers.length;
  return Math.max(-1, Math.min(1, 2 * (pct - 0.5)));
}

function tailReturn(closes, lookback) {
  const n = closes.length;
  if (n < lookback + 1) return null;
  const prev = closes[n - 1 - lookback];
  if (!prev) return null;
  return (closes[n - 1] - prev) / prev;
}
