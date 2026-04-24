// ─── Broad-market context features (Phase 4.5) ────────────────────────────
// Three features that need top-150 crypto market data:
//   1. XS momentum rank [slot 9] — BTC's 14d-return percentile within the
//      top-150 basket. Replaces the n=1-dead state on btc universe. At 150
//      coins the rank has 0.67% resolution (vs 2.5% on 40-coin watchlist),
//      enough to resolve the Liu-Tsyvinski 2022 cross-sectional effect if
//      it's present.
//   2. Real BTC dominance z [slot 7] — BTC's market-cap share of the
//      top-150 sum, z-scored against rolling history. Replaces the self-
//      referential "BTC-14d-return-as-dominance-proxy" that was zero on
//      btc universe. Orthogonal to TS-momentum: a coin can have positive
//      return while DOMINANCE falls (alt-season), or negative return with
//      dominance RISING (flight to BTC during broad selloff).
//   3. Market breadth [slot 14] — % of top-150 with positive 14d return,
//      centred at 0 via (pct − 0.5) × 2. Captures overall crypto-market
//      risk-on/risk-off temperature — orthogonal to dominance (which is
//      about BTC's share) and XS rank (BTC's position in the distribution).
//
// Data strategy — MAXIMIZE POLYGON USAGE:
//   CoinGecko /coins/markets is used ONLY for the symbol list + supplies
//   (which Polygon doesn't provide — it's price/bar data not metadata).
//   All actual BAR data goes through fetchHistoricalBars which routes
//   Polygon-first if the user has a key. CoinGecko is a metadata layer,
//   not a price layer — no downgrade from Polygon-paid fetching.

const COINGECKO_BASE = "https://api.coingecko.com/api/v3";
const CACHE_TTL_MS = 10 * 60 * 1000;
const snapshotCache = { data: null, fetchedAt: 0 };

// Stablecoins + wrapped tokens to exclude from the cross-section. These
// either have ~zero 14d returns (stablecoins by design) or are tethered
// to BTC/ETH price (wrapped), both of which would skew rank/breadth.
const EXCLUDE_SYMBOLS = new Set([
  // Stablecoins
  "usdt","usdc","dai","busd","fdusd","tusd","usdd","frax","gusd","usdp",
  "usdn","husd","sai","lusd","pyusd","usde","usdy","usd1",
  // Wrapped / liquid-staking tethered to BTC/ETH
  "wbtc","weth","steth","wsteth","reth","cbeth","tbtc","renbtc","meth",
  "lseth","rseth","ezeth","weeth","wbeth",
  // Self — we're ranking BTC against these, not including BTC itself
  "btc",
]);

// Fetch top-N crypto snapshot from CoinGecko. Returns:
//   [{ id, symbol, yahooTicker, marketCap, circulatingSupply, ret14dPct }]
// where yahooTicker = "${SYMBOL}-USD" ready for fetchHistoricalBars.
// Session-cached 10 min TTL — CoinGecko rate limits are generous but this
// call fires once per sim so caching is purely a latency win on warm state.
export async function fetchTopCryptoSnapshot(n = 150) {
  const now = Date.now();
  if (snapshotCache.data && now - snapshotCache.fetchedAt < CACHE_TTL_MS) {
    return snapshotCache.data;
  }
  try {
    const url = `${COINGECKO_BASE}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=${n}&page=1&price_change_percentage=14d`;
    const res = await fetch(url, { signal: AbortSignal.timeout(10000) });
    if (!res.ok) return null;
    const data = await res.json();
    if (!Array.isArray(data) || data.length < 20) return null;
    const filtered = data
      .filter(d => !EXCLUDE_SYMBOLS.has((d.symbol || "").toLowerCase()))
      .map(d => ({
        id: d.id,
        symbol: (d.symbol || "").toUpperCase(),
        yahooTicker: `${(d.symbol || "").toUpperCase()}-USD`,
        marketCap: parseFloat(d.market_cap) || 0,
        circulatingSupply: parseFloat(d.circulating_supply) || 0,
        ret14dPct: parseFloat(d.price_change_percentage_14d_in_currency),
      }))
      .filter(d => d.marketCap > 0 && d.circulatingSupply > 0);
    if (filtered.length < 20) return null;
    snapshotCache.data = filtered;
    snapshotCache.fetchedAt = now;
    return filtered;
  } catch {
    return null;
  }
}

// Fetch historical bars for each of the top-N coins using the existing
// fetchHistoricalBars pipeline (Polygon-first if keyed, Yahoo fallback).
// Returns a Map<yahooTicker, bars> where bars = { closes, timestamps, ... }.
// Coins that fail to fetch are silently dropped — at N=150 we can easily
// lose 10-20 and still have plenty for rank/breadth resolution.
//
// Concurrency cap 15 — parallel enough to be fast with Polygon (~100ms/call)
// but not so high that we blow through Polygon's per-second rate limits on
// the paid tier. Total first-fetch time: ~15s with Polygon, ~60s without.
// Subsequent sims hit the bars cache so runs 2-20 are free.
export async function fetchTopCryptoBars(topList, fetchBarsFn, daysAgo, polygonKey, interval) {
  const out = new Map();
  const queue = [...topList];
  let polygonHits = 0, yahooHits = 0, failed = 0;
  async function worker() {
    while (queue.length) {
      const coin = queue.shift();
      try {
        const fetched = await fetchBarsFn(coin.yahooTicker, daysAgo, polygonKey, interval);
        if (fetched?.bars) {
          out.set(coin.yahooTicker, { ...coin, bars: fetched.bars, source: fetched.source });
          if (fetched.source === "polygon") polygonHits++;
          else if (fetched.source === "yahoo") yahooHits++;
        } else {
          failed++;
        }
      } catch {
        failed++;
      }
    }
  }
  await Promise.all(Array.from({ length: 15 }, () => worker()));
  // Diagnostic: surface fetch-source breakdown so the user knows when
  // Polygon dominates vs when Yahoo fallback is carrying the load. Both
  // are fine for data-correctness but Polygon is faster and the user is
  // paying for it.
  console.info(`[broadMarket] fetched ${out.size}/${topList.length} coins — polygon ${polygonHits}, yahoo ${yahooHits}, failed ${failed}`);
  return out;
}

// Precompute per-coin 14-bar returns array aligned to bars.timestamps.
// Same shape as backtest.js universeReturns — callable with the existing
// binary search pattern. ret14[i] = (closes[i] - closes[i-14]) / closes[i-14]
// or null if i < 14 or closes invalid.
export function precomputeReturns14d(coinBarsMap) {
  const byTicker = new Map();
  for (const [ticker, coin] of coinBarsMap) {
    const bars = coin.bars;
    if (!bars?.closes || bars.closes.length < 15) continue;
    const ret14 = new Array(bars.closes.length).fill(null);
    for (let i = 14; i < bars.closes.length; i++) {
      const prev = bars.closes[i - 14];
      if (prev > 0) ret14[i] = (bars.closes[i] - prev) / prev;
    }
    byTicker.set(ticker, {
      timestamps: bars.timestamps,
      ret14,
      closes: bars.closes,
      supply: coin.circulatingSupply,
    });
  }
  return byTicker;
}

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

// Collect every coin's 14d return AT timestampSec (nearest bar ≤ t). Returns
// an array of numbers with null entries stripped. Used by xs-rank + breadth.
function peerReturnsAt(returnsMap, timestampSec) {
  const out = [];
  for (const [, series] of returnsMap) {
    const idx = findIdxAtOrBefore(series.timestamps, timestampSec);
    if (idx >= 14 && series.ret14[idx] != null) out.push(series.ret14[idx]);
  }
  return out;
}

// XS momentum rank: BTC's percentile within peer returns at this timestamp.
// Returns ±1-clipped: +1 = BTC outperforming all peers, −1 = underperforming
// all, 0 = median.
export function xsRankAt(btcBars, btcIdx, returnsMap) {
  if (!btcBars?.closes || btcIdx < 14) return 0;
  const btcPrev = btcBars.closes[btcIdx - 14];
  if (!btcPrev) return 0;
  const btcRet = (btcBars.closes[btcIdx] - btcPrev) / btcPrev;
  const peers = peerReturnsAt(returnsMap, btcBars.timestamps[btcIdx]);
  if (peers.length < 20) return 0;
  let below = 0;
  for (const r of peers) if (r < btcRet) below++;
  const pct = below / peers.length;
  return Math.max(-1, Math.min(1, 2 * (pct - 0.5)));
}

// Market breadth: fraction of peers with positive 14d return. Centred at 0
// via (pct − 0.5) × 2. +1 = every peer up, −1 = every peer down, 0 = half.
export function breadthAt(btcBars, btcIdx, returnsMap) {
  if (!btcBars?.timestamps || btcIdx < 14) return 0;
  const peers = peerReturnsAt(returnsMap, btcBars.timestamps[btcIdx]);
  if (peers.length < 20) return 0;
  const positive = peers.filter(r => r > 0).length;
  const pct = positive / peers.length;
  return Math.max(-1, Math.min(1, 2 * (pct - 0.5)));
}

// Real BTC dominance at a given timestamp, z-scored against a rolling window.
// Dominance(t) = btc_cap(t) / sum(all_top_150_caps(t)) where each coin's cap
// at time t is approximated as price(t) × current_supply. Supplies change
// slowly (BTC block rewards are ~0.009% of supply per day; most alts have
// stable supplies on daily horizon) so using current supply as a constant
// introduces negligible error over a sim window.
//
// Returns ±1-clipped z-score vs trailing 60-day window of dominance values.
// Computes the full history up front so per-entry lookup is O(1) after
// precompute. Returns { at(timestampSec) => z } as a closure.
export function makeDominanceZLookup(btcBars, btcSupply, returnsMap) {
  if (!btcBars?.closes?.length || !btcSupply || btcSupply <= 0) {
    return () => 0;
  }
  // For each BTC bar, compute the full top-150 market-cap sum at that
  // timestamp. We need supplies per coin from the snapshot — stored in
  // returnsMap entries. Peers at timestamp t: each coin's close-price ×
  // supply, summed.
  const domSeries = new Array(btcBars.closes.length).fill(null);
  for (let i = 14; i < btcBars.closes.length; i++) {
    const t = btcBars.timestamps[i];
    let sumCap = btcBars.closes[i] * btcSupply;
    let btcCap = btcBars.closes[i] * btcSupply;
    for (const [, series] of returnsMap) {
      const idx = findIdxAtOrBefore(series.timestamps, t);
      if (idx >= 0 && series.closes[idx] > 0 && series.supply > 0) {
        sumCap += series.closes[idx] * series.supply;
      }
    }
    if (sumCap > 0) domSeries[i] = btcCap / sumCap;
  }

  // Rolling z-score of dominance vs trailing 60-bar window.
  return function dominanceZAt(timestampSec) {
    const idx = findIdxAtOrBefore(btcBars.timestamps, timestampSec);
    if (idx < 30 || domSeries[idx] == null) return 0;
    const loBound = Math.max(14, idx - 60);
    const history = [];
    for (let k = loBound; k < idx; k++) {
      if (domSeries[k] != null) history.push(domSeries[k]);
    }
    if (history.length < 10) return 0;
    const mean = history.reduce((a, b) => a + b, 0) / history.length;
    const variance = history.reduce((a, b) => a + (b - mean) ** 2, 0) / history.length;
    const sd = Math.sqrt(variance);
    if (sd < 1e-12) return 0;
    const z = (domSeries[idx] - mean) / sd;
    return Math.max(-1, Math.min(1, z / 2));
  };
}

// Live-path helpers — use the CoinGecko snapshot directly without needing
// historical bars. These compute the SAME quantities at "now" from the
// snapshot's 14d-return + market-cap fields.

// Live XS rank from snapshot's ret14dPct values.
export function xsRankLive(btcReturn14Pct, snapshot) {
  if (!snapshot || snapshot.length < 20 || !Number.isFinite(btcReturn14Pct)) return 0;
  const peers = snapshot
    .map(s => s.ret14dPct)
    .filter(v => Number.isFinite(v));
  if (peers.length < 20) return 0;
  let below = 0;
  for (const r of peers) if (r < btcReturn14Pct) below++;
  const pct = below / peers.length;
  return Math.max(-1, Math.min(1, 2 * (pct - 0.5)));
}

// Live breadth from snapshot.
export function breadthLive(snapshot) {
  if (!snapshot || snapshot.length < 20) return 0;
  const valid = snapshot.map(s => s.ret14dPct).filter(v => Number.isFinite(v));
  if (valid.length < 20) return 0;
  const positive = valid.filter(r => r > 0).length;
  const pct = positive / valid.length;
  return Math.max(-1, Math.min(1, 2 * (pct - 0.5)));
}

// Live dominance (un-z-scored — just current share). Caller can treat this
// as raw; the model feature is the z-scored backtest version, so live just
// gets the raw value pinned to 0 until historical z can be built from
// session data. Returns 0-1 share (BTC / top-150 total).
export function dominanceLive(snapshot, btcEntry) {
  if (!snapshot || !btcEntry?.marketCap) return 0;
  let sumCap = btcEntry.marketCap;
  for (const s of snapshot) sumCap += s.marketCap;
  if (sumCap <= 0) return 0;
  return btcEntry.marketCap / sumCap;
}
