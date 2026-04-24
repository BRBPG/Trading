// ─── Perpetual funding rate z-score (Phase 3d step 2) ──────────────────────
// Binance perpetual funding rates — the 8-hour premium paid between longs
// and shorts to keep the perpetual price tethered to spot. Structurally
// NOT derived from OHLCV — reflects DERIVATIVES market positioning, which
// is the one public data source most likely to carry signal orthogonal
// to anything a price-only technical model can learn.
//
// Documented effects:
//   • Extreme positive funding → crowd heavily long → contrarian bearish
//     (Hazel, Ammous, Gogolin 2021: "The Cryptocurrency Funding Rate as
//     a Sentiment Indicator")
//   • Extreme negative → crowd heavily short → contrarian bullish
//   • Z-score against rolling window captures "extreme-ness" while
//     normalising for per-asset baseline funding levels
//
// Not every symbol in the watchlist has an active Binance perp — the
// legacy/DeFi alts (BCH, ETC, MKR, CRV etc) may be missing. For those
// symbols fundingZ = 0 and the feature is uninformative; the GBM will
// learn to ignore it for those symbols.

// Binance public fapi — perpetual futures historical funding.
// CORS: Binance public endpoints historically allow cross-origin GET.
// Falls back to corsproxy if direct fetch fails.
const FUNDING_BASE = "https://fapi.binance.com/fapi/v1/fundingRate";
const CORS_PROXIES = [
  u => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
  u => `https://corsproxy.io/?${encodeURIComponent(u)}`,
];

// Symbol map — Yahoo/Polygon ticker → Binance perp. Binance uses USDT
// quote for perps, so XRP-USD → XRPUSDT etc. Symbols without active
// perps on Binance return null and are skipped.
const BINANCE_PERP_MAP = {
  "BTC-USD":  "BTCUSDT",
  "ETH-USD":  "ETHUSDT",
  "SOL-USD":  "SOLUSDT",
  "BNB-USD":  "BNBUSDT",
  "XRP-USD":  "XRPUSDT",
  "ADA-USD":  "ADAUSDT",
  "AVAX-USD": "AVAXUSDT",
  "LINK-USD": "LINKUSDT",
  "DOGE-USD": "DOGEUSDT",
  "POL-USD":  "POLUSDT",    // formerly MATIC; Binance renamed perp Sep 2024
  "TRX-USD":  "TRXUSDT",
  "LTC-USD":  "LTCUSDT",
  "DOT-USD":  "DOTUSDT",
  "ATOM-USD": "ATOMUSDT",
  "UNI-USD":  "UNIUSDT",
  "NEAR-USD": "NEARUSDT",
  "APT-USD":  "APTUSDT",
  "ARB-USD":  "ARBUSDT",
  "OP-USD":   "OPUSDT",
  "AAVE-USD": "AAVEUSDT",
  "BCH-USD":  "BCHUSDT",
  "ETC-USD":  "ETCUSDT",
  "XLM-USD":  "XLMUSDT",
  "HBAR-USD": "HBARUSDT",
  "ICP-USD":  "ICPUSDT",
  "FIL-USD":  "FILUSDT",
  "ALGO-USD": "ALGOUSDT",
  "VET-USD":  "VETUSDT",
  "STX-USD":  "STXUSDT",
  "IMX-USD":  "IMXUSDT",
  "MKR-USD":  "MKRUSDT",
  "CRV-USD":  "CRVUSDT",
  "LDO-USD":  "LDOUSDT",
  "SNX-USD":  "SNXUSDT",
  "COMP-USD": "COMPUSDT",
  "GRT-USD":  "GRTUSDT",
  "SUI-USD":  "SUIUSDT",
  "SEI-USD":  "SEIUSDT",
  "TIA-USD":  "TIAUSDT",
  "RUNE-USD": "RUNEUSDT",
};

export function toBinancePerp(symbol) {
  return BINANCE_PERP_MAP[symbol] || null;
}

// Session-scoped cache — funding doesn't change retroactively so once
// fetched the history is good for the session.
const fundingCache = new Map();  // binanceSym → { records: [{time, rate}], fetchedAt }
const FUNDING_CACHE_TTL_MS = 10 * 60 * 1000;

// Fetch up to 1000 most recent funding records (≈333 days at 3/day).
// Returns array of { time: seconds, rate: number } sorted ascending.
// null if both direct and proxy fetches fail.
export async function fetchFundingHistory(symbol, opts = {}) {
  const { limit = 1000 } = opts;
  const perp = toBinancePerp(symbol);
  if (!perp) return null;

  const cached = fundingCache.get(perp);
  if (cached && Date.now() - cached.fetchedAt < FUNDING_CACHE_TTL_MS) {
    return cached.records;
  }

  const url = `${FUNDING_BASE}?symbol=${perp}&limit=${limit}`;
  // Try direct first, then CORS proxies. Binance public endpoints usually
  // allow cross-origin but some networks/browsers block them.
  const attempts = [url, ...CORS_PROXIES.map(p => p(url))];
  for (const u of attempts) {
    try {
      const res = await fetch(u, { signal: AbortSignal.timeout(8000) });
      if (!res.ok) continue;
      const data = await res.json();
      if (!Array.isArray(data)) continue;
      const records = data
        .map(r => ({
          time: Math.floor((r.fundingTime || 0) / 1000),
          rate: parseFloat(r.fundingRate || 0),
        }))
        .filter(r => r.time > 0 && Number.isFinite(r.rate))
        .sort((a, b) => a.time - b.time);
      if (records.length < 10) continue;
      fundingCache.set(perp, { records, fetchedAt: Date.now() });
      return records;
    } catch { /* try next */ }
  }
  console.warn(`[funding] fetch_failed for ${symbol} (${perp}) — all endpoints errored`);
  return null;
}

// Bulk-fetch for a list of symbols with concurrency limit. Binance has
// weight-based rate limits but funding endpoint is 1 weight/request; 40
// parallel requests is well under the 2400/min public limit.
export async function fetchFundingForUniverse(symbols, opts = {}) {
  const { concurrency = 10 } = opts;
  const out = new Map();
  const queue = [...symbols];
  async function worker() {
    while (queue.length) {
      const sym = queue.shift();
      const records = await fetchFundingHistory(sym, opts);
      if (records) out.set(sym, records);
    }
  }
  await Promise.all(Array.from({ length: concurrency }, () => worker()));
  return out;
}

// Binary search: find the latest funding record at or before `timestampSec`.
// Returns -1 if no record fits (e.g. timestamp predates the fetched history).
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

// Z-score of the current funding rate vs a rolling-window history at
// point-in-time timestampSec. Clipped to ±1 for the feature vector.
//
// window: number of preceding records to use for baseline mean/std.
// Default 21 ≈ 7 days of 8-hour records, matching the "recent regime"
// rather than "lifetime baseline" — funding regimes shift, a rolling
// z captures current extreme-ness not historical average.
export function fundingZAt(records, timestampSec, window = 21) {
  if (!records?.length) return 0;
  const idx = findRecordAtOrBefore(records, timestampSec);
  if (idx < window) return 0;

  const slice = records.slice(idx - window + 1, idx + 1);
  const rates = slice.map(r => r.rate);
  const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
  const variance = rates.reduce((a, b) => a + (b - mean) ** 2, 0) / rates.length;
  const sd = Math.sqrt(variance);
  if (sd === 0) return 0;

  const current = records[idx].rate;
  const z = (current - mean) / sd;
  // Clip to ±1 for the feature vector convention (model.js clip1).
  // Typical z values at extremes are ±2-3; clipping sacrifices the
  // extra magnitude but keeps the feature scale comparable to others.
  return Math.max(-1, Math.min(1, z / 2));
}

// Live-path version — most recent funding z against rolling window.
// records: same shape as fetchFundingHistory output.
export function fundingZLive(records, window = 21) {
  if (!records?.length) return 0;
  return fundingZAt(records, records[records.length - 1].time, window);
}
