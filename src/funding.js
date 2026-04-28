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
// Routed through the backend /api/proxy because direct browser fetch is
// blocked by CORS on most networks. Hetzner egress is geo-blocked by
// Binance (HTTP 451) so a Coinalyze fallback is wired in below; the
// fallback only triggers BTC-USD because Coinalyze's free tier coverage
// for the long-tail of alt perps is uneven and we'd rather report 0 than
// stitch heterogeneous histories per-symbol.
const FUNDING_BASE = "https://fapi.binance.com/fapi/v1/fundingRate";
const COINALYZE_BASE = "https://api.coinalyze.net/v1";

// Coinalyze symbol map for the perp aggregates we actually fall back on.
// `.A` suffix = aggregated across exchanges' BTCUSDT-margined perps, which
// is close enough to Binance's series for z-score purposes. Only BTC is
// covered intentionally (see comment above).
const COINALYZE_PERP_MAP = {
  "BTC-USD": "BTCUSDT_PERP.A",
};

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
const FUNDING_CACHE_TTL_MS = 60 * 60 * 1000;  // 60 min, matches bars cache

// Coinalyze fallback — daily aggregated funding history for the BTC perp.
// Returns the same { time, rate } shape so callers don't branch.
// Resolves to null if no key is set, the symbol isn't covered, or the
// upstream fails. Logs at most one warn per session per failure mode.
const coinalyzeWarned = new Set();
async function fetchFundingFromCoinalyze(symbol, daysAgo = 365) {
  const caSym = COINALYZE_PERP_MAP[symbol];
  if (!caSym) return null;
  const key = import.meta.env?.VITE_COINALYZE_KEY;
  if (!key) {
    if (!coinalyzeWarned.has("nokey")) {
      console.warn("[funding] VITE_COINALYZE_KEY not set — Binance-blocked symbols will report funding=0");
      coinalyzeWarned.add("nokey");
    }
    return null;
  }
  const to = Math.floor(Date.now() / 1000);
  const from = to - daysAgo * 86400;
  const url = `${COINALYZE_BASE}/funding-rate-history?symbols=${caSym}&interval=daily&from=${from}&to=${to}&api_key=${encodeURIComponent(key)}`;
  try {
    const res = await fetch(`/api/proxy?url=${encodeURIComponent(url)}`, {
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) {
      console.warn(`[funding] coinalyze fallback for ${symbol} returned ${res.status}`);
      return null;
    }
    const data = await res.json();
    const block = Array.isArray(data) ? data[0] : null;
    const history = block?.history;
    if (!Array.isArray(history) || history.length < 10) return null;
    const records = history
      .map(h => ({
        time: Math.floor(h.t || 0),
        rate: parseFloat(h.c ?? h.close ?? 0),
      }))
      .filter(r => r.time > 0 && Number.isFinite(r.rate))
      .sort((a, b) => a.time - b.time);
    return records.length >= 10 ? records : null;
  } catch {
    console.warn(`[funding] coinalyze fallback for ${symbol} errored`);
    return null;
  }
}

// Fetch up to 1000 most recent funding records (≈333 days at 3/day).
// Returns array of { time: seconds, rate: number } sorted ascending.
// null if both Binance and Coinalyze fail.
export async function fetchFundingHistory(symbol, opts = {}) {
  const { limit = 1000 } = opts;
  const perp = toBinancePerp(symbol);
  if (!perp) return null;

  const cached = fundingCache.get(perp);
  if (cached && Date.now() - cached.fetchedAt < FUNDING_CACHE_TTL_MS) {
    return cached.records;
  }

  // Primary: Binance via /api/proxy. May 451 from Hetzner egress.
  const url = `${FUNDING_BASE}?symbol=${perp}&limit=${limit}`;
  let records = null;
  try {
    const res = await fetch(`/api/proxy?url=${encodeURIComponent(url)}`, {
      signal: AbortSignal.timeout(15000),
    });
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data)) {
        records = data
          .map(r => ({
            time: Math.floor((r.fundingTime || 0) / 1000),
            rate: parseFloat(r.fundingRate || 0),
          }))
          .filter(r => r.time > 0 && Number.isFinite(r.rate))
          .sort((a, b) => a.time - b.time);
        if (records.length < 10) records = null;
      }
    }
  } catch { /* fall through to Coinalyze */ }

  // Fallback: Coinalyze (BTC only — see COINALYZE_PERP_MAP).
  if (!records) {
    records = await fetchFundingFromCoinalyze(symbol);
  }

  if (!records) return null;
  fundingCache.set(perp, { records, fetchedAt: Date.now() });
  return records;
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
