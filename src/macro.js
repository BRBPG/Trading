// ─── Macro context module ───────────────────────────────────────────────────
// Single-symbol technicals (RSI/MACD/ATR/etc.) capture only what's happening
// on that one chart. Everything interesting in real markets is MULTI-ASSET:
// a rate shock shows up in TNX before equities react; a risk-off regime
// shows up in VIX + HYG/LQD before individual names break; commodity stress
// shows up in DXY + oil.
//
// This module fetches a handful of macro reference tickers from Yahoo (free,
// already proven in the existing live-bar path), caches them for 2 minutes,
// and exposes both a LIVE snapshot (for the dashboard refresh loop) and a
// HISTORICAL-ALIGNED version (for the backtester to consume when it asks
// "what was VIX doing at timestamp T").
//
// Reference tickers chosen per the research findings:
//   ^VIX     30-day implied vol on SPX
//   ^VIX9D   9-day vol; VIX9D/VIX > 1 = short-term stress spike
//   ^SKEW    crash risk premium
//   DX-Y.NYB US Dollar Index
//   ^TNX     10-year Treasury yield
//   CL=F     WTI crude oil
//   GC=F     Gold
//   HYG      high-yield credit ETF (risk proxy)
//   LQD      investment-grade credit ETF

const YAHOO_PROXIES = [
  u => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
  u => `https://corsproxy.io/?${encodeURIComponent(u)}`,
];

export const MACRO_SYMBOLS = {
  vix:    "^VIX",
  vix9d:  "^VIX9D",
  skew:   "^SKEW",
  dxy:    "DX-Y.NYB",
  tnx:    "^TNX",
  oil:    "CL=F",
  gold:   "GC=F",
  hyg:    "HYG",
  lqd:    "LQD",
};

const macroCache = { data: null, fetchedAt: 0 };
const MACRO_CACHE_MS = 2 * 60 * 1000;

async function fetchYahooSymbol(sym, range = "5d", interval = "15m") {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(sym)}?interval=${interval}&range=${range}`;
  for (const proxy of YAHOO_PROXIES) {
    try {
      const res = await fetch(proxy(url), { signal: AbortSignal.timeout(6000) });
      if (!res.ok) continue;
      const data = await res.json();
      const r = data?.chart?.result?.[0];
      if (!r) continue;
      const m = r.meta || {};
      const q = r.indicators?.quote?.[0] || {};
      const ts = r.timestamp || [];
      const closes = [], timestamps = [];
      for (let i = 0; i < ts.length; i++) {
        if (q.close?.[i] == null) continue;
        closes.push(q.close[i]);
        timestamps.push(ts[i]);
      }
      return {
        price: m.regularMarketPrice,
        prevClose: m.chartPreviousClose ?? m.previousClose,
        closes,
        timestamps,
      };
    } catch { /* try next proxy */ }
  }
  return null;
}

// ─── Live snapshot for the dashboard refresh loop ──────────────────────────
// Returns { vix, vix9d, ..., vixTermStructure, vixZ, dxyMom5, ... }
// Cached for 2 minutes so refreshAll isn't pounding Yahoo every 30s.
export async function fetchMacroSnapshot() {
  const now = Date.now();
  if (macroCache.data && now - macroCache.fetchedAt < MACRO_CACHE_MS) return macroCache.data;

  // Fan out. Run in parallel — 9 requests through 2 proxies concurrently.
  const entries = await Promise.all(
    Object.entries(MACRO_SYMBOLS).map(async ([key, sym]) => {
      const r = await fetchYahooSymbol(sym, "5d", "15m");
      return [key, r];
    })
  );
  const raw = Object.fromEntries(entries.filter(([, v]) => v != null));

  // Derived signals — these are what goes into the feature vector
  const derived = {};

  // VIX level + its 60-bar z-score (regime: is current vol extreme vs recent)
  if (raw.vix?.closes?.length > 20) {
    const c = raw.vix.closes.slice(-60);
    const mean = c.reduce((a, b) => a + b, 0) / c.length;
    const sd   = Math.sqrt(c.reduce((a, b) => a + (b - mean) ** 2, 0) / c.length);
    derived.vixLevel = raw.vix.price;
    derived.vixZ = sd > 0 ? (raw.vix.price - mean) / sd : 0;
  }

  // Short-end stress: VIX9D / VIX > 1.0 = near-term panic bid
  if (raw.vix9d?.price && raw.vix?.price) {
    derived.vixTerm = raw.vix9d.price / raw.vix.price;
  }

  // SKEW: 100 = neutral, >130 = heavy out-the-money put bid (crash hedging)
  if (raw.skew?.price) derived.skew = raw.skew.price;

  // Cross-asset 5-bar (≈ 75 min of 15m bars) momentum — small numbers, sign matters
  const mom5 = (r) => {
    if (!r?.closes || r.closes.length < 6) return 0;
    const a = r.closes.slice(-6);
    return (a[a.length - 1] - a[0]) / a[0];
  };
  derived.dxyMom5 = mom5(raw.dxy);
  derived.tnxMom5 = mom5(raw.tnx);
  derived.oilMom5 = mom5(raw.oil);
  derived.goldMom5 = mom5(raw.gold);

  // Credit-risk proxy: HYG/LQD ratio change. HYG falling faster than LQD =
  // credit stress leaking into equities soon.
  if (raw.hyg?.price && raw.lqd?.price && raw.hyg?.closes?.length > 5 && raw.lqd?.closes?.length > 5) {
    const hygC = raw.hyg.closes.slice(-6);
    const lqdC = raw.lqd.closes.slice(-6);
    const hygMom = (hygC[hygC.length - 1] - hygC[0]) / hygC[0];
    const lqdMom = (lqdC[lqdC.length - 1] - lqdC[0]) / lqdC[0];
    derived.creditSpreadMom = hygMom - lqdMom; // negative = credit widening
  }

  const out = { raw, ...derived, fetchedAt: now };
  macroCache.data = out;
  macroCache.fetchedAt = now;
  return out;
}

// ─── Historical-aligned macro for backtesting ──────────────────────────────
// Fetches macro series covering the same window as the symbol backtest and
// returns a function `at(timestamp) → { vixZ, vixTerm, dxyMom5, ... }` that
// the backtester calls per-trade to get point-in-time macro context.
//
// Alignment rule: for a given decision timestamp T, we use the MOST RECENT
// macro bar with timestamp ≤ T. This is the honest point-in-time behaviour
// and prevents look-ahead leakage from macro feeds that update on a
// different cadence than the symbol bars.
export async function fetchMacroHistorical(daysAgo = 30) {
  const range = `${Math.max(daysAgo + 1, 5)}d`;
  const interval = daysAgo <= 60 ? "15m" : "1h";
  const entries = await Promise.all(
    Object.entries(MACRO_SYMBOLS).map(async ([key, sym]) => {
      const r = await fetchYahooSymbol(sym, range, interval);
      return [key, r];
    })
  );
  const raw = Object.fromEntries(entries.filter(([, v]) => v != null));

  // Pre-compute rolling z-scores for VIX so at() is a cheap O(1) lookup.
  const vixZSeries = [];
  if (raw.vix?.closes?.length > 20) {
    const c = raw.vix.closes;
    const window = 60;
    for (let i = 0; i < c.length; i++) {
      const start = Math.max(0, i - window + 1);
      const slice = c.slice(start, i + 1);
      const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
      const sd   = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length);
      vixZSeries.push(sd > 0 ? (c[i] - mean) / sd : 0);
    }
  }

  // Binary search for greatest macro-bar index with timestamp ≤ t.
  const findIdx = (timestamps, t) => {
    if (!timestamps?.length || t < timestamps[0]) return -1;
    let lo = 0, hi = timestamps.length - 1, ans = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (timestamps[mid] <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
    }
    return ans;
  };

  // 5-bar momentum lookup (non-leaky — uses only bars with timestamp ≤ t).
  const mom5At = (series, t) => {
    if (!series?.closes?.length || !series?.timestamps?.length) return 0;
    const i = findIdx(series.timestamps, t);
    if (i < 5) return 0;
    const a = series.closes[i - 5], b = series.closes[i];
    return a > 0 ? (b - a) / a : 0;
  };

  const at = (t) => {
    const out = {};
    // t is expected in seconds (like the backtest's bars.timestamps entries)
    if (raw.vix?.closes?.length) {
      const i = findIdx(raw.vix.timestamps, t);
      if (i >= 0) {
        out.vixLevel = raw.vix.closes[i];
        out.vixZ = vixZSeries[i] ?? 0;
      }
    }
    if (raw.vix9d && raw.vix) {
      const i9 = findIdx(raw.vix9d.timestamps, t);
      const iV = findIdx(raw.vix.timestamps, t);
      if (i9 >= 0 && iV >= 0 && raw.vix.closes[iV] > 0) {
        out.vixTerm = raw.vix9d.closes[i9] / raw.vix.closes[iV];
      }
    }
    if (raw.skew) {
      const i = findIdx(raw.skew.timestamps, t);
      if (i >= 0) out.skew = raw.skew.closes[i];
    }
    out.dxyMom5  = mom5At(raw.dxy,  t);
    out.tnxMom5  = mom5At(raw.tnx,  t);
    out.oilMom5  = mom5At(raw.oil,  t);
    out.goldMom5 = mom5At(raw.gold, t);

    if (raw.hyg && raw.lqd) {
      const hygMom = mom5At(raw.hyg, t);
      const lqdMom = mom5At(raw.lqd, t);
      out.creditSpreadMom = hygMom - lqdMom;
    }
    return out;
  };

  return { at, raw };
}
