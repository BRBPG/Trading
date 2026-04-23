// ─── Polygon.io historical bar adapter ──────────────────────────────────────
// Yahoo caps 5-min candle history at ~60 days. For proper walk-forward
// training we want months-to-years of data, which Polygon's paid tier
// provides. $29/mo Starter gives 5 calls/min (plenty for 15 symbols) and
// unlimited historical range.
//
// Endpoint: GET /v2/aggs/ticker/{sym}/range/5/minute/{from}/{to}?apiKey=X
//   Returns up to 50,000 bars per call — 365 days of 5-min = ~28k bars,
//   fits in one call per symbol.
//
// Graceful degradation: if no key is provided, callers fall back to Yahoo
// and the existing 7-day training horizon.

const POLYGON_BASE = "https://api.polygon.io";

export function hasPolygonKey(key) {
  return typeof key === "string" && key.length > 10;
}

// Polygon uses different ticker-prefix conventions per asset class:
//   Stocks   → plain ticker (AAPL)
//   Crypto   → "X:BTCUSD" format (prefix X:, no dash)
//   Forex    → "C:EURUSD"
// Our watchlist uses the Yahoo-style dash format (BTC-USD) for crypto
// symbols. This helper normalises to Polygon's expected format, so the same
// symbol string from the watchlist works whether it's routed to Polygon
// (crypto subscription) or Yahoo (proxy).
function toPolygonTicker(symbol) {
  const s = symbol.toUpperCase();
  // Crypto: BTC-USD → X:BTCUSD; BTC-USDT → X:BTCUSDT. The Polygon
  // Currencies Standard plan ($49/mo) covers both crypto (X: prefix) and
  // forex (C: prefix, e.g. C:EURUSD) at real-time with ~100 calls/min
  // rate limit — well above Stocks Starter's 5/min ceiling.
  if (/-USD(T)?$/.test(s)) return "X:" + s.replace("-", "");
  // Forex support reserved for future watchlist additions (e.g. EUR-USD
  // routed as C:EURUSD). Not used in current watchlists but the prefix
  // is here so adding FX pairs later doesn't require code changes.
  if (/^EUR-|^GBP-|^JPY-|^AUD-|^CAD-|^CHF-|^NZD-/.test(s)) return "C:" + s.replace("-", "");
  return s;
}

// Fetch bars for the past {daysAgo} days at the requested interval.
// interval is "5m" (5-min bars) or "1d" (daily bars). The URL path differs:
//   5m → /range/5/minute/
//   1d → /range/1/day/
// Same output shape either way: { closes, highs, lows, volumes, timestamps }
// where timestamps are Unix seconds.
export async function fetchPolygonBars(symbol, daysAgo = 90, apiKey, interval = "5m") {
  if (!hasPolygonKey(apiKey)) return null;

  const polyTicker = toPolygonTicker(symbol);

  // Polygon uses YYYY-MM-DD format. Use UTC to avoid local-time edge cases.
  const to = new Date();
  const from = new Date(to.getTime() - daysAgo * 24 * 60 * 60 * 1000);
  const fmt = d => d.toISOString().slice(0, 10);

  const range = interval === "1d" ? "1/day" : "5/minute";

  // adjusted=true applies split/dividend adjustments. sort=asc gives oldest
  // first which is what all our indicators assume.
  const url = `${POLYGON_BASE}/v2/aggs/ticker/${encodeURIComponent(polyTicker)}/range/${range}/${fmt(from)}/${fmt(to)}?adjusted=true&sort=asc&limit=50000&apiKey=${apiKey}`;

  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(15000) });
    if (!res.ok) {
      // 403 usually means the subscription doesn't cover this symbol class;
      // 429 is rate limit. Either way — null signals caller to fall back.
      return null;
    }
    const data = await res.json();
    if (data?.status === "ERROR" || !Array.isArray(data?.results)) return null;

    const closes = [], highs = [], lows = [], volumes = [], timestamps = [];
    for (const bar of data.results) {
      // Polygon: o=open, h=high, l=low, c=close, v=volume, t=timestamp_ms
      if (bar.c == null || bar.h == null || bar.l == null) continue;
      closes.push(bar.c);
      highs.push(bar.h);
      lows.push(bar.l);
      volumes.push(bar.v ?? 0);
      timestamps.push(Math.floor(bar.t / 1000)); // → seconds to match Yahoo shape
    }
    if (closes.length < 30) return null;
    return { closes, highs, lows, volumes, timestamps };
  } catch {
    return null;
  }
}

// Probe the key with a tiny request so we can tell the user upfront whether
// their Polygon key actually works, rather than failing silently mid-backtest.
export async function validatePolygonKey(apiKey) {
  if (!hasPolygonKey(apiKey)) return { ok: false, reason: "Missing key" };
  const url = `${POLYGON_BASE}/v3/reference/tickers/AAPL?apiKey=${apiKey}`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(6000) });
    if (res.status === 401 || res.status === 403) return { ok: false, reason: "Unauthorised — check subscription" };
    if (res.status === 429) return { ok: false, reason: "Rate limited — try again in a minute" };
    if (!res.ok) return { ok: false, reason: `HTTP ${res.status}` };
    const data = await res.json();
    return { ok: data?.results != null, reason: data?.results == null ? "Unexpected response" : "ok" };
  } catch (err) {
    return { ok: false, reason: err.message || "Network error" };
  }
}
