// ─── Polygon.io historical bar adapter ──────────────────────────────────────
// Yahoo caps 5-min candle history at ~60 days. For proper walk-forward
// training we want months-to-years of data, which Polygon's paid tier
// provides. $29/mo Starter gives 5 calls/min (plenty for 15 symbols) and
// unlimited historical range.
//
// Endpoint: GET /v2/aggs/ticker/{sym}/range/5/minute/{from}/{to}?apiKey=X
//   In practice Polygon caps the response body at ~10,000 bars per page
//   regardless of the limit= query param being honoured (queryCount echoes
//   50000 but resultsCount is 10000). Additional pages are exposed via the
//   `next_url` field on the JSON response — we follow it until empty so a
//   180-day 5-min request actually returns ~52k bars (~6 pages) instead of
//   silently truncating to ~35 days.
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
  const firstUrl = `${POLYGON_BASE}/v2/aggs/ticker/${encodeURIComponent(polyTicker)}/range/${range}/${fmt(from)}/${fmt(to)}?adjusted=true&sort=asc&limit=50000&apiKey=${apiKey}`;

  // Crypto Standard plan ≈ 100 req/min; pace at 700ms between pages to stay
  // well under the limit. Stocks Starter (5/min) callers should set their
  // own outer pacing — this delay only protects pagination within one call.
  const PAGE_PACING_MS = 700;
  // Hard cap on pages followed: defends against a runaway next_url loop on
  // an unexpected Polygon response. 5m × 365d ≈ 105k bars / 10k = ~11 pages
  // worst case, so 30 is comfortable headroom without being unbounded.
  const MAX_PAGES = 30;

  const closes = [], highs = [], lows = [], volumes = [], timestamps = [];
  let url = firstUrl;
  let pages = 0;

  try {
    while (url && pages < MAX_PAGES) {
      const res = await fetch(url, { signal: AbortSignal.timeout(20000) });
      if (!res.ok) {
        // 403 = subscription doesn't cover this symbol class (common for
        // niche crypto pairs); 429 = rate limit. Surface the reason so the
        // user can see WHY a symbol dropped out in devtools.
        if (pages === 0) {
          console.warn(`[polygon] ${polyTicker} → HTTP ${res.status} (${symbol}). Falling back.`);
          return null;
        }
        // Mid-flight failure: keep partial result rather than discard work.
        console.warn(`[polygon] ${polyTicker} → page ${pages + 1} HTTP ${res.status}; returning ${closes.length} partial bars`);
        break;
      }
      const data = await res.json();
      if (data?.status === "ERROR") {
        if (pages === 0) {
          console.warn(`[polygon] ${polyTicker} → ${data.error || "ERROR status"} (${symbol})`);
          return null;
        }
        console.warn(`[polygon] ${polyTicker} → page ${pages + 1} ERROR (${data.error}); returning ${closes.length} partial bars`);
        break;
      }
      if (!Array.isArray(data?.results)) {
        if (pages === 0) {
          console.warn(`[polygon] ${polyTicker} → no results array (${symbol}). Response:`, data?.resultsCount, data?.status);
          return null;
        }
        console.warn(`[polygon] ${polyTicker} → page ${pages + 1} no results array; returning ${closes.length} partial bars`);
        break;
      }

      for (const bar of data.results) {
        // Polygon: o=open, h=high, l=low, c=close, v=volume, t=timestamp_ms
        if (bar.c == null || bar.h == null || bar.l == null) continue;
        closes.push(bar.c);
        highs.push(bar.h);
        lows.push(bar.l);
        volumes.push(bar.v ?? 0);
        timestamps.push(Math.floor(bar.t / 1000)); // → seconds to match Yahoo shape
      }
      pages++;

      // next_url is the same /v2/aggs path with a cursor query param baked in
      // and DOES NOT include apiKey — append it. When absent or empty there
      // are no more bars in the requested range.
      const nextUrl = data.next_url;
      if (!nextUrl) { url = null; break; }
      url = nextUrl.includes("apiKey=") ? nextUrl : `${nextUrl}&apiKey=${apiKey}`;
      await new Promise(r => setTimeout(r, PAGE_PACING_MS));
    }

    if (pages >= MAX_PAGES && url) {
      console.warn(`[polygon] ${polyTicker} → hit MAX_PAGES=${MAX_PAGES}; returning ${closes.length} bars (more available)`);
    }

    if (closes.length < 30) {
      console.warn(`[polygon] ${polyTicker} → only ${closes.length} bars, need ≥30 (${symbol})`);
      return null;
    }
    return { closes, highs, lows, volumes, timestamps };
  } catch (err) {
    if (closes.length >= 30) {
      console.warn(`[polygon] ${polyTicker} → fetch error after ${pages} pages: ${err.message}; returning ${closes.length} partial bars`);
      return { closes, highs, lows, volumes, timestamps };
    }
    console.warn(`[polygon] ${polyTicker} → fetch error: ${err.message} (${symbol})`);
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
