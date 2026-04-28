import { Router } from "express";
import { fetchHistoricalBars } from "../training/simulation.js";
import { hasPolygonKey } from "../../src/polygon.js";

const router = Router();

const QUOTE_TTL_MS = 30 * 1000;
const BARS_TTL_MS  = 60 * 1000;
const quoteCache = new Map();
const barsCache  = new Map();

function rangeForDays(days) {
  if (days <= 1)  return "1d";
  if (days <= 7)  return "7d";
  if (days <= 59) return "60d";
  return "1y";
}

async function fetchYahooChart(symbol, interval, range) {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=${interval}&range=${range}`;
  const res = await fetch(url, {
    signal: AbortSignal.timeout(15000),
    headers: { "User-Agent": "Mozilla/5.0" },
  });
  if (!res.ok) throw new Error(`yahoo http ${res.status}`);
  const data = await res.json();
  const result = data?.chart?.result?.[0];
  if (!result) throw new Error("yahoo empty result");
  return result;
}

router.get("/quote/:symbol", async (req, res) => {
  const symbol = req.params.symbol;
  const cached = quoteCache.get(symbol);
  if (cached && Date.now() - cached.fetchedAt < QUOTE_TTL_MS) {
    return res.json(cached.payload);
  }

  try {
    const result = await fetchYahooChart(symbol, "1m", "1d");
    const meta = result.meta || {};
    const price     = meta.regularMarketPrice;
    const prevClose = meta.chartPreviousClose ?? meta.previousClose;
    if (price == null || prevClose == null) {
      return res.status(503).json({ status: "unavailable", reason: "missing price/prevClose" });
    }

    const timestamps = result.timestamp || [];
    const q = result.indicators?.quote?.[0] || {};
    const closes = [], highs = [], lows = [], volumes = [];
    for (let i = 0; i < timestamps.length; i++) {
      if (q.close?.[i] == null) continue;
      closes.push(q.close[i]);
      highs.push(q.high?.[i]   ?? q.close[i]);
      lows.push(q.low?.[i]     ?? q.close[i]);
      volumes.push(q.volume?.[i] ?? 0);
    }

    const payload = {
      symbol,
      price,
      prevClose,
      dayHigh:  meta.regularMarketDayHigh ?? null,
      dayLow:   meta.regularMarketDayLow  ?? null,
      high52:   meta.fiftyTwoWeekHigh     ?? null,
      low52:    meta.fiftyTwoWeekLow      ?? null,
      volume:   meta.regularMarketVolume  ?? null,
      currency: meta.currency             ?? null,
      bars: closes.length >= 30 ? { closes, highs, lows, volumes } : null,
    };

    quoteCache.set(symbol, { payload, fetchedAt: Date.now() });
    res.json(payload);
  } catch (err) {
    res.status(503).json({ status: "unavailable", reason: err.message || "fetch failed" });
  }
});

router.get("/bars/:symbol", async (req, res) => {
  const symbol   = req.params.symbol;
  const interval = ["1m", "5m", "1d"].includes(req.query.interval) ? req.query.interval : "5m";
  const days     = Math.max(1, parseInt(req.query.days, 10) || 1);
  const key = `${symbol}|${interval}|${days}`;
  const cached = barsCache.get(key);
  if (cached && Date.now() - cached.fetchedAt < BARS_TTL_MS) {
    return res.json(cached.payload);
  }

  const isCrypto = /-USD(T)?$/i.test(symbol);
  const polyKey  = process.env.POLYGON_KEY;

  try {
    if (isCrypto && days > 7 && hasPolygonKey(polyKey)) {
      const r = await fetchHistoricalBars(symbol, days, polyKey, interval);
      if (r?.bars?.closes?.length) {
        const { closes, highs, lows, volumes, timestamps } = r.bars;
        const payload = { closes, highs, lows, volumes, timestamps, source: r.source };
        barsCache.set(key, { payload, fetchedAt: Date.now() });
        return res.json(payload);
      }
      return res.status(503).json({ status: "unavailable", reason: "polygon empty" });
    }

    const result = await fetchYahooChart(symbol, interval, rangeForDays(days));
    const timestamps = result.timestamp || [];
    const q = result.indicators?.quote?.[0] || {};
    const closes = [], highs = [], lows = [], volumes = [], ts = [];
    for (let i = 0; i < timestamps.length; i++) {
      if (q.close?.[i] == null) continue;
      closes.push(q.close[i]);
      highs.push(q.high?.[i]   ?? q.close[i]);
      lows.push(q.low?.[i]     ?? q.close[i]);
      volumes.push(q.volume?.[i] ?? 0);
      ts.push(Math.floor(timestamps[i]));
    }
    if (!closes.length) {
      return res.status(503).json({ status: "unavailable", reason: "no bars" });
    }
    const payload = { closes, highs, lows, volumes, timestamps: ts, source: "yahoo" };
    barsCache.set(key, { payload, fetchedAt: Date.now() });
    res.json(payload);
  } catch (err) {
    res.status(503).json({ status: "unavailable", reason: err.message || "fetch failed" });
  }
});

export default router;
