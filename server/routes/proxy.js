import { Router } from "express";

const router = Router();

const ALLOWED_HOSTS = [
  "query1.finance.yahoo.com",
  "query2.finance.yahoo.com",
  "fapi.binance.com",
  "api.binance.com",
  "www.deribit.com",
  "open-api.coinglass.com",
  "api.coinalyze.net",
  "api.coingecko.com",
  "api.coincap.io",
  "api.coinbase.com",
];

const CACHE_TTL_MS = 30 * 1000;
const cache = new Map();

router.get("/proxy", async (req, res) => {
  const raw = req.query.url;
  if (!raw || typeof raw !== "string") {
    return res.status(400).json({ error: "missing url query param" });
  }

  let target;
  try {
    target = new URL(raw);
  } catch {
    return res.status(400).json({ error: "invalid url" });
  }

  const host = target.hostname;
  if (!ALLOWED_HOSTS.some(h => host === h || host.endsWith("." + h))) {
    return res.status(403).json({ error: "host not allowed", host });
  }

  const cacheKey = target.toString();
  const cached = cache.get(cacheKey);
  if (cached && Date.now() - cached.fetchedAt < CACHE_TTL_MS) {
    res.status(cached.status);
    if (cached.contentType) res.set("Content-Type", cached.contentType);
    return res.send(cached.body);
  }

  try {
    const upstream = await fetch(cacheKey, {
      signal: AbortSignal.timeout(15000),
      headers: { "User-Agent": "Mozilla/5.0" },
    });
    const contentType = upstream.headers.get("content-type") || "application/octet-stream";
    const body = Buffer.from(await upstream.arrayBuffer());

    // Cache successful responses only — don't cache transient upstream errors.
    if (upstream.ok) {
      cache.set(cacheKey, { status: upstream.status, contentType, body, fetchedAt: Date.now() });
    }

    res.status(upstream.status);
    res.set("Content-Type", contentType);
    res.send(body);
  } catch (err) {
    res.status(503).json({ status: "unavailable", reason: err.message || "fetch failed" });
  }
});

export default router;
