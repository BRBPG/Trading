// ─── Earnings calendar + PEAD signal ───────────────────────────────────────
// Post-Earnings-Announcement-Drift (PEAD) is one of the most persistent
// documented equity anomalies (Bernard & Thomas 1989, still working in 2024
// per multiple recent replication studies). The effect: a stock that BEATS
// earnings consensus drifts up for ~30-60 days post-announcement, and a
// MISS drifts down. Retail-tractable because:
//   1. Earnings dates are free (Finnhub /stock/earnings endpoint)
//   2. The effect is driven by underreaction, not arbitrage, so it still
//      works despite being public knowledge
//   3. It's additive — stacks cleanly alongside other features
//
// This module:
//   1. Fetches up-to-4-quarters of historical earnings per symbol (beats
//      / misses / exact surprise)
//   2. Computes two features per-symbol: daysSinceEarnings and surpriseSign
//      (weighted by magnitude, decayed with days-since to model drift)
//   3. Caches per symbol in localStorage keyed by a weekly refresh — earnings
//      dates don't change except when a company announces a new date. A
//      30-day cache is a fine heuristic.

const CACHE_KEY = "trader_earnings_cache_v1";
const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 1 week

function loadCache() {
  try {
    const raw = JSON.parse(localStorage.getItem(CACHE_KEY) || "{}");
    return raw && typeof raw === "object" ? raw : {};
  } catch { return {}; }
}

function saveCache(cache) {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(cache));
  } catch { /* quota etc. */ }
}

// Fetch the last few reported earnings for a symbol from Finnhub.
// Response shape: [ { symbol, period, actual, estimate, surprise,
//                    surprisePercent, ... }, ... ]
// Returns null if key is missing, symbol not covered, or fetch fails.
async function fetchEarningsFromFinnhub(symbol, finnhubKey) {
  if (!finnhubKey) return null;
  const url = `https://finnhub.io/api/v1/stock/earnings?symbol=${encodeURIComponent(symbol)}&token=${finnhubKey}`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(6000) });
    if (!res.ok) return null;
    const data = await res.json();
    if (!Array.isArray(data)) return null;
    // Sort most-recent first; normalise to what we need downstream.
    return data
      .map(d => ({
        period: d.period,                          // "2024-10-31" etc.
        actual: d.actual,
        estimate: d.estimate,
        surprisePercent: d.surprisePercent,        // e.g. +5.2 = beat by 5.2%
      }))
      .filter(d => d.period && d.actual != null && d.estimate != null)
      .sort((a, b) => new Date(b.period) - new Date(a.period));
  } catch {
    return null;
  }
}

// Get earnings data for a symbol, using cache when fresh.
export async function getEarnings(symbol, finnhubKey) {
  if (!symbol) return null;
  const cache = loadCache();
  const entry = cache[symbol];
  const now = Date.now();
  if (entry && now - entry.fetchedAt < CACHE_TTL_MS) return entry.data;

  const data = await fetchEarningsFromFinnhub(symbol, finnhubKey);
  if (data) {
    cache[symbol] = { data, fetchedAt: now };
    saveCache(cache);
  }
  return data;
}

// Batch-fetch earnings for all symbols. Returns a { symbol → earnings[] } map.
export async function getEarningsBatch(symbols, finnhubKey) {
  const results = await Promise.all(
    symbols.map(async sym => [sym, await getEarnings(sym, finnhubKey)])
  );
  return Object.fromEntries(results);
}

// ─── PEAD feature computation ──────────────────────────────────────────────
// Given an earnings history and a reference timestamp, compute two scalars:
//   daysSinceEarnings: number of calendar days since the last earnings that
//                      happened BEFORE the reference timestamp (-1 if none)
//   surpriseDecayed:   surprisePercent of that last earnings, decayed by
//                      exp(-daysSince / 30) so the effect is strongest in
//                      the first month and fades over ~60 days.
//
// Both features scaled into [-1, 1] for the feature vector.
export function computePeadFeatures(earningsHistory, referenceTimeMs = Date.now()) {
  if (!earningsHistory || !earningsHistory.length) {
    return { daysSinceEarnings: 0, surpriseDecayed: 0 };
  }

  // Find the most recent earnings event BEFORE the reference time.
  const past = earningsHistory
    .filter(e => new Date(e.period).getTime() <= referenceTimeMs)
    .sort((a, b) => new Date(b.period) - new Date(a.period));

  if (!past.length) return { daysSinceEarnings: 0, surpriseDecayed: 0 };

  const last = past[0];
  const daysSince = (referenceTimeMs - new Date(last.period).getTime()) / 86_400_000;

  // Feature 1: days-since, capped at 90 and normalised.
  // PEAD drift primarily happens in days 1-60 post-announcement; beyond that
  // the effect has faded so we saturate the feature.
  const clampedDays = Math.min(90, Math.max(0, daysSince));
  const daysFeature = (clampedDays / 90) * 2 - 1; // [-1, 1], -1 = just earned, +1 = 90d+ ago

  // Feature 2: surprise% × exp-decay. surprisePercent typically ±0.5-10%.
  // Scale by /5 so ±5% surprise = ±1 before decay.
  const decay = Math.exp(-daysSince / 30); // half-life ~21 days
  const surpriseScaled = Math.max(-1, Math.min(1, (last.surprisePercent || 0) / 5));
  const surpriseDecayed = surpriseScaled * decay;

  return {
    daysSinceEarnings: daysFeature,
    surpriseDecayed,
    // Raw values for debugging / UI display
    lastEarnings: last.period,
    lastSurprisePct: last.surprisePercent,
    daysSinceRaw: daysSince,
  };
}

// Clear cache (for forced refresh from UI if needed).
export function resetEarningsCache() {
  localStorage.removeItem(CACHE_KEY);
}
