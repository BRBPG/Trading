// ─── Data-cleaning pipeline ─────────────────────────────────────────────────
// Sits between the raw candle generator and the indicator pipeline. Goal is to
// stop one bad tick, one zero-volume gap, or one split-mis-adjustment from
// contaminating RSI/MACD/ATR/EMA and cascading into the model score.
//
// Techniques used (in order):
//   1. Zero-volume-bar fill during regular-hours sessions (halts, feed drops).
//   2. Hampel filter on closes — rolling median + MAD outlier detector, the
//      industry-standard tick validator (Bloomberg, Refinitiv use variants).
//   3. Winsorisation on bar-to-bar returns — caps at the 0.1st/99.9th
//      percentile so a single flash-crash tick can't blow up σ/ATR.
//   4. OHLC consistency clamp — after cleaning, ensure low ≤ close ≤ high so
//      ATR math stays sensible.
//   5. Quality flag (clean | suspect | stale) so the model AND the AI can see
//      how trustworthy this quote is.

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) return 0;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

function mad(arr, med) {
  return median(arr.map(v => Math.abs(v - med)));
}

// ─── Hampel filter ──────────────────────────────────────────────────────────
// For each point: symmetric window → median + MAD → flag if |x - med| > k·σ
// where σ = 1.4826 · MAD (Gaussian-consistent scale factor). Replace flagged
// points with the local median.
export function hampelFilter(series, windowSize = 7, nSigmas = 3) {
  if (series.length < windowSize) return { cleaned: [...series], flagged: [] };
  const cleaned = [...series];
  const flagged = [];
  const half = Math.floor(windowSize / 2);
  const k = 1.4826;
  for (let i = half; i < series.length - half; i++) {
    const window = series.slice(i - half, i + half + 1);
    const m = median(window);
    const sigma = k * mad(window, m);
    if (sigma > 0 && Math.abs(series[i] - m) > nSigmas * sigma) {
      cleaned[i] = m;
      flagged.push(i);
    }
  }
  return { cleaned, flagged };
}

// ─── Winsorise bar-to-bar returns ───────────────────────────────────────────
// Cap returns at the given percentiles, then re-integrate to produce a cleaned
// close series. Protects rolling-σ (Bollinger) and ATR from a single outlier.
export function winsorizeReturns(closes, lowerPct = 0.001, upperPct = 0.999) {
  if (closes.length < 10) return { closes: [...closes], capped: 0 };
  const rets = [];
  for (let i = 1; i < closes.length; i++) {
    const prev = closes[i - 1];
    rets.push(prev > 0 ? (closes[i] - prev) / prev : 0);
  }
  const sorted = [...rets].sort((a, b) => a - b);
  const loIdx = Math.max(0, Math.floor(rets.length * lowerPct));
  const hiIdx = Math.min(rets.length - 1, Math.floor(rets.length * upperPct));
  const lo = sorted[loIdx];
  const hi = sorted[hiIdx];
  const out = [closes[0]];
  let capped = 0;
  for (let i = 1; i < closes.length; i++) {
    const prev = out[i - 1];
    const r = prev > 0 ? (closes[i] - prev) / prev : 0;
    if (r < lo)       { out.push(prev * (1 + lo)); capped++; }
    else if (r > hi)  { out.push(prev * (1 + hi)); capped++; }
    else              { out.push(closes[i]); }
  }
  return { closes: out, capped };
}

// ─── Zero-volume / halt / frozen-price bar handling ────────────────────────
// During OPEN session, these shapes are almost always data problems:
//   - zero volume        = feed gap or halt
//   - high == low        = tape frozen (LULD band, auction, halt)
//   - price unchanged across 3+ consecutive bars with any volume = suspect
// Forward-fill the OHLC so indicators don't see phantom prints. Cap the
// consecutive fill length at MAX_CONSEC_FFILL to avoid propagating bad data
// across long halts — if a symbol halts for 30+ bars it's no longer safe
// to compute VWAP / MACD / BB from the pre-halt baseline. Pre/post-market
// minutes can legitimately be empty, so we skip this rule there.
const MAX_CONSEC_FFILL = 3;

export function cleanZeroVolumeBars({ closes, highs, lows, volumes }, session = "OPEN") {
  if (session !== "OPEN") {
    return { closes: [...closes], highs: [...highs], lows: [...lows], volumes: [...volumes], zeroVolFilled: 0, haltBars: 0, frozenBars: 0, ffillAborted: 0 };
  }
  const c = [...closes], h = [...highs], l = [...lows], v = [...volumes];
  let filled = 0, haltBars = 0, frozenBars = 0, ffillAborted = 0, consecFill = 0;
  for (let i = 1; i < v.length; i++) {
    const zeroVol = !v[i];
    const frozenBar = h[i] != null && l[i] != null && h[i] === l[i];
    const needsFill = zeroVol || (frozenBar && v[i] > 0);
    if (needsFill) {
      if (consecFill >= MAX_CONSEC_FFILL) {
        // Don't keep propagating; leave the bar untouched so downstream
        // quality assessment sees it as real. Count as an abort event.
        ffillAborted++;
        consecFill = 0;
        continue;
      }
      consecFill++;
      if (zeroVol) haltBars++;
      if (frozenBar && !zeroVol) frozenBars++;
      c[i] = c[i - 1];
      h[i] = Math.max(h[i] || c[i - 1], c[i - 1]);
      l[i] = Math.min(l[i] || c[i - 1], c[i - 1]);
      filled++;
    } else {
      consecFill = 0;
    }
  }
  return { closes: c, highs: h, lows: l, volumes: v, zeroVolFilled: filled, haltBars, frozenBars, ffillAborted };
}

// ─── OHLC consistency clamp ─────────────────────────────────────────────────
// After the close series has been filtered, highs/lows may be tighter than
// the cleaned close. Clamp so low ≤ close ≤ high always holds.
function clampOHLC(closes, highs, lows) {
  const h = [...highs], l = [...lows];
  for (let i = 0; i < closes.length; i++) {
    if (closes[i] > h[i]) h[i] = closes[i];
    if (closes[i] < l[i]) l[i] = closes[i];
  }
  return { highs: h, lows: l };
}

// ─── Quality assessment ─────────────────────────────────────────────────────
//   clean   — nothing needed fixing
//   suspect — something was filtered/filled/capped; treat with caution
//   stale   — quote hasn't updated in longer than expected for this session
export function assessQuality({ flagged, capped, zeroVolFilled, lastFetched, session }) {
  const age = Date.now() - (lastFetched || Date.now());
  const staleMs = session === "OPEN" ? 90_000 : 30 * 60_000;
  if (age > staleMs) return "stale";
  if ((flagged || 0) + (capped || 0) + (zeroVolFilled || 0) === 0) return "clean";
  return "suspect";
}

// ─── Full cleaning pipeline ─────────────────────────────────────────────────
// anchors: { first, last } — the real prevClose and current price from the
// live feed. We preserve these after cleaning so the series still agrees with
// reality at its endpoints.
export function cleanBars(bars, session = "OPEN", anchors = {}) {
  const zv   = cleanZeroVolumeBars(bars, session);
  const hamp = hampelFilter(zv.closes, 7, 3);
  const win  = winsorizeReturns(hamp.cleaned, 0.001, 0.999);

  const closes = win.closes;
  if (anchors.first != null && closes.length > 0) closes[0] = anchors.first;
  if (anchors.last  != null && closes.length > 0) closes[closes.length - 1] = anchors.last;

  const { highs, lows } = clampOHLC(closes, zv.highs, zv.lows);

  return {
    closes,
    highs,
    lows,
    volumes: zv.volumes,
    cleaning: {
      hampelFlagged: hamp.flagged.length,
      winsorised:    win.capped,
      zeroVolFilled: zv.zeroVolFilled,
      haltBars:      zv.haltBars || 0,
      frozenBars:    zv.frozenBars || 0,
      ffillAborted:  zv.ffillAborted || 0,
      totalTouched:  hamp.flagged.length + win.capped + zv.zeroVolFilled,
    },
  };
}

// ─── Pretty one-liner for the AI context ────────────────────────────────────
export function cleaningSummary(cleaning, quality) {
  if (!cleaning) return `Quality: ${quality || "unknown"}`;
  const { hampelFlagged, winsorised, zeroVolFilled, totalTouched } = cleaning;
  if (totalTouched === 0) return `Quality: ${quality} — 0 bars touched, pristine.`;
  const parts = [];
  if (hampelFlagged) parts.push(`${hampelFlagged} outlier${hampelFlagged > 1 ? "s" : ""} filtered`);
  if (winsorised)    parts.push(`${winsorised} return${winsorised > 1 ? "s" : ""} winsorised`);
  if (zeroVolFilled) parts.push(`${zeroVolFilled} zero-vol bar${zeroVolFilled > 1 ? "s" : ""} filled`);
  return `Quality: ${quality} — ${parts.join(", ")}.`;
}
