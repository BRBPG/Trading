// ─── Position sizing ────────────────────────────────────────────────────────
// Three sizing methods, layered so the final size = min(fixed%, vol-target,
// model-confidence). Whichever constraint is tightest wins.
//
// 1. FIXED-FRACTIONAL — never risk more than X% of account per trade. The
//    default retail heuristic. Determines MAX size; other rules can only
//    shrink it.
//
// 2. VOLATILITY-TARGETED (Moreira & Muir 2017, Journal of Finance) —
//    size_multiplier = target_vol / realised_vol. When vol doubles, size
//    halves. Raises Sharpe by 0.3-0.5 in documented backtests vs constant
//    sizing. Moreira-Muir implementation uses short-window realised vol
//    in the denominator so the system deleverages into vol spikes.
//
// 3. MODEL-CONFIDENCE (fractional Kelly) — scale by (2·p - 1) where p is
//    model's bullish probability, using a fraction of full Kelly (0.25x
//    per López de Prado's recommendation, because full Kelly is too
//    aggressive under estimation uncertainty).
//
// The output is a % of account to allocate. Downstream the trader UI
// converts this to notional $ given an account size the user enters.

const TARGET_ANN_VOL = 0.15;        // 15% annualised target — classic level
const KELLY_FRACTION = 0.25;        // fractional Kelly
const MAX_RISK_PER_TRADE = 0.02;    // 2% fixed-fractional ceiling

// Realised vol from a close series, annualised assuming 252 trading days.
// Shorter window (20 bars) is more responsive; the Moreira-Muir paper uses
// daily data but the math is the same on 5-min bars provided we annualise
// correctly.
export function annualisedVol(closes, barsPerYear = 252 * 78) {
  if (!closes || closes.length < 21) return null;
  const rets = [];
  for (let i = Math.max(1, closes.length - 20); i < closes.length; i++) {
    const prev = closes[i - 1], cur = closes[i];
    if (prev > 0) rets.push((cur - prev) / prev);
  }
  if (rets.length < 5) return null;
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / rets.length;
  return Math.sqrt(variance) * Math.sqrt(barsPerYear);
}

// Moreira-Muir volatility-targeted size. Returns a multiplier 0..maxMult.
// When realised_vol equals target, mult = 1. When double, mult = 0.5. Etc.
export function volTargetMultiplier(realisedVol, targetVol = TARGET_ANN_VOL, maxMult = 2.0) {
  if (!realisedVol || realisedVol <= 0) return 1;
  return Math.min(maxMult, targetVol / realisedVol);
}

// Fractional-Kelly size from a model probability.
// For a setup with fixed R:R = reward/risk:
//   full Kelly = (p * (R+1) - 1) / R
// where p is probability of winning. Returns fraction of capital 0..1.
export function kellyFraction(prob, rewardToRisk = 3, fraction = KELLY_FRACTION) {
  if (prob == null || prob <= 0.5 || rewardToRisk <= 0) return 0;
  const R = rewardToRisk;
  const full = (prob * (R + 1) - 1) / R;
  if (full <= 0) return 0;
  return Math.max(0, Math.min(1, full * fraction));
}

// Combined recommendation. Returns the account-% to risk on this trade,
// plus a breakdown so the UI can explain why it picked that number.
//
//   bullishProb: from scoreSetup(...).compositeProb (0..1 after /100)
//   atr: average true range from the quote
//   price: current price
//   closes: for realised-vol computation
//   rr: reward-to-risk used in the planned trade (default 3 matching
//       the decision tree's 3-ATR target / 1.5-ATR stop → 2)
//
// Output:
//   { sizePct, stopPct, explanation: { kelly, volMult, fixed, binding } }
//
// stopPct is the % distance from entry to stop — used to convert account
// risk to notional position size: notional = (sizePct * accountSize) / stopPct.
export function recommendSize(q, model, opts = {}) {
  const {
    rr = 2,
    target = TARGET_ANN_VOL,
    maxRisk = MAX_RISK_PER_TRADE,
  } = opts;

  if (!q || !q.price || !q.atr) return { sizePct: 0, stopPct: 0, explanation: null };

  // Pick stop distance matching the intended horizon. Default "swing" since
  // that's the primary horizon per the current project direction; callers
  // can pass horizon: "intraday" to use the tighter 1.5× intraday ATR.
  const { horizon = "swing" } = opts;
  const stopDist = horizon === "intraday"
    ? 1.5 * q.atr
    : 2 * q.atr * Math.sqrt(78);           // daily-equivalent, matches SWING SUGGESTED LEVELS
  const stopPct = (stopDist / q.price) * 100;

  // Confidence direction — vol-targeting applies to the *magnitude* of
  // edge, so take |p - 0.5| as the basis.
  const p = model?.compositeProb != null ? parseFloat(model.compositeProb) / 100 : 0.5;
  const edge = Math.abs(p - 0.5) * 2;   // 0..1

  // Three size signals:
  const fixed  = maxRisk;                          // hard ceiling
  const kelly  = kellyFraction(p > 0.5 ? p : 1 - p, rr);
  const vol    = annualisedVol(q.closes);
  const volMult = volTargetMultiplier(vol, target);

  // Scale Kelly by vol-target multiplier — this combines edge-sizing with
  // regime-aware deleveraging. In high vol, even a high-conviction trade
  // gets a smaller position.
  const combined = kelly * volMult;

  // Apply the fixed-fractional ceiling. The binding constraint (smallest)
  // determines the final size.
  const sizePct = Math.min(fixed, combined);
  const binding = sizePct === fixed
    ? "fixed 2% cap"
    : combined < kelly ? "vol-target shrink"
    : edge < 0.1 ? "low model confidence"
    : "fractional Kelly";

  return {
    sizePct,
    stopPct,
    notionalPct: stopPct > 0 ? (sizePct * 100) / stopPct : 0,  // $ exposure = (riskPct/stopPct) of account
    explanation: {
      modelProb: p,
      edge,
      annualisedVol: vol,
      volMultiplier: volMult,
      kelly,
      fixedCap: fixed,
      binding,
    },
  };
}
