// ─── Regime-switching ensemble ──────────────────────────────────────────────
// The features that predict returns in calm markets are different from the
// ones that predict returns in panics. Training a single model on both
// regimes averages those signals and dilutes both. Solution: train TWO
// models — one per regime — and gate between them at inference using a
// regime indicator (here: VIX z-score sign).
//
// Approach (simplest viable, per published literature):
//   1. Tag each training sample with its regime (low_vol vs high_vol)
//      based on macro.vixZ at that timestamp.
//   2. Train a GBM on each subset independently.
//   3. At inference, compute current regime, route to corresponding model.
//   4. If sample count in either regime is too low, fall back to the
//      single-regime production GBM.
//
// Why GBM not NN: GBMs handle small-sample regime subsets better than NNs
// (which need more data per model) and Phase 2d already established GBM
// as the strongest single model.

import { trainGBM, predictGBM } from "./gbm";

function regimeKeyFor(universe = "equities") {
  // btc-only won't typically train regime models (too few trades per regime
  // on a single symbol) but keep the key namespace isolated anyway so a
  // future switch doesn't collide with multi-crypto state.
  if (universe === "btc")    return "trader_regime_v3_btc";
  if (universe === "crypto") return "trader_regime_v2_crypto";
  return "trader_regime_v1";
}
const MIN_SAMPLES_PER_REGIME = 30;

// Regime classifier: returns "high_vol" | "low_vol" | "neutral"
// Threshold is VIX z-score relative to a 60-bar rolling window. Above 0 =
// vol elevated vs recent past. Above 1.0 = clearly in a stress regime.
export function classifyRegime(macro) {
  if (!macro || macro.vixZ == null) return "neutral";
  if (macro.vixZ >= 0.5) return "high_vol";
  if (macro.vixZ <= -0.5) return "low_vol";
  return "neutral";
}

// Train two regime-conditional GBMs from sim trades. Each trade carries
// its own macro.vixZ at entry-time (set by the backtester via the modelCtx
// pipeline) — but if the sim trades don't include that field directly, we
// approximate using the sample's stored features (vix_z is feature index 7).
export function trainRegimeModels(simTrades, universe = "equities") {
  if (!simTrades || simTrades.length < MIN_SAMPLES_PER_REGIME * 2) {
    return {
      error: `Need ≥${MIN_SAMPLES_PER_REGIME * 2} sim trades total, got ${simTrades?.length || 0}`,
    };
  }

  // Split by regime using the vix_z feature (index 7 in our 16-dim vector).
  // vix_z is already z-score normalised (clipped to [-1,1] in extractFeatures
  // via `clip1(macro.vixZ / 2)`), so a feature value of >=0.25 corresponds
  // roughly to vixZ >= 0.5 (high vol) and similarly for low.
  const high = [], low = [];
  for (const t of simTrades) {
    if (!t.outcome || !t.features) continue;
    const vixZFeat = t.features[7] ?? 0;
    if (vixZFeat >= 0.25) high.push(t);
    else if (vixZFeat <= -0.25) low.push(t);
    // neutral samples (-0.25 < vixZFeat < 0.25) are excluded — the production
    // GBM (single-regime) will handle those at inference time.
  }

  const result = { high: null, low: null, counts: { high: high.length, low: low.length } };

  // Direction-based label — matches what compositeProb predicts.
  // See backtest.js labelBullish comment for why verdict/outcome alone
  // would produce conflicting labels under random-direction bootstrap.
  const yOf = t => (t.labelBullish === 0 || t.labelBullish === 1)
    ? t.labelBullish
    : ((t.verdict === "BUY" && t.outcome === "WIN") ||
       (t.verdict === "SELL" && t.outcome === "LOSS") ? 1 : 0);

  if (high.length >= MIN_SAMPLES_PER_REGIME) {
    const samples = high.map(t => ({ x: t.features, y: yOf(t) }));
    result.high = trainGBM(samples, { nRounds: 80, maxDepth: 4 });
  }
  if (low.length >= MIN_SAMPLES_PER_REGIME) {
    const samples = low.map(t => ({ x: t.features, y: yOf(t) }));
    result.low = trainGBM(samples, { nRounds: 80, maxDepth: 4 });
  }

  if (!result.high && !result.low) {
    return { error: `Neither regime had ≥${MIN_SAMPLES_PER_REGIME} samples (high=${high.length}, low=${low.length})`, counts: result.counts };
  }

  // Persist together so loadRegimeModels gets both at once
  const payload = {
    high: result.high?.trees ? result.high : null,
    low:  result.low?.trees  ? result.low  : null,
    counts: result.counts,
    updatedAt: new Date().toISOString(),
  };
  localStorage.setItem(regimeKeyFor(universe), JSON.stringify(payload));

  return {
    ok: true,
    counts: result.counts,
    highTrained: !!result.high?.trees,
    lowTrained:  !!result.low?.trees,
    highRounds: result.high?.rounds,
    lowRounds:  result.low?.rounds,
    highLoss:   result.high?.finalLoss,
    lowLoss:    result.low?.finalLoss,
  };
}

export function loadRegimeModels(universe = "equities") {
  try {
    const saved = JSON.parse(localStorage.getItem(regimeKeyFor(universe)) || "null");
    if (saved && (saved.high || saved.low)) return saved;
  } catch { /* fall through */ }
  return null;
}

export function resetRegimeModels(universe = "equities") {
  localStorage.removeItem(regimeKeyFor(universe));
}

// Predict using the regime-appropriate model. Returns { prob, regime, used }
// where `used` indicates which model was consulted (or null if neither
// regime model exists / regime is neutral).
export function predictRegime(features, macro, universe = "equities") {
  const models = loadRegimeModels(universe);
  if (!models) return null;

  const regime = classifyRegime(macro);
  if (regime === "high_vol" && models.high) {
    return { prob: predictGBM(models.high, features), regime, used: "high_vol" };
  }
  if (regime === "low_vol" && models.low) {
    return { prob: predictGBM(models.low, features), regime, used: "low_vol" };
  }
  // Neutral or missing-model regime → no regime prediction; caller falls
  // back to the production single-regime GBM.
  return null;
}

export function getRegimeInfo(universe = "equities") {
  const models = loadRegimeModels(universe);
  if (!models) return null;
  return {
    highTrained: !!models.high,
    lowTrained:  !!models.low,
    highSamples: models.counts?.high ?? 0,
    lowSamples:  models.counts?.low ?? 0,
    highRounds:  models.high?.rounds ?? 0,
    lowRounds:   models.low?.rounds ?? 0,
    updatedAt:   models.updatedAt,
  };
}
