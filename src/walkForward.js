// ─── Walk-forward cross-validation ──────────────────────────────────────────
// The honest way to evaluate a time-series model: at each point in time, the
// model can only learn from the PAST. In-sample metrics (the existing
// "trained on 150 trades, evaluated on the same 150 trades") tell you
// nothing about generalisation — a model that has memorised the training
// set looks identical to a model that has learned something real.
//
// Walk-forward splits the trades chronologically into K folds and, for each
// fold, trains a fresh NN on everything before it and evaluates on the
// held-out fold. The aggregate out-of-sample metrics are the honest read.
//
// Folds:   |  1  |  2  |  3  |  4  |  5  |
// Train:   |  -  | 1   | 1-2 | 1-3 | 1-4 |
// Test:    |  -  | 2   |  3  |  4  |  5  |
//
// Fold 1 has no earlier data so it's skipped for evaluation — we roll the
// training window forward and collect OOS predictions from folds 2..K.

import { trainNN as trainNNRaw, scoreWithWeights } from "./nn";
import { trainGBM, predictGBM } from "./gbm";

// Convert sim trades to the {x, y, ageDays} shape expected by the NN.
// Mirrors trainNNFromSim in model.js — direction-based labels (labelBullish
// with verdict/outcome fallback). Walk-forward MUST use the same labelling
// the production trainers use, otherwise OOS metrics measure a different
// target from what compositeProb represents.
function toSamples(simTrades) {
  return simTrades
    .filter(d => d.outcome && d.features)
    .map(d => ({
      x: d.features,
      y: (d.labelBullish === 0 || d.labelBullish === 1)
         ? d.labelBullish
         : ((d.verdict === "BUY" && d.outcome === "WIN") ||
            (d.verdict === "SELL" && d.outcome === "LOSS") ? 1 : 0),
      ageDays: d.ageDays || 0,
      timestamp: d.timestamp || 0,
    }));
}

// Log-loss = -mean(y·log(ŷ) + (1-y)·log(1-ŷ)). The canonical binary
// classification metric; lower is better. Random guessing ≈ 0.693.
//
// Preds are now expected to already be clipped to [0.01, 0.99] by the
// source (predictNN / predictGBM / logisticScoreFromFeatures / scoreWith-
// Weights) but belt-and-braces — re-clip here. Worst-case per-sample
// loss is bounded by -log(0.01) ≈ 4.605 under this convention, so one
// confidently-wrong point can't dominate the mean. Kaggle's platform
// clips at 1e-15; 0.01 is the tighter bound used for small-sample
// finance where overfit-extreme outputs are the default not the edge.
function logLoss(preds) {
  if (!preds.length) return null;
  const LO = 0.01, HI = 0.99;
  const s = preds.reduce((acc, p) => {
    const y = p.y;
    const yh = Math.max(LO, Math.min(HI, p.yHat));
    return acc - (y * Math.log(yh) + (1 - y) * Math.log(1 - yh));
  }, 0);
  return s / preds.length;
}

// Brier score = mean((y - ŷ)²). Measures calibration — how close predicted
// probabilities are to actual outcomes. 0 is perfect, 0.25 is random.
function brierScore(preds) {
  if (!preds.length) return null;
  return preds.reduce((a, p) => a + (p.y - p.yHat) ** 2, 0) / preds.length;
}

// Accuracy at 0.5 threshold. Blunt but interpretable.
function accuracy(preds) {
  if (!preds.length) return null;
  return preds.filter(p => (p.yHat >= 0.5 ? 1 : 0) === p.y).length / preds.length;
}

// AUC via the Mann-Whitney U statistic approach. Measures ranking quality:
// does the model give higher probability to actual winners than losers?
// 0.5 = random, 1.0 = perfect. The single best metric for an uncalibrated
// binary model because it's threshold-free.
function auc(preds) {
  const pos = preds.filter(p => p.y === 1).map(p => p.yHat);
  const neg = preds.filter(p => p.y === 0).map(p => p.yHat);
  if (!pos.length || !neg.length) return null;
  let wins = 0;
  for (const a of pos) for (const b of neg) {
    if (a > b) wins += 1;
    else if (a === b) wins += 0.5;
  }
  return wins / (pos.length * neg.length);
}

// Mask the specified feature indices in a sample's x vector by setting
// them to zero. Used by runAblationStudy to measure per-feature AUC delta.
function maskFeatures(sample, slotsToZero) {
  if (!slotsToZero || !slotsToZero.length) return sample;
  const x = [...sample.x];
  for (const i of slotsToZero) if (i >= 0 && i < x.length) x[i] = 0;
  return { ...sample, x };
}

export function runWalkForward(simTrades, opts = {}) {
  const { folds = 5, epochs = 80, embargoSec = 3 * 60 * 60, modelKind = "nn", maskSlots = [] } = opts;
  // maskSlots: array of feature-vector indices to zero before both training
  // and prediction. Enables leave-one-out feature ablation — runAblation-
  // Study wraps this by calling runWalkForward multiple times with
  // different masks and diffing the resulting AUCs.
  // modelKind: "nn" (default, equity) or "gbm" (crypto sanity check).
  //   425-param NN on 24-sample folds is massively over-parameterised and
  //   will reliably memorise noise, which — combined with near-zero crypto
  //   feature signal — shows up as confidently-anti-predictive AUC (0.44
  //   instead of 0.50 on the null). GBM with val-loss early stopping
  //   truncates the tree list back to bestRound when val loss stops
  //   improving, so on signal-free folds it emits ~0 trees and outputs
  //   the prior ≈ 0.5 everywhere — honest null AUC 0.50 ± sampling noise
  //   rather than the NN's systematic anti-prediction.
  //
  // embargoSec: gap in seconds between the LAST timestamp in the train set and
  // the FIRST allowed timestamp in the test set. This plugs a subtle leak:
  // our trades have 3-hour holds, so two entries 30 minutes apart have
  // OVERLAPPING forward windows. If one is in train and the other in test, the
  // model effectively sees the label of a future test trade during training.
  // Default embargo = one max-hold window = 3h = 10800s, which purges any
  // overlap. This is Marcos López de Prado's "purged k-fold with embargo".

  // Chronological sort — walk-forward only makes sense in time order.
  // Apply feature-mask BEFORE sort so the mask propagates into every
  // downstream train/test path consistently.
  const all = toSamples(simTrades)
    .map(s => maskFeatures(s, maskSlots))
    .sort((a, b) => a.timestamp - b.timestamp);
  if (all.length < folds * 8) {
    return {
      error: `Not enough samples for ${folds}-fold walk-forward (need ≥${folds * 8}, have ${all.length}). Run a sim with more days.`,
      samples: all.length,
    };
  }

  const foldSize = Math.floor(all.length / folds);
  const foldResults = [];
  const allPreds = [];
  // Pooled meta-predictions across folds — shape:
  //   { y, yHat, metaYHat, metaTruth }
  //   y         = ground-truth direction (1=up)
  //   yHat      = primary model's direction prob
  //   metaYHat  = meta model's "primary-is-trustworthy" prob
  //   metaTruth = did primary actually get this direction right (0/1)
  const allMetaPreds = [];
  let totalPurged = 0;

  // Fold i: train on [0 .. i*foldSize), test on [i*foldSize .. (i+1)*foldSize),
  // PURGING any train sample whose timestamp is within embargoSec of any test
  // sample's timestamp. We skip fold 0 because there's no earlier training.
  for (let i = 1; i < folds; i++) {
    const rawTrain = all.slice(0, i * foldSize);
    const testSet  = all.slice(i * foldSize, (i + 1) * foldSize);
    if (rawTrain.length < 8 || testSet.length === 0) continue;

    // Find the earliest test-set timestamp and purge train samples whose
    // timestamp (representing the ENTRY — forward label extends after) could
    // overlap with the test set's label window.
    const firstTestT = testSet[0].timestamp;
    const trainSet = rawTrain.filter(s => s.timestamp <= firstTestT - embargoSec);
    totalPurged += rawTrain.length - trainSet.length;
    if (trainSet.length < 8) continue;

    let preds, foldMeta, primaryModel;
    if (modelKind === "gbm") {
      // Slightly more conservative defaults for small-sample crypto folds.
      primaryModel = trainGBM(trainSet, {
        nRounds: 100,
        maxDepth: 3,             // shallower than default 4
        learningRate: 0.08,
        earlyStopRounds: 8,
      });
      if (!primaryModel?.trees) continue;
      preds = testSet.map(s => ({ y: s.y, yHat: predictGBM(primaryModel, s.x) }));
      foldMeta = { epochs: primaryModel.rounds, trainLoss: primaryModel.finalLoss, valSize: primaryModel.valSize };
    } else {
      const t = trainNNRaw(trainSet, { isolated: true, epochs });
      if (!t.weights) continue;
      preds = scoreWithWeights(t.weights, testSet);
      primaryModel = t.weights;  // stored for meta step below (NN weight set)
      foldMeta = { epochs: t.epochs, trainLoss: t.loss, valSize: t.valSize };
    }

    // ─── META-LABELING (AFML Ch. 4) ─────────────────────────────────
    // Decouple direction from bet-sizing. The PRIMARY model has already
    // predicted direction. The META model learns, from the SAME features,
    // whether the primary is trustworthy ON THIS SETUP — i.e. "would the
    // primary have been right?"
    //
    //   y_meta[s] = 1 if primary_dir(s) === actual_dir(s), else 0
    //   meta model:  features → P(primary will be right)
    //
    // In deployment: take the trade only when meta > threshold. Unlike the
    // conviction filter (which just trusts |prob−0.5| as a confidence
    // proxy), the meta model can learn regime-specific trustworthiness —
    // e.g. "primary is reliable when funding-z > 0 AND TS-mom > 0,
    // noise elsewhere". If THAT pattern exists, meta-labeling finds it.
    let metaPreds = null;
    const primPredOnTrain = trainSet.map(s =>
      modelKind === "gbm"
        ? predictGBM(primaryModel, s.x)
        : scoreWithWeights(primaryModel, [s])[0].yHat);
    const metaTrainSet = trainSet.map((s, idx) => {
      const pDir = primPredOnTrain[idx] > 0.5 ? 1 : 0;
      return {
        x: s.x,
        y: pDir === s.y ? 1 : 0,   // 1 = primary got this right
        ageDays: s.ageDays,
        timestamp: s.timestamp,
      };
    });
    // Only train meta if train set has both classes (primary wasn't uniformly
    // right or wrong — otherwise there's nothing to learn).
    const metaPos = metaTrainSet.filter(s => s.y === 1).length;
    const metaMix = metaPos > 2 && metaPos < metaTrainSet.length - 2;
    if (metaMix) {
      // Meta always uses GBM regardless of primary — small footprint, val-
      // loss truncation protects against overfit on same fold size.
      const metaModel = trainGBM(metaTrainSet, {
        nRounds: 80,
        maxDepth: 3,
        learningRate: 0.06,
        earlyStopRounds: 6,
      });
      if (metaModel?.trees) {
        metaPreds = testSet.map((s, idx) => ({
          y: s.y,
          yHat: preds[idx].yHat,                          // primary's output (direction)
          metaYHat: predictGBM(metaModel, s.x),           // meta's "trust primary" prob
          // ground truth of "was primary right" for meta-AUC computation
          metaTruth: (preds[idx].yHat > 0.5 ? 1 : 0) === s.y ? 1 : 0,
        }));
      }
    }
    if (metaPreds) allMetaPreds.push(...metaPreds);

    allPreds.push(...preds);

    foldResults.push({
      fold: i,
      trainSize: trainSet.length,
      testSize: testSet.length,
      ...foldMeta,
      testLoss: logLoss(preds),
      testAccuracy: accuracy(preds),
      testAUC: auc(preds),
      testBrier: brierScore(preds),
      metaTrained: !!metaPreds,
    });
  }

  if (foldResults.length === 0) {
    return { error: "No evaluable folds (train or test sets empty). Increase samples.", samples: all.length };
  }

  // ─── Conviction-stratified metrics (meta-labeling, López de Prado AFML §3) ─
  // A forced-trade backtest dilutes measurable edge: if the model has real
  // signal on only the top X% most-confident predictions and is near-random
  // on the rest, overall AUC collapses toward 0.5 even when the policy
  // "trade only the high-conviction subset" has a genuine edge. Stratifying
  // by |yHat − 0.5| tells you exactly where the signal lives and gives the
  // user an actionable selectivity threshold.
  //
  // Percentiles computed on the pooled OOS prediction set (not per fold —
  // fold-level top-10% of 5 preds is too noisy to be meaningful).
  const topByConviction = (preds, q) => {
    const sorted = [...preds].sort((a, b) =>
      Math.abs(b.yHat - 0.5) - Math.abs(a.yHat - 0.5));
    const take = Math.max(1, Math.floor(preds.length * q));
    return sorted.slice(0, take);
  };
  const top50 = topByConviction(allPreds, 0.50);
  const top30 = topByConviction(allPreds, 0.30);
  const top10 = topByConviction(allPreds, 0.10);
  const convictionThreshold = (subset) => {
    // Minimum |yHat − 0.5| in the subset — the "clear this line to take
    // a trade" threshold that reproduces this AUC in production.
    if (!subset.length) return null;
    return Math.min(...subset.map(p => Math.abs(p.yHat - 0.5)));
  };

  // ─── Meta-labeling metrics (AFML Ch. 4) ────────────────────────────
  // Pooled meta-predictions across folds, then evaluate:
  //
  // 1. Meta-AUC: can the meta model predict primary-correctness?
  //    AUC with (y=metaTruth, yHat=metaYHat). If this is substantially
  //    above 0.5, the meta model has learned to identify setup regimes
  //    where the primary is reliable. If ~0.5, the primary's errors are
  //    unstructured noise — meta has nothing to learn.
  //
  // 2. Meta-gated primary metrics: primary's AUC/log-loss ONLY on test
  //    samples where meta > threshold. This is the actionable output —
  //    "take trades only when meta says primary is trustworthy".
  //    Compare gated AUC to ungated; if gated >> ungated with tight CI,
  //    the trading filter IS the edge. Thresholds 0.50, 0.55, 0.60 give
  //    successively more selective filters.
  let metaMetrics = null;
  if (allMetaPreds.length >= 30) {
    const metaForAUC = allMetaPreds.map(p => ({ y: p.metaTruth, yHat: p.metaYHat }));
    const metaAUC = auc(metaForAUC);

    const gated = (thr) => {
      const kept = allMetaPreds.filter(p => p.metaYHat >= thr);
      if (kept.length < 5) return { n: kept.length, auc: null, logLoss: null, accuracy: null, threshold: thr, kept: kept.length / allMetaPreds.length };
      const primOnKept = kept.map(p => ({ y: p.y, yHat: p.yHat }));
      return {
        n: kept.length,
        auc: auc(primOnKept),
        logLoss: logLoss(primOnKept),
        accuracy: accuracy(primOnKept),
        threshold: thr,
        kept: kept.length / allMetaPreds.length,
      };
    };

    metaMetrics = {
      samples: allMetaPreds.length,
      metaAUC,
      metaLogLoss: logLoss(metaForAUC),
      // How often the primary was right at baseline — if this is near 0.5
      // the meta model is working with minimal class imbalance. Very far
      // from 0.5 means primary is systematically biased one way.
      primaryAccuracyBase: allMetaPreds.filter(p => p.metaTruth === 1).length / allMetaPreds.length,
      gated: {
        t50: gated(0.50),
        t55: gated(0.55),
        t60: gated(0.60),
      },
    };
  }

  // Aggregate OOS metrics over all folds' predictions pooled together.
  return {
    samples: all.length,
    purgedByEmbargo: totalPurged,
    embargoSec,
    folds: foldResults,
    overall: {
      oosSamples: allPreds.length,
      oosLogLoss: logLoss(allPreds),
      oosAccuracy: accuracy(allPreds),
      oosAUC: auc(allPreds),
      oosBrier: brierScore(allPreds),
      // Average in-sample (train) loss across folds — if train loss is much
      // lower than test loss, the model is OVERFITTING.
      avgTrainLoss: foldResults.reduce((a, f) => a + (f.trainLoss || 0), 0) / foldResults.length,
      avgTestLoss:  foldResults.reduce((a, f) => a + (f.testLoss  || 0), 0) / foldResults.length,
      // Conviction-stratified — where the signal actually lives.
      // Each entry: { n, auc, logLoss, accuracy, threshold }
      byConviction: {
        all:   { n: allPreds.length, auc: auc(allPreds), logLoss: logLoss(allPreds), accuracy: accuracy(allPreds), threshold: 0 },
        top50: { n: top50.length, auc: auc(top50), logLoss: logLoss(top50), accuracy: accuracy(top50), threshold: convictionThreshold(top50) },
        top30: { n: top30.length, auc: auc(top30), logLoss: logLoss(top30), accuracy: accuracy(top30), threshold: convictionThreshold(top30) },
        top10: { n: top10.length, auc: auc(top10), logLoss: logLoss(top10), accuracy: accuracy(top10), threshold: convictionThreshold(top10) },
      },
      meta: metaMetrics,
    },
  };
}

// ─── Ablation study (Phase 4 Commit 7) ───────────────────────────────────
// For each target feature, run walk-forward with that slot zeroed and
// measure AUC delta vs the baseline (all features active). Leave-one-out
// feature importance — measures each feature's MARGINAL contribution.
//
// Interpretation:
//   delta > 0.005 with tight CI → feature carries real signal
//   delta ≈ 0                   → feature is redundant or dead
//   delta < -0.005              → feature is actively hurting (rare)
//
// Cost: N+1 × single walk-forward runtime. At 5 folds × ~500ms each, a
// 10-feature ablation is ~30s. Cheaper than a full multi-sim.
//
// targets: array of { slot: number, name: string } — the indices to
// ablate and human-readable labels for the report.
export function runAblationStudy(simTrades, baseOpts = {}, targets = []) {
  if (!simTrades?.length) return { error: "no trades provided" };
  // Baseline run — all features active.
  const baseline = runWalkForward(simTrades, baseOpts);
  if (baseline.error) return { error: `baseline: ${baseline.error}` };
  const baseAUC = baseline.overall?.oosAUC;
  if (baseAUC == null) return { error: "baseline produced no AUC" };

  const results = [];
  for (const t of targets) {
    const masked = runWalkForward(simTrades, { ...baseOpts, maskSlots: [t.slot] });
    if (masked.error) {
      results.push({ ...t, auc: null, delta: null, error: masked.error });
      continue;
    }
    const maskedAUC = masked.overall?.oosAUC;
    if (maskedAUC == null) {
      results.push({ ...t, auc: null, delta: null, error: "no auc" });
      continue;
    }
    // delta = baseline AUC − masked AUC. Positive delta means removing
    // the feature HURT the model, so the feature was contributing. The
    // bigger the positive delta, the more important the feature.
    results.push({
      ...t,
      aucWithout: maskedAUC,
      aucBaseline: baseAUC,
      delta: baseAUC - maskedAUC,
    });
  }
  // Sort by delta descending — most important features at the top.
  results.sort((a, b) => (b.delta ?? -Infinity) - (a.delta ?? -Infinity));

  return {
    baselineAUC: baseAUC,
    baselineLogLoss: baseline.overall?.oosLogLoss,
    baselineSamples: baseline.overall?.oosSamples,
    results,
  };
}

// Short verdict derivable from the overall OOS metrics, for the UI headline.
export function interpretWF(overall) {
  if (!overall) return { label: "—", color: "#888" };
  const auc = overall.oosAUC;
  const gap = overall.avgTestLoss - overall.avgTrainLoss;

  if (auc == null) return { label: "Inconclusive", color: "#888" };

  // Heavy overfitting check first — high AUC with a big train/test gap is
  // a mirage.
  if (gap > 0.15) {
    return { label: "Severe overfitting — train ≪ test loss", color: "#E74C3C" };
  }
  if (auc >= 0.60) return { label: "Real out-of-sample edge detected", color: "#2ECC71" };
  if (auc >= 0.55) return { label: "Marginal edge — more data needed", color: "#7FD8A6" };
  if (auc >= 0.50) return { label: "No edge — coin flip", color: "#C9A84C" };
  return { label: "Inverted — model picks the WRONG side", color: "#E74C3C" };
}
