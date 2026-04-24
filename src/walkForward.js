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

export function runWalkForward(simTrades, opts = {}) {
  const { folds = 5, epochs = 80, embargoSec = 3 * 60 * 60, modelKind = "nn" } = opts;
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
  const all = toSamples(simTrades).sort((a, b) => a.timestamp - b.timestamp);
  if (all.length < folds * 8) {
    return {
      error: `Not enough samples for ${folds}-fold walk-forward (need ≥${folds * 8}, have ${all.length}). Run a sim with more days.`,
      samples: all.length,
    };
  }

  const foldSize = Math.floor(all.length / folds);
  const foldResults = [];
  const allPreds = [];
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

    let preds, foldMeta;
    if (modelKind === "gbm") {
      // Slightly more conservative defaults for small-sample crypto folds.
      const model = trainGBM(trainSet, {
        nRounds: 100,
        maxDepth: 3,             // shallower than default 4
        learningRate: 0.08,
        earlyStopRounds: 8,
      });
      if (!model?.trees) continue;
      preds = testSet.map(s => ({ y: s.y, yHat: predictGBM(model, s.x) }));
      foldMeta = { epochs: model.rounds, trainLoss: model.finalLoss, valSize: model.valSize };
    } else {
      const t = trainNNRaw(trainSet, { isolated: true, epochs });
      if (!t.weights) continue;
      preds = scoreWithWeights(t.weights, testSet);
      foldMeta = { epochs: t.epochs, trainLoss: t.loss, valSize: t.valSize };
    }
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
    },
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
