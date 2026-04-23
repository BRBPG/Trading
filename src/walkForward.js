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

// Convert sim trades to the {x, y, ageDays} shape expected by the NN.
// Mirrors trainNNFromSim in model.js but does NOT touch persisted weights.
function toSamples(simTrades) {
  return simTrades
    .filter(d => d.outcome && d.features)
    .map(d => ({
      x: d.features,
      y: d.outcome === "WIN" ? 1 : 0,
      ageDays: d.ageDays || 0,
      timestamp: d.timestamp || 0,
    }));
}

// Log-loss = -mean(y·log(ŷ) + (1-y)·log(1-ŷ)). The canonical binary
// classification metric; lower is better. Random guessing ≈ 0.693.
function logLoss(preds) {
  if (!preds.length) return null;
  const eps = 1e-9;
  const s = preds.reduce((acc, p) =>
    acc - (p.y * Math.log(p.yHat + eps) + (1 - p.y) * Math.log(1 - p.yHat + eps)), 0);
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
  const { folds = 5, epochs = 80 } = opts;

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

  // Fold i: train on [0 .. i*foldSize), test on [i*foldSize .. (i+1)*foldSize).
  // We skip fold 0 because there's no earlier training set.
  for (let i = 1; i < folds; i++) {
    const trainSet = all.slice(0, i * foldSize);
    const testSet  = all.slice(i * foldSize, (i + 1) * foldSize);
    if (trainSet.length < 8 || testSet.length === 0) continue;

    const t = trainNNRaw(trainSet, { isolated: true, epochs });
    if (!t.weights) continue;

    const preds = scoreWithWeights(t.weights, testSet);
    allPreds.push(...preds);

    foldResults.push({
      fold: i,
      trainSize: trainSet.length,
      testSize: testSet.length,
      epochs: t.epochs,
      trainLoss: t.loss,
      testLoss: logLoss(preds),
      testAccuracy: accuracy(preds),
      testAUC: auc(preds),
      testBrier: brierScore(preds),
    });
  }

  if (foldResults.length === 0) {
    return { error: "No evaluable folds (train or test sets empty). Increase samples.", samples: all.length };
  }

  // Aggregate OOS metrics over all folds' predictions pooled together.
  return {
    samples: all.length,
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
