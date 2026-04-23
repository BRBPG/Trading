// ─── Gradient Boosted Trees (XGBoost-style, browser-native) ────────────────
// Minimal second-order gradient-boosted-decision-trees for binary
// classification. Not the full XGBoost framework — but mathematically
// identical algorithm:
//   - Logistic loss with second-order Newton updates (g = gradient of loss,
//     h = Hessian, both per-sample)
//   - Regression trees fitted to minimize a Taylor-approximated loss
//   - Hessian-weighted split gain:
//       gain = 0.5 * [(Σg_L)²/(Σh_L+λ) + (Σg_R)²/(Σh_R+λ) - (Σg)²/(Σh+λ)] - γ
//   - L2 regularization on leaf values
//   - Shrinkage (learning rate) between rounds
//   - Max tree depth as primary complexity control
//
// Why GBT over the existing small NN:
//   - Gradient-boosted trees consistently outperform small neural nets on
//     tabular financial data (Chen & Guestrin 2016, plus ~every subsequent
//     Kaggle competition on tabular problems).
//   - Captures feature interactions (e.g. VIX × RSI) that a 16→16→8→1 NN
//     can't represent well.
//   - Robust to feature scaling — no normalisation needed.
//   - Interpretable: feature importance falls out of split statistics.
//
// Storage: trees are plain JS objects {feature, threshold, left, right} or
// {value} for leaves — trivially JSON-serialisable to localStorage.

// Per-universe storage — GBM weights diverge cleanly between equities and
// crypto because the features in effect differ (equity-only slots zeroed
// on crypto) and the training sets are disjoint.
function gbmKeyFor(universe = "equities") {
  return universe === "crypto" ? "trader_gbm_v1_crypto" : "trader_gbm_v1";
}
const N_ROUNDS_DEFAULT = 100;
const MAX_DEPTH_DEFAULT = 4;
const LEARNING_RATE_DEFAULT = 0.1;
const L2_LAMBDA_DEFAULT = 1.0;          // leaf-value regularization
const MIN_CHILD_HESS_DEFAULT = 1.0;      // minimum summed Hessian per leaf
const SUBSAMPLE_DEFAULT = 0.8;           // row subsample per tree
const COLSAMPLE_DEFAULT = 0.8;           // feature subsample per tree

// ─── Tree fitting ──────────────────────────────────────────────────────────
// X: number[n][d], g/h: number[n]
function growTree(X, g, h, depth, opts) {
  const { maxDepth, minChildHess, l2, featureSubset } = opts;
  const n = X.length;

  const sumG = g.reduce((a, b) => a + b, 0);
  const sumH = h.reduce((a, b) => a + b, 0);

  // Leaf value: -sumG / (sumH + lambda). Standard XGBoost closed-form.
  const leafValue = sumH + l2 > 0 ? -sumG / (sumH + l2) : 0;
  const leaf = { value: leafValue };

  if (depth >= maxDepth || n <= 2 || sumH < 2 * minChildHess) return leaf;

  // Try splits across the feature subset (colsample)
  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;
  let bestLeftIdx = null, bestRightIdx = null;

  const currentGainBase = (sumG * sumG) / (sumH + l2);

  for (const f of featureSubset) {
    // Get indices sorted by X[i][f]
    const sortedIdx = [...Array(n).keys()].sort((a, b) => X[a][f] - X[b][f]);

    // Scan candidate thresholds left-to-right, accumulating running sums
    let gLeft = 0, hLeft = 0;
    for (let i = 0; i < n - 1; i++) {
      const idx = sortedIdx[i];
      gLeft += g[idx];
      hLeft += h[idx];
      const gRight = sumG - gLeft;
      const hRight = sumH - hLeft;

      // Skip if either side's Hessian is below minimum
      if (hLeft < minChildHess || hRight < minChildHess) continue;

      // Skip duplicate values — can't split between identical thresholds
      const vHere = X[idx][f];
      const vNext = X[sortedIdx[i + 1]][f];
      if (vHere === vNext) continue;

      const gainLeft  = (gLeft  * gLeft ) / (hLeft  + l2);
      const gainRight = (gRight * gRight) / (hRight + l2);
      const gain = 0.5 * (gainLeft + gainRight - currentGainBase);

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = f;
        bestThreshold = (vHere + vNext) / 2;
        bestLeftIdx = sortedIdx.slice(0, i + 1);
        bestRightIdx = sortedIdx.slice(i + 1);
      }
    }
  }

  if (bestFeature < 0 || bestGain <= 0) return leaf;

  const leftX = bestLeftIdx.map(i => X[i]);
  const leftG = bestLeftIdx.map(i => g[i]);
  const leftH = bestLeftIdx.map(i => h[i]);
  const rightX = bestRightIdx.map(i => X[i]);
  const rightG = bestRightIdx.map(i => g[i]);
  const rightH = bestRightIdx.map(i => h[i]);

  return {
    feature: bestFeature,
    threshold: bestThreshold,
    left: growTree(leftX, leftG, leftH, depth + 1, opts),
    right: growTree(rightX, rightG, rightH, depth + 1, opts),
  };
}

function predictTree(tree, x) {
  let node = tree;
  while (node.feature !== undefined) {
    node = x[node.feature] <= node.threshold ? node.left : node.right;
  }
  return node.value;
}

// ─── Training loop ─────────────────────────────────────────────────────────
// samples: [{ x: number[d], y: 0|1 }]
export function trainGBM(samples, opts = {}) {
  const {
    nRounds = N_ROUNDS_DEFAULT,
    maxDepth = MAX_DEPTH_DEFAULT,
    learningRate = LEARNING_RATE_DEFAULT,
    l2 = L2_LAMBDA_DEFAULT,
    minChildHess = MIN_CHILD_HESS_DEFAULT,
    subsample = SUBSAMPLE_DEFAULT,
    colsample = COLSAMPLE_DEFAULT,
    earlyStopRounds = 10,
    verbose = false,
  } = opts;

  if (!samples || samples.length < 20) {
    return { trained: 0, rounds: 0, reason: `Need ≥20 samples, got ${samples?.length || 0}` };
  }

  const X = samples.map(s => s.x);
  const y = samples.map(s => s.y);
  const n = X.length;
  const d = X[0].length;

  // Class balancing: weight minority class to equal influence
  const pos = y.filter(v => v === 1).length;
  const neg = n - pos;
  const wPos = pos > 0 ? n / (2 * pos) : 1;
  const wNeg = neg > 0 ? n / (2 * neg) : 1;
  const sampleW = y.map(v => v === 1 ? wPos : wNeg);

  // Predictions start at log-odds of prior (class-balanced so ~0)
  const prior = Math.log((pos + 1) / (neg + 1));
  const predictions = new Array(n).fill(prior);

  const trees = [];
  const history = [];
  let bestLoss = Infinity;
  let stagnantRounds = 0;

  for (let round = 0; round < nRounds; round++) {
    // Row subsample for this round
    const subN = Math.max(20, Math.floor(n * subsample));
    const subIdx = [];
    const seen = new Set();
    while (subIdx.length < subN) {
      const i = Math.floor(Math.random() * n);
      if (!seen.has(i)) { seen.add(i); subIdx.push(i); }
    }

    // Feature subsample for this round
    const nFeat = Math.max(1, Math.floor(d * colsample));
    const featureSubset = [...Array(d).keys()]
      .sort(() => Math.random() - 0.5)
      .slice(0, nFeat);

    // Compute gradient and Hessian for logistic loss, weighted by class balance
    const subX = subIdx.map(i => X[i]);
    const subG = subIdx.map(i => {
      const p = sigmoid(predictions[i]);
      return sampleW[i] * (p - y[i]);             // dL/dz
    });
    const subH = subIdx.map(i => {
      const p = sigmoid(predictions[i]);
      return sampleW[i] * p * (1 - p);            // d²L/dz²
    });

    const tree = growTree(subX, subG, subH, 0, {
      maxDepth, minChildHess, l2, featureSubset,
    });
    trees.push(tree);

    // Update predictions on ALL samples (not just subsample)
    for (let i = 0; i < n; i++) {
      predictions[i] += learningRate * predictTree(tree, X[i]);
    }

    // Compute training loss (BCE over full set)
    let loss = 0;
    const eps = 1e-9;
    for (let i = 0; i < n; i++) {
      const p = sigmoid(predictions[i]);
      loss -= sampleW[i] * (y[i] * Math.log(p + eps) + (1 - y[i]) * Math.log(1 - p + eps));
    }
    loss /= n;
    history.push(loss);

    if (verbose && round % 10 === 0) console.log(`round ${round} loss=${loss.toFixed(4)}`);

    // Early stopping
    if (loss < bestLoss - 1e-4) {
      bestLoss = loss;
      stagnantRounds = 0;
    } else {
      stagnantRounds++;
      if (stagnantRounds >= earlyStopRounds) break;
    }
  }

  return {
    trees,
    prior,
    learningRate,
    trainedOn: n,
    rounds: trees.length,
    finalLoss: history[history.length - 1],
    history,
    reason: trees.length === nRounds ? "completed all rounds" : "early stop (loss plateau)",
  };
}

// ─── Inference ─────────────────────────────────────────────────────────────
export function predictGBM(model, x) {
  if (!model || !model.trees || !model.trees.length) return null;
  let logOdds = model.prior;
  for (const tree of model.trees) {
    logOdds += model.learningRate * predictTree(tree, x);
  }
  return sigmoid(logOdds);
}

// ─── Persistence ───────────────────────────────────────────────────────────
export function saveGBM(model, universe = "equities") {
  if (!model) return;
  localStorage.setItem(gbmKeyFor(universe), JSON.stringify({ ...model, updatedAt: new Date().toISOString() }));
}

export function loadGBM(universe = "equities") {
  try {
    const saved = JSON.parse(localStorage.getItem(gbmKeyFor(universe)) || "null");
    if (saved?.trees && saved.trees.length > 0) return saved;
  } catch { /* fall through */ }
  return null;
}

export function resetGBM(universe = "equities") {
  localStorage.removeItem(gbmKeyFor(universe));
}

export function getGBMInfo(universe = "equities") {
  const m = loadGBM(universe);
  if (!m) return { trainedOn: 0, rounds: 0, finalLoss: null };
  return {
    trainedOn: m.trainedOn || 0,
    rounds: m.rounds || 0,
    finalLoss: m.finalLoss,
    updatedAt: m.updatedAt,
  };
}

// ─── Feature importance (gain-based) ───────────────────────────────────────
// Walks all trees and aggregates total gain attributed to each feature.
// Useful for debugging: which features is the model actually using?
export function featureImportance(model, nFeatures) {
  const totals = new Array(nFeatures).fill(0);
  if (!model?.trees) return totals;
  function walk(node) {
    if (node.feature === undefined) return;
    // Gain isn't stored on split nodes in this simplified impl — use count
    // as a proxy. Real XGBoost stores gain at the node; ours doesn't.
    totals[node.feature] += 1;
    walk(node.left);
    walk(node.right);
  }
  for (const t of model.trees) walk(t);
  return totals;
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
