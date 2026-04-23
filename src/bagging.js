// ─── Bagged logistic regression ensemble ────────────────────────────────────
// A single trained model gives a point estimate of probability. That estimate
// has error bars that the model itself can't tell you — you could have 65%
// confidence that's tight (every model agrees) or 65% that's loose (models
// disagree wildly, true prob could be anywhere 45-85%). Without uncertainty
// you can't size positions honestly.
//
// Bagging (bootstrap aggregation) gives us both a better point estimate AND
// calibrated uncertainty:
//   1. Train N models, each on a bootstrap resample of the training set
//      (n samples drawn WITH replacement from the n training trades).
//   2. At inference, predict with every model. The mean is the ensemble
//      prediction; the std is the uncertainty.
//
// Why LR and not more NNs? Three reasons:
//   - LR training is O(samples * epochs * features), essentially free. We
//     can train 30 bags in under a second. NN bagging would be 30x slower.
//   - LR is convex — each bag converges to its global optimum, so the
//     ensemble variance is genuinely sampling uncertainty, not training
//     randomness.
//   - The NN is already the flexible model in the stack. LR bagging
//     complements it by giving us calibration for the NN's prediction.
//
// Stored as an array of weight vectors under a dedicated localStorage key;
// independent of the single-LR weights used by the composite (model.js).

function bagKeyFor(universe = "equities") {
  return universe === "crypto" ? "trader_lr_bag_v3_crypto" : "trader_lr_bag_v2";
}
const N_BAGS = 30;
const BAG_EPOCHS = 50;
const BAG_LR = 0.05;

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// Fit a single LR to a weighted set of samples via gradient descent on BCE.
// Samples: [{ x: [features], y: 0|1, weight?: number }]
function fitLR(samples, featureDim, opts = {}) {
  const { epochs = BAG_EPOCHS, lr = BAG_LR, l2 = 0.01 } = opts;
  const w = new Array(featureDim).fill(0);
  let b = 0;
  for (let e = 0; e < epochs; e++) {
    // Shuffle order each epoch (SGD)
    const order = [...Array(samples.length).keys()].sort(() => Math.random() - 0.5);
    for (const i of order) {
      const s = samples[i];
      const sw = s.weight ?? 1;
      let dot = b;
      for (let j = 0; j < featureDim; j++) dot += s.x[j] * w[j];
      const yHat = sigmoid(dot);
      const err = (yHat - s.y) * sw;
      for (let j = 0; j < featureDim; j++) w[j] -= lr * (err * s.x[j] + l2 * w[j]);
      b -= lr * err;
    }
  }
  return { w, b };
}

// Draw a bootstrap resample: n samples with replacement from input of size n.
function bootstrap(samples) {
  const n = samples.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = samples[Math.floor(Math.random() * n)];
  return out;
}

// Train the ensemble. Returns the trained bag (N weight vectors + metadata).
// Caller decides whether to persist or evaluate in-memory.
export function trainBag(samples, opts = {}) {
  const { nBags = N_BAGS } = opts;
  if (!samples || samples.length < 10) {
    return { error: `Need ≥10 samples, got ${samples?.length || 0}` };
  }
  const featureDim = samples[0].x.length;
  const bags = [];
  for (let i = 0; i < nBags; i++) {
    const resample = bootstrap(samples);
    bags.push(fitLR(resample, featureDim, opts));
  }
  return {
    bags,
    nBags,
    featureDim,
    trainedOn: samples.length,
    updatedAt: new Date().toISOString(),
  };
}

// Predict with the ensemble. Returns mean probability + std + min/max band.
// The std is the sampling-uncertainty proxy — wider = less confident.
export function predictBag(bag, features) {
  if (!bag?.bags?.length) return null;
  const probs = bag.bags.map(({ w, b }) => {
    let dot = b;
    const n = Math.min(features.length, w.length);
    for (let i = 0; i < n; i++) dot += features[i] * w[i];
    return sigmoid(dot);
  });
  const mean = probs.reduce((a, b) => a + b, 0) / probs.length;
  const variance = probs.reduce((a, b) => a + (b - mean) ** 2, 0) / probs.length;
  const std = Math.sqrt(variance);
  return {
    mean,
    std,
    min: Math.min(...probs),
    max: Math.max(...probs),
    nBags: probs.length,
  };
}

export function saveBag(bag, universe = "equities") {
  if (!bag) return;
  localStorage.setItem(bagKeyFor(universe), JSON.stringify({ ...bag, updatedAt: new Date().toISOString() }));
}

export function loadBag(universe = "equities") {
  try {
    const saved = JSON.parse(localStorage.getItem(bagKeyFor(universe)) || "null");
    if (saved?.bags?.length && saved.featureDim) return saved;
  } catch { /* fall through */ }
  return null;
}

export function resetBag(universe = "equities") {
  localStorage.removeItem(bagKeyFor(universe));
}

// Convenience for the UI — trains from sim trades with the same shape
// trainNNFromSim expects ({ x, y, ageDays }). Labels via direction
// (labelBullish) not verdict-outcome — see backtest.js for why.
export function trainBagFromSim(simTrades, universe = "equities") {
  const samples = simTrades
    .filter(d => d.outcome && d.features)
    .map(d => ({
      x: d.features,
      y: (d.labelBullish === 0 || d.labelBullish === 1)
         ? d.labelBullish
         : ((d.verdict === "BUY" && d.outcome === "WIN") ||
            (d.verdict === "SELL" && d.outcome === "LOSS") ? 1 : 0),
      weight: 1, // time-decay weighting could go here but keep simple for now
    }));
  if (samples.length < 10) return { error: `Need ≥10 samples, got ${samples.length}` };
  const bag = trainBag(samples);
  if (bag.error) return bag;
  saveBag(bag, universe);
  return { ok: true, trainedOn: bag.trainedOn, nBags: bag.nBags };
}
