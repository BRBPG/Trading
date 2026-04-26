import { storageGet, storageSet, storageRemove } from "./storage.js";

function weightsKeyFor(universe) {
  if (universe === "btc")    return "trader_lr_weights_v5_btc";
  if (universe === "crypto") return "trader_lr_weights_v4_crypto";
  return "trader_lr_weights_v3";
}

const DEFAULT_WEIGHTS_EQUITIES = [-0.52, 1.28, 1.61, -0.74, 0.91, 1.05, 0.22, 0, 0, 0, 0, 0, 0, 0, 0, 0.50];
const DEFAULT_BIAS = 0.04;

function defaultFor(universe) {
  if (universe === "btc" || universe === "crypto")
    return { weights: new Array(16).fill(0), bias: 0.0 };
  return { weights: [...DEFAULT_WEIGHTS_EQUITIES], bias: DEFAULT_BIAS };
}

export function loadWeights(universe = "equities") {
  try {
    const saved = JSON.parse(storageGet(weightsKeyFor(universe)) || "null");
    if (saved?.weights?.length === 16) return { weights: saved.weights, bias: saved.bias };
  } catch { /* corrupt */ }
  return defaultFor(universe);
}

export function saveWeights(weights, bias, universe = "equities") {
  storageSet(weightsKeyFor(universe), JSON.stringify({ weights, bias, updatedAt: new Date().toISOString() }));
}

export function resetWeights(universe = "equities") {
  storageRemove(weightsKeyFor(universe));
}

// Gradient descent LR update using server-side persistence.
// Mirrors adaptWeights from src/model.js but reads/writes via file I/O.
export function adaptWeightsServer(reviewedLog, lr = 0.08, epochs = 40, universe = "equities") {
  const { weights, bias } = loadWeights(universe);
  const w = [...weights];
  let b = bias;

  const samples = reviewedLog
    .filter(d => d.reviewed && d.outcome && d.features)
    .map(d => ({
      x: d.features,
      y: (d.verdict === "BUY" && d.outcome === "WIN") ||
         (d.verdict === "SELL" && d.outcome === "WIN") ? 1 : 0,
    }));

  if (samples.length < 2) return { weights: w, bias: b, trained: 0 };

  for (let e = 0; e < epochs; e++) {
    for (const { x, y } of samples) {
      const dot = x.reduce((s, v, i) => s + v * w[i], b);
      const pred = 1 / (1 + Math.exp(-dot));
      const err = pred - y;
      for (let i = 0; i < w.length; i++) w[i] -= lr * err * x[i];
      b -= lr * err;
    }
  }

  saveWeights(w, b, universe);
  return { weights: w, bias: b, trained: samples.length };
}
