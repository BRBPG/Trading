// ─── Multi-layer perceptron with backprop — trading edition ────────────────
// Architecture: 7 → 8 → 4 → 1 (105 parameters). Feeds from the same feature
// vector as the LR so the two can be ensembled directly.
//
// Trading-specific choices, explained:
//   • L2 weight-decay (λ=0.003) — prevents over-fit on small (~100-sample)
//     training sets, which is the realistic regime for a retail trade log.
//   • Time-decay sample weighting — trades older than 30 days are down-
//     weighted by 0.5^(age/30) so yesterday's regime counts more than last
//     quarter's.
//   • Class balancing — WIN/LOSS ratio is normalised to 50/50 via inverse-
//     frequency weighting (typical trade logs skew win-heavy after pruning).
//   • Early stopping on loss plateau — if loss doesn't improve by >0.1% for
//     5 epochs in a row, stop. Stops wasting compute and prevents over-fit.
//   • Feature normalisation — z-score using train-set means/stds, stored
//     with the weights so inference normalises with the same stats.
//
// Not included (intentional scope limits for now):
//   • Adam / RMSProp optimiser — vanilla SGD is adequate for 105 params and
//     makes the learning-rate tuning story simple.
//   • Dropout — sample counts too small for it to add signal.
//   • Batch norm — overkill at this scale.

// v3 architecture: 16 → 16 → 8 → 1 (433 params). Two more inputs for the
// PEAD (earnings drift) features on top of v2's 14 inputs. Storage key
// bumped so any v2 weights are discarded cleanly rather than feeding the
// wrong shape into inference.
// Per-universe storage. Equities keeps the bare v3 key (back-compat with
// weights users have already trained); crypto gets a suffix so it trains
// independently. Architecture (16→16→8→1) is shared across universes —
// only the trained weights diverge.
function nnKeyFor(universe = "equities") {
  return universe === "crypto"
    ? "trader_nn_weights_v3_crypto"
    : "trader_nn_weights_v3";
}
const NN_INPUT_DIM = 16;
const NN_HIDDEN1 = 16;
const NN_HIDDEN2 = 8;

// ─── Linear algebra helpers (tiny, no dep on libraries) ───────────────────
function randn() {
  // Box-Muller for Gaussian init
  const u = Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u || 1e-9)) * Math.cos(2 * Math.PI * v);
}

function xavierMatrix(rows, cols) {
  // Xavier/Glorot: var = 1/cols. Good starting point for tanh/sigmoid-ish nets.
  const scale = Math.sqrt(1 / cols);
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => randn() * scale));
}

function zeros(n) { return Array(n).fill(0); }
function matVec(M, v) { return M.map(row => row.reduce((s, w, i) => s + w * v[i], 0)); }
function relu(x) { return x > 0 ? x : 0; }
function reluD(x) { return x > 0 ? 1 : 0; }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// ─── Weight init / persistence ─────────────────────────────────────────────
function initWeights() {
  return {
    W1: xavierMatrix(NN_HIDDEN1, NN_INPUT_DIM), b1: zeros(NN_HIDDEN1),
    W2: xavierMatrix(NN_HIDDEN2, NN_HIDDEN1),   b2: zeros(NN_HIDDEN2),
    W3: xavierMatrix(1, NN_HIDDEN2),            b3: zeros(1),
    means: null,  // set after first training
    stds: null,
    trainedOn: 0,
    epochs: 0,
    finalLoss: null,
    updatedAt: null,
  };
}

export function loadNN(universe = "equities") {
  try {
    const saved = JSON.parse(localStorage.getItem(nnKeyFor(universe)) || "null");
    if (saved?.W1 && saved.W1.length === NN_HIDDEN1 && saved.W1[0].length === NN_INPUT_DIM) return saved;
  } catch { /* fall through */ }
  return initWeights();
}

function saveNN(W, universe = "equities") {
  localStorage.setItem(nnKeyFor(universe), JSON.stringify({ ...W, updatedAt: new Date().toISOString() }));
}

export function resetNN(universe = "equities") {
  localStorage.removeItem(nnKeyFor(universe));
}

// ─── Feature normalisation ────────────────────────────────────────────────
function computeStats(X) {
  const n = X.length, d = X[0].length;
  const means = zeros(d), stds = zeros(d);
  for (const x of X) for (let j = 0; j < d; j++) means[j] += x[j] / n;
  for (const x of X) for (let j = 0; j < d; j++) stds[j] += (x[j] - means[j]) ** 2 / n;
  for (let j = 0; j < d; j++) stds[j] = Math.sqrt(stds[j]) || 1;
  return { means, stds };
}

function normalise(x, means, stds) {
  if (!means || !stds) return x;
  return x.map((v, i) => (v - means[i]) / stds[i]);
}

// ─── Forward pass ─────────────────────────────────────────────────────────
function forward(W, x) {
  const xn = normalise(x, W.means, W.stds);
  const z1 = matVec(W.W1, xn).map((v, i) => v + W.b1[i]);
  const a1 = z1.map(relu);
  const z2 = matVec(W.W2, a1).map((v, i) => v + W.b2[i]);
  const a2 = z2.map(relu);
  const z3 = matVec(W.W3, a2).map((v, i) => v + W.b3[i]);
  const a3 = z3.map(sigmoid);
  return { xn, z1, a1, z2, a2, z3, a3 };
}

export function predictNN(features, universe = "equities") {
  const W = loadNN(universe);
  if (!W.trainedOn) return null; // Untrained — callers should fall back to LR
  return forward(W, features).a3[0];
}

// Score a batch of samples with a given weight set (does NOT touch the
// persisted NN). Used by walk-forward validation where we train a fresh NN
// per fold and then evaluate it on held-out data without clobbering the
// production NN.
export function scoreWithWeights(W, samples) {
  return samples.map(s => ({
    y: s.y,
    yHat: forward(W, s.x).a3[0],
  }));
}

export function getNNInfo(universe = "equities") {
  const W = loadNN(universe);
  return {
    trainedOn: W.trainedOn || 0,
    epochs: W.epochs || 0,
    finalLoss: W.finalLoss,
    updatedAt: W.updatedAt,
  };
}

// ─── Training — backprop with L2 + time-decay + class balancing ────────────
// samples: [{ x: [7 features], y: 0|1, ageDays?: number }]
export function trainNN(samples, opts = {}) {
  const {
    epochs = 120,
    lr = 0.05,
    l2 = 0.003,
    halfLifeDays = 30,
    patience = 5,
    minDeltaPct = 0.001,
    // When true, train into a fresh weight set and RETURN it without
    // persisting to localStorage. Used by walk-forward so each fold trains
    // independently and doesn't clobber the production NN.
    isolated = false,
    universe = "equities",
  } = opts;

  if (samples.length < 8) {
    return { trained: 0, epochs: 0, loss: null, history: [], reason: "Need at least 8 samples to train." };
  }

  // Class balancing — reweight minority class to equal influence
  const pos = samples.filter(s => s.y === 1).length;
  const neg = samples.length - pos;
  const wPos = pos > 0 ? samples.length / (2 * pos) : 1;
  const wNeg = neg > 0 ? samples.length / (2 * neg) : 1;

  // Time-decay weight per sample
  const timeWeight = (ageDays = 0) => Math.pow(0.5, ageDays / halfLifeDays);

  // Initialise / carry forward. When isolated, always start from fresh random
  // weights — mixing persisted weights into a per-fold training set would
  // leak information across folds and defeat the point of walk-forward.
  let W = isolated ? initWeights() : loadNN(universe);
  if (!W.trainedOn) W = initWeights();

  // Fit normalisation on current batch
  const X = samples.map(s => s.x);
  const { means, stds } = computeStats(X);
  W.means = means;
  W.stds = stds;

  const history = [];
  let bestLoss = Infinity, stagnantEpochs = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const order = [...samples.keys()].sort(() => Math.random() - 0.5);
    let lossSum = 0, weightSum = 0;

    for (const i of order) {
      const s = samples[i];
      const classW = s.y === 1 ? wPos : wNeg;
      const tW = timeWeight(s.ageDays || 0);
      const sampleW = classW * tW;

      const fwd = forward(W, s.x);
      const yHat = fwd.a3[0];
      const eps = 1e-9;
      const loss = -(s.y * Math.log(yHat + eps) + (1 - s.y) * Math.log(1 - yHat + eps));
      lossSum += loss * sampleW;
      weightSum += sampleW;

      // ─── Backward pass ─────────────────────────────────────────────
      // Output (sigmoid + BCE): dL/dz3 = (ŷ - y) · sampleW
      const dz3 = [(yHat - s.y) * sampleW];
      // ∂L/∂W3, ∂L/∂b3
      const dW3 = dz3.map(dz => fwd.a2.map(a => dz * a));
      const db3 = [...dz3];
      // Back through ReLU layer 2
      const da2 = fwd.a2.map((_, j) => dz3.reduce((acc, dz, k) => acc + dz * W.W3[k][j], 0));
      const dz2 = da2.map((d, j) => d * reluD(fwd.z2[j]));
      const dW2 = dz2.map(dz => fwd.a1.map(a => dz * a));
      const db2 = [...dz2];
      // Back through ReLU layer 1
      const da1 = fwd.a1.map((_, j) => dz2.reduce((acc, dz, k) => acc + dz * W.W2[k][j], 0));
      const dz1 = da1.map((d, j) => d * reluD(fwd.z1[j]));
      const dW1 = dz1.map(dz => fwd.xn.map(a => dz * a));
      const db1 = [...dz1];

      // ─── SGD update with L2 decay ──────────────────────────────────
      for (let r = 0; r < W.W1.length; r++)
        for (let c = 0; c < W.W1[0].length; c++)
          W.W1[r][c] -= lr * (dW1[r][c] + l2 * W.W1[r][c]);
      for (let j = 0; j < W.b1.length; j++) W.b1[j] -= lr * db1[j];

      for (let r = 0; r < W.W2.length; r++)
        for (let c = 0; c < W.W2[0].length; c++)
          W.W2[r][c] -= lr * (dW2[r][c] + l2 * W.W2[r][c]);
      for (let j = 0; j < W.b2.length; j++) W.b2[j] -= lr * db2[j];

      for (let r = 0; r < W.W3.length; r++)
        for (let c = 0; c < W.W3[0].length; c++)
          W.W3[r][c] -= lr * (dW3[r][c] + l2 * W.W3[r][c]);
      for (let j = 0; j < W.b3.length; j++) W.b3[j] -= lr * db3[j];
    }

    const avgLoss = lossSum / (weightSum || 1);
    history.push(avgLoss);

    // Early stopping
    if (bestLoss - avgLoss > minDeltaPct * bestLoss) {
      bestLoss = avgLoss;
      stagnantEpochs = 0;
    } else {
      stagnantEpochs++;
      if (stagnantEpochs >= patience) {
        W.epochs = (W.epochs || 0) + epoch + 1;
        W.trainedOn = samples.length;
        W.finalLoss = avgLoss;
        if (!isolated) saveNN(W, universe);
        return { trained: samples.length, epochs: epoch + 1, loss: avgLoss, history, reason: "Early stop — loss plateau", weights: isolated ? W : undefined };
      }
    }
  }

  W.epochs = (W.epochs || 0) + epochs;
  W.trainedOn = samples.length;
  W.finalLoss = history[history.length - 1];
  if (!isolated) saveNN(W, universe);
  return { trained: samples.length, epochs, loss: W.finalLoss, history, reason: "Completed all epochs", weights: isolated ? W : undefined };
}
