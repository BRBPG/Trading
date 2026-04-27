// ─── Cross-device state persistence ─────────────────────────────────────────
// Everything the trader system "learns" — the decision log and the trained
// model state — lives in the browser's localStorage. That's per-device,
// per-browser. Open the app on a different laptop and you start from scratch.
//
// To make state portable without standing up a backend, we expose two manual
// operations:
//   exportState() → produces a JSON blob the user downloads as a file
//   importState(json) → restores all keys from a previously-exported file
//
// API keys are deliberately NOT included in the export — those would be a
// security hole if the file were ever shared or stored in the wrong place.
//
// Universe is hard-pinned to "btc" (App.jsx). Storage keys must match the
// per-universe keys in nn.js / gbm.js / model.js / bagging.js / regime.js,
// otherwise EXPORT/IMPORT silently round-trips empty slots.

const KEYS = {
  log:        "trader_decision_log",
  lrWeights:  "trader_lr_weights_v5_btc",
  nnWeights:  "trader_nn_weights_v5_btc",
  lrBag:      "trader_lr_bag_v4_btc",
  gbm:        "trader_gbm_v3_btc",
  activeMask: "trader_active_mask_v1_btc",
  regime:     "trader_regime_v3_btc",
  earnings:   "trader_earnings_cache_v1",
};

// v4: BTC-only payload covering NN, GBM, LR, LR-bag, regime, and active
// mask. v3 and earlier exports were equity-keyed and are no longer
// readable — they round-tripped empty slots anyway, so nothing of value
// is lost by rejecting them.
const SCHEMA_VERSION = 4;
const COMPATIBLE_SCHEMAS = [4];

export function exportState() {
  return {
    schemaVersion: SCHEMA_VERSION,
    exportedAt: new Date().toISOString(),
    appName: "the-trader",
    universe: "btc",
    log:        safeParse(localStorage.getItem(KEYS.log)) ?? [],
    lrWeights:  safeParse(localStorage.getItem(KEYS.lrWeights)) ?? null,
    nnWeights:  safeParse(localStorage.getItem(KEYS.nnWeights)) ?? null,
    lrBag:      safeParse(localStorage.getItem(KEYS.lrBag)) ?? null,
    gbm:        safeParse(localStorage.getItem(KEYS.gbm)) ?? null,
    activeMask: safeParse(localStorage.getItem(KEYS.activeMask)) ?? null,
    regime:     safeParse(localStorage.getItem(KEYS.regime)) ?? null,
  };
}

export function downloadExport() {
  const payload = exportState();
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  a.href = url;
  a.download = `the-trader-state-${stamp}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  return payload;
}

// Restore all keys from a payload object.
// mode: "replace" wipes the existing log; "merge" appends new entries by id.
export function importState(payload, { mode = "replace" } = {}) {
  if (!payload || typeof payload !== "object") {
    throw new Error("Invalid import: not an object.");
  }
  if (!COMPATIBLE_SCHEMAS.includes(payload.schemaVersion)) {
    throw new Error(`Unsupported schema version ${payload.schemaVersion}; expected ${SCHEMA_VERSION}. Older exports were equity-keyed and contain no BTC model state.`);
  }

  const restored = {
    log: 0,
    lrWeights: false,
    nnWeights: false,
    lrBag: false,
    gbm: false,
    activeMask: false,
    regime: false,
  };

  if (Array.isArray(payload.log)) {
    if (mode === "merge") {
      const existing = safeParse(localStorage.getItem(KEYS.log)) ?? [];
      const seen = new Set(existing.map(d => d.id));
      const additions = payload.log.filter(d => !seen.has(d.id));
      const merged = [...additions, ...existing]
        .sort((a, b) => (b.id || 0) - (a.id || 0))
        .slice(0, 200);
      localStorage.setItem(KEYS.log, JSON.stringify(merged));
      restored.log = additions.length;
    } else {
      localStorage.setItem(KEYS.log, JSON.stringify(payload.log.slice(0, 200)));
      restored.log = payload.log.length;
    }
  }

  // LR weights — 16-dim BTC feature vector.
  if (payload.lrWeights && Array.isArray(payload.lrWeights.weights) && payload.lrWeights.weights.length === 16) {
    localStorage.setItem(KEYS.lrWeights, JSON.stringify(payload.lrWeights));
    restored.lrWeights = true;
  }

  // NN weights — 16→16→8→1; W1 is 16×16.
  if (payload.nnWeights && Array.isArray(payload.nnWeights.W1)
      && payload.nnWeights.W1.length === 16
      && payload.nnWeights.W1[0]?.length === 16) {
    localStorage.setItem(KEYS.nnWeights, JSON.stringify(payload.nnWeights));
    restored.nnWeights = true;
  }

  if (payload.gbm && Array.isArray(payload.gbm.trees) && payload.gbm.trees.length > 0) {
    localStorage.setItem(KEYS.gbm, JSON.stringify(payload.gbm));
    restored.gbm = true;
  }

  if (payload.lrBag && Array.isArray(payload.lrBag.bags) && payload.lrBag.bags.length > 0) {
    localStorage.setItem(KEYS.lrBag, JSON.stringify(payload.lrBag));
    restored.lrBag = true;
  }

  if (payload.activeMask && Array.isArray(payload.activeMask.slots)) {
    localStorage.setItem(KEYS.activeMask, JSON.stringify(payload.activeMask));
    restored.activeMask = true;
  }

  if (payload.regime && typeof payload.regime === "object") {
    localStorage.setItem(KEYS.regime, JSON.stringify(payload.regime));
    restored.regime = true;
  }

  return restored;
}

function safeParse(s) {
  if (!s) return null;
  try { return JSON.parse(s); } catch { return null; }
}
