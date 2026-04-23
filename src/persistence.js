// ─── Cross-device state persistence ─────────────────────────────────────────
// Everything the trader system "learns" — the decision log, the LR weights,
// the NN weights — lives in the browser's localStorage. That's per-device,
// per-browser. Open the app on a different laptop and you start from scratch.
//
// To make state portable without standing up a backend, we expose two manual
// operations:
//   exportState() → produces a JSON blob the user downloads as a file
//   importState(json) → restores all three keys from a previously-exported file
//
// API keys are deliberately NOT included in the export — those would be a
// security hole if the file were ever shared or stored in the wrong place.

const KEYS = {
  log:       "trader_decision_log",
  lrWeights: "trader_lr_weights",
  nnWeights: "trader_nn_weights_v1",
};

const SCHEMA_VERSION = 1;

export function exportState() {
  const payload = {
    schemaVersion: SCHEMA_VERSION,
    exportedAt: new Date().toISOString(),
    appName: "the-trader",
    log:       safeParse(localStorage.getItem(KEYS.log)) ?? [],
    lrWeights: safeParse(localStorage.getItem(KEYS.lrWeights)) ?? null,
    nnWeights: safeParse(localStorage.getItem(KEYS.nnWeights)) ?? null,
  };
  return payload;
}

// Trigger a browser file download for the export blob.
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

// Restore all three keys from a payload object.
// mode: "replace" wipes the existing log; "merge" appends new entries by id.
export function importState(payload, { mode = "replace" } = {}) {
  if (!payload || typeof payload !== "object") {
    throw new Error("Invalid import: not an object.");
  }
  if (payload.schemaVersion !== SCHEMA_VERSION) {
    throw new Error(`Unsupported schema version ${payload.schemaVersion}; expected ${SCHEMA_VERSION}.`);
  }

  const restored = { log: 0, lrWeights: false, nnWeights: false };

  // Decision log
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

  // Logistic-regression weights
  if (payload.lrWeights && Array.isArray(payload.lrWeights.weights) && payload.lrWeights.weights.length === 7) {
    localStorage.setItem(KEYS.lrWeights, JSON.stringify(payload.lrWeights));
    restored.lrWeights = true;
  }

  // Neural network weights
  if (payload.nnWeights && Array.isArray(payload.nnWeights.W1) && payload.nnWeights.W1.length === 8) {
    localStorage.setItem(KEYS.nnWeights, JSON.stringify(payload.nnWeights));
    restored.nnWeights = true;
  }

  return restored;
}

// Small helper so a corrupt entry in any one slot doesn't take the whole
// importer down.
function safeParse(s) {
  if (!s) return null;
  try { return JSON.parse(s); } catch { return null; }
}

// One-line summary of what's currently stored locally — useful for the UI.
export function describeLocalState() {
  const log = safeParse(localStorage.getItem(KEYS.log)) ?? [];
  const lr  = safeParse(localStorage.getItem(KEYS.lrWeights));
  const nn  = safeParse(localStorage.getItem(KEYS.nnWeights));
  return {
    logCount: log.length,
    reviewedCount: log.filter(d => d.reviewed).length,
    lrUpdatedAt: lr?.updatedAt || null,
    nnTrainedOn: nn?.trainedOn || 0,
    nnEpochs: nn?.epochs || 0,
  };
}
