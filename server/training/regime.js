export { classifyRegime, predictRegime } from "../../src/regime.js";
import { storageGet, storageSet, storageRemove } from "./storage.js";
import { trainGBM } from "../../src/gbm.js";

const MIN_SAMPLES_PER_REGIME = 15;

function regimeKeyFor(universe) {
  return `trader_regime_v1_${universe}`;
}

export function saveRegimeModels(models, universe = "equities") {
  storageSet(regimeKeyFor(universe), JSON.stringify({ ...models, updatedAt: new Date().toISOString() }));
}

export function loadRegimeModels(universe = "equities") {
  try {
    const saved = JSON.parse(storageGet(regimeKeyFor(universe)) || "null");
    if (saved?.high || saved?.low) return saved;
  } catch { /* corrupt */ }
  return null;
}

export function resetRegimeModels(universe = "equities") {
  storageRemove(regimeKeyFor(universe));
}

export function getRegimeInfo(universe = "equities") {
  const m = loadRegimeModels(universe);
  if (!m) return { trained: false };
  return { trained: true, highTrained: !!m.high, lowTrained: !!m.low, updatedAt: m.updatedAt };
}

// Mirrors src/regime.js trainRegimeModels but uses file I/O persistence.
export function trainRegimeModels(simTrades, universe = "equities") {
  if (!simTrades || simTrades.length < MIN_SAMPLES_PER_REGIME * 2) {
    return { error: `Need ≥${MIN_SAMPLES_PER_REGIME * 2} trades, got ${simTrades?.length || 0}` };
  }

  const high = [], low = [];
  for (const t of simTrades) {
    if (!t.outcome || !t.features) continue;
    const vixZFeat = t.features[7] ?? 0;
    if (vixZFeat >= 0.25) high.push(t);
    else if (vixZFeat <= -0.25) low.push(t);
  }

  const yOf = t => (t.labelBullish === 0 || t.labelBullish === 1)
    ? t.labelBullish
    : ((t.verdict === "BUY" && t.outcome === "WIN") ||
       (t.verdict === "SELL" && t.outcome === "LOSS") ? 1 : 0);

  const result = { high: null, low: null, counts: { high: high.length, low: low.length } };

  if (high.length >= MIN_SAMPLES_PER_REGIME) {
    result.high = trainGBM(high.map(t => ({ x: t.features, y: yOf(t) })), { nRounds: 80, maxDepth: 4 });
  }
  if (low.length >= MIN_SAMPLES_PER_REGIME) {
    result.low = trainGBM(low.map(t => ({ x: t.features, y: yOf(t) })), { nRounds: 80, maxDepth: 4 });
  }

  if (!result.high && !result.low) {
    return { error: `Neither regime had ≥${MIN_SAMPLES_PER_REGIME} samples`, counts: result.counts };
  }

  const payload = {
    high: result.high?.trees ? result.high : null,
    low:  result.low?.trees  ? result.low  : null,
    counts: result.counts,
    updatedAt: new Date().toISOString(),
  };
  saveRegimeModels(payload, universe);

  return {
    ok: true,
    counts: result.counts,
    highTrained: !!result.high?.trees,
    lowTrained:  !!result.low?.trees,
  };
}
