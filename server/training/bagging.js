import { trainBag as trainBagRaw, predictBag } from "../../src/bagging.js";
import { storageGet, storageSet, storageRemove } from "./storage.js";

export { predictBag };
export { trainBagRaw as trainBag };

function bagKeyFor(universe) {
  return `trader_bag_v1_${universe}`;
}

export function saveBag(bag, universe = "equities") {
  if (!bag) return;
  storageSet(bagKeyFor(universe), JSON.stringify({ ...bag, updatedAt: new Date().toISOString() }));
}

export function loadBag(universe = "equities") {
  try {
    const saved = JSON.parse(storageGet(bagKeyFor(universe)) || "null");
    if (saved?.bags?.length && saved.featureDim) return saved;
  } catch { /* corrupt */ }
  return null;
}

export function resetBag(universe = "equities") {
  storageRemove(bagKeyFor(universe));
}

export function getBagInfo(universe = "equities") {
  const b = loadBag(universe);
  if (!b) return { trained: false };
  return { trained: true, nBags: b.bags?.length ?? 0, updatedAt: b.updatedAt };
}

export function trainBagFromSim(simTrades, universe = "equities") {
  const samples = simTrades
    .filter(d => d.outcome && d.features)
    .map(d => ({
      x: d.features,
      y: (d.labelBullish === 0 || d.labelBullish === 1)
         ? d.labelBullish
         : ((d.verdict === "BUY" && d.outcome === "WIN") ||
            (d.verdict === "SELL" && d.outcome === "LOSS") ? 1 : 0),
    }));
  if (samples.length < 10) return { error: `Need ≥10 samples, got ${samples.length}` };
  const bag = trainBagRaw(samples);
  saveBag(bag, universe);
  return bag;
}
