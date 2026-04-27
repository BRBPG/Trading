export { trainGBM, predictGBM, featureImportance } from "../../src/gbm.js";
import { storageGet, storageSet, storageRemove } from "./storage.js";

function gbmKeyFor(universe) {
  if (universe === "btc")    return "trader_gbm_v1_btc";
  if (universe === "crypto") return "trader_gbm_v1_crypto";
  return "trader_gbm_v1";
}
function maskKeyFor(universe) {
  return `trader_gbm_mask_${universe}`;
}

export function saveGBM(model, universe = "equities") {
  storageSet(gbmKeyFor(universe), JSON.stringify({ ...model, updatedAt: new Date().toISOString() }));
}

export function loadGBM(universe = "equities") {
  try {
    const saved = JSON.parse(storageGet(gbmKeyFor(universe)) || "null");
    if (saved?.trees?.length) return saved;
  } catch { /* corrupt */ }
  return null;
}

export function resetGBM(universe = "equities") {
  storageRemove(gbmKeyFor(universe));
}

export function getGBMInfo(universe = "equities") {
  const m = loadGBM(universe);
  if (!m) return { trained: false };
  return { trained: true, nTrees: m.trees?.length ?? 0, updatedAt: m.updatedAt };
}

export function getActiveMask(universe = "equities") {
  try {
    const raw = storageGet(maskKeyFor(universe));
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed?.slots) ? parsed.slots : null;
  } catch { return null; }
}

export function setActiveMask(universe, slots, meta = {}) {
  storageSet(maskKeyFor(universe), JSON.stringify({ slots, ...meta, updatedAt: new Date().toISOString() }));
}

export function clearActiveMask(universe = "equities") {
  storageRemove(maskKeyFor(universe));
}

export function getActiveMaskInfo(universe = "equities") {
  try {
    const raw = storageGet(maskKeyFor(universe));
    if (!raw) return { hasMask: false };
    const parsed = JSON.parse(raw);
    return { hasMask: true, slots: parsed.slots, updatedAt: parsed.updatedAt };
  } catch { return { hasMask: false }; }
}
