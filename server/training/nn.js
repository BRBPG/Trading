export { trainNN, predictNN, scoreWithWeights } from "../../src/nn.js";
import { storageGet, storageSet, storageRemove } from "./storage.js";

function nnKeyFor(universe) {
  if (universe === "btc")    return "trader_nn_v2_btc";
  if (universe === "crypto") return "trader_nn_v2_crypto";
  return "trader_nn_v2";
}

export function loadNN(universe = "equities") {
  try {
    const saved = JSON.parse(storageGet(nnKeyFor(universe)) || "null");
    if (saved?.W1) return saved;
  } catch { /* corrupt */ }
  return null;
}

export function saveNN(W, universe = "equities") {
  storageSet(nnKeyFor(universe), JSON.stringify({ ...W, updatedAt: new Date().toISOString() }));
}

export function resetNN(universe = "equities") {
  storageRemove(nnKeyFor(universe));
}

export function getNNInfo(universe = "equities") {
  const w = loadNN(universe);
  if (!w) return { trained: false };
  return { trained: true, updatedAt: w.updatedAt };
}
