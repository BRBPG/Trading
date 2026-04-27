import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const watchlistConfig = require("../config/watchlist.json");

import { fetchHistoricalBars, generateSimTrades, clearSimCache } from "./simulation.js";
import { trainGBM, saveGBM, loadGBM, getActiveMask } from "./gbm.js";
import { trainNN, saveNN, loadNN } from "./nn.js";
import { trainRegimeModels } from "./regime.js";
import { trainBagFromSim } from "./bagging.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_DIR  = join(__dirname, "../data/weights");
const LOG_PATH  = join(__dirname, "../data/weights/last_run.json");

if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });

let runLog = [];

function log(msg) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  console.log(line);
  runLog.push(line);
}

// Polygon Starter: 5 req/min — wait 13s between requests.
const POLYGON_DELAY_MS = 13_000;

async function fetchAllBars(symbols, polygonKey) {
  const allBars = {};
  for (const sym of symbols) {
    const result = await fetchHistoricalBars(sym, 180, polygonKey, "5m");
    if (result?.bars) {
      allBars[sym] = result.bars;
      log(`  fetched ${sym} (${result.source}, ${result.bars.closes.length} bars)`);
    } else {
      log(`  WARN: no bars for ${sym}`);
    }
    if (polygonKey) await new Promise(r => setTimeout(r, POLYGON_DELAY_MS));
  }
  return allBars;
}

async function buildSimTrades(universe, polygonKey) {
  const symbols = watchlistConfig[universe] ?? [];
  log(`[${universe}] Fetching bars for ${symbols.length} symbols...`);
  const allBars = await fetchAllBars(symbols, polygonKey);

  const simTrades = [];
  for (const [symbol, bars] of Object.entries(allBars)) {
    const trades = generateSimTrades(symbol, bars);
    simTrades.push(...trades);
  }
  log(`[${universe}] Generated ${simTrades.length} sim trades from ${Object.keys(allBars).length} symbols`);
  return simTrades;
}

// Train GBM with warm-start (matching trainGBMFromSim behaviour).
function trainGBMForUniverse(simTrades, universe) {
  const activeMask = getActiveMask(universe) ?? [];
  const maskSet = new Set(activeMask);

  const samples = simTrades
    .filter(d => d.outcome && d.features)
    .map(d => {
      let x = d.features;
      if (maskSet.size > 0) {
        x = x.slice();
        for (const s of maskSet) if (s < x.length) x[s] = 0;
      }
      return { x, y: d.labelBullish ?? (d.outcome === "WIN" ? 1 : 0) };
    });

  if (samples.length < 20) return { trained: 0, reason: `Need ≥20 samples, got ${samples.length}` };

  const continueFrom = loadGBM(universe);
  const result = trainGBM(samples, { continueFrom, maxTreesTotal: 300 });
  if (result.trees) saveGBM(result, universe);
  return result;
}

// Train NN (matching trainNNFromSim behaviour).
function trainNNForUniverse(simTrades, universe) {
  const isCrypto = universe === "crypto" || universe === "btc";
  const samples = simTrades
    .filter(d => d.outcome && d.features)
    .map(d => ({
      x: d.features,
      y: d.labelBullish ?? (d.outcome === "WIN" ? 1 : 0),
      ageDays: d.ageDays || 0,
    }));

  if (samples.length < 20) return { trained: 0 };
  const result = trainNN(samples, { universe, ...(isCrypto ? { l2: 0.01, epochs: 100 } : {}) });
  if (result?.W1) saveNN(result, universe);
  return result;
}

export async function runTrainingPipeline(polygonKey) {
  runLog = [];
  const startedAt = new Date().toISOString();
  log("=== Nightly training pipeline started ===");

  const results = {};
  clearSimCache();

  for (const universe of ["btc"]) {
    log(`\n--- Universe: ${universe} ---`);
    try {
      const simTrades = await buildSimTrades(universe, polygonKey);
      if (simTrades.length < 30) {
        log(`[${universe}] SKIP: only ${simTrades.length} trades (need ≥30)`);
        results[universe] = { skipped: true, reason: "insufficient_trades", count: simTrades.length };
        continue;
      }

      const gbm    = trainGBMForUniverse(simTrades, universe);
      const nn     = trainNNForUniverse(simTrades, universe);
      const regime = trainRegimeModels(simTrades, universe);
      const bag    = trainBagFromSim(simTrades, universe);

      log(`[${universe}] GBM: ${gbm?.trees?.length ?? gbm?.reason ?? "?"} trees`);
      log(`[${universe}] NN: trained=${!!nn?.W1}`);
      log(`[${universe}] Regime: high=${regime?.highTrained}, low=${regime?.lowTrained}`);
      log(`[${universe}] Bag: ${bag?.bags?.length ?? bag?.error ?? "?"} bags`);

      results[universe] = {
        simTrades: simTrades.length,
        gbm:    { trees: gbm?.trees?.length, trained: !!gbm?.trees },
        nn:     { trained: !!nn?.W1 },
        regime: { highTrained: regime?.highTrained, lowTrained: regime?.lowTrained },
        bag:    { nBags: bag?.bags?.length, trained: !!bag?.bags },
        promotedAt: new Date().toISOString(),
      };
    } catch (err) {
      log(`[${universe}] ERROR: ${err.message}\n${err.stack}`);
      results[universe] = { error: err.message };
    }
  }

  const report = { startedAt, completedAt: new Date().toISOString(), results, log: runLog };
  writeFileSync(LOG_PATH, JSON.stringify(report, null, 2), "utf8");
  log("\n=== Pipeline complete ===");
  return report;
}

export function getLastRunReport() {
  try {
    if (!existsSync(LOG_PATH)) return null;
    return JSON.parse(readFileSync(LOG_PATH, "utf8"));
  } catch { return null; }
}

export function getRunLog() { return runLog; }
