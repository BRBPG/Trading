import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const watchlistConfig = require("../config/watchlist.json");

import { fetchHistoricalBars, generateSimTrades, clearSimCache } from "./simulation.js";
import { buildBtcLookups } from "./lookups.js";
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

async function fetchAllBars(symbols, polygonKey, daysAgo) {
  const allBars = {};
  for (const sym of symbols) {
    const result = await fetchHistoricalBars(sym, daysAgo, polygonKey, "5m");
    if (result?.bars) {
      allBars[sym] = result.bars;
      log(`  fetched ${sym} (${result.source}, ${result.bars.closes.length} bars)`);
    } else {
      log(`  ERROR: no bars for ${sym} (daysAgo=${daysAgo}, polygonKey=${polygonKey ? "present" : "MISSING"})`);
    }
    if (polygonKey) await new Promise(r => setTimeout(r, POLYGON_DELAY_MS));
  }
  return allBars;
}

async function buildSimTrades(universe, polygonKey, daysAgo) {
  const symbols = watchlistConfig[universe] ?? [];
  log(`[${universe}] Fetching bars for ${symbols.length} symbols (daysAgo=${daysAgo})...`);
  const allBars = await fetchAllBars(symbols, polygonKey, daysAgo);

  // Build crypto-context lookups for slots 7-15 once per universe per run.
  // Uses BTC-USD's bars as the daily-aggregation source for dominance/XS/RV/
  // Parkinson/breakout. Funding/DVOL/OI come from external APIs (Coinalyze /
  // Deribit / Binance-fapi-with-Coinalyze-fallback).
  let lookups = {};
  if (universe === "btc" && allBars["BTC-USD"]) {
    log(`[${universe}] Fetching crypto-context lookups (funding/DVOL/OI/dominance/XS)...`);
    const built = await buildBtcLookups(allBars["BTC-USD"]);
    lookups = built.lookups;
    for (const [name, reason] of Object.entries(built.diagnostics)) {
      log(`  lookup ${name}: ${reason}`);
    }
  }

  const simTrades = [];
  for (const [symbol, bars] of Object.entries(allBars)) {
    const trades = generateSimTrades(symbol, bars, lookups);
    simTrades.push(...trades);
  }

  // Per-slot population diagnostics: count non-zero values in each of
  // slots 7-15 across all generated trades. Helps verify which lookups
  // populated successfully vs degraded to zero.
  if (simTrades.length > 0) {
    const slotCounts = new Array(16).fill(0);
    for (const t of simTrades) {
      for (let s = 0; s < 16; s++) if (t.features[s] !== 0) slotCounts[s]++;
    }
    const slotNames = [
      "RSI","MACD","Mom","BB","EMA9","EMA21","Vol",
      "BTC_dom_z","TS_mom_z","XS_rank","Fund_z",
      "DVOL-RV_z","OI_z","RV_ratio","Park_ratio","Breakout",
    ];
    const slotDigest = slotCounts.slice(7).map((c, i) =>
      `${slotNames[7 + i]}=${c}/${simTrades.length}`).join(" ");
    log(`[${universe}] feature-slot population (slots 7-15): ${slotDigest}`);
  }

  // Verdict balance: confirm BUY/SELL coin flip is actually splitting the trades.
  const buys = simTrades.filter(t => t.verdict === "BUY").length;
  const sells = simTrades.length - buys;
  log(`[${universe}] Generated ${simTrades.length} sim trades from ${Object.keys(allBars).length} symbols (BUY=${buys}, SELL=${sells})`);
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
  // isolated=true returns weights in result.weights without calling browser-only
  // localStorage; the server persists via the file-backed saveNN below.
  const result = trainNN(samples, { universe, isolated: true, ...(isCrypto ? { l2: 0.01, epochs: 100 } : {}) });
  const W = result?.weights;
  if (W?.W1) saveNN(W, universe);
  return W ? { ...result, ...W } : result;
}

export async function runTrainingPipeline(polygonKey, options = {}) {
  runLog = [];
  const startedAt = new Date().toISOString();

  const daysAgo = options.daysAgo
    ?? (process.env.TRAINING_DAYS_AGO ? parseInt(process.env.TRAINING_DAYS_AGO, 10) : null)
    ?? 730;

  log(`=== Nightly training pipeline started (daysAgo=${daysAgo}, polygonKey=${polygonKey ? "present" : "MISSING"}) ===`);

  if (!polygonKey) {
    const error = "polygon_key_missing";
    log(`ERROR: ${error} — pipeline aborted (no fallback for live training)`);
    const report = { startedAt, completedAt: new Date().toISOString(), error, results: {}, log: runLog };
    writeFileSync(LOG_PATH, JSON.stringify(report, null, 2), "utf8");
    return report;
  }

  const results = {};
  clearSimCache();
  let totalSymbolsWithBars = 0;

  for (const universe of ["btc"]) {
    log(`\n--- Universe: ${universe} ---`);
    try {
      const simTrades = await buildSimTrades(universe, polygonKey, daysAgo);
      if (simTrades.length > 0) totalSymbolsWithBars++;
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

  const report = { startedAt, completedAt: new Date().toISOString(), daysAgo, results, log: runLog };
  if (totalSymbolsWithBars === 0) {
    report.error = "no_bars_fetched";
    log(`ERROR: no_bars_fetched — every symbol returned zero bars`);
  }
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
