import { Router } from "express";
import { loadGBM, getGBMInfo } from "../training/gbm.js";
import { loadNN, getNNInfo } from "../training/nn.js";
import { loadWeights } from "../training/lr.js";
import { loadRegimeModels, getRegimeInfo } from "../training/regime.js";
import { loadBag, getBagInfo } from "../training/bagging.js";
import { getLastRunReport } from "../training/pipeline.js";

const router = Router();

router.get("/weights", (req, res) => {
  const universes = ["btc"];
  const weights = {};
  for (const u of universes) {
    weights[u] = {
      gbm:    loadGBM(u),
      nn:     loadNN(u),
      lr:     loadWeights(u),
      regime: loadRegimeModels(u),
      bag:    loadBag(u),
    };
  }
  res.json(weights);
});

router.get("/status", (req, res) => {
  const report = getLastRunReport();
  const universes = ["btc"];
  const models = {};
  for (const u of universes) {
    models[u] = {
      gbm:    getGBMInfo(u),
      nn:     getNNInfo(u),
      regime: getRegimeInfo(u),
      bag:    getBagInfo(u),
    };
  }
  res.json({ lastRun: report, models, uptime: process.uptime() });
});

export default router;
