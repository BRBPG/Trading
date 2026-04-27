import { Router } from "express";
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { adaptWeightsServer } from "../training/lr.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUTCOMES_DIR = join(__dirname, "../data/outcomes");
const LOG_PATH     = join(OUTCOMES_DIR, "log.jsonl");

if (!existsSync(OUTCOMES_DIR)) mkdirSync(OUTCOMES_DIR, { recursive: true });

const router = Router();

function getRollingWinRate(universe) {
  try {
    if (!existsSync(LOG_PATH)) return null;
    const lines = readFileSync(LOG_PATH, "utf8").trim().split("\n").filter(Boolean);
    const recent = lines
      .slice(-20)
      .map(l => { try { return JSON.parse(l); } catch { return null; } })
      .filter(e => e && e.universe === universe && e.outcome);
    if (recent.length < 10) return null;
    return recent.filter(e => e.outcome === "WIN").length / recent.length;
  } catch { return null; }
}

router.post("/outcome", (req, res) => {
  const { symbol, verdict, outcome, features, universe = "equities" } = req.body;
  if (!symbol || !outcome || !Array.isArray(features)) {
    return res.status(400).json({ error: "Missing required fields: symbol, outcome, features (array)" });
  }

  const entry = { symbol, verdict, outcome, features, universe, ts: new Date().toISOString() };
  appendFileSync(LOG_PATH, JSON.stringify(entry) + "\n", "utf8");

  const winRate = getRollingWinRate(universe);
  if (winRate !== null && winRate < 0.40) {
    return res.json({ logged: true, updated: false, gate: "frozen", winRate });
  }

  try {
    const reviewedLog = [{ reviewed: true, outcome, verdict, features }];
    const result = adaptWeightsServer(reviewedLog, 0.04, 10, universe);
    res.json({ logged: true, updated: result.trained > 0, winRate: winRate ?? "pending" });
  } catch (err) {
    res.json({ logged: true, updated: false, error: err.message });
  }
});

export default router;
