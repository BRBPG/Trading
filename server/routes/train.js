import { Router } from "express";
import { runTrainingPipeline, getRunLog } from "../training/pipeline.js";

const router = Router();
let isRunning = false;

router.post("/train/trigger", async (req, res) => {
  if (isRunning) return res.status(409).json({ error: "Training already in progress" });

  const { daysAgo } = req.body ?? {};
  if (daysAgo !== undefined) {
    if (!Number.isInteger(daysAgo) || daysAgo < 30 || daysAgo > 1825) {
      return res.status(400).json({ error: "daysAgo must be an integer in [30, 1825]" });
    }
  }

  isRunning = true;
  res.json({ started: true, daysAgo: daysAgo ?? null, message: "Training pipeline started" });
  try {
    await runTrainingPipeline(process.env.POLYGON_KEY, { daysAgo });
  } catch (err) {
    console.error("[train] Pipeline error:", err.message);
  } finally {
    isRunning = false;
  }
});

router.get("/train/log", (req, res) => {
  res.json({ log: getRunLog(), isRunning });
});

export default router;
