import { Router } from "express";
import { runTrainingPipeline, getRunLog } from "../training/pipeline.js";

const router = Router();
let isRunning = false;

router.post("/train/trigger", async (req, res) => {
  if (isRunning) return res.status(409).json({ error: "Training already in progress" });
  isRunning = true;
  res.json({ started: true, message: "Training pipeline started" });
  try {
    await runTrainingPipeline(process.env.POLYGON_KEY);
  } finally {
    isRunning = false;
  }
});

router.get("/train/log", (req, res) => {
  res.json({ log: getRunLog(), isRunning });
});

export default router;
