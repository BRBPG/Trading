import dotenv from "dotenv";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: join(__dirname, ".env") });

import express from "express";
import cron from "node-cron";
import { runTrainingPipeline } from "./training/pipeline.js";
import weightsRouter from "./routes/weights.js";
import trainRouter   from "./routes/train.js";
import outcomeRouter from "./routes/outcome.js";
import quoteRouter   from "./routes/quote.js";
import proxyRouter   from "./routes/proxy.js";

console.log("[boot] POLYGON_KEY:", process.env.POLYGON_KEY ? "present" : "MISSING");

const app  = express();
const PORT = process.env.PORT || 3001;

app.use(express.json({ limit: "1mb" }));

app.use("/api", weightsRouter);
app.use("/api", trainRouter);
app.use("/api", outcomeRouter);
app.use("/api", quoteRouter);
app.use("/api", proxyRouter);

app.get("/api/health", (_req, res) =>
  res.json({ ok: true, uptime: process.uptime(), ts: new Date().toISOString() })
);

const cronExpr = process.env.TRAINING_CRON || "0 2 * * *";
cron.schedule(cronExpr, () => {
  console.log("[cron] POLYGON_KEY:", process.env.POLYGON_KEY ? "present" : "MISSING");
  console.log("[cron] Starting nightly training...");
  runTrainingPipeline(process.env.POLYGON_KEY).catch(err => {
    console.error("[cron] Pipeline error:", err.message);
    process.stderr.write(`[cron] Pipeline error: ${err.stack || err.message}\n`);
  });
});

app.listen(PORT, () => {
  console.log(`Trading server running on port ${PORT}`);
  console.log(`Nightly training scheduled: ${cronExpr}`);
});
