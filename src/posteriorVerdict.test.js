import { test } from "node:test";
import assert from "node:assert/strict";
import {
  normalNormalPosterior,
  pNeg,
  mapTier,
  estimateSigmaObs,
  sampleNormal,
  TIER,
} from "./posteriorVerdict.js";
import { mulberry32 } from "./seededRandom.js";

// Helper — sample a Normal(mean, sigma) using a mulberry32 RNG via
// Box-Muller. Lets tests be deterministic.
function makeNormalSampler(seed) {
  const r = mulberry32(seed);
  return function (mean = 0, sigma = 1) {
    const u = r() || 1e-9;
    const v = r();
    return mean + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

test("normalNormalPosterior returns prior when n=0", () => {
  const post = normalNormalPosterior([], 0.05, 0.02);
  assert.strictEqual(post.mean, 0);
  assert.ok(Math.abs(post.variance - 0.05 * 0.05) < 1e-12);
  assert.strictEqual(post.n, 0);
});

test("normalNormalPosterior shrinks toward sample mean as n grows", () => {
  const obsVar = 0.02;
  const priorVar = 0.05;
  const deltas = [-0.03, -0.025, -0.035, -0.028, -0.031, -0.027, -0.029, -0.033];
  const post = normalNormalPosterior(deltas, priorVar, obsVar);
  // Sample mean ≈ -0.0298. Posterior should be very close because n=8
  // and obsVar/n = 0.0025 ≪ priorVar = 0.0025. With more data the
  // posterior collapses around the sample mean.
  const sampleMean = deltas.reduce((s, x) => s + x, 0) / deltas.length;
  assert.ok(Math.abs(post.mean - sampleMean) < 0.01,
    `posterior mean ${post.mean} too far from sample mean ${sampleMean}`);
  assert.ok(post.variance < priorVar * priorVar);
  assert.strictEqual(post.n, 8);
});

test("pNeg is 0.5 at the prior", () => {
  const post = normalNormalPosterior([], 0.05, 0.02);
  assert.ok(Math.abs(pNeg(post) - 0.5) < 1e-9);
});

test("pNeg approaches 1 for strongly negative posterior", () => {
  const post = { mean: -0.1, variance: 0.001, n: 8 };
  assert.ok(pNeg(post) > 0.99);
});

test("pNeg approaches 0 for strongly positive posterior", () => {
  const post = { mean: 0.1, variance: 0.001, n: 8 };
  assert.ok(pNeg(post) < 0.01);
});

test("mapTier returns INSUFFICIENT when n < 4", () => {
  assert.strictEqual(mapTier(0, 0.99), TIER.INSUFFICIENT);
  assert.strictEqual(mapTier(3, 0.5), TIER.INSUFFICIENT);
});

test("mapTier returns DROP / WATCH / KEEP per the spec thresholds at n>=4", () => {
  assert.strictEqual(mapTier(4, 0.90), TIER.DROP);
  assert.strictEqual(mapTier(4, 0.86), TIER.DROP);
  assert.strictEqual(mapTier(4, 0.85), TIER.WATCH);
  assert.strictEqual(mapTier(4, 0.70), TIER.WATCH);
  assert.strictEqual(mapTier(4, 0.56), TIER.WATCH);
  assert.strictEqual(mapTier(4, 0.55), TIER.KEEP);
  assert.strictEqual(mapTier(4, 0.20), TIER.KEEP);
});

test("estimateSigmaObs falls back to 0.02 when fewer than 10 observations", () => {
  assert.strictEqual(estimateSigmaObs([0.01, -0.01, 0.005]), 0.02);
  assert.strictEqual(estimateSigmaObs([]), 0.02);
});

test("estimateSigmaObs uses pooled SD once at least 10 observations exist", () => {
  // Synthetic: 10 obs around 0, sd ~ 0.04
  const arr = [-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05];
  const sigma = estimateSigmaObs(arr);
  // Sample SD of this list is ~ 0.030, so estimateSigmaObs should be in
  // [0.025, 0.040].
  assert.ok(sigma >= 0.02 && sigma <= 0.06,
    `estimateSigmaObs returned ${sigma}, outside expected band`);
});

test("null calibration: under mu=0 with n=8, DROP rate is ~15% with ±5% slack", () => {
  // Generate 1000 synthetic features each with 8 deltas drawn N(0, 0.02^2),
  // run the verdict mapper, count tiers.
  const rand = makeNormalSampler(20260427);
  let drop = 0, watch = 0, keep = 0;
  const N = 1000;
  for (let i = 0; i < N; i++) {
    const deltas = [];
    for (let j = 0; j < 8; j++) deltas.push(rand(0, 0.02));
    const post = normalNormalPosterior(deltas, 0.05, 0.02);
    const t = mapTier(post.n, pNeg(post));
    if (t === TIER.DROP) drop++;
    else if (t === TIER.WATCH) watch++;
    else if (t === TIER.KEEP) keep++;
  }
  const dropRate = drop / N;
  const watchRate = watch / N;
  const keepRate = keep / N;
  assert.ok(Math.abs(dropRate - 0.15) <= 0.05, `dropRate=${dropRate}`);
  assert.ok(Math.abs(watchRate - 0.30) <= 0.07, `watchRate=${watchRate}`);
  assert.ok(Math.abs(keepRate - 0.55) <= 0.07, `keepRate=${keepRate}`);
});

test("power: under mu=-0.03 with n=8, DROP rate exceeds 80%", () => {
  const rand = makeNormalSampler(99);
  let drop = 0;
  const N = 1000;
  for (let i = 0; i < N; i++) {
    const deltas = [];
    for (let j = 0; j < 8; j++) deltas.push(rand(-0.03, 0.02));
    const post = normalNormalPosterior(deltas, 0.05, 0.02);
    if (mapTier(post.n, pNeg(post)) === TIER.DROP) drop++;
  }
  const rate = drop / N;
  assert.ok(rate > 0.80, `DROP rate under negative effect = ${rate}, expected > 0.80`);
});

test("sampleNormal centred on the posterior mean", () => {
  const r = mulberry32(123);
  let sum = 0;
  const N = 5000;
  for (let i = 0; i < N; i++) sum += sampleNormal(0.5, 0.001, r);
  const mean = sum / N;
  assert.ok(Math.abs(mean - 0.5) < 0.02, `mean=${mean}`);
});
