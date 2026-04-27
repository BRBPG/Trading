# Calibrated Feature-Pruning Verdict Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the AND-of-thresholds DROP rule that has parked every feature in `KEEP` with a calibrated three-tier verdict (`DROP` / `WATCH` / `KEEP` / `INSUFFICIENT`) driven by a closed-form normal-normal posterior over each feature's true mean ablation Δ. Switch within-cycle ablation comparisons from weakly-paired to fully-paired by wrapping the baseline and per-feature `runWalkForward` calls in a deterministic-seed `Math.random` shim.

**Architecture:** Two new pure modules — `seededRandom.js` (PRNG + scope wrapper) and `posteriorVerdict.js` (posterior accumulator, σ_obs estimator, verdict mapper, normal Thompson sampler). Surgical edits to `runAblationStudy` and the continuous-train loop in `App.jsx` to wire them in. Tests run via Node's built-in test runner (`node --test`) with `node:assert` — no new dependencies.

**Tech Stack:** React 19 + Vite 6 frontend (already in place), Node 18 with built-in `node:test` and `node:assert`, ESM (`"type": "module"` already set in `package.json`). No new packages.

**Spec:** `/root/Trading/docs/superpowers/specs/2026-04-27-pruning-verdict-redesign-design.md`

---

## File structure

**New files:**

- `src/seededRandom.js` — `mulberry32(seed)` and `withSeededRandom(seed, fn)`.
- `src/seededRandom.test.js` — determinism, restoration, throw-safety, Math.random isolation.
- `src/posteriorVerdict.js` — `normalNormalPosterior`, `pNeg`, `mapTier`, `estimateSigmaObs`, `sampleNormal`.
- `src/posteriorVerdict.test.js` — null calibration, power, INSUFFICIENT guard, σ_obs fallback.

**Modified files:**

- `src/walkForward.js` — `runAblationStudy` accepts an optional `seed` and wraps each `runWalkForward` invocation in `withSeededRandom(seed, …)`.
- `src/App.jsx` — replace per-cycle Beta-Thompson sampling with a normal-normal Thompson analog (sample μ_f from posterior, include slot if μ_f > 0); replace the post-run verdict block; update the verdicts render block; narrow the apply-verdict mask union to DROP-only; remove now-unused constants and `sampleBeta`.
- `package.json` — add `"test": "node --test src/seededRandom.test.js src/posteriorVerdict.test.js"`.

**Out of scope for this plan** (already deferred to follow-ups):
- `SHOW_LEGACY_PANELS` gates removal.
- Deep `universe === "crypto"` / `"equities"` JSX branch removal.
- Any change to feature engineering, the Project B AUC<0.5 diagnostic, or non-pruning UI.

---

### Task 1: Add npm test script

**Files:**
- Modify: `/root/Trading/package.json` (scripts block)

- [ ] **Step 1: Read current package.json scripts.**

Run: `grep -n "scripts" /root/Trading/package.json`
Expected: line showing the `"scripts": {` block.

- [ ] **Step 2: Add a `test` script.**

Replace the existing `"scripts"` block in `/root/Trading/package.json` with:

```json
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "lint": "eslint .",
    "preview": "vite preview",
    "test": "node --test src/seededRandom.test.js src/posteriorVerdict.test.js"
  },
```

(Order preserved; `test` appended last.)

- [ ] **Step 3: Verify the script is parseable (test files don't exist yet).**

Run: `cd /root/Trading && npm test`
Expected: failure of the form `Cannot find module ...seededRandom.test.js`. The script ITSELF must parse — the failure must be at file-resolution time, not before. If `npm` prints `unknown script`, the JSON edit is wrong; fix it.

- [ ] **Step 4: Commit.**

```bash
cd /root/Trading
git add package.json
git commit -m "chore: add npm test script for plain Node test runner

Wires up node --test against the upcoming seededRandom and
posteriorVerdict test files. No new deps — uses Node 18's
built-in test runner with ESM ('type': 'module' already set)."
```

---

### Task 2: Implement `seededRandom.js` and tests (TDD)

**Files:**
- Create: `/root/Trading/src/seededRandom.js`
- Create: `/root/Trading/src/seededRandom.test.js`

- [ ] **Step 1: Write the failing test file.**

Create `/root/Trading/src/seededRandom.test.js` with this exact content:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { mulberry32, withSeededRandom } from "./seededRandom.js";

test("mulberry32 produces deterministic sequence for a given seed", () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  for (let i = 0; i < 10; i++) {
    assert.strictEqual(a(), b());
  }
});

test("mulberry32 differs across seeds", () => {
  const a = mulberry32(1);
  const b = mulberry32(2);
  assert.notStrictEqual(a(), b());
});

test("mulberry32 outputs are in [0, 1)", () => {
  const r = mulberry32(7);
  for (let i = 0; i < 1000; i++) {
    const x = r();
    assert.ok(x >= 0 && x < 1, `out of range: ${x}`);
  }
});

test("withSeededRandom installs a deterministic Math.random for the duration of fn", async () => {
  const captured = [];
  await withSeededRandom(42, async () => {
    captured.push(Math.random(), Math.random(), Math.random());
  });
  const expected = [];
  await withSeededRandom(42, async () => {
    expected.push(Math.random(), Math.random(), Math.random());
  });
  assert.deepStrictEqual(captured, expected);
});

test("withSeededRandom restores the original Math.random after fn returns", async () => {
  const before = Math.random;
  await withSeededRandom(42, async () => {
    // intentionally empty — we only care about restoration
  });
  assert.strictEqual(Math.random, before, "Math.random reference was not restored");
});

test("withSeededRandom restores the original Math.random when fn throws", async () => {
  const before = Math.random;
  await assert.rejects(
    withSeededRandom(42, async () => {
      throw new Error("boom");
    }),
    /boom/,
  );
  assert.strictEqual(Math.random, before, "Math.random reference was not restored on throw");
});

test("withSeededRandom returns the value of fn", async () => {
  const out = await withSeededRandom(42, async () => "hello");
  assert.strictEqual(out, "hello");
});
```

- [ ] **Step 2: Run tests to verify they fail with module-not-found.**

Run: `cd /root/Trading && npm test`
Expected: failure with `Cannot find module './seededRandom.js'` (or similar). The error must come from the import, not a syntax error in the test file.

- [ ] **Step 3: Implement `seededRandom.js`.**

Create `/root/Trading/src/seededRandom.js` with this exact content:

```js
// Deterministic PRNG + scoped Math.random shim. Used by runAblationStudy
// to make the baseline and each masked walk-forward call see the same
// stochastic sequence within a cycle (paired comparison), while still
// producing independent draws across cycles (different seeds).
//
// mulberry32 is a 32-bit PRNG with period 2^32. Good enough for paired
// ablation noise control; not cryptographic.

export function mulberry32(seed) {
  let s = seed >>> 0;
  return function next() {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Run `fn` with Math.random temporarily replaced by a mulberry32 stream
// seeded by `seed`. Restores the original Math.random in a finally block
// so a thrown error doesn't leak the shim into subsequent code.
export async function withSeededRandom(seed, fn) {
  const original = Math.random;
  const rng = mulberry32(seed);
  Math.random = rng;
  try {
    return await fn();
  } finally {
    Math.random = original;
  }
}
```

- [ ] **Step 4: Run tests to verify they all pass.**

Run: `cd /root/Trading && npm test`
Expected: all 7 tests pass. Output should include lines like `# pass 7` and `# fail 0`.

- [ ] **Step 5: Commit.**

```bash
cd /root/Trading
git add src/seededRandom.js src/seededRandom.test.js
git commit -m "feat: add seededRandom module for paired ablation comparisons

mulberry32 PRNG + withSeededRandom scope wrapper that swaps
Math.random for the duration of an async callback and restores
it (even on throw). Used in the next commit to wrap the baseline
and per-feature masked runWalkForward calls inside a cycle, so
init/bootstrap/minibatch RNG matches between the two — turning
delta into a true paired difference."
```

---

### Task 3: Implement `posteriorVerdict.js` and tests (TDD)

**Files:**
- Create: `/root/Trading/src/posteriorVerdict.js`
- Create: `/root/Trading/src/posteriorVerdict.test.js`

- [ ] **Step 1: Write the failing test file.**

Create `/root/Trading/src/posteriorVerdict.test.js` with this exact content:

```js
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
```

- [ ] **Step 2: Run tests to verify they fail with module-not-found.**

Run: `cd /root/Trading && npm test`
Expected: failure on the import of `./posteriorVerdict.js`.

- [ ] **Step 3: Implement `posteriorVerdict.js`.**

Create `/root/Trading/src/posteriorVerdict.js` with this exact content:

```js
// Closed-form normal-normal posterior over each feature's true mean
// ablation Δ. Replaces the Beta(α, β) machinery that drove the previous
// AND-of-thresholds verdict rule. See spec at
// docs/superpowers/specs/2026-04-27-pruning-verdict-redesign-design.md
// for derivation.
//
// Conventions:
//   prior:       mu_f ~ N(0, priorSigma^2)            priorSigma = 0.05
//   likelihood:  delta_i | mu_f ~ N(mu_f, obsSigma^2)
//   posterior:   mu_f | data    ~ N(meanHat, tauSquared)
// pNeg(post) = P(mu_f < 0 | data) via the normal CDF at 0.

export const TIER = Object.freeze({
  INSUFFICIENT: "INSUFFICIENT",
  DROP: "DROP",
  WATCH: "WATCH",
  KEEP: "KEEP",
});

const DROP_PNEG_MIN = 0.85;
const WATCH_PNEG_MIN = 0.55;
const MIN_N_FOR_VERDICT = 4;
const SIGMA_OBS_FALLBACK = 0.02;
const MIN_OBS_FOR_EMPIRICAL_SIGMA = 10;

// Closed-form normal-normal conjugate update. Returns { mean, variance, n }.
// `priorSigma` is the prior std-dev on mu_f (centred at 0). `obsSigma` is
// the per-observation noise std-dev (one observation = one paired Δ).
export function normalNormalPosterior(deltas, priorSigma, obsSigma) {
  const priorVar = priorSigma * priorSigma;
  const obsVar = obsSigma * obsSigma;
  const n = deltas.length;
  if (n === 0) {
    return { mean: 0, variance: priorVar, n: 0 };
  }
  const sum = deltas.reduce((s, x) => s + x, 0);
  const sampleMean = sum / n;
  const tauSquared = 1 / (1 / priorVar + n / obsVar);
  const meanHat = tauSquared * (n * sampleMean / obsVar);
  return { mean: meanHat, variance: tauSquared, n };
}

// P(mu_f < 0 | data). Uses Φ(-meanHat / sqrt(variance)) via the
// Abramowitz-Stegun rational approximation for the normal CDF.
export function pNeg(post) {
  const sd = Math.sqrt(post.variance);
  if (sd === 0) return post.mean < 0 ? 1 : 0;
  return normalCdf(-post.mean / sd);
}

// Three-tier verdict. INSUFFICIENT when n is too small to render a
// confident verdict at all; otherwise DROP / WATCH / KEEP from the
// posterior tail probability.
export function mapTier(n, pNegValue) {
  if (n < MIN_N_FOR_VERDICT) return TIER.INSUFFICIENT;
  if (pNegValue > DROP_PNEG_MIN) return TIER.DROP;
  if (pNegValue > WATCH_PNEG_MIN) return TIER.WATCH;
  return TIER.KEEP;
}

// Pooled per-observation noise estimate. Below the threshold of evidence
// we fall back to a fixed plausible default rather than over-fit a tiny
// sample; once enough evidence exists we use the sample SD.
export function estimateSigmaObs(allDeltasFlat) {
  const valid = allDeltasFlat.filter(Number.isFinite);
  if (valid.length < MIN_OBS_FOR_EMPIRICAL_SIGMA) return SIGMA_OBS_FALLBACK;
  const mean = valid.reduce((s, x) => s + x, 0) / valid.length;
  const variance = valid.reduce((s, x) => s + (x - mean) * (x - mean), 0) / (valid.length - 1);
  return Math.max(1e-6, Math.sqrt(variance));
}

// Sample one value from N(mean, variance). `randFn` is a [0,1) RNG; pass
// Math.random by default. Uses Box-Muller.
export function sampleNormal(mean, variance, randFn = Math.random) {
  const u = randFn() || 1e-9;
  const v = randFn();
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return mean + Math.sqrt(variance) * z;
}

// ─── Numerical helpers ────────────────────────────────────────────────────
// Abramowitz & Stegun 26.2.17 — max abs error ≈ 7.5e-8 over the real line.
function normalCdf(x) {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return 0.5 * (1 + sign * y);
}
```

- [ ] **Step 4: Run tests to verify they all pass.**

Run: `cd /root/Trading && npm test`
Expected: all tests in both `seededRandom.test.js` and `posteriorVerdict.test.js` pass. Output includes `# fail 0`.

- [ ] **Step 5: Commit.**

```bash
cd /root/Trading
git add src/posteriorVerdict.js src/posteriorVerdict.test.js
git commit -m "feat: add posteriorVerdict module — normal-normal posterior + verdict mapper

Closed-form conjugate update of mu_f given paired deltas, plus a
three-tier verdict (DROP/WATCH/KEEP) from the posterior tail
probability and an INSUFFICIENT guard at n<4. Includes a pooled
sigma_obs estimator with a fallback for thin data, and a Box-Muller
normal sampler for cycle-to-cycle Thompson exploration.

Tests cover null calibration (DROP rate ~15% under mu=0, n=8,
sigma=0.02 across 1000 synthetic features) and power (DROP rate
>80% under mu=-0.03, n=8) per the spec acceptance criteria."
```

---

### Task 4: Wire seededRandom into `runAblationStudy`

**Files:**
- Modify: `/root/Trading/src/walkForward.js` (function `runAblationStudy` near the bottom)

- [ ] **Step 1: Add a regression test that pairing reduces variance.**

Append the following test at the end of `/root/Trading/src/seededRandom.test.js`:

```js
import { mulberry32 as _m } from "./seededRandom.js";

test("integration: paired calls produce identical Math.random sequences across two withSeededRandom invocations", async () => {
  const a = [];
  const b = [];
  await withSeededRandom(123, async () => {
    for (let i = 0; i < 5; i++) a.push(Math.random());
  });
  await withSeededRandom(123, async () => {
    for (let i = 0; i < 5; i++) b.push(Math.random());
  });
  assert.deepStrictEqual(a, b);
});
```

(Note: the duplicate `mulberry32 as _m` import is intentional — it's a no-op alias to avoid a linter complaint about an unused import; use plain re-import if your linter accepts it.)

- [ ] **Step 2: Run test to verify it passes.**

Run: `cd /root/Trading && npm test`
Expected: all tests pass.

- [ ] **Step 3: Locate the existing `runAblationStudy` block.**

Run: `grep -n "export async function runAblationStudy" /root/Trading/src/walkForward.js`
Expected: a single match around line 429. If multiple matches or none, stop and inspect.

- [ ] **Step 4: Modify `runAblationStudy` to accept a seed and wrap calls.**

In `/root/Trading/src/walkForward.js`, find:

```js
import { trainGBM, predictGBM, getActiveMask } from "./gbm.js";
```

(or whatever the existing top-of-file import block looks like) and add this line near the other imports:

```js
import { withSeededRandom } from "./seededRandom.js";
```

Then replace the `runAblationStudy` function body. Find the existing function — it begins with:

```js
export async function runAblationStudy(simTrades, baseOpts = {}, targets = [], onProgress = null) {
```

Replace the entire function (from `export async function runAblationStudy` through its closing `}` brace) with:

```js
export async function runAblationStudy(simTrades, baseOpts = {}, targets = [], onProgress = null) {
  if (!simTrades?.length) return { error: "no trades provided" };

  // Seed for paired comparisons within this cycle. Caller passes
  // baseOpts.seed; if absent we derive a deterministic seed from the
  // call timestamp — better than nothing, but the caller SHOULD pass a
  // per-cycle seed so different cycles produce independent paired
  // observations.
  const seed = (baseOpts.seed >>> 0) || (Date.now() & 0xffffffff);

  const baseline = await withSeededRandom(seed, () => runWalkForward(simTrades, baseOpts));
  if (baseline.error) return { error: `baseline: ${baseline.error}` };
  const baseAUC = baseline.overall?.oosAUC;
  if (baseAUC == null) return { error: "baseline produced no AUC" };
  if (onProgress) {
    await new Promise(r => setTimeout(r, 0));
    onProgress({ phase: "baseline_done", idx: 0, total: targets.length + 1 });
  }

  const results = [];
  for (let ti = 0; ti < targets.length; ti++) {
    const t = targets[ti];
    const base = baseOpts.maskSlots || [];
    const unioned = Array.from(new Set([...base, t.slot]));
    const masked = await withSeededRandom(seed, () =>
      runWalkForward(simTrades, { ...baseOpts, maskSlots: unioned })
    );
    if (masked.error) {
      results.push({ ...t, auc: null, delta: null, error: masked.error });
    } else {
      const maskedAUC = masked.overall?.oosAUC;
      if (maskedAUC == null) {
        results.push({ ...t, auc: null, delta: null, error: "no auc" });
      } else {
        results.push({
          ...t,
          aucWithout: maskedAUC,
          aucBaseline: baseAUC,
          delta: baseAUC - maskedAUC,
        });
      }
    }
    if (onProgress) {
      await new Promise(r => setTimeout(r, 0));
      onProgress({ phase: "feature_done", slot: t.slot, name: t.name, idx: ti + 1, total: targets.length + 1 });
    }
  }
  results.sort((a, b) => (b.delta ?? -Infinity) - (a.delta ?? -Infinity));

  return {
    baselineAUC: baseAUC,
    baselineLogLoss: baseline.overall?.oosLogLoss,
    baselineSamples: baseline.overall?.oosSamples,
    baselineGap: (baseline.overall?.avgTestLoss != null && baseline.overall?.avgTrainLoss != null)
      ? baseline.overall.avgTestLoss - baseline.overall.avgTrainLoss : null,
    results,
  };
}
```

- [ ] **Step 5: Verify the build still compiles.**

Run: `cd /root/Trading && npm run build`
Expected: build succeeds. If TypeScript-like errors appear about unused `mulberry32 as _m` import in the test file, change that import line in `src/seededRandom.test.js` to:

```js
import { mulberry32 } from "./seededRandom.js";  // for explicit determinism reference
```

and use `mulberry32` somewhere trivial in that test — or just remove the line if your linter is fine.

- [ ] **Step 6: Run the test suite again.**

Run: `cd /root/Trading && npm test`
Expected: all tests pass.

- [ ] **Step 7: Commit.**

```bash
cd /root/Trading
git add src/walkForward.js src/seededRandom.test.js
git commit -m "feat(walkForward): wrap ablation runs in withSeededRandom for paired Δ

Each cycle picks one seed; baseline runWalkForward and every
masked runWalkForward in that cycle execute under the same Math.random
stream. This makes bootstrap row sampling, column subsample shuffle,
NN init, and minibatch shuffle identical between the with-feature
and without-feature runs — turning the per-feature delta into a
genuine paired difference. Caller passes baseOpts.seed; the next
commit threads a per-cycle seed in from App.jsx."
```

---

### Task 5: Replace per-cycle mask sampling in `App.jsx` (Thompson on normal-normal)

**Files:**
- Modify: `/root/Trading/src/App.jsx` (continuous-train loop, near `posteriors[t.slot] = { alpha: PRIOR_ALPHA, beta: PRIOR_BETA }` and the `sampleBeta` block)

- [ ] **Step 1: Add a top-of-file import for the new modules.**

In `/root/Trading/src/App.jsx`, find the existing model imports near the top:

```js
import { runWalkForward, interpretWF, runAblationStudy } from "./walkForward";
```

Add this line directly below it:

```js
import { normalNormalPosterior, pNeg, mapTier, estimateSigmaObs, sampleNormal, TIER } from "./posteriorVerdict.js";
```

- [ ] **Step 2: Locate the per-cycle posterior init + Thompson block.**

Run: `grep -n "PRIOR_ALPHA\|sampleBeta\|posteriors\[t.slot\] = { alpha" /root/Trading/src/App.jsx`
Expected: matches at the const block (PRIOR_ALPHA / PRIOR_BETA), at the init loop (`posteriors[t.slot] = { alpha: PRIOR_ALPHA, beta: PRIOR_BETA };`), and at the `sampleBeta` definition.

- [ ] **Step 3: Replace `PRIOR_ALPHA`, `PRIOR_BETA`, `POS_THRESHOLD`, `NEG_THRESHOLD`, `BASELINE_QUALITY_MIN`, `GAP_QUALITY_MAX` with new posterior config.**

Find this block (line numbers approximate, anchor on `POS_THRESHOLD = 0.02`):

```js
    const POS_THRESHOLD = 0.02;
    const NEG_THRESHOLD = -0.02;
```

Replace it with:

```js
    // Normal-normal posterior parameters. priorSigma is the prior SD on
    // mu_f (the true mean ablation Δ); a 5% AUC swing is "large." obsSigma
    // is recomputed from the empirical pooled SD once enough Δs exist;
    // until then it falls back to 0.02 (see posteriorVerdict.js).
    const PRIOR_SIGMA = 0.05;
```

Find this block:

```js
    const BASELINE_QUALITY_MIN = 0.05;  // |baselineAUC − 0.5|
    const GAP_QUALITY_MAX = 0.15;        // train-test gap
```

Delete it entirely. (The baseline-noise gate downstream goes away with the alpha/beta ratchet — see Step 6.)

Find this block:

```js
    const PRIOR_ALPHA = 2;
    const PRIOR_BETA = 2;
```

Delete it entirely.

- [ ] **Step 4: Replace the per-slot init line.**

Find:

```js
      posteriors[t.slot] = { alpha: PRIOR_ALPHA, beta: PRIOR_BETA };
```

Replace with:

```js
      posteriors[t.slot] = { mean: 0, variance: PRIOR_SIGMA * PRIOR_SIGMA, n: 0, deltas: [] };
```

- [ ] **Step 5: Replace the `sampleBeta` definition and Thompson sampling.**

Find the `sampleBeta` definition (a block beginning with `const sampleBeta = (a, b) => {`) and delete the entire `const sampleBeta = …;` declaration (it spans roughly 8–15 lines depending on style).

Find the Thompson-sample call site:

```js
        const p = sampleBeta(posteriors[t.slot].alpha, posteriors[t.slot].beta);
        const postMean = posteriors[t.slot].alpha / (posteriors[t.slot].alpha + posteriors[t.slot].beta);
```

Replace both lines with:

```js
        const post = posteriors[t.slot];
        const sampledMu = sampleNormal(post.mean, post.variance);
        const postMean = post.mean;
```

Then below those lines, find the existing inclusion test that uses `p` and `postMean`. The original test was `if (p >= 0.5) { ... include ... }`. Replace any occurrence of:

```js
if (p >= 0.5)
```

with:

```js
if (sampledMu >= 0)
```

(Sampled μ_f ≥ 0 means "this feature's posterior says it's beneficial in this draw," which is the normal-normal analog of Beta-Thompson p ≥ 0.5.)

- [ ] **Step 6: Replace the alpha/beta ratchet block with a normal-normal posterior update.**

Find this block (anchor on `if (d.delta > POS_THRESHOLD)`):

```js
            const post = posteriors[d.slot];
            if (!post) continue;
            // Skip posterior update on noise-regime cycles. Delta archive
            // still grows; the post-run median naturally absorbs noise by
            // centering on zero for truly noisy features.
            if (baselineIsNoise) continue;
            if (d.delta > POS_THRESHOLD) {
              post.alpha += 1;
              posteriorUpdates.push({ slot: d.slot, verdict: "helpful", delta: d.delta });
            } else if (d.delta < NEG_THRESHOLD) {
              post.beta += 1;
              posteriorUpdates.push({ slot: d.slot, verdict: "harmful", delta: d.delta });
            }
          }
        }
```

Replace it with:

```js
            const post = posteriors[d.slot];
            if (!post) continue;
            // Clip absurd Δs (likely a botched fold, not feature signal)
            // before they pull the posterior. ±0.10 covers any plausible
            // single-feature ablation effect.
            const clipped = Math.max(-0.10, Math.min(0.10, d.delta));
            post.deltas.push(clipped);
            // Recompute σ_obs from all Δs accumulated so far across all
            // features. With <10 total observations the estimator falls
            // back to 0.02; thereafter it uses the pooled empirical SD.
            const allDeltas = Object.values(posteriors)
              .flatMap(p => (p.deltas ?? []))
              .filter(Number.isFinite);
            const sigmaObs = estimateSigmaObs(allDeltas);
            const updated = normalNormalPosterior(post.deltas, PRIOR_SIGMA, sigmaObs);
            post.mean = updated.mean;
            post.variance = updated.variance;
            post.n = updated.n;
            posteriorUpdates.push({
              slot: d.slot,
              verdict: clipped > 0 ? "helpful" : clipped < 0 ? "harmful" : "neutral",
              delta: clipped,
              mean: updated.mean,
              n: updated.n,
            });
          }
        }
```

Note: this drops the `baselineIsNoise` early-exit because we no longer have a discretized ratchet that gets corrupted by one noisy cycle. The clipping at ±0.10 plus the conjugate update is robust to a single bad cycle without needing the gate.

- [ ] **Step 7: Run the build to verify the file compiles.**

Run: `cd /root/Trading && npm run build`
Expected: build succeeds with no missing-import or undefined-symbol errors. Common failure modes: forgot to remove `BASELINE_QUALITY_MIN` reference downstream — search and remove.

- [ ] **Step 8: Run the test suite (no new tests yet, but it confirms nothing got broken).**

Run: `cd /root/Trading && npm test`
Expected: all tests pass.

- [ ] **Step 9: Commit.**

```bash
cd /root/Trading
git add src/App.jsx
git commit -m "feat(App): switch per-cycle Thompson sampling to normal-normal posterior

Replaces the Beta(alpha, beta) per-feature posterior with a closed-form
normal-normal update of mu_f from the paired Δs accumulated across
cycles. Per-cycle inclusion now samples mu_f from N(meanHat, tauSquared)
and includes the slot if the sample is positive — the natural analog
of Beta-Thompson p>=0.5. Δs are clipped to ±0.10 before update for
robustness to a single bad cycle; sigma_obs is recomputed from pooled
empirical SD once at least 10 observations exist."
```

---

### Task 6: Replace post-run verdict computation in `App.jsx`

**Files:**
- Modify: `/root/Trading/src/App.jsx` (post-run verdict block — anchor on `dropMedianEvidence`)

- [ ] **Step 1: Locate the verdict block.**

Run: `grep -n "dropMedianEvidence\|dropPosteriorEvidence\|verdicts.push" /root/Trading/src/App.jsx`
Expected: matches around line ~2296 (verdict computation) and ~2314 (push).

- [ ] **Step 2: Replace the verdict computation block.**

Find this block (anchor on `const obs = allDeltasPerSlot[t.slot]`):

```js
      const obs = allDeltasPerSlot[t.slot].filter(Number.isFinite);
      const n = obs.length;
      const median = n === 0 ? null
        : (() => {
          const s = obs.slice().sort((a, b) => a - b);
          const m = Math.floor(s.length / 2);
          return s.length % 2 ? s[m] : (s[m-1] + s[m]) / 2;
        })();
      const postMean = posteriors[t.slot].alpha / (posteriors[t.slot].alpha + posteriors[t.slot].beta);

      // Binary decision, asymmetric ("innocent until proven guilty").
      // DROP requires BOTH the median delta AND the posterior mean to
      // show meaningful negative evidence — not just hair-below-zero.
      // Without this, noise medians like -0.005 on 6 observations were
      // enough to drop a feature, leading to the pathology where every
      // feature gets DROP because 6 noise samples cluster below zero
      // for everything.
      //
      // Thresholds tuned to 5-fold WF noise floor (std-dev ~0.03 AUC
      // per fold average → ~0.015 std-err across 5 folds → require the
      // observed median to clear ~1.5σ before counting as evidence).
      const DROP_MEDIAN_MAX = -0.02;      // median must be below this
      const DROP_POSTERIOR_MAX = 0.45;     // posterior mean must be below this
      const dropMedianEvidence = (median ?? 0) < DROP_MEDIAN_MAX;
      const dropPosteriorEvidence = postMean < DROP_POSTERIOR_MAX;
      const verdict = (dropMedianEvidence && dropPosteriorEvidence) ? "DROP" : "KEEP";

      // Confidence: HIGH requires both signals strongly past threshold
      // with >= 6 observations (at 20 cycles and ablate-every=3, that's
      // every ablation cycle counting). MED if signals agree but one is
      // weaker. LOW on disagreement or thin evidence.
      const medianPositive = (median ?? 0) > 0;
      const posteriorPositive = postMean > 0.5;
      const strongMedian = Math.abs(median ?? 0) > 0.03;
      const strongPosterior = Math.abs(postMean - 0.5) > 0.15;
      const confidence = (n >= 6 && strongMedian && strongPosterior && (medianPositive === posteriorPositive))
        ? "HIGH"
        : (n >= 4 && (strongMedian || strongPosterior))
          ? "MED"
          : "LOW";

      verdicts.push({
        slot: t.slot,
        name: t.name,
        verdict,
        confidence,
        median,
        postMean,
        n,
      });
    }
```

Replace it with:

```js
      const post = posteriors[t.slot];
      const obs = (post?.deltas ?? []).filter(Number.isFinite);
      const n = obs.length;
      const median = n === 0 ? null
        : (() => {
          const s = obs.slice().sort((a, b) => a - b);
          const m = Math.floor(s.length / 2);
          return s.length % 2 ? s[m] : (s[m-1] + s[m]) / 2;
        })();
      const postSnapshot = post
        ? { mean: post.mean, variance: post.variance, n: post.n }
        : { mean: 0, variance: PRIOR_SIGMA * PRIOR_SIGMA, n: 0 };
      const pNegValue = pNeg(postSnapshot);
      const tier = mapTier(postSnapshot.n, pNegValue);

      verdicts.push({
        slot: t.slot,
        name: t.name,
        verdict: tier,           // "DROP" | "WATCH" | "KEEP" | "INSUFFICIENT"
        pNeg: pNegValue,         // posterior tail probability — UI confidence number
        median,                  // retained for the sparkline render
        postMean: postSnapshot.mean,
        postVariance: postSnapshot.variance,
        n: postSnapshot.n,
      });
    }
```

- [ ] **Step 3: Replace the verdict sort key.**

Find:

```js
    // Sort: KEEP with HIGH confidence first, then KEEP LOW, DROP LOW, DROP HIGH
    const sortKey = (v) => {
      const baseScore = v.verdict === "KEEP" ? 0 : 2;
      const confBoost = v.confidence === "HIGH" ? 0 : v.confidence === "MED" ? 0.5 : 1;
      return v.verdict === "KEEP" ? baseScore + confBoost : baseScore + (1 - confBoost);
    };
    verdicts.sort((a, b) => sortKey(a) - sortKey(b));
```

Replace with:

```js
    // Sort: most actionable items first.
    //   DROP-high → DROP-low → WATCH-high → WATCH-low → KEEP-low → KEEP-high → INSUFFICIENT.
    // Within a tier, items closer to the decision boundary are less
    // certain — placing the high-pNeg DROPs and the high-pNeg KEEPs
    // (i.e. lowest pNeg KEEPs) at the EXTREMES makes the "what should
    // I act on?" question read top-to-bottom.
    const tierOrder = (t) =>
      t === TIER.DROP ? 0 :
      t === TIER.WATCH ? 1 :
      t === TIER.KEEP ? 2 : 3;
    verdicts.sort((a, b) => {
      const dt = tierOrder(a.verdict) - tierOrder(b.verdict);
      if (dt !== 0) return dt;
      // Within a tier: higher pNeg first for DROP/WATCH (more confident
      // it's bad), lower pNeg first for KEEP (still mostly trusted).
      if (a.verdict === TIER.KEEP) return a.pNeg - b.pNeg;
      return b.pNeg - a.pNeg;
    });
```

- [ ] **Step 4: Run the build.**

Run: `cd /root/Trading && npm run build`
Expected: build succeeds. If `confidence` or `dropMedianEvidence` references remain elsewhere in the file, search and adjust — they should not exist after this task.

- [ ] **Step 5: Run tests.**

Run: `cd /root/Trading && npm test`
Expected: all tests pass.

- [ ] **Step 6: Commit.**

```bash
cd /root/Trading
git add src/App.jsx
git commit -m "feat(App): three-tier post-run verdict from normal-normal posterior tail

Replaces the AND-of-thresholds DROP rule (median<-0.02 AND postMean<0.45)
that had parked every feature in KEEP. New rule:
  pNeg = P(mu_f < 0 | data) from the posterior CDF
  tier = INSUFFICIENT (n<4) | DROP (pNeg>0.85) | WATCH (>0.55) | KEEP

Calibrated by construction: under mu_f=0 the pNeg is uniform on [0,1],
giving expected DROP/WATCH/KEEP ≈ 15/30/55%. Sort order rationalised
to put most-actionable items at the top."
```

---

### Task 7: UI render — three tiers + confidence number + INSUFFICIENT

**Files:**
- Modify: `/root/Trading/src/App.jsx` (verdict row render — anchor on `v.verdict === "KEEP" ? "#2ECC71"`)

- [ ] **Step 1: Locate the render block.**

Run: `grep -n 'verdict === "KEEP"' /root/Trading/src/App.jsx`
Expected: a match at ~line 3896.

- [ ] **Step 2: Replace the colour mapping and add a confidence number.**

Find:

```js
                                  const col = v.verdict === "KEEP" ? "#2ECC71" : "#E74C3C";
```

Replace with:

```js
                                  const col =
                                    v.verdict === TIER.KEEP ? "#2ECC71"
                                    : v.verdict === TIER.WATCH ? "#C9A84C"
                                    : v.verdict === TIER.DROP ? "#E74C3C"
                                    : "#888";  // INSUFFICIENT
                                  const confPct = v.verdict === TIER.INSUFFICIENT
                                    ? null
                                    : Math.round((v.pNeg ?? 0) * 100);
```

- [ ] **Step 3: Render the verdict label with the confidence percentage.**

In the same row-render block (immediately below the line you just edited; you should see JSX that uses `v.verdict` as a label string), find the JSX that displays the verdict badge. It looks something like:

```jsx
                                  <span style={{color:col,fontWeight:700}}>{v.verdict}</span>
```

Replace it with:

```jsx
                                  <span style={{color:col,fontWeight:700}}>
                                    {v.verdict}{confPct != null ? ` — ${confPct}%` : ""}
                                  </span>
                                  {v.n != null && (
                                    <span style={{color:"#555",fontSize:8,marginLeft:6}}>n={v.n}</span>
                                  )}
```

If your local render uses different surrounding markup, preserve the parent element and just substitute the verdict-display span as shown above.

- [ ] **Step 4: Build.**

Run: `cd /root/Trading && npm run build`
Expected: build succeeds.

- [ ] **Step 5: Manual smoke test.**

Run the dev server in the background:

```bash
cd /root/Trading && npm run dev
```

Open the URL printed by Vite in a browser. Navigate to the MODEL tab, run a few continuous-train cycles with ABLATE_EVERY=3 (the existing default). After the run completes:

- Confirm at least one feature row shows `INSUFFICIENT` while n<4.
- Confirm at least one feature row shows `KEEP — XX%` once n>=4.
- Confirm WATCH rows render in amber, DROP in red, KEEP in green, INSUFFICIENT in grey.

Stop the dev server (Ctrl+C in the npm run dev shell).

- [ ] **Step 6: Commit.**

```bash
cd /root/Trading
git add src/App.jsx
git commit -m "feat(App): render three-tier verdict with confidence% and n annotation

WATCH rows show amber, DROP red, KEEP green, INSUFFICIENT grey.
Confidence number is pNeg*100 rounded — answers 'how sure is this
DROP?' without forcing the user to read the spec."
```

---

### Task 8: Apply-verdict — narrow mask union to DROP only

**Files:**
- Modify: `/root/Trading/src/App.jsx` (apply-verdict block — anchor on `verdicts.filter(v => v.verdict === "DROP")`)

- [ ] **Step 1: Locate.**

Run: `grep -n 'newDropSlots\|verdict === "DROP"' /root/Trading/src/App.jsx`
Expected: a match around line 2338 building `newDropSlots`.

- [ ] **Step 2: Confirm the existing filter already targets only DROP.**

The existing code is:

```js
    const newDropSlots = new Set(verdicts.filter(v => v.verdict === "DROP").map(v => v.slot));
```

Since `v.verdict` is now one of `"DROP" | "WATCH" | "KEEP" | "INSUFFICIENT"`, this filter already narrows correctly to DROP only — WATCH and INSUFFICIENT are excluded. **No code change is needed if this single line is the only consumer.**

Run: `grep -n "newDropSlots" /root/Trading/src/App.jsx`
Expected: 2-3 references to `newDropSlots` in the same neighbourhood. Confirm they all do union-with-existing-mask, not anything more elaborate.

- [ ] **Step 3 (if step 2 turned up additional references): replace any that re-derive the drop set without the filter.**

Any line of the form `verdicts.map(v => v.slot)` (without filter) inside the apply-verdict path needs the same `.filter(v => v.verdict === "DROP")` interposed. If no such line exists, skip.

- [ ] **Step 4: Build.**

Run: `cd /root/Trading && npm run build`
Expected: build succeeds.

- [ ] **Step 5: Commit only if Step 3 changed anything.**

```bash
cd /root/Trading
git diff --stat src/App.jsx
# If changes are non-zero:
git add src/App.jsx
git commit -m "fix(App): apply-verdict mask union excludes WATCH and INSUFFICIENT

WATCH is informational-only and INSUFFICIENT is below the n>=4 evidence
threshold. Only DROP slots get added to the persistent mask."
```

If Step 3 made no changes, this task is verifying-only — no commit required.

---

### Task 9: Cleanup unused symbols

**Files:**
- Modify: `/root/Trading/src/App.jsx`

- [ ] **Step 1: Search for orphan references.**

Run:

```bash
cd /root/Trading
for sym in PRIOR_ALPHA PRIOR_BETA sampleBeta POS_THRESHOLD NEG_THRESHOLD BASELINE_QUALITY_MIN GAP_QUALITY_MAX dropMedianEvidence dropPosteriorEvidence baselineIsNoise; do
  echo "=== $sym ==="
  grep -n "$sym" src/App.jsx || echo "  (none)"
done
```

Expected: most should print `(none)`. If any remain, those are leftover references to the deleted symbols.

- [ ] **Step 2: Remove any remaining references.**

For each symbol that still appears, locate the surrounding block and decide whether the line still has meaning without the symbol:

- If the line was the symbol's declaration: delete the line.
- If the line was guarded by the symbol (e.g. `if (baselineIsNoise) continue;`): the guard is no longer needed (Task 5 removed it). Delete the line.
- If the line referenced the symbol's value (e.g. `const x = POS_THRESHOLD * 2;`): inline the literal value the symbol previously held, OR — if the surrounding code is also dead — delete the block.

When unsure, run `git blame` on the line to read the original intent before deleting.

- [ ] **Step 3: Re-run the orphan search.**

Repeat the loop from Step 1. Every entry should now print `(none)`.

- [ ] **Step 4: Build.**

Run: `cd /root/Trading && npm run build`
Expected: build succeeds.

- [ ] **Step 5: Run tests.**

Run: `cd /root/Trading && npm test`
Expected: all tests pass.

- [ ] **Step 6: Commit.**

```bash
cd /root/Trading
git add src/App.jsx
git commit -m "chore(App): remove unused alpha/beta-era constants and helpers

PRIOR_ALPHA, PRIOR_BETA, sampleBeta, POS_THRESHOLD, NEG_THRESHOLD,
BASELINE_QUALITY_MIN, GAP_QUALITY_MAX and the baselineIsNoise gate
are no longer referenced after the switch to normal-normal posterior."
```

---

### Task 10: Build, deploy, and manual end-to-end smoke

**Files:**
- None modified — verification only.

- [ ] **Step 1: Final clean build.**

Run: `cd /root/Trading && npm run build 2>&1 | tail -20`
Expected: build succeeds, dist/index.html and dist/assets/index-*.js update.

- [ ] **Step 2: Confirm dist is fresh.**

Run: `stat -c "%y %n" /root/Trading/dist/index.html /root/Trading/src/App.jsx`
Expected: dist/index.html mtime is later than src/App.jsx mtime.

- [ ] **Step 3: Open the served URL and verify three-tier behaviour.**

The frontend is served by nginx at `http://5.161.246.161` (per project memory). Open it in a browser. Navigate to the MODEL tab. Run a continuous-train cycle of ~10 cycles. After completion:

- At least one feature row shows `INSUFFICIENT` if any feature accumulated <4 Δs.
- Feature rows show colours per Task 7 (green/amber/red/grey).
- Each non-INSUFFICIENT row shows `<TIER> — NN%` with `n=N` annotation.
- Sort order: DROP at top, then WATCH, then KEEP, INSUFFICIENT last.
- Apply-verdict button still works; clicking it adds only DROP slots to the mask.

- [ ] **Step 4: Final all-tests run.**

Run: `cd /root/Trading && npm test`
Expected: all tests pass (no regressions).

- [ ] **Step 5: No commit — verification only.**

This task confirms end-to-end behaviour and produces no diffs. The implementation is complete.

---

## Self-review notes

- **Spec coverage.** Tasks 2–7 implement the spec components 1, 2, 3, 4, 5 in order. Task 8 covers component 5 (apply-verdict). Task 9 covers component 6 (migration cleanup). Tasks 2 and 3 cover all the unit-test acceptance criteria from §"Goal" and §"Testing". Task 4 covers the seeded-pairing variance reduction logged-smoke (acceptance #4 — recorded, no hard threshold per spec).
- **Type/name consistency.** `TIER` is exported from `posteriorVerdict.js` and imported in App.jsx. `mapTier` returns members of `TIER`. UI render and sort key compare against `TIER.*`. All consistent.
- **No placeholders.** Every step has either exact code or a precise diagnostic command. The one judgment-call step (Task 9 Step 2) carries explicit decision rules.
- **Spec drift addressed.** The spec did not explicitly cover the per-cycle Thompson sampling that the alpha/beta posterior also drove. Task 5 replaces it with the natural normal-normal analog — sample μ_f, include if positive — and documents this in the commit message. Without this, dropping `sampleBeta` would have left the cycle-to-cycle exploration undefined.
