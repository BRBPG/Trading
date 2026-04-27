# Calibrated Feature-Pruning Verdict Redesign

**Date:** 2026-04-27
**Status:** Design — awaiting user review
**Scope:** Project A only. Project B (AUC inversion / no-edge diagnostic) and Project C (codebase cleanup pass) are tracked separately.

## Problem

The post-run feature-pruning verdict at `src/App.jsx:2386–2451` currently parks **every** feature in `KEEP`. The current rule is:

```js
const verdict = (dropMedianEvidence && dropPosteriorEvidence) ? "DROP" : "KEEP";
```

with `dropMedianEvidence = median Δ < −0.02` and `dropPosteriorEvidence = posteriorMean < 0.45`. Both thresholds sit ~1.5σ outside the empirical noise floor, and they are combined with `AND`, so a feature is dropped only when *both* a median test *and* a Beta-posterior test independently clear stringent cutoffs in the negative direction. With only n=6 ablation observations available per run, that compound condition is essentially never satisfied — even features with genuinely negative effect rarely produce verdicts.

This is the symmetric inversion of the original pathology (everything dropped on noise medians like −0.005 across 6 samples). Neither extreme is useful.

The user's stated goal is **calibrated** verdicts: under zero true effect, the rule should not lock at one corner — and verdicts should give a calibrated *confidence number*, not just a binary. Three tiers (`DROP` / `WATCH` / `KEEP`) with a per-feature posterior-tail-probability are the agreed shape.

## Goal

Replace the current AND-of-thresholds DROP rule with a calibrated three-tier verdict driven by a closed-form posterior over each feature's true mean ablation Δ. As a necessary precondition, switch the within-cycle ablation comparison from "weakly paired" (shared trades and folds, but unseeded model RNG) to "fully paired" (shared trades, folds, AND model RNG seed) so that paired statistics at n ≈ 6 are statistically legitimate.

Acceptance criteria:

1. Under simulated null (μ_f = 0, Δ ~ N(0, σ²)), empirical verdict distribution across many synthetic features lands within ±2% of the target {DROP ≈ 15%, WATCH ≈ 30%, KEEP ≈ 55%}.
2. Under simulated true negative effect (μ_f = −0.03, σ = 0.02, n = 8), DROP rate exceeds 80%.
3. The seeded-RNG wrapper, when applied around a `runWalkForward` invocation, restores the global `Math.random` reference even when the inner callback throws.
4. With seeded paired comparisons, observed σ_obs across cycles drops *meaningfully* versus the unpaired baseline on a fixture replay. The exact fraction is unknown until measured; acceptance is "documented number, sanity-checked, not regressed." A naive expectation is ~20–50% reduction, but this spec does not promise a specific figure.

## Non-goals (deliberate)

- Recency weighting / regime-change adaptation. The posterior accumulates all cycles equally for now; a v2 can add discount factors.
- Multi-feature joint posterior. Features are treated as independent. Defensible at the current ~9-feature scale.
- Auto-application of DROP verdicts. The user-driven `APPLY VERDICT` button stays gated on human review.
- Changes to which features are computed or how. This spec only changes the *evaluation and verdict* path, not feature engineering.
- Anything to do with the underlying AUC < 0.5 problem. That is Project B.

## Architecture

Three components plus migration cleanup. Each is independently testable.

### Component 1 — Seeded RNG infrastructure (new)

**New file:** `src/seededRandom.js`

Two exports:

- `mulberry32(seed: number): () => number` — pure 7-line PRNG, period 2³², range `[0, 1)`.
- `withSeededRandom(seed: number, fn: () => Promise<T>): Promise<T>` — saves the current `Math.random` binding, replaces it with `mulberry32(seed)`, awaits `fn`, then restores the original binding in a `finally` block (so it restores on throw too).

**Edit site:** `src/walkForward.js` `runAblationStudy` (currently at line 429).

For each ablation cycle, derive a single deterministic seed (e.g. hash of cycle index plus a run-level salt). Wrap the baseline `runWalkForward` call AND every per-feature masked `runWalkForward` call in `withSeededRandom(cycleSeed, …)` using the same `cycleSeed` value. This makes:

- `gbm.js:242` — bootstrap row sampling
- `gbm.js:249` — column subsample shuffle
- `nn.js:45` — Box–Muller weight init
- `nn.js:241` — minibatch shuffle

…all produce identical sequences in the baseline call and in each masked call within the same cycle. The only difference between baseline and masked runs is the feature mask itself. Per-feature Δ becomes a true paired difference; most fold-internal stochasticity cancels.

Across cycles the seed advances, so per-cycle Δs remain independent draws — the n=6 sample is still i.i.d. from the per-feature Δ distribution.

### Component 2 — Posterior accumulator (replaces existing Beta machinery)

Replace the per-slot `posteriors[t.slot] = { alpha, beta }` state at `App.jsx:2113` (and the alpha/beta ratchet logic at `~App.jsx:2310`) with a closed-form normal-normal posterior:

```
prior:       μ_f ~ N(0, σ_prior²)               σ_prior = 0.05
likelihood:  Δ_i | μ_f ~ N(μ_f, σ_obs²)
posterior:   μ_f | {Δ_i}_{i=1..n} ~ N(μ̂, τ²)

τ² = 1 / (1/σ_prior² + n/σ_obs²)
μ̂ = τ² · (n · mean(Δ) / σ_obs²)
```

`σ_obs` is estimated **empirically** from the pooled across-feature standard deviation of Δs once a few cycles' worth of data exist. With seeded paired comparisons this number is much smaller than the raw ~0.03 documented in the existing comments, so the posterior is informative even at modest n. Until at least, say, 10 (feature × cycle) Δs are available, fall back to a fixed `σ_obs = 0.02` so early cycles still produce a usable posterior; switch to the empirical estimate once data is sufficient.

State shape: `posteriors[slot] = { mean, variance, n, deltas }`. `deltas` retained for the empirical σ_obs estimate, for diagnostics, and for the existing post-run trimmed-median computation.

### Component 3 — Verdict mapper (replaces `App.jsx:2421` block)

For each feature compute `pNeg = P(μ_f < 0 | data) = Φ(0; μ̂, τ²) = Φ(-μ̂/τ)` from the standard normal CDF.

Tier definitions:

| Tier             | Condition                            |
|------------------|--------------------------------------|
| `INSUFFICIENT`   | `n < 4`                              |
| `DROP`           | `n ≥ 4` and `pNeg > 0.85`            |
| `WATCH`          | `n ≥ 4` and `0.55 < pNeg ≤ 0.85`     |
| `KEEP`           | `n ≥ 4` and `pNeg ≤ 0.55`            |

**Calibration property.** Under the null (μ_f = 0 truly) with a flat prior, `pNeg` is uniform on `[0, 1]` across repeats of the experiment (the standard Bayesian-frequentist correspondence: a well-specified posterior tail is a valid p-value under the null). Hence expected verdict rates in the null world are:

- DROP ≈ 1 − 0.85 = **15%**
- WATCH ≈ 0.85 − 0.55 = **30%**
- KEEP ≈ 0.55 = **55%**

With a *non-flat* prior centred at zero (which we use), `pNeg` is biased slightly toward 0.5 at small n — i.e. mildly conservative on DROP. The acceptance criterion in §"Goal" allows ±2% slack to cover this.

The thresholds 0.55 and 0.85 are tunable knobs: lower the DROP threshold to 0.80 to drop more aggressively, raise to 0.90 to be more conservative. The math doesn't change.

Under a true negative effect of typical real magnitude (e.g. μ_f = −0.03, σ = 0.02, n = 8), `pNeg` rises sharply: this is the non-null behaviour the test in §"Verification" §2 is designed to confirm.

The output object per feature also carries `pNeg`, `μ̂`, `τ`, and `n` so the UI can render a confidence number alongside the tier.

### Component 4 — UI changes

**Edit site:** the verdict sort + render block at `App.jsx:2447–2451` and the per-row render at `~App.jsx:4019`.

- `KEEP` → green (`#2ECC71`, current).
- `WATCH` → amber (`#C9A84C`).
- `DROP` → red (`#E74C3C`, current).
- `INSUFFICIENT` → grey (`#888`).
- Each row shows `"<TIER> — XX%"` where the percentage is `pNeg * 100` rounded to integer.
- A small `n=N` annotation accompanies the row.
- Sort order: KEEP-high-confidence → KEEP-low-confidence → WATCH → DROP-low-confidence → DROP-high-confidence → INSUFFICIENT.

### Component 5 — Apply-verdict semantics

At the post-run preview-retrain logic (~`App.jsx:2451`), the union currently masks `existingMask ∪ dropVerdicts`. After this change, only features in tier `DROP` are added to `dropVerdicts`. `WATCH` is informational only and does not affect the mask. No structural change to the retrain code path; only the input set narrows.

### Component 6 — Migration cleanup (in-scope)

Remove now-unused symbols and their comments:

- `PRIOR_ALPHA`, `PRIOR_BETA`
- `sampleBeta` (and any imports)
- `POS_THRESHOLD`, `NEG_THRESHOLD`
- `BASELINE_QUALITY_MIN`, `GAP_QUALITY_MAX` — but **only after confirming** they aren't referenced elsewhere; the baseline-noise gating logic at `~App.jsx:2300` is partially independent of the verdict logic and may need a smaller equivalent kept.
- The explanatory comment block at `App.jsx:2386–2421` is replaced with a short comment describing the posterior-tail rule.

Anything *not* listed above is left alone. This spec is not authority for unrelated refactors.

## Data flow (after change)

```
runAblationStudy (per cycle)
   ├── pick cycleSeed
   ├── withSeededRandom(cycleSeed, runWalkForward(baselineOpts))
   │     → baselineAUC
   └── for each target feature:
         withSeededRandom(cycleSeed, runWalkForward({...baselineOpts, maskSlots: union(base, target)}))
            → maskedAUC
         delta = baselineAUC − maskedAUC

continuousLoop (each ablation cycle)
   ├── deltas = runAblationStudy(...)
   ├── for each delta:
   │     posteriors[slot].deltas.push(delta)
   │     posteriors[slot].n += 1
   │     update (μ̂, τ²) from posteriors[slot].deltas using current σ_obs estimate
   └── recompute σ_obs from cross-feature pooled SD if n_total ≥ 10

postRunVerdictPass
   for each feature:
      pNeg = normalCDF(-μ̂ / τ)
      tier = mapTier(n, pNeg)
      render row
```

## Testing

Unit tests live alongside their modules. New test files: `src/seededRandom.test.js`, `src/posteriorVerdict.test.js`.

Implementation step zero: check `package.json` and the source tree for an existing test runner (vitest, jest, node:test). If one exists, use it. If none exists, ship the new tests as plain Node scripts (`node src/posteriorVerdict.test.js`) using `node:assert` — **do not** introduce a test framework as part of this spec; that decision belongs to a separate scope.

1. **Seeded RNG round-trip.** `withSeededRandom(42, fn)` followed by `Math.random()` returns a value drawn from the *original* (non-seeded) source. Holds even when `fn` throws.
2. **Seeded RNG determinism.** Two identical `withSeededRandom(42, …)` calls invoking the same inner work produce byte-identical results.
3. **Null calibration.** Generate 1000 synthetic features each with 8 Δs drawn i.i.d. from `N(0, 0.02²)`. Run the verdict mapper. Empirical {DROP, WATCH, KEEP} rates within ±2% of {15%, 30%, 55%}.
4. **Power under true effect.** μ_f = −0.03, σ = 0.02, n = 8 → DROP rate > 80% across 1000 reps.
5. **Insufficient guard.** n = 3 with any data → tier is `INSUFFICIENT`, no `pNeg` rendered.
6. **σ_obs fallback.** With < 10 total Δs in the system, σ_obs uses the fixed 0.02 fallback; with ≥ 10 it switches to empirical pooled SD.
7. **Pairing variance reduction (integration).** Run `runAblationStudy` twice on the same fixture — once with seeded wrappers, once without — and *record* the cross-cycle σ of per-feature Δs in both runs. No hard threshold: this is a logged smoke check that the pairing is measurably tighter, not a numeric gate.

## Verification before completion

- All unit tests pass locally.
- Manual smoke: launch the dev frontend, run a continuous loop with ABLATE_EVERY=3 for ~10 cycles, confirm at least one feature exits `INSUFFICIENT` and lands in a real tier; confirm percentages render.
- Verify the deployed `/root/Trading/dist` reflects the new behaviour after `npm run build`.

## Risks

- **σ_obs misestimation.** If features genuinely have heterogeneous variances, the pooled estimator will under- or over-cover. Mitigation: per-feature variance estimation once n_f ≥ 4. Out of scope for v1; flag for v2.
- **Posterior is normal-normal.** Heavy tails or outlier Δs (e.g. one cycle with a botched fold) will pull the mean. Mitigation: clip Δ to ±0.10 before posterior update, on the basis that any larger move is structural noise rather than feature signal.
- **Seed collisions across runs.** Using `cycle index` alone as seed means run 2 of the day reuses run 1's seeds. Acceptable — independence is wanted *within a run* across cycles, not across runs. But document it.
- **Drift between this spec and Project B.** If Project B reveals the AUC pipeline is structurally broken, the σ_obs empirical estimate is meaningless. The spec still ships value (calibrated logic + paired RNG) but the *numbers* will be uninformative until Project B lands. Document this dependency.

## Out of scope, captured for follow-up

- Project B: AUC < 0.5 root-cause investigation. Held in a parallel diagnostic thread.
- Project C: codebase contradictions/cruft cleanup. Held in a parallel automated pass.
- v2 of this spec: recency-weighted posteriors, joint feature posteriors, auto-apply with audit log.

## Open questions

None. All clarifications resolved during brainstorming:

- Goal style: calibrated posterior with three tiers (user choice **C**).
- Pairing strategy: weak today, made fully paired via seeded RNG (user requested code investigation; confirmed).
- Scope split: Project A standalone; B and C parallel (user choice **A**).
