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
