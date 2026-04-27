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
