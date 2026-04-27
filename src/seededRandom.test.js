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
