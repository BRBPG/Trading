// ─── Calendar features ──────────────────────────────────────────────────────
// Documented intraday/calendar effects per the literature review:
//   - FOMC drift (Lucca & Moench 2015, JF): equity returns unusually high in
//     the 24h pre-FOMC. ~50 bps/meeting effect.
//   - OpEx week: 3rd-Friday option expiry. Vol compresses pre-, expands post-.
//   - U-shaped intraday volume/volatility: open and close bring spikes.
//   - Day-of-week: mild Monday effect, stronger Friday reversal tendency.
//
// These aren't silver bullets — each is a small edge — but they're free to
// encode and feed into the model as additional features.

// FOMC meeting dates 2024-2026 (scheduled). Source: Federal Reserve calendar.
// Dates are ISO (YYYY-MM-DD) of the second day of each 2-day meeting.
const FOMC_DATES = [
  "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
  "2024-09-18", "2024-11-07", "2024-12-18",
  "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
  "2025-09-17", "2025-10-29", "2025-12-10",
  "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
  "2026-09-16", "2026-10-28", "2026-12-16",
];

// Distance in days from now to the nearest FOMC meeting (signed: negative =
// past, positive = upcoming). The pre-FOMC drift effect is ±2 days.
export function fomcDistanceDays(now = new Date()) {
  const nowMs = now.getTime();
  let bestAbs = Infinity, bestSigned = null;
  for (const d of FOMC_DATES) {
    const diffDays = (new Date(d + "T14:00:00Z").getTime() - nowMs) / 86_400_000;
    if (Math.abs(diffDays) < bestAbs) {
      bestAbs = Math.abs(diffDays);
      bestSigned = diffDays;
    }
  }
  return bestSigned;
}

// Feature: 1.0 within 2 days before an FOMC meeting, else 0. Captures the
// Lucca-Moench pre-FOMC drift window.
export function isPreFomcWindow(now = new Date()) {
  const d = fomcDistanceDays(now);
  return d != null && d > 0 && d <= 2 ? 1 : 0;
}

// Third-Friday OpEx detection. Monthly equity options expire the 3rd Friday.
// Returns 1 if we're in OpEx week (Mon-Fri of that week), else 0.
export function isOpExWeek(now = new Date()) {
  const y = now.getUTCFullYear();
  const m = now.getUTCMonth();
  // Find third Friday: first Friday + 14 days.
  const first = new Date(Date.UTC(y, m, 1));
  const firstFridayOffset = (5 - first.getUTCDay() + 7) % 7;
  const thirdFridayDate = 1 + firstFridayOffset + 14;
  const third = new Date(Date.UTC(y, m, thirdFridayDate));
  // OpEx week = Monday before the 3rd Friday up to that Friday.
  const monday = new Date(third);
  monday.setUTCDate(third.getUTCDate() - 4);
  return now >= monday && now <= third ? 1 : 0;
}

// Time-of-day bucket as a [0,1] position within the regular US session
// (09:30–16:00 ET = 390 minutes). Before open = 0, after close = 1. Used
// raw as a feature; the NN can learn non-linear shape (U-curve etc).
export function timeOfDayPosition(now = new Date()) {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone: "America/New_York",
      hour: "numeric", minute: "numeric", hour12: false,
    }).formatToParts(now).map(p => [p.type, p.value])
  );
  const h = parseInt(parts.hour, 10) % 24;
  const m = parseInt(parts.minute, 10);
  const mins = h * 60 + m;
  const open = 9 * 60 + 30, close = 16 * 60;
  if (mins <= open)  return 0;
  if (mins >= close) return 1;
  return (mins - open) / (close - open);
}

// Distance from nearest session edge (0 = edge, 0.5 = midday). The U-shaped
// intraday activity profile means edges are structurally different from
// midday — encoding as "edge-ness" gives the model a simpler signal than
// raw position.
export function edgeness(now = new Date()) {
  const p = timeOfDayPosition(now);
  if (p <= 0 || p >= 1) return 1;
  return Math.min(p, 1 - p) / 0.5; // 0 at edges, 1 at midday
}

// Day-of-week as one-hot isn't efficient in a tiny LR; encode as a single
// scalar: Monday=0, Tuesday=0.2, ..., Friday=0.8. Lets the LR pick up any
// monotonic weekday drift. (Not a strong signal but free to include.)
export function dayOfWeekScalar(now = new Date()) {
  const d = now.getDay();
  if (d === 0 || d === 6) return 0; // weekend
  return (d - 1) / 4;
}

// Bundle all calendar features into one object for consumption by the
// feature extractor. Keep it flat to match macro module's shape.
export function calendarFeatures(now = new Date()) {
  return {
    preFomc:  isPreFomcWindow(now),
    opexWeek: isOpExWeek(now),
    todPos:   timeOfDayPosition(now),
    todEdge:  edgeness(now),
    dowScale: dayOfWeekScalar(now),
  };
}

// Same shape but point-in-time for a given timestamp (in seconds) —
// used by the backtest when walking historical bars.
export function calendarFeaturesAt(timestampSec) {
  return calendarFeatures(new Date(timestampSec * 1000));
}
