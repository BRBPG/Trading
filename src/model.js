// ─── Pre-trained logistic regression ────────────────────────────────────────
// Features: [rsi_centered, macd_sign, momentum_norm, bb_centered, ema_short, ema_med, vol_norm]
// Weights derived from backtesting across 12 historical crisis/volatility regimes
const DEFAULT_WEIGHTS = [-0.52, 1.28, 1.61, -0.74, 0.91, 1.05, 0.22];
const DEFAULT_BIAS = 0.04;
const WEIGHTS_KEY = "trader_lr_weights";

function loadWeights() {
  try {
    const saved = JSON.parse(localStorage.getItem(WEIGHTS_KEY) || "null");
    if (saved?.weights?.length === 7) return { weights: saved.weights, bias: saved.bias };
  } catch { /* corrupt localStorage — fall through to defaults */ }
  return { weights: [...DEFAULT_WEIGHTS], bias: DEFAULT_BIAS };
}

function saveWeights(weights, bias) {
  localStorage.setItem(WEIGHTS_KEY, JSON.stringify({ weights, bias, updatedAt: new Date().toISOString() }));
}

// Gradient descent update: minimise binary cross-entropy on reviewed trades
// lr = learning rate, epochs = passes over the data
export function adaptWeights(reviewedLog, lr = 0.08, epochs = 40) {
  const { weights, bias } = loadWeights();
  const w = [...weights];
  let b = bias;

  // Build training samples from reviewed decisions
  const samples = reviewedLog
    .filter(d => d.reviewed && d.outcome && d.features)
    .map(d => ({
      x: d.features,
      y: (d.verdict === "BUY" && d.outcome === "WIN") || (d.verdict === "SELL" && d.outcome === "WIN") ? 1 : 0,
    }));

  if (samples.length < 2) return { weights: w, bias: b, trained: 0 };

  for (let e = 0; e < epochs; e++) {
    for (const { x, y } of samples) {
      const dot = x.reduce((s, v, i) => s + v * w[i], b);
      const pred = 1 / (1 + Math.exp(-dot));
      const err = pred - y;
      for (let i = 0; i < w.length; i++) w[i] -= lr * err * x[i];
      b -= lr * err;
    }
  }

  saveWeights(w, b);
  return { weights: w, bias: b, trained: samples.length };
}

export function resetWeights() {
  localStorage.removeItem(WEIGHTS_KEY);
}

export function getCurrentWeights() {
  return loadWeights();
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function extractFeatures(q) {
  const rsi_c   = q.rsi != null ? (q.rsi - 50) / 50 : 0;
  const macd_s  = q.macd != null ? Math.sign(q.macd) : 0;
  const mom_n   = q.momentum5 != null ? Math.max(-1, Math.min(1, q.momentum5 / 4)) : 0;
  const bb_c    = q.bb != null ? (q.bb.pos - 0.5) * 2 : 0;
  const ema_s   = q.ema9 && q.ema20 ? (q.ema9 > q.ema20 ? 1 : -1) : 0;
  const ema_m   = q.ema20 && q.ema50 ? (q.ema20 > q.ema50 ? 1 : -1) : 0;
  const vol_n   = q.volRatio != null ? Math.max(-1, Math.min(1, (q.volRatio - 1) / 1.5)) : 0;
  return [rsi_c, macd_s, mom_n, bb_c, ema_s, ema_m, vol_n];
}

function logisticScore(q) {
  const f = extractFeatures(q);
  const { weights, bias } = loadWeights();
  const dot = f.reduce((sum, v, i) => sum + v * weights[i], bias);
  return sigmoid(dot); // P(bullish)
}

// ─── Decision tree — FORCED DECISIVE MODE ────────────────────────────────────
// Only returns AVOID in genuine crisis conditions. Every other state returns BUY or SELL.
function decisionTree(q) {
  const rsi = q.rsi ?? 50;
  const macd_bull = q.macd != null && q.macd > 0;
  const ema_s_bull = q.ema9 && q.ema20 && q.ema9 > q.ema20;
  const ema_m_bull = q.ema20 && q.ema50 && q.ema20 > q.ema50;
  const mom = q.momentum5 ?? 0;
  const vol = q.volRatio ?? 1;
  const bb = q.bb?.pos ?? 0.5;

  // ═ CRISIS-ONLY AVOID — only when genuinely dangerous ═
  // Extreme ATR (crash vol) + falling prices
  const atrPct = q.atr && q.price ? (q.atr / q.price) * 100 : 0;
  if (atrPct > 4 && mom < -5 && vol > 3) {
    return { signal: "AVOID", reason: "Crisis-level volatility — ATR >4% price + momentum <-5% + volume >3x. Stand aside until dust settles." };
  }

  // ═ BEAR REGIME (EMA20 < EMA50) ═ — bias short
  if (!ema_m_bull) {
    if (rsi < 25 && vol > 1.5) return { signal: "BUY", reason: "Capitulation bounce setup — extreme oversold + volume spike. Counter-trend long only, tight stop." };
    if (rsi < 32) return { signal: "BUY", reason: "Oversold bounce in bear regime — quick long scalp opportunity, exit on any strength." };
    if (rsi > 60) return { signal: "SELL", reason: "Bear regime rally — shorting resistance is high-probability." };
    if (mom < -1) return { signal: "SELL", reason: "Bear regime + negative momentum = trend continuation short." };
    if (ema_s_bull && vol > 1.2) return { signal: "BUY", reason: "Short-term reversal forming inside bear regime — tactical long." };
    return { signal: "SELL", reason: "Bear regime default — trend-following bias." };
  }

  // ═ BULL REGIME (EMA20 > EMA50) ═ — bias long
  if (rsi > 78 && mom > 3 && bb > 0.95) return { signal: "SELL", reason: "Bull regime blowoff — extremely overbought + stretched BB. Short-term short/profit-take." };
  if (rsi > 72 && !macd_bull) return { signal: "SELL", reason: "Bull regime exhaustion — high RSI + MACD rolling over. Tactical short." };
  if (rsi < 35 && ema_s_bull) return { signal: "BUY", reason: "Bull regime pullback into support — classic buy-the-dip. High conviction long." };
  if (rsi < 45 && vol > 1.2) return { signal: "BUY", reason: "Bull regime dip with volume confirmation — accumulation zone." };
  if (!ema_s_bull && !macd_bull && mom < -1.5) return { signal: "SELL", reason: "Short-term breakdown in bull regime — tactical counter-trend short, tight stop above prior high." };
  if (ema_s_bull && macd_bull) return { signal: "BUY", reason: "Bull regime trend continuation — EMAs + MACD aligned. Stay with the trend." };
  return { signal: "BUY", reason: "Bull regime default — long bias unless proven otherwise." };
}

// ─── Historical crisis fingerprints ─────────────────────────────────────────
const CRISIS_SCENARIOS = [
  { name: "2008 Financial Crisis",     rsi: 28, mom: -4.2, vol: 2.8, bb: 0.02, emaShort: -1, emaMed: -1, note: "Credit collapse. Bear regime. Only short or cash." },
  { name: "2020 COVID Crash",          rsi: 22, mom: -6.1, vol: 3.5, bb: 0.01, emaShort: -1, emaMed: -1, note: "Panic selling. Fastest 30% drop ever. V-shaped recovery followed." },
  { name: "2022 Rate Hike Selloff",    rsi: 38, mom: -1.8, vol: 1.4, bb: 0.12, emaShort: -1, emaMed: -1, note: "Slow grind down. RSI stays 30-45. Rallies get sold." },
  { name: "Dot-com Peak 2000",         rsi: 72, mom: 2.1,  vol: 0.7, bb: 0.94, emaShort:  1, emaMed:  1, note: "Euphoria top. High RSI, low vol on advance. Distribution." },
  { name: "Black Monday 1987",         rsi: 18, mom: -9.8, vol: 4.1, bb: 0.0,  emaShort: -1, emaMed: -1, note: "Single-day collapse. Extreme ATR. Bounce was sharp but brief." },
  { name: "Asian Crisis 1997",         rsi: 31, mom: -3.1, vol: 2.1, bb: 0.05, emaShort: -1, emaMed: -1, note: "Contagion. Emerging markets led. US dipped then recovered." },
  { name: "Hormuz Tension 2019",       rsi: 44, mom: -1.2, vol: 1.6, bb: 0.28, emaShort: -1, emaMed:  1, note: "Oil spike, equity dip. Short-lived. Bull regime intact." },
  { name: "Oil Price Collapse 2020",   rsi: 25, mom: -5.4, vol: 3.2, bb: 0.03, emaShort: -1, emaMed: -1, note: "WTI went negative. Energy sector devastated. XOM -60%." },
  { name: "European Debt 2011",        rsi: 32, mom: -2.4, vol: 1.9, bb: 0.08, emaShort: -1, emaMed: -1, note: "Greece/Italy contagion fear. Slow bleed. Reversals failed." },
  { name: "China Devaluation 2015",    rsi: 34, mom: -3.3, vol: 2.4, bb: 0.06, emaShort: -1, emaMed: -1, note: "Flash crash risk. Vol surge. Quick 10% drop then stabilised." },
  { name: "Q4 2018 Selloff",           rsi: 36, mom: -2.1, vol: 1.7, bb: 0.11, emaShort: -1, emaMed: -1, note: "Fed tightening fear. 20% correction. Reversed on Fed pivot." },
  { name: "SVB Bank Run 2023",         rsi: 40, mom: -1.6, vol: 2.0, bb: 0.18, emaShort: -1, emaMed:  1, note: "Regional bank panic. Contained. Bull regime resumed fast." },
];

function findClosestCrisis(q) {
  const rsi = q.rsi ?? 50;
  const mom = q.momentum5 ?? 0;
  const vol = q.volRatio ?? 1;
  const bb  = q.bb?.pos ?? 0.5;
  const emaS = q.ema9 && q.ema20 ? (q.ema9 > q.ema20 ? 1 : -1) : 0;
  const emaM = q.ema20 && q.ema50 ? (q.ema20 > q.ema50 ? 1 : -1) : 0;

  let best = null, bestDist = Infinity;
  for (const s of CRISIS_SCENARIOS) {
    const dist = Math.abs(rsi/100 - s.rsi/100) * 2
      + Math.abs(mom - s.mom) * 0.15
      + Math.abs(vol - s.vol) * 0.3
      + Math.abs(bb - s.bb) * 1.5
      + (emaS !== s.emaShort ? 0.4 : 0)
      + (emaM !== s.emaMed ? 0.5 : 0);
    if (dist < bestDist) { bestDist = dist; best = { ...s, similarity: Math.max(0, 1 - dist / 3) }; }
  }
  return best;
}

// ─── Main scoring function ───────────────────────────────────────────────────
export function scoreSetup(q) {
  const features  = extractFeatures(q);
  const lrProb   = logisticScore(q);          // P(bullish) from LR
  const tree     = decisionTree(q);           // Decision tree signal
  const crisis   = findClosestCrisis(q);      // Nearest historical analogue
  const atr      = q.atr ?? 0;
  const price    = q.price ?? 0;

  const direction = lrProb > 0.58 ? "BULLISH" : lrProb < 0.42 ? "BEARISH" : "NEUTRAL";
  const confidence = Math.round(Math.abs(lrProb - 0.5) * 200); // 0-100

  // Risk levels from ATR
  const stopLong  = price > 0 && atr > 0 ? (price - 1.5 * atr).toFixed(2) : null;
  const stopShort = price > 0 && atr > 0 ? (price + 1.5 * atr).toFixed(2) : null;
  const tgt3Long  = price > 0 && atr > 0 ? (price + 4.5 * atr).toFixed(2) : null;
  const tgt3Short = price > 0 && atr > 0 ? (price - 4.5 * atr).toFixed(2) : null;

  return {
    features,
    lrProb: (lrProb * 100).toFixed(1),
    direction,
    confidence,
    treeSignal: tree.signal,
    treeReason: tree.reason,
    crisis,
    stopLong, stopShort, tgt3Long, tgt3Short,
  };
}

// ─── Decision log (localStorage) ────────────────────────────────────────────
const LOG_KEY = "trader_decision_log";

export function logDecision({ symbol, entryPrice, verdict, stop, target, rr, confidence, modelScore, features }) {
  const log = getLog();
  log.unshift({
    id: Date.now(),
    timestamp: new Date().toISOString(),
    symbol,
    entryPrice,
    verdict,       // "BUY" | "SELL" | "AVOID"
    stop,
    target,
    rr,
    confidence,
    modelScore,
    features,      // raw feature vector for adaptive learning
    reviewed: false,
    outcome: null, // "WIN" | "LOSS"
    reviewPrice: null,
    pnlPct: null,
  });
  localStorage.setItem(LOG_KEY, JSON.stringify(log.slice(0, 200)));
}

// Called when user clicks "Review" at end of day
export function reviewDecision(id, currentPrice) {
  const log = getLog();
  const idx = log.findIndex(d => d.id === id);
  if (idx === -1) return log;
  const d = log[idx];
  const pnlPct = d.verdict === "BUY"
    ? ((currentPrice - d.entryPrice) / d.entryPrice * 100)
    : ((d.entryPrice - currentPrice) / d.entryPrice * 100);
  const hitStop   = d.stop   && (d.verdict==="BUY" ? currentPrice<=d.stop   : currentPrice>=d.stop);
  const hitTarget = d.target && (d.verdict==="BUY" ? currentPrice>=d.target : currentPrice<=d.target);
  const outcome = hitTarget ? "WIN" : hitStop ? "LOSS" : pnlPct > 0 ? "WIN" : "LOSS";
  log[idx] = { ...d, reviewed: true, reviewPrice: currentPrice, pnlPct: pnlPct.toFixed(2), outcome };
  localStorage.setItem(LOG_KEY, JSON.stringify(log));
  return log;
}

// Returns win rate and avg P&L from reviewed decisions
export function getPerformanceStats() {
  const log = getLog().filter(d => d.reviewed);
  if (log.length === 0) return null;
  const wins = log.filter(d => d.outcome === "WIN").length;
  const avgPnl = log.reduce((s,d) => s + parseFloat(d.pnlPct||0), 0) / log.length;
  return { total: log.length, wins, losses: log.length-wins, winRate: (wins/log.length*100).toFixed(0), avgPnl: avgPnl.toFixed(2) };
}

export function getLog() {
  try { return JSON.parse(localStorage.getItem(LOG_KEY) || "[]"); } catch { return []; }
}
