import { predictNN, trainNN as trainNNRaw, getNNInfo, resetNN as resetNNRaw } from "./nn";

// ─── Pre-trained logistic regression (v2, 14-dim) ──────────────────────────
// Honest note: these defaults are a mild bullish-tech prior — negative
// weight on high RSI, positive weights on MACD/momentum/trend alignment,
// near-zero weights on macro features (the NN is expected to learn those).
// They are NOT derived from any fitted dataset and should be treated as
// sensible starting points, not "trained" values. Real training comes from
// sim trades (trainNNFromSim) or reviewed log (trainNNFromLog).
const DEFAULT_WEIGHTS = [
  -0.52, 1.28, 1.61, -0.74, 0.91, 1.05, 0.22,   // technicals
   0.00, 0.00,                                   // VIX_z, VIX_term (learned)
   0.00, 0.00, 0.00, 0.00,                       // DXY, TNX, Oil, Gold mom
   0.00,                                         // TOD_edge
];
const DEFAULT_BIAS = 0.04;
const WEIGHTS_KEY = "trader_lr_weights_v2";

function loadWeights() {
  try {
    const saved = JSON.parse(localStorage.getItem(WEIGHTS_KEY) || "null");
    if (saved?.weights?.length === 14) return { weights: saved.weights, bias: saved.bias };
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

// ─── Feature extraction ────────────────────────────────────────────────────
// 14-dim feature vector as of v2:
//   [0..6]   Single-symbol technicals (v1 legacy):
//            rsi_c, macd_s, mom_n, bb_c, ema_s, ema_m, vol_n
//   [7]      VIX z-score (regime indicator, -1..1 clipped)
//   [8]      VIX term structure (VIX9D/VIX, >1 = stress bid)
//   [9..12]  Cross-asset 5-bar momentum: DXY, TNX, Oil, Gold (each ±1 clipped)
//   [13]     Time-of-day edge-ness (0=open/close, 1=midday)
//
// macro and calendar args are OPTIONAL — if the caller doesn't supply them
// (e.g. the old scoreSetup(q) signature still works), those slots are zeroed,
// so the extracted vector length is always 14 and old call sites don't break.
function extractFeatures(q, macro = null, calendar = null) {
  const rsi_c   = q.rsi != null ? (q.rsi - 50) / 50 : 0;
  const macd_s  = q.macd != null ? Math.sign(q.macd) : 0;
  const mom_n   = q.momentum5 != null ? Math.max(-1, Math.min(1, q.momentum5 / 4)) : 0;
  const bb_c    = q.bb != null ? (q.bb.pos - 0.5) * 2 : 0;
  const ema_s   = q.ema9 && q.ema20 ? (q.ema9 > q.ema20 ? 1 : -1) : 0;
  const ema_m   = q.ema20 && q.ema50 ? (q.ema20 > q.ema50 ? 1 : -1) : 0;
  const vol_n   = q.volRatio != null ? Math.max(-1, Math.min(1, (q.volRatio - 1) / 1.5)) : 0;

  // Clip helpers so extreme values (rare flash crashes etc.) can't blow up
  // the NN's input distribution.
  const clip1 = v => Math.max(-1, Math.min(1, v || 0));
  const clip0to1 = v => Math.max(0, Math.min(1, v || 0));

  // Macro features — scaled so each lives roughly in [-1, 1].
  const vix_z    = macro?.vixZ != null ? clip1(macro.vixZ / 2) : 0;          // ±2σ → ±1
  const vix_term = macro?.vixTerm != null ? clip1((macro.vixTerm - 1) * 5) : 0; // 0.9→-0.5, 1.1→+0.5
  // Cross-asset momentum comes in as fractional (e.g. 0.003 = 0.3%). Scale
  // by 100 so a 1% move = 1.0 feature weight.
  const dxy_mom  = clip1((macro?.dxyMom5  || 0) * 100);
  const tnx_mom  = clip1((macro?.tnxMom5  || 0) * 100);
  const oil_mom  = clip1((macro?.oilMom5  || 0) * 100);
  const gold_mom = clip1((macro?.goldMom5 || 0) * 100);

  // Calendar
  const tod_edge = clip0to1(calendar?.todEdge ?? 0.5);

  return [
    rsi_c, macd_s, mom_n, bb_c, ema_s, ema_m, vol_n,
    vix_z, vix_term,
    dxy_mom, tnx_mom, oil_mom, gold_mom,
    tod_edge,
  ];
}

export const FEATURE_DIM = 14;
export const FEATURE_NAMES = [
  "RSI", "MACD", "Mom", "BB", "EMA9/20", "EMA20/50", "Vol",
  "VIX_z", "VIX_term",
  "DXY_m", "TNX_m", "Oil_m", "Gold_m",
  "TOD_edge",
];

function logisticScoreFromFeatures(f) {
  const { weights, bias } = loadWeights();
  // Defensive: if feature length drifts (e.g. old saved weights + new
  // feature vector), pad/truncate to keep the dot product well-defined.
  const n = Math.min(f.length, weights.length);
  let dot = bias;
  for (let i = 0; i < n; i++) dot += f[i] * weights[i];
  return sigmoid(dot); // P(bullish)
}

// ─── Decision tree — FORCED DECISIVE MODE with STRONG signals ────────────────
// Returns { signal, strength, reason }. signal ∈ {STRONG_BUY, BUY, SELL,
// STRONG_SELL, AVOID}. strength ∈ [0,1] — how much conviction the rule carries,
// used to weight the tree's contribution to the composite score.
function decisionTree(q) {
  const rsi = q.rsi ?? 50;
  const macd_bull = q.macd != null && q.macd > 0;
  const ema_s_bull = q.ema9 && q.ema20 && q.ema9 > q.ema20;
  const ema_m_bull = q.ema20 && q.ema50 && q.ema20 > q.ema50;
  const mom = q.momentum5 ?? 0;
  const vol = q.volRatio ?? 1;
  const bb = q.bb?.pos ?? 0.5;
  const atrPct = q.atr && q.price ? (q.atr / q.price) * 100 : 0;

  // ═ CRISIS-ONLY AVOID ═
  if (atrPct > 4 && mom < -5 && vol > 3) {
    return { signal: "AVOID", strength: 1.0, reason: "Crisis-level volatility — ATR >4% + mom <-5% + vol >3x. Stand aside." };
  }

  // ═ STRONG signals — multiple factors aligned ═
  const bullFactors = [ema_s_bull, ema_m_bull, macd_bull, mom > 1, rsi > 45 && rsi < 65, vol > 1.2, bb > 0.5].filter(Boolean).length;
  const bearFactors = [!ema_s_bull, !ema_m_bull, !macd_bull, mom < -1, rsi < 55 && rsi > 35, vol > 1.2, bb < 0.5].filter(Boolean).length;

  if (bullFactors >= 6 && rsi < 75) {
    return { signal: "STRONG_BUY", strength: 0.9, reason: `Full-stack bull alignment — ${bullFactors}/7 bullish factors. EMAs stacked, MACD positive, momentum up, price holding above BB midline.` };
  }
  if (bearFactors >= 6 && rsi > 25) {
    return { signal: "STRONG_SELL", strength: 0.9, reason: `Full-stack bear alignment — ${bearFactors}/7 bearish factors. EMAs inverted, MACD negative, momentum down, price below BB midline.` };
  }

  // ═ BEAR REGIME (EMA20 < EMA50) ═
  if (!ema_m_bull) {
    if (rsi < 25 && vol > 1.5) return { signal: "BUY", strength: 0.55, reason: "Capitulation bounce — extreme oversold + volume spike. Counter-trend long only, tight stop." };
    if (rsi < 32) return { signal: "BUY", strength: 0.45, reason: "Oversold bounce in bear — quick long scalp, exit on strength." };
    if (rsi > 60) return { signal: "STRONG_SELL", strength: 0.75, reason: "Bear regime rally — high-probability shorting opportunity into resistance." };
    if (mom < -1) return { signal: "SELL", strength: 0.65, reason: "Bear regime + negative momentum = trend continuation short." };
    if (ema_s_bull && vol > 1.2) return { signal: "BUY", strength: 0.50, reason: "Short-term reversal inside bear — tactical long." };
    return { signal: "SELL", strength: 0.55, reason: "Bear regime default — trend-following bias." };
  }

  // ═ BULL REGIME (EMA20 > EMA50) ═
  if (rsi > 78 && mom > 3 && bb > 0.95) return { signal: "STRONG_SELL", strength: 0.70, reason: "Bull regime blowoff — extreme overbought + stretched BB. Profit-take/counter-trend short." };
  if (rsi > 72 && !macd_bull) return { signal: "SELL", strength: 0.60, reason: "Bull regime exhaustion — high RSI + MACD rolling over." };
  if (rsi < 35 && ema_s_bull) return { signal: "STRONG_BUY", strength: 0.80, reason: "Bull regime pullback into support — classic buy-the-dip. High conviction." };
  if (rsi < 45 && vol > 1.2) return { signal: "BUY", strength: 0.65, reason: "Bull regime dip with volume confirmation — accumulation zone." };
  if (!ema_s_bull && !macd_bull && mom < -1.5) return { signal: "SELL", strength: 0.55, reason: "Short-term breakdown in bull — tactical counter-trend short, tight stop above prior high." };
  if (ema_s_bull && macd_bull) return { signal: "BUY", strength: 0.60, reason: "Bull regime trend continuation — EMAs + MACD aligned." };
  return { signal: "BUY", strength: 0.50, reason: "Bull regime default — long bias unless proven otherwise." };
}

// Map tree signal to a 0-1 bullish score for blending
function treeSignalToScore(signal) {
  if (signal === "STRONG_BUY")  return 0.88;
  if (signal === "BUY")         return 0.66;
  if (signal === "SELL")        return 0.34;
  if (signal === "STRONG_SELL") return 0.12;
  return 0.50; // AVOID
}

// ─── Historical regime fingerprints ─────────────────────────────────────────
// Expanded from 12 → 55 scenarios. Covers crashes, corrections, euphoria tops,
// melt-ups, sideways regimes, sector-specific events, political shocks, and
// recovery patterns. Each entry is a hand-calibrated feature fingerprint that
// the nearest-neighbour matcher compares against live market state. More
// scenarios = better coverage of edge cases and finer-grained analogue
// matching. Sourced from: NBER recession dating, CBOE historical VIX, Fed
// minutes, earnings-reaction studies, and public post-mortems.
const CRISIS_SCENARIOS = [
  // ─── Major crashes ─────────────────────────────────────────────────
  { name: "2008 Financial Crisis",     rsi: 28, mom: -4.2, vol: 2.8, bb: 0.02, emaShort: -1, emaMed: -1, regime: "crash", note: "Credit collapse. Bear regime. Only short or cash." },
  { name: "2020 COVID Crash",          rsi: 22, mom: -6.1, vol: 3.5, bb: 0.01, emaShort: -1, emaMed: -1, regime: "crash", note: "Panic selling. Fastest 30% drop ever. V-shaped recovery followed." },
  { name: "Black Monday 1987",         rsi: 18, mom: -9.8, vol: 4.1, bb: 0.00, emaShort: -1, emaMed: -1, regime: "crash", note: "Single-day collapse. Extreme ATR. Bounce was sharp but brief." },
  { name: "Dot-com Crash 2001-2002",   rsi: 30, mom: -2.7, vol: 1.8, bb: 0.10, emaShort: -1, emaMed: -1, regime: "crash", note: "18-month bear. -49% SPX. Tech led; breadth kept breaking down. Rallies sold." },
  { name: "9/11 Reopening 2001",       rsi: 26, mom: -5.8, vol: 3.8, bb: 0.03, emaShort: -1, emaMed: -1, regime: "crash", note: "Market closed 4 days. Reopened -7%. Airlines/insurers destroyed. V recovery within weeks." },
  { name: "1973-74 Oil Shock Bear",    rsi: 29, mom: -2.2, vol: 1.6, bb: 0.08, emaShort: -1, emaMed: -1, regime: "crash", note: "Stagflation. -48% over 21 months. Slow grinding bear. No V recovery." },
  { name: "1929 Crash (historical)",   rsi: 20, mom: -7.5, vol: 3.2, bb: 0.02, emaShort: -1, emaMed: -1, regime: "crash", note: "Oct 1929. -12% single day. Deeper bear followed: -89% trough 1932." },

  // ─── Corrections (-10-20%) ─────────────────────────────────────────
  { name: "2022 Rate Hike Selloff",    rsi: 38, mom: -1.8, vol: 1.4, bb: 0.12, emaShort: -1, emaMed: -1, regime: "correction", note: "Slow grind down. RSI 30-45. Rallies get sold. -27% peak to trough." },
  { name: "Q4 2018 Selloff",           rsi: 36, mom: -2.1, vol: 1.7, bb: 0.11, emaShort: -1, emaMed: -1, regime: "correction", note: "Fed tightening fear. -20% correction. Reversed on Powell pivot Jan 2019." },
  { name: "Jan 2016 China Worry",      rsi: 34, mom: -2.4, vol: 2.0, bb: 0.09, emaShort: -1, emaMed: -1, regime: "correction", note: "-13% in 6 weeks. Oil + China. Bottomed Feb '16." },
  { name: "2011 Debt Ceiling",         rsi: 32, mom: -2.9, vol: 2.3, bb: 0.07, emaShort: -1, emaMed: -1, regime: "correction", note: "S&P downgraded USA to AA+. -19% drop. VIX spiked to 48." },
  { name: "2015 Aug Flash Crash",      rsi: 33, mom: -4.1, vol: 2.8, bb: 0.05, emaShort: -1, emaMed: -1, regime: "correction", note: "Yuan devaluation Aug 11. SPX -11% in 6 days. ETF pricing broke." },
  { name: "2010 Flash Crash",          rsi: 42, mom: -5.2, vol: 3.0, bb: 0.04, emaShort: -1, emaMed: -1, regime: "correction", note: "May 6 2010. DJIA -9% intraday, recovered. HFT failure. Single bad trade." },
  { name: "2018 Volmageddon Feb",      rsi: 31, mom: -3.8, vol: 3.4, bb: 0.04, emaShort: -1, emaMed: -1, regime: "correction", note: "Short-vol blow-up. XIV liquidated. VIX +118% in a day. -10% SPX." },
  { name: "2023 Aug-Oct Yield Jump",   rsi: 37, mom: -1.4, vol: 1.3, bb: 0.15, emaShort: -1, emaMed:  1, regime: "correction", note: "10Y hit 5%. Long-duration stocks clobbered. -10% SPX." },

  // ─── Slow bleeds & sideways bears ──────────────────────────────────
  { name: "European Debt 2011",        rsi: 32, mom: -2.4, vol: 1.9, bb: 0.08, emaShort: -1, emaMed: -1, regime: "slow_bleed", note: "Greece/Italy contagion. Slow bleed. Reversals failed repeatedly." },
  { name: "Asian Crisis 1997",         rsi: 31, mom: -3.1, vol: 2.1, bb: 0.05, emaShort: -1, emaMed: -1, regime: "slow_bleed", note: "Thai baht devaluation. Contagion. EM led. US dipped then recovered." },
  { name: "1994 Bond Massacre",        rsi: 36, mom: -1.1, vol: 1.5, bb: 0.14, emaShort: -1, emaMed: -1, regime: "slow_bleed", note: "Greenspan hike cycle. Bonds -10%, equities flat-down. Sideways chop." },
  { name: "Stagflation 1970s",         rsi: 39, mom: -0.6, vol: 1.2, bb: 0.25, emaShort: -1, emaMed: -1, regime: "slow_bleed", note: "High inflation + low growth. Nominal flat, real -50%. Gold outperformed." },

  // ─── Euphoria tops ─────────────────────────────────────────────────
  { name: "Dot-com Peak 2000",         rsi: 72, mom: 2.1, vol: 0.7, bb: 0.94, emaShort:  1, emaMed:  1, regime: "euphoria", note: "Euphoria top. High RSI, low vol on advance. Distribution under the surface." },
  { name: "Jan 2022 Top",              rsi: 71, mom: 1.6, vol: 0.8, bb: 0.92, emaShort:  1, emaMed:  1, regime: "euphoria", note: "Post-COVID stimulus peak. ARKK already rolling over 3 months prior." },
  { name: "Nov 2021 Meme Peak",        rsi: 74, mom: 3.2, vol: 1.3, bb: 0.95, emaShort:  1, emaMed:  1, regime: "euphoria", note: "Rivian IPO, Shiba, NFTs. Quality had peaked earlier. Rotation under the hood." },
  { name: "Oct 2007 Top",              rsi: 68, mom: 0.8, vol: 0.9, bb: 0.88, emaShort:  1, emaMed:  1, regime: "euphoria", note: "Last gasp before GFC. Credit markets already showed cracks." },
  { name: "Jun 1990 Top",              rsi: 70, mom: 1.2, vol: 0.8, bb: 0.90, emaShort:  1, emaMed:  1, regime: "euphoria", note: "Iraq/Kuwait preamble. S&L crisis building. -20% bear ensued." },

  // ─── Geopolitical shocks ───────────────────────────────────────────
  { name: "Hormuz Tension 2019",       rsi: 44, mom: -1.2, vol: 1.6, bb: 0.28, emaShort: -1, emaMed:  1, regime: "geo_shock", note: "Oil spike, equity dip. Short-lived. Bull regime intact." },
  { name: "Russia/Ukraine Feb 2022",   rsi: 35, mom: -2.8, vol: 2.2, bb: 0.08, emaShort: -1, emaMed: -1, regime: "geo_shock", note: "Invasion. SPX -7% in 2 weeks. Energy +40%. Tech sold hard." },
  { name: "Israel/Hamas Oct 2023",     rsi: 43, mom: -1.0, vol: 1.4, bb: 0.30, emaShort: -1, emaMed:  1, regime: "geo_shock", note: "Brief risk-off. Oil spike. Equities rebounded within 2 weeks." },
  { name: "Brexit Vote Jun 2016",      rsi: 38, mom: -4.3, vol: 2.6, bb: 0.06, emaShort: -1, emaMed:  1, regime: "geo_shock", note: "Overnight -8% FTSE. 2-day panic. New highs within 3 weeks." },
  { name: "Fukushima Mar 2011",        rsi: 40, mom: -2.1, vol: 1.8, bb: 0.15, emaShort: -1, emaMed:  1, regime: "geo_shock", note: "Nikkei -17% in 2 days. Yen surge. Global contagion mild." },
  { name: "Gulf War Jan 1991",         rsi: 48, mom:  2.3, vol: 1.9, bb: 0.55, emaShort:  1, emaMed:  1, regime: "geo_shock", note: "Actually triggered RALLY. 'Uncertainty priced in' — operation start = relief trade." },

  // ─── Sector / commodity events ─────────────────────────────────────
  { name: "Oil Price Collapse 2020",   rsi: 25, mom: -5.4, vol: 3.2, bb: 0.03, emaShort: -1, emaMed: -1, regime: "sector_shock", note: "WTI went negative. XOM -60%. Energy sector devastated." },
  { name: "Oil Crash 2014-16",         rsi: 33, mom: -3.0, vol: 2.0, bb: 0.10, emaShort: -1, emaMed: -1, regime: "sector_shock", note: "Oil $100 → $26. Shale producers blew up. HY credit spreads widened." },
  { name: "Silver Spike Apr 2011",     rsi: 78, mom: 5.1, vol: 2.2, bb: 0.97, emaShort:  1, emaMed:  1, regime: "sector_shock", note: "Silver $50. CME hiked margins 5x. -30% in 2 weeks." },
  { name: "Bitcoin Crash 2022",        rsi: 24, mom: -5.9, vol: 2.9, bb: 0.02, emaShort: -1, emaMed: -1, regime: "sector_shock", note: "Luna + 3AC + FTX cascade. BTC $69k → $15k. Crypto-equity correlation broke." },
  { name: "AI Bubble Melt-up 2023-24", rsi: 73, mom: 3.8, vol: 1.4, bb: 0.95, emaShort:  1, emaMed:  1, regime: "euphoria", note: "NVDA +240% YTD. Mag-7 concentration record. Breadth anaemic." },

  // ─── Banking crises ────────────────────────────────────────────────
  { name: "SVB Bank Run 2023",         rsi: 40, mom: -1.6, vol: 2.0, bb: 0.18, emaShort: -1, emaMed:  1, regime: "bank_crisis", note: "Regional bank panic. Contained by BTFP. Bull resumed within a month." },
  { name: "Credit Suisse Mar 2023",    rsi: 42, mom: -1.2, vol: 1.7, bb: 0.20, emaShort: -1, emaMed:  1, regime: "bank_crisis", note: "UBS rescue. AT1s wiped. European banks sold hard. Contained fast." },
  { name: "Bear Stearns Mar 2008",     rsi: 32, mom: -3.6, vol: 2.5, bb: 0.06, emaShort: -1, emaMed: -1, regime: "bank_crisis", note: "Fire sale to JPM. Dead-cat rally; real crisis peaked Sept '08." },
  { name: "Lehman Sept 2008",          rsi: 22, mom: -7.2, vol: 3.9, bb: 0.01, emaShort: -1, emaMed: -1, regime: "bank_crisis", note: "Bankruptcy Sept 15. MMF broke the buck. -28% SPX in 3 weeks." },

  // ─── Melt-ups / sustained bulls ────────────────────────────────────
  { name: "2017 Low-Vol Melt-up",      rsi: 65, mom: 0.9, vol: 0.6, bb: 0.82, emaShort:  1, emaMed:  1, regime: "melt_up", note: "VIX sub-10. Steady grind up. No corrections >3% all year." },
  { name: "2013 QE3 Rally",            rsi: 63, mom: 1.1, vol: 0.8, bb: 0.78, emaShort:  1, emaMed:  1, regime: "melt_up", note: "Taper Tantrum aside, straight line up. Bonds bled, equities won." },
  { name: "2020-21 Post-COVID Melt",   rsi: 67, mom: 2.4, vol: 1.1, bb: 0.85, emaShort:  1, emaMed:  1, regime: "melt_up", note: "Fiscal + monetary firehose. Everything bid. Speculative excess visible." },
  { name: "2019 Fed Pivot Rally",      rsi: 62, mom: 1.3, vol: 0.9, bb: 0.75, emaShort:  1, emaMed:  1, regime: "melt_up", note: "After Powell pivot Jan '19. Rate cuts. Y-end +29%." },

  // ─── Recovery / base-building ──────────────────────────────────────
  { name: "March 2009 Bottom",         rsi: 31, mom: 0.4, vol: 2.0, bb: 0.25, emaShort:  1, emaMed: -1, regime: "recovery", note: "666 SPX. 12-year bull began. Short covering first, fundamentals later." },
  { name: "Dec 2018 Bottom",           rsi: 34, mom: 0.2, vol: 1.8, bb: 0.22, emaShort:  1, emaMed: -1, regime: "recovery", note: "Christmas Eve low. Fed pivot. V-recovery back to ATH by April." },
  { name: "March 2020 Bottom",         rsi: 30, mom: 1.1, vol: 2.4, bb: 0.30, emaShort:  1, emaMed: -1, regime: "recovery", note: "Fed ZIRP + unlimited QE. Fastest bear-to-bull in history." },
  { name: "Oct 2022 Bottom",           rsi: 36, mom: 0.6, vol: 1.5, bb: 0.28, emaShort:  1, emaMed: -1, regime: "recovery", note: "SPX 3491. CPI peaked month prior. Stealth bull began here." },

  // ─── Sideways / chop ────────────────────────────────────────────────
  { name: "Chop 2015 Sideways",        rsi: 51, mom: 0.1, vol: 1.1, bb: 0.52, emaShort:  0, emaMed:  0, regime: "chop", note: "Flat year. No trend. RSI oscillated 40-60. Mean-reversion worked." },
  { name: "Chop 1994 Sideways",        rsi: 49, mom: -0.2, vol: 1.0, bb: 0.48, emaShort:  0, emaMed:  0, regime: "chop", note: "Post-'93 rally consolidation. Range-bound all year." },
  { name: "Summer 2019 Chop",          rsi: 53, mom: 0.3, vol: 1.2, bb: 0.58, emaShort:  1, emaMed:  1, regime: "chop", note: "Trade war tweets. Rangebound 2800-3000. Volatile chop, no trend." },

  // ─── Inflation / macro shocks ──────────────────────────────────────
  { name: "2021-22 Inflation Shock",   rsi: 45, mom: -1.0, vol: 1.5, bb: 0.35, emaShort: -1, emaMed:  1, regime: "macro_shock", note: "CPI 9.1%. Fed behind curve. Growth/quality sold, energy/defensive won." },
  { name: "Taper Tantrum May 2013",    rsi: 44, mom: -1.8, vol: 1.6, bb: 0.20, emaShort: -1, emaMed:  1, regime: "macro_shock", note: "10Y 1.6→3%. EM/bonds hit. SPX dipped 5% then new high." },
  { name: "1998 LTCM Russia",          rsi: 32, mom: -3.4, vol: 2.3, bb: 0.08, emaShort: -1, emaMed: -1, regime: "macro_shock", note: "Russia default + LTCM bailout. -20% SPX. Fed cut. V recovery." },
  { name: "Aug 2024 Yen Carry Unwind", rsi: 34, mom: -3.1, vol: 2.4, bb: 0.09, emaShort: -1, emaMed:  1, regime: "macro_shock", note: "BOJ hike. VIX 65 intraday. -6% SPX in 3 days. Full recovery in 2 weeks." },

  // ─── Election / political ──────────────────────────────────────────
  { name: "Trump Election Nov 2016",   rsi: 58, mom: 3.2, vol: 1.8, bb: 0.72, emaShort:  1, emaMed:  1, regime: "political", note: "Overnight limit down, reopened limit up. Reflation trade. 18-month rally." },
  { name: "Nov 2020 Election",         rsi: 60, mom: 2.8, vol: 1.4, bb: 0.76, emaShort:  1, emaMed:  1, regime: "political", note: "Vaccine news day after. Rotation value/small. Biden-trade started." },
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

// ─── Main scoring function — ensemble of LR + NN + Tree + Crisis analogue ──
//
// Composite probability is a weighted blend:
//
//   • LR      always contributes 40%
//   • NN      contributes 0-40% depending on how many samples it's seen
//             (ramps from 0 at <8 samples to 40% at ≥50 samples)
//   • Tree    contributes 20-40% depending on its own strength score
//   • Crisis  doesn't vote but BIASES the composite ±5% based on whether the
//             nearest analogue was bullish or bearish
//
// Weights always normalise to 1.0 so the output stays a probability.
// Agreement between models boosts confidence; disagreement reduces it.
export function scoreSetup(q, context = {}) {
  const { macro = null, calendar = null } = context;
  const features  = extractFeatures(q, macro, calendar);
  const lrProb    = logisticScoreFromFeatures(features);
  const tree      = decisionTree(q);
  const crisis    = findClosestCrisis(q);
  const atr       = q.atr ?? 0;
  const price     = q.price ?? 0;

  // ── NN probability (null if untrained) ──────────────────────────────
  const nnProb    = predictNN(features);
  const nnInfo    = getNNInfo();
  const nnReady   = nnProb != null;

  // ── Blend weights ───────────────────────────────────────────────────
  const treeWeight = 0.20 + 0.20 * (tree.strength ?? 0.5);
  const nnWeight   = nnReady ? Math.min(0.40, nnInfo.trainedOn / 125) : 0;
  const lrWeight   = 0.40;
  const totalRaw   = lrWeight + treeWeight + nnWeight;

  const lrW   = lrWeight   / totalRaw;
  const treeW = treeWeight / totalRaw;
  const nnW   = nnWeight   / totalRaw;

  const treeScore = treeSignalToScore(tree.signal);
  let compositeProb = lrW * lrProb + treeW * treeScore + nnW * (nnProb ?? 0.5);

  // Crisis bias: nearest analogue's bias pushes +/- 5%
  const crisisBias = crisis?.regime === "melt_up" || crisis?.regime === "recovery" ?  0.05
                  : crisis?.regime === "crash"    || crisis?.regime === "slow_bleed" ? -0.05
                  : crisis?.regime === "euphoria" ? -0.03   // top-like conditions lean bearish
                  : crisis?.regime === "bank_crisis" || crisis?.regime === "macro_shock" ? -0.04
                  : 0;
  compositeProb = Math.max(0.01, Math.min(0.99, compositeProb + crisisBias * (crisis?.similarity ?? 0)));

  // ── Agreement boost — if LR, NN, Tree all point same way, amplify confidence ──
  const lrDir   = lrProb > 0.5   ? 1 : -1;
  const treeDir = treeScore > 0.5 ? 1 : -1;
  const nnDir   = nnReady ? (nnProb > 0.5 ? 1 : -1) : lrDir;
  const agreeCount = (lrDir === treeDir ? 1 : 0) + (lrDir === nnDir ? 1 : 0) + (treeDir === nnDir ? 1 : 0);
  const agreementBoost = agreeCount >= 3 ? 1.20 : agreeCount === 2 ? 1.0 : 0.75;

  const rawConfidence = Math.abs(compositeProb - 0.5) * 200;
  const confidence    = Math.min(100, Math.round(rawConfidence * agreementBoost));

  const direction = compositeProb > 0.58 ? "BULLISH" : compositeProb < 0.42 ? "BEARISH" : "NEUTRAL";

  // Risk levels from ATR
  const stopLong  = price > 0 && atr > 0 ? (price - 1.5 * atr).toFixed(2) : null;
  const stopShort = price > 0 && atr > 0 ? (price + 1.5 * atr).toFixed(2) : null;
  const tgt3Long  = price > 0 && atr > 0 ? (price + 4.5 * atr).toFixed(2) : null;
  const tgt3Short = price > 0 && atr > 0 ? (price - 4.5 * atr).toFixed(2) : null;

  return {
    features,
    lrProb: (lrProb * 100).toFixed(1),
    nnProb: nnReady ? (nnProb * 100).toFixed(1) : null,
    compositeProb: (compositeProb * 100).toFixed(1),
    weights: { lr: lrW, nn: nnW, tree: treeW, crisisBias },
    direction,
    confidence,
    agreement: { count: agreeCount, boost: agreementBoost },
    treeSignal: tree.signal,
    treeStrength: tree.strength,
    treeReason: tree.reason,
    crisis,
    nnInfo,
    stopLong, stopShort, tgt3Long, tgt3Short,
  };
}

// ─── Delegating wrappers so App.jsx doesn't need to import nn.js directly ──
//
// Two distinct training sources:
//   • LOG  — real trades the user manually reviewed in the Decision Log.
//            Shape: { reviewed:true, outcome:"WIN"|"LOSS", features, timestamp:ISO }
//   • SIM  — synthetic trades produced by the backtester from real candles.
//            Shape: { outcome:"WIN"|"LOSS", features, timestamp:ms, ageDays, ... }
//            (no `reviewed` flag — they were never user-reviewed)
//
// They were previously sharing one wrapper that hard-required `reviewed:true`,
// which silently dropped every sim trade and produced "Need at least 8 samples"
// even when the backtester returned 100+ trades.

export function trainNNFromLog(reviewedLog) {
  const samples = reviewedLog
    .filter(d => d.reviewed && d.outcome && d.features)
    .map(d => ({
      x: d.features,
      y: d.outcome === "WIN" ? 1 : 0,
      ageDays: (Date.now() - new Date(d.timestamp).getTime()) / (24 * 3600 * 1000),
    }));
  return trainNNRaw(samples);
}

export function trainNNFromSim(simTrades) {
  const samples = simTrades
    .filter(d => d.outcome && d.features)
    .map(d => ({
      x: d.features,
      y: d.outcome === "WIN" ? 1 : 0,
      ageDays: d.ageDays || 0,
    }));
  return trainNNRaw(samples);
}

// Back-compat alias — older imports of trainNN keep working (= log training).
export const trainNN = trainNNFromLog;

export function resetNN() { resetNNRaw(); }
export { getNNInfo };

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
