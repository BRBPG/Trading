import React, { useState, useRef, useEffect, useCallback } from "react";
import { generateMockQuote, generateLiveIndicators, computeIndicators } from "./mockData";
import { scoreSetup, logDecision, reviewDecision, getPerformanceStats, getLog, adaptWeights, resetWeights, getCurrentWeights, trainNNFromLog, trainNNFromSim, resetNN, getNNInfo, trainGBMFromSim, trainGBMFromLog, resetGBM, getGBMInfo, trainRegimeFromSim, resetRegime, FEATURE_NAMES } from "./model";
import { computeSimMetrics, summariseEdge } from "./simMetrics";
import { runWalkForward, interpretWF, runAblationStudy } from "./walkForward";
import { runBacktest } from "./backtest";
import { calcADX, calcWilliamsR, calcStochastic, calcROC, calcZScore,
         calcCMF, calcMaxDrawdown, calcSharpe, calcBeta, engineerFeatures,
         fetchAllNews } from "./quant";
import { BUFFETT_SYSTEM_PROMPT, buildBuffettContext } from "./buffett";
import { cleanBars, assessQuality, cleaningSummary } from "./cleaning";
import { downloadExport, importState } from "./persistence";
import { fetchMacroSnapshot } from "./macro";
import { fetchBTCDominance, timeSeriesMomentum, xsMomRankLive, rvRatioLive, dayOfWeekSinAt } from "./crypto";
import { fetchTopCryptoSnapshot, xsRankLive as xsRankLiveBroad, breadthLive } from "./broadMarket";
import { fetchFundingForUniverse, fundingZLive } from "./funding";
import { calendarFeatures } from "./calendar";
import { getEarningsBatch, computePeadFeatures } from "./earnings";
import { recommendSize } from "./sizing";
import { trainBagFromSim, loadBag, predictBag } from "./bagging";

const FH_QUOTE  = (sym, key) => `https://finnhub.io/api/v1/quote?symbol=${sym}&token=${key}`;
const FH_METRIC = (sym, key) => `https://finnhub.io/api/v1/stock/metric?symbol=${sym}&metric=all&token=${key}`;

// Watchlist selected by the user. Mix of:
//   - Broad index ETFs (SPY, QQQ)
//   - Mega-cap tech (AAPL, MSFT, AMZN)
//   - Semiconductors (NVDA, AMD, TSM — TSM is the foundry that makes silicon
//     for the other two; all three move together on semi-cycle news)
//   - EV / autos (TSLA)
//   - Quantum computing (IONQ, RGTI — pure-play, high-vol speculative names)
//   - Airline (UAL) — left in for consumer-discretionary / travel exposure
//   - Commodities (USO = WTI oil ETF, BNO = Brent oil ETF, GLD = gold ETF)
//   - LSE (TW.L = Taylor Wimpey, UK housebuilder) — routed through Yahoo
// ─── Universe: equities vs crypto ──────────────────────────────────────────
// Two parallel asset universes share the infrastructure (backtest, walk-
// forward, multi-sim, ensemble models) but have different watchlists,
// session behaviour, cost defaults, and eventually feature vectors.
// Universe is a runtime-switchable state — see universe selector in the
// top-bar toggle.
const EQUITY_WATCHLIST = ["SPY","QQQ","AAPL","MSFT","AMZN","NVDA","AMD","TSM","TSLA","IONQ","RGTI","UAL","USO","BNO","GLD","TW.L"];

// Crypto universe — expanded from 10 → 20 to make cross-sectional momentum
// rank (Liu-Tsyvinski 2022) statistically meaningful. At 8 symbols (when
// BNB silent-drops) XS-rank has only 8 possible discrete percentile values,
// which degenerates to near-random. 20 symbols give 5% resolution on
// percentile, enough to pick up the documented XS-momentum edge if it's
// present. Selected for:
//   (a) Polygon Currencies Standard coverage (all serve as X:{SYM}USD)
//   (b) Yahoo -USD fallback for redundancy
//   (c) ≥1 year continuous trading history (no recent IPOs that lack the
//       warmup buffer at 90-180d sim windows)
//   (d) Sector diversity — L1s, DeFi, L2s, exchange tokens, legacy coins
// MATIC was rebranded to POL in September 2024 when the Polygon ecosystem
// migrated the native token. Polygon.io's ticker is now X:POLUSD and Yahoo
// has migrated to POL-USD.
// ─── BTC-only single-asset universe (Phase 4) ──────────────────────────────
// After the 40-coin multi-symbol crypto pipeline landed three independent
// diagnostics at null (overall AUC, conviction-stratified AUC, meta-labeling
// AUC all at ~0.50), the pivot is to single-asset BTC focus. Rationale:
//   - Retail feature universe is the ceiling on 40-coin universes where
//     cross-sectional momentum degenerates (Liu-Tsyvinski 2022 used 1800+).
//   - BTC has the richest orthogonal retail-accessible data (Deribit DVOL,
//     Binance funding, on-chain metrics via free APIs) — most published
//     retail-ML-for-crypto successes focus on BTC specifically.
//   - Single-asset simplifies: no XS-rank degeneracy, no universe-level
//     silent drops, no per-symbol feature normalization headaches.
// Storage keys fully isolated from both "crypto" (multi-symbol weights) and
// "equities" (different asset class entirely) — see model.js/nn.js/gbm.js
// for btc-specific key routing.
const BTC_WATCHLIST = ["BTC-USD"];

const CRYPTO_WATCHLIST = [
  // ─── Tier 1: Top 10 by market cap / liquidity (original 3a universe) ───
  "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD",
  "ADA-USD","AVAX-USD","LINK-USD","DOGE-USD","POL-USD",
  // ─── Tier 2: Large-cap alts, multi-year history (3d step 1 expansion) ─
  "TRX-USD","LTC-USD","DOT-USD","ATOM-USD","UNI-USD",
  "NEAR-USD","APT-USD","ARB-USD","OP-USD","AAVE-USD",
  // ─── Tier 3: Further expansion to ~40 symbols for XS-rank resolution ──
  // Targeting ~2.5% percentile granularity which is where XS-momentum
  // genuinely resolves in the literature (Liu-Tsyvinski used 1800+; we
  // can't get there on a retail data plan, but 40 is the next meaningful
  // step up from 20). All covered by Polygon Currencies Standard.
  "BCH-USD",   // Bitcoin Cash — legacy fork, ~8y
  "ETC-USD",   // Ethereum Classic — legacy fork, ~9y
  "XLM-USD",   // Stellar — payments L1, ~11y
  "HBAR-USD",  // Hedera — hashgraph L1, ~5y
  "ICP-USD",   // Internet Computer — ~5y, Dfinity
  "FIL-USD",   // Filecoin — decentralised storage, ~5y
  "ALGO-USD",  // Algorand — L1, ~6y
  "VET-USD",   // VeChain — supply-chain L1, ~7y
  "STX-USD",   // Stacks — Bitcoin L2, ~4y
  "IMX-USD",   // Immutable X — gaming L2, ~5y
  "MKR-USD",   // Maker — CDP/DAI, DeFi bluechip, ~8y
  "CRV-USD",   // Curve — stableswap AMM, ~5y
  "LDO-USD",   // Lido — liquid staking, ~5y
  "SNX-USD",   // Synthetix — synthetic assets, ~6y
  "COMP-USD",  // Compound — DeFi lending, ~5y
  "GRT-USD",   // The Graph — decentralised indexing, ~5y
  "SUI-USD",   // Sui — Move-based L1, ~2.5y
  "SEI-USD",   // Sei — trading L1, ~2.5y
  "TIA-USD",   // Celestia — modular DA, ~2.5y
  "RUNE-USD",  // THORChain — cross-chain AMM, ~5y
];

// Keep the legacy WATCHLIST symbol exported for anywhere that needs a
// compile-time default — defaults to equities, will be overridden by the
// React state when the user switches universe.
const WATCHLIST = EQUITY_WATCHLIST;

function isCryptoSymbol(sym) {
  // Yahoo crypto pairs end in -USD or -USDT. Distinguishes from LSE dot-
  // suffix and plain US tickers.
  return /-USD(T)?$/.test(sym);
}

// Single-source truth for "is this universe a crypto-style asset class?"
// Used everywhere we need to route features + 24/7 session handling + crypto-
// specific model configuration. Both multi-symbol "crypto" and single-asset
// "btc" pass this check — storage keys are still segregated per universe in
// model.js/nn.js/gbm.js so they don't cross-contaminate.
function isCryptoUniverse(universe) {
  return universe === "crypto" || universe === "btc";
}

function watchlistFor(universe) {
  if (universe === "btc")    return BTC_WATCHLIST;
  if (universe === "crypto") return CRYPTO_WATCHLIST;
  return EQUITY_WATCHLIST;
}

function backtestSymbolsFor(universe) {
  // Equities: drop dot-suffix (non-US session-misaligned).
  // Crypto / BTC: use all — 24/7 markets, no session mismatch.
  const list = watchlistFor(universe);
  return isCryptoUniverse(universe)
    ? [...list]
    : list.filter(s => !s.includes("."));
}

// Per-universe default cost model. Crypto costs are dominated by spread
// + taker fees, much higher than US equities.
//   Equities: 15 bps round-trip (retail on liquid US equities)
//   Crypto:   30 bps default; BTC/ETH are lower (~20), alts higher (~40-60).
//             Phase 3b will add per-asset cost table; for 3a a flat 30 is
//             an honest middle estimate. Used by the universe-switch side
//             effect to reset costBps when the user flips universe.
// eslint-disable-next-line no-unused-vars
function defaultCostBpsFor(universe) {
  // BTC on a top-tier venue (Coinbase Pro, Kraken, Binance) clears at ~15-20
  // bps round-trip at retail size — tighter than the broad crypto average
  // because BTC is the highest-liquidity pair by an order of magnitude.
  if (universe === "btc")    return 18;
  if (universe === "crypto") return 30;
  return 15;
}

// Subset of the watchlist fed into the backtest / sim / training pipeline.
// Non-US listings (anything with a dot-suffix, currently just TW.L) are
// deliberately excluded because:
//   1. Their session hours don't align with US-session macro features.
//     VIX, SPY, TNX, DXY all publish US-hours timestamps; TW.L trades
//     8am-4:30pm GMT. The macro.at(t) lookup for a TW.L bar timestamp
//     would return VIX from 3am-11:30am ET (pre-open for most of it),
//     producing spurious cross-asset inputs.
//   2. Currency mismatch — TW.L is quoted in GBX (pence), the cost model
//     assumes USD-like scale for the 15bps round-trip default.
//   3. The user explicitly added TW.L for live reference viewing (their
//     father's interest), not for systematic trading.
// The live dashboard still fetches, displays, and allows manual ANALYZE
// on TW.L via the regular WATCHLIST iteration — only the backtest loop
// skips it.
// Legacy compile-time defaults — runtime code paths call backtestSymbolsFor
// and high-vol list is looked up per-universe via highVolSymbolsFor.
const BACKTEST_SYMBOLS = backtestSymbolsFor("equities");
// Symbols whose per-bar vol is high enough to materially destabilise sims:
// quantum names introduce ±5-6% per-bar moves that dominate the variance
// across multi-sim runs. Excluding them via the diagnostic toggle isolates
// whether the apparent edge is robust or a quantum-name-day artefact.
// High-vol speculative names per universe. Equity: small-cap quantum.
// Crypto: meme coins + highly speculative alts whose swings dominate
// multi-sim variance the way IONQ/RGTI did for equities.
const HIGH_VOL_EQUITIES = ["IONQ", "RGTI"];
const HIGH_VOL_CRYPTO   = ["DOGE-USD", "AVAX-USD"];  // tune with experience
const HIGH_VOL_BTC      = [];                         // n/a for single-asset
function highVolSymbolsFor(universe) {
  if (universe === "btc")    return HIGH_VOL_BTC;
  if (universe === "crypto") return HIGH_VOL_CRYPTO;
  return HIGH_VOL_EQUITIES;
}
const HIGH_VOL_SYMBOLS = HIGH_VOL_EQUITIES;  // back-compat default

// ─── Market-session detection (pure, client-side) ───────────────────────────
// US stocks:   premarket 04:00-09:30 ET, open 09:30-16:00 ET, after 16:00-20:00 ET
// LSE (.L):    premarket 07:00-08:00 UK, open 08:00-16:30 UK, after 16:30-17:15 UK
function getMarketSession(symbol = "") {
  // Crypto markets are 24/7 with no session distinction. Applying US-hours
  // logic to BTC-USD would produce "CLOSED" during overnight hours when
  // crypto is in fact trading normally — which would then corrupt quality
  // tagging, stop extended-hours-move detection from working, and misalign
  // macro point-in-time lookups.
  if (isCryptoSymbol(symbol)) return "OPEN";
  const isUK = symbol.toUpperCase().endsWith(".L");
  const tz = isUK ? "Europe/London" : "America/New_York";
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone: tz, weekday: "short", hour: "numeric", minute: "numeric", hour12: false,
    }).formatToParts(new Date()).map(p => [p.type, p.value])
  );
  const h = parseInt(parts.hour, 10) % 24;
  const m = parseInt(parts.minute, 10);
  const mins = h * 60 + m;
  if (parts.weekday === "Sat" || parts.weekday === "Sun") return "CLOSED";
  if (isUK) {
    if (mins >= 7*60   && mins < 8*60)     return "PREMARKET";
    if (mins >= 8*60   && mins < 16*60+30) return "OPEN";
    if (mins >= 16*60+30 && mins < 17*60+15) return "AFTERHOURS";
    return "CLOSED";
  }
  if (mins >= 4*60   && mins < 9*60+30) return "PREMARKET";
  if (mins >= 9*60+30 && mins < 16*60)  return "OPEN";
  if (mins >= 16*60  && mins < 20*60)   return "AFTERHOURS";
  return "CLOSED";
}

function sessionLabel(s) {
  return s === "PREMARKET" ? "PRE-MARKET"
       : s === "AFTERHOURS" ? "AFTER-HOURS"
       : s === "CLOSED" ? "CLOSED"
       : "OPEN";
}

function sessionColor(s) {
  return s === "OPEN" ? "#2ECC71"
       : s === "PREMARKET" ? "#5AACDF"
       : s === "AFTERHOURS" ? "#C9A84C"
       : "#888";
}

const SYSTEM_PROMPT = `You are THE TRADER — the composite mind of five legendary traders. You have been given a pre-trained quantitative model score alongside live market data. Your job is to give ONE clear, decisive verdict. Not "it could go either way." A real verdict.

═══ ABSOLUTE DIRECTIVE — READ THIS FIRST ═══
Every session you MUST produce an actionable BUY or SELL. At every moment money can be made — either long or short. Markets always favour one direction more than the other. "AVOID" and "WAIT" are BANNED except in one specific case: crisis-level volatility where the model shows ATR >4%, 5-bar momentum <-5%, AND volume >3x average simultaneously. Anything else is a BUY or a SELL. If your first instinct is AVOID, you haven't done the work — look harder at both sides and pick the one with marginally better odds. A 55/45 edge played with discipline makes money over 1000 trades. Passivity does not.

═══ THE FIVE MINDS ═══

JESSE LIVERMORE (1877–1940): Made and lost $100M multiple times. Rules he died by:
- "The market is never wrong. Opinions often are." — Never fight the tape.
- "There is only one side of the market — the right side." — Direction first, always.
- Pivotal points: stocks consolidate, then break. Enter ONLY on confirmed breakout with volume.
- Never average a losing position. Ever. "Doubling down is amateur hour."
- Ran SILENT periods: if a trade doesn't move immediately, it's wrong. Exit.
- He shorted the 1929 crash and made $100M. He was FULLY SHORT before Black Thursday.
- Key tell: if a stock makes a new high on LOWER volume than the previous new high — distribution. Sell.

PAUL TUDOR JONES (born 1954): 5:1 minimum R/R. Never lost a year in 30+ years of trading.
- "Every day I assume every position I have is wrong." — Defense first. Always.
- 200-day MA is the line of demarcation: above = bull, below = bear. Non-negotiable.
- He predicted and profited from Black Monday 1987 by watching the 1929 chart overlay.
- "The most important rule is to play great defense, not great offense."
- His 5:1 rule: if you can't define a stop that gives you 5x reward potential, don't trade.
- Macro context: what is the Fed doing? What is the dollar doing? Stocks don't trade in isolation.
- "Losers average losers." Cut the position, not the stop.

RICHARD DENNIS (born 1949): The Turtle Experiment — proved trading can be taught systematically.
- 20-day high breakout = entry signal. 10-day low = exit signal (for that system).
- Pyramid into winners: add 1 unit per 0.5 ATR move in your favour, max 4 units.
- Stop = 2 ATR from entry. Always. No discretion on stops.
- Never risk more than 2% of account on any single trade.
- "The markets are the same now as they were five to ten years ago because they keep changing."
- Trend is everything. Don't predict reversals — follow what IS happening.
- He turned $400 into $200M. The method: breakouts, pyramiding, rigid stops.

JIM SIMONS (1938–2024): Medallion Fund: +66% annually before fees for 30 years.
- Pure quant. Emotion = noise. Data = signal.
- Look for statistical divergences: when RSI says one thing and price says another, that's alpha.
- Relative strength: what is this stock doing vs its sector vs the market? Divergence = opportunity.
- Volume precedes price. Always. A move without volume is a rumour, not a fact.
- Mean reversion AND momentum exist simultaneously in different timeframes — know which regime you're in.
- "We don't override the model." — When the data says X, trade X.
- Pattern frequency: how often has THIS exact setup (RSI level + EMA position + volume) resolved bullish vs bearish historically?

LARRY WILLIAMS (born 1942): Won World Cup Trading Championship 1987, turning $10k into $1.1M in 12 months.
- Williams %R: his own indicator. Oversold below -80, overbought above -20.
- Volume exhaustion: when a downmove happens on DECLINING volume, the sellers are exhausted. Buy.
- "The market punishes the majority." — When everyone is bearish, look for the buy.
- Commitment of Traders (COT): commercials are always right at extremes. Follow the smart money.
- Seasonal patterns matter: certain stocks/sectors perform predictably at certain times of year.
- Open interest tells the story. Rising OI + rising price = strong bull. Rising OI + falling price = distribution.
- Key rule: only trade when 3+ of his signals align. One signal = noise.

═══ WHAT YOU MUST DO ═══

You are given: live price data + a pre-trained model score (logistic regression + decision tree trained on 12 historical crises). The model has already made a call. You must either:
A) CONFIRM the model's call with your qualitative analysis
B) OVERRIDE the model's call and explain exactly why (which trader's rule is being violated)

The user's message will include a MODE directive and a strict OUTPUT FORMAT template. Follow it exactly. Do not add preamble. Do not add a closing paragraph. Produce the format and nothing else.

═══ INVARIANT RULES — NON-NEGOTIABLE ═══
- FINAL verdict must be BUY or SELL. No AVOID, no WAIT, no HOLD, no "no trade today." Pick a side.
- The ONLY exception: genuine crisis conditions (ATR >4% of price AND momentum <-5% AND volume >3x). Only then may you output AVOID.
- R/R floor: 2:1 minimum. If a 2:1 can't be constructed with the current ATR, use a 1.5 ATR stop and 3.5 ATR target.
- When the setup is genuinely mixed, default to the decision-tree model's direction, then justify it through the trader lens with the cleanest entry you can find.
- Always give specific dollar levels for entry, stop and target. Entry should usually be the current price unless you want a specific limit level — say which.
- If RSI is 40-60 and EMA is flat: do NOT say WAIT. Look at Williams %R, Stochastic, CMF, volume trend. One will tip the bias — use it.
- Confidence calibration: HIGH = 3+ legends agree and model agrees. MEDIUM = 2 agree. LOW = mixed but you still pick a side.
- When asked "best opportunity" across multiple stocks: rank them, pick ONE, commit to the trade.`;

// ─── CRYPTO PERSONAS ────────────────────────────────────────────────────────
// Parallel to SYSTEM_PROMPT but with crypto-native voices. Picked because
// (a) each has a documented, distinct analytical school that transfers to
// crypto analysis, (b) all are publicly-known practitioners with crypto
// skin in the game, and (c) together they cover macro / on-chain / classical
// TA / derivatives / cycle analysis without overlap. Replaces the equity
// five in crypto mode so Claude isn't asked to LARP equity legends on
// assets they never traded.
const CRYPTO_SYSTEM_PROMPT = `You are THE CRYPTO TRADER — the composite mind of five contemporary crypto-native analysts. You have been given a pre-trained quantitative model score alongside live market data. Your job is to give ONE clear, decisive verdict. Not "it could go either way." A real verdict.

═══ ABSOLUTE DIRECTIVE ═══
Every session you MUST produce an actionable BUY or SELL. Crypto markets trade 24/7 — at every moment there is directional opportunity, long or short. "AVOID" and "WAIT" are BANNED except in one specific case: crisis-level conditions where BTC realized vol has annualised >150%, 7-day momentum <-20%, AND funding rates are at extreme negative (>1% sustained). Anything else is a BUY or a SELL. If your first instinct is AVOID, look harder — pick the side with marginally better odds. A 55/45 edge played with discipline compounds.

═══ THE FIVE MINDS ═══

RAOUL PAL (born 1969, founder of Real Vision Group, ex-Goldman macro hedge fund manager):
- Core framework: macro-to-crypto bridge. The "Everything Code" and "Exponential Age" — central bank liquidity is the dominant driver of crypto returns. M2 money supply, Fed balance sheet, DXY, real yields → BTC.
- "Network Value" thinking: Metcalfe's Law applied to crypto — value scales with active-addresses squared. ETH and SOL as "digital economies," not currencies.
- Cycle overlay: Global Business Cycle (4-year) + BTC halving cycle (4-year) are synced. "We're in the summer of the cycle" / "banana zone" etc.
- Horizon: multi-month to multi-year. Rarely a day-trader; takes macro-scale views.
- Quotes: "This is the greatest macro trade of our lifetimes." "The business cycle drives everything."
- Translation to BUY/SELL: macro-regime framing, risk-on vs risk-off, broad crypto beta rather than alt-specific picks.

WILLY WOO (born ~1971, on-chain analyst, ex-software engineer turned Bitcoin researcher):
- Core framework: Bitcoin is a network, trading is about flows through it. On-chain metrics > chart patterns.
- Key indicators: Realized Cap (aggregate cost basis of all BTC), NVT ratio (network value to transactions), HODL Waves (coin age distribution), RHODL (short-term vs long-term holder ratio), Spent Output Profit Ratio (SOPR — capitulation signal when <1).
- "Smart money vs dumb money" via on-chain cohort analysis. Old coins moving = whales positioning.
- Horizon: weeks to months. Data-driven like Simons but from blockchain data not order flow.
- Quotes: "Price follows realized cap." "HODLers control supply dynamics."
- Translation to BUY/SELL: on-chain accumulation/distribution signals, realized-cap floor, holder-cohort positioning.

PETER BRANDT (born 1947, 50-year classical chartist, one of the oldest actively-trading chart analysts):
- Core framework: classical technical analysis — head-and-shoulders, wedges, flags, pennants, triangles. "I don't care what's traded, I care what the chart shows."
- Applied equity/commodity TA to crypto starting 2013. Famously called BTC's 2017 top via "parabolic blow-off" and 2021 H&S top in real-time.
- Prefers WEEKLY charts. "Daily is noise. Weekly is signal. Monthly is trend."
- 50%-retracement rule: trends often retest the 50% Fibonacci of their move.
- Horizon: swing to position (weeks to months). Takes his time; won't chase.
- Quotes: "Let the market prove you right." "I am in the business of not losing money."
- Translation to BUY/SELL: pattern recognition on daily/weekly, confirmed-break entries, wide stops on weekly structure.

ARTHUR HAYES (born 1985, co-founder and ex-CEO of BitMEX, derivatives specialist):
- Core framework: monetary mechanics + derivatives positioning. Follow the dealer. Follow the funding.
- Key tools: perpetual funding rates (positive = bullish bias crowd, negative = bearish), basis (spot-perp spread), open interest changes, liquidation cascades. "Funding tells you where retail is; basis tells you where institutions are."
- Macro-liquidity framing: USD strength, JGB yields, Japanese carry trade, Fed RRP balances → BTC cycles.
- Writes the "Crypto Trader Digest" blog; direct, profane, aggressive.
- Horizon: days to weeks, tactical. Often uses options. Aggressive sizing when conviction is high.
- Quotes: "Max pain for the most participants." "I'm a degenerate gambler with a PhD in finance."
- Translation to BUY/SELL: contrarian on extreme funding, aggressive on liquidation cascade setups, basis-arb context.

BENJAMIN COWEN (born 1987, founder of Into The Cryptoverse, PhD physics, quantitative analyst):
- Core framework: cycle analysis + log regression. BTC price lives inside a log-regression band; "low-risk" zones accumulate, "high-risk" zones trim.
- Key tools: Risk Metric (0-1 scale per asset), ETH/BTC ratio analysis, Bitcoin Dominance trends, halving-cycle timing (2012→2016→2020→2024→2028).
- Measured, quantitative, deliberately unexciting. "I don't care about your cope."
- Data over narrative: "If my model says accumulation zone, I accumulate regardless of sentiment."
- Horizon: multi-month to full-cycle (years). Thinks in 4-year BTC halving cycles.
- Quotes: "Time in the market > timing the market, except at cycle extremes." "Risk is high → take profit; risk is low → accumulate."
- Translation to BUY/SELL: risk-band position (low-risk = BUY, high-risk = SELL), cycle phase awareness, dominance regime.

═══ WHAT YOU MUST DO ═══

You are given: live crypto price data + a pre-trained quantitative model score (LR + GBM ensemble, neural net, regime-switching). The model has already made a directional call. You must either:
A) CONFIRM the model's call with your qualitative analysis
B) OVERRIDE the model's call and explain exactly why (which analyst's framework is being violated)

YOUR OUTPUT FORMAT — follow this exactly, every time:

📊 TAPE READING
[2-3 sentences. What is BTC (and the selected alt if different) doing right now? Above/below key MA? Range-bound or breaking out? Dominance trend? Funding regime? Be specific.]

🤖 MODEL SIGNAL: [BUY/SELL] [confidence]% — [one line explanation]

🧠 ANALYST CONSENSUS
PAL:    [one specific opinion — macro backdrop, liquidity regime, cycle phase]
WOO:    [one specific opinion — on-chain positioning, HODL cohort behavior, realized-cap context]
BRANDT: [one specific opinion — weekly chart pattern, 50%-retracement, breakout confirmation]
HAYES:  [one specific opinion — funding rate, basis, OI, liquidation setup]
COWEN:  [one specific opinion — risk-band position, cycle phase, ETH/BTC + dominance]

⚡ FINAL VERDICT: BUY or SELL (pick one — no other options)
Entry: $X.XX | Stop: $X.XX | Target: $X.XX | R/R: X:1
Confidence: [HIGH/MEDIUM/LOW]

⚠️ RISK: [One sentence. Position size (1-2% of account), crypto-specific risks — liquidation cascade, funding flip, weekend gap, acknowledge you can be wrong.]

CRITICAL RULES — NON-NEGOTIABLE:
- FINAL VERDICT must be BUY or SELL. No AVOID, no WAIT, no HOLD. Pick a side.
- The ONLY exception: genuine crisis conditions (annualised realized vol >150%, 7d momentum <-20%, sustained negative funding >1%).
- R/R floor: 2:1 minimum on swing (daily) horizon, 1.5:1 on intraday.
- When mixed, default to the model's compositeProb direction, then justify through whichever analyst's lens fits best.
- Cost awareness: spread + funding on crypto is 20-50 bps round-trip. Entry and target must clear those costs.
- Specific dollar levels for entry, stop, target ALWAYS. No "around $X" — commit to a number.
- Confidence calibration: HIGH = 3+ analysts agree AND model agrees. MEDIUM = 2 agree. LOW = mixed but you still pick a side.
- When asked "best opportunity" across coins: rank them, pick ONE, commit to the trade.
- Do NOT reference Livermore, Tudor Jones, Dennis, Simons, or Williams. You are the crypto panel. They did not trade crypto.`;

// ─── Output-format directives — injected into the user message per call ─────
// Keeping these out of the system prompt so we can switch modes without
// rebuilding the persona on every request.
const QUICK_DIRECTIVE = `
═══ MODE: QUICK — OUTPUT EXACTLY FOUR LINES, NOTHING ELSE ═══
⚡ {SYMBOL} — {BUY|SELL} @ \${entry}
SL \${stop} | TP \${target} | R/R {n}:1 | {HIGH|MED|LOW}
🤖 Model: {BUY|SELL} {probability}%
🧠 Consensus: {majority trader direction, one short phrase}

No tape reading. No per-trader breakdown. No risk paragraph. No markdown. Four lines. End.`;

const DEEP_DIRECTIVE = `
═══ MODE: IN-DEPTH — STRUCTURED BUT COMPACT, EACH SECTION ONE LINE ═══
📊 TAPE: {trend, volume, one key level — one line}
🤖 MODEL: {BUY|SELL} {pct}% — {one line why}

LIVERMORE: {one line}
JONES: {one line — does price hold 200MA?}
DENNIS: {one line — breakout? 2-ATR stop?}
SIMONS: {one line — statistical divergence?}
WILLIAMS: {one line — is the crowd wrong?}

⚡ {BUY|SELL} @ \${entry} | SL \${stop} | TP \${target} | R/R {n}:1 | {HIGH|MED|LOW}
⚠️ {one line — position size 1-2%, acknowledge downside}

No prose padding. No "let me analyse..." preamble. No closing summary. ~10 lines total.`;


// ─── Yahoo Finance fallback ─────────────────────────────────────────────────
// Finnhub's free tier only covers US stocks. For UK / non-US tickers we hit
// Yahoo's public chart endpoint via a CORS proxy (same pattern we use for the
// RSS news feed). Returns the same shape fetchQuote needs: price, prevClose,
// optionally a bar series and day high/low. Works for any Yahoo-supported
// symbol — .L (LSE), .DE (Xetra), .PA (Paris), .HK (HKEX), etc.
const YAHOO_PROXIES = [
  u => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
  u => `https://corsproxy.io/?${encodeURIComponent(u)}`,
];

async function fetchYahoo(symbol) {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1m&range=1d`;
  for (const proxy of YAHOO_PROXIES) {
    try {
      const res = await fetch(proxy(url), { signal: AbortSignal.timeout(10000) });
      if (!res.ok) continue;
      const data = await res.json();
      const r = data?.chart?.result?.[0];
      if (!r) continue;
      const m = r.meta || {};
      const q = r.indicators?.quote?.[0] || {};
      const price = m.regularMarketPrice;
      const prevClose = m.chartPreviousClose ?? m.previousClose;
      if (!price || !prevClose) continue;

      // Filter out nulls that Yahoo leaves for halted/pre-open minutes.
      const rawCloses  = (q.close  || []).filter(v => v != null);
      const rawHighs   = (q.high   || []).filter(v => v != null);
      const rawLows    = (q.low    || []).filter(v => v != null);
      const rawVolumes = (q.volume || []).map(v => v == null ? 0 : v);

      return {
        price, prevClose,
        dayHigh: m.regularMarketDayHigh,
        dayLow:  m.regularMarketDayLow,
        high52:  m.fiftyTwoWeekHigh,
        low52:   m.fiftyTwoWeekLow,
        volume:  m.regularMarketVolume,
        currency: m.currency,
        bars: rawCloses.length >= 30
          ? { closes: rawCloses, highs: rawHighs, lows: rawLows, volumes: rawVolumes }
          : null,
      };
    } catch { /* try next proxy */ }
  }
  return null;
}

function shouldUseYahoo(symbol) {
  // Route by symbol suffix:
  //   - Dot-suffix (TW.L, SAP.DE) = non-US exchange → Yahoo (Finnhub free
  //     doesn't cover)
  //   - Dash-USD suffix (BTC-USD, ETH-USD) = crypto → Yahoo (Finnhub doesn't
  //     do crypto spot; Yahoo chart endpoint covers majors natively)
  //   - Everything else → Finnhub (real-time US equities)
  if (isCryptoSymbol(symbol)) return true;
  return /\.[A-Z]{1,3}$/.test(symbol.toUpperCase());
}

// ─── Real intraday bar cache for US tickers ────────────────────────────────
// The indicator pipeline (RSI, MACD, EMA, BB, ATR, VWAP, ADX, Williams %R,
// Stochastic, CMF, etc.) needs a bar SERIES, not just a last price. Finnhub's
// free tier doesn't include the /stock/candle endpoint, so until this commit
// the US path was falling back to a Math.random() walker — meaning every
// "live" indicator was being computed on simulated bars. Fix: pull real
// intraday bars from Yahoo (which has them free for US names too), cache
// them for 3 minutes per symbol, and override the last close with Finnhub's
// real-time price each refresh so the endpoint stays current.
const barsCache = new Map();               // symbol → { bars, fetchedAt }
const BARS_CACHE_MS = 3 * 60 * 1000;        // 3 minutes

async function fetchIntradayBars(symbol) {
  const now = Date.now();
  const cached = barsCache.get(symbol);
  if (cached && now - cached.fetchedAt < BARS_CACHE_MS) return cached.bars;

  // 5-min bars over 1 day = ~78 bars; enough for all indicators
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=5m&range=1d`;
  for (const proxy of YAHOO_PROXIES) {
    try {
      const res = await fetch(proxy(url), { signal: AbortSignal.timeout(8000) });
      if (!res.ok) continue;
      const data = await res.json();
      const r = data?.chart?.result?.[0];
      if (!r) continue;
      const q = r.indicators?.quote?.[0] || {};
      const closes = [], highs = [], lows = [], volumes = [];
      const len = r.timestamp?.length || 0;
      for (let i = 0; i < len; i++) {
        // Skip nulls (halted minutes, pre-open, post-close gaps)
        if (q.close?.[i] == null || q.high?.[i] == null || q.low?.[i] == null) continue;
        closes.push(q.close[i]);
        highs.push(q.high[i]);
        lows.push(q.low[i]);
        volumes.push(q.volume?.[i] ?? 0);
      }
      if (closes.length < 30) continue; // need enough for EMA50, RSI14, BB20

      const bars = { closes, highs, lows, volumes };
      barsCache.set(symbol, { bars, fetchedAt: now });
      return bars;
    } catch { /* try next proxy */ }
  }
  // If the fetch failed BUT we still have a cached copy (even if expired),
  // return the stale copy — better than falling back to synthetic data.
  return cached?.bars || null;
}

// Run the raw candle series through the cleaning pipeline, then recompute
// indicators on the CLEANED bars so RSI/MACD/ATR/EMA reflect validated data.
function cleanAndRecompute(live, session, anchors, { skipCleaning = false } = {}) {
  // When skipCleaning=true (synthetic fallback bars), we bypass Hampel +
  // winsorisation. The synthetic walker generates a Gaussian random walk
  // with tiny per-bar moves — there are no real outliers to find, and
  // winsorising the anchor-induced jump was the original cause of the GLD
  // "suspect on every refresh" bug. Still applies anchors + clampOHLC.
  let closes, highs, lows, volumes, cleaning;
  if (skipCleaning) {
    closes = [...live.closes];
    highs = [...live.highs];
    lows = [...live.lows];
    volumes = [...live.volumes];
    if (anchors.last != null && closes.length > 0) closes[closes.length - 1] = anchors.last;
    if (anchors.first != null && closes.length > 0) closes[0] = anchors.first;
    // Clamp OHLC for consistency
    for (let i = 0; i < closes.length; i++) {
      if (closes[i] > highs[i]) highs[i] = closes[i];
      if (closes[i] < lows[i])  lows[i]  = closes[i];
    }
    cleaning = { hampelFlagged: 0, winsorised: 0, zeroVolFilled: 0, haltBars: 0, frozenBars: 0, ffillAborted: 0, totalTouched: 0, skipped: true };
  } else {
    const cleanedBars = cleanBars(
      { closes: live.closes, highs: live.highs, lows: live.lows, volumes: live.volumes },
      session,
      anchors,
    );
    ({ closes, highs, lows, volumes, cleaning } = cleanedBars);
  }
  const indicators = computeIndicators(closes, highs, lows, volumes);
  return {
    closes, highs, lows, volumes, ...indicators,
    sparkline: closes.slice(-30),
    dayHigh: Math.max(...highs.slice(-78)),
    dayLow:  Math.min(...lows.slice(-78)),
    cleaning,
  };
}

async function fetchQuote(symbol, finnhubKey) {
  const session = getMarketSession(symbol);

  // Non-US tickers (e.g. TW.L) go to Yahoo regardless of Finnhub key — free
  // Finnhub doesn't cover those exchanges.
  if (shouldUseYahoo(symbol)) {
    const y = await fetchYahoo(symbol);
    if (!y) {
      const mk = generateMockQuote(symbol);
      return { ...mk, session, quality: "suspect" };
    }
    const change = y.price - y.prevClose;
    const changePct = (change / y.prevClose) * 100;

    // Build a candle series: prefer Yahoo's real bars if we got enough,
    // otherwise fall back to the synthetic candle generator anchored to the
    // real Yahoo price/prevClose so indicators still compute.
    const rawLive = y.bars
      ? { closes: y.bars.closes, highs: y.bars.highs, lows: y.bars.lows, volumes: y.bars.volumes }
      : generateLiveIndicators(symbol, y.price, y.prevClose);
    const live = cleanAndRecompute(rawLive, session, { last: y.price });
    const { closes, highs, lows, volumes, cleaning } = live;

    const quant = {
      adx:        calcADX(highs, lows, closes),
      williamsR:  calcWilliamsR(highs, lows, closes),
      stochastic: calcStochastic(highs, lows, closes),
      roc:        calcROC(closes),
      zScore:     calcZScore(closes),
      cmf:        calcCMF(highs, lows, closes, volumes),
      maxDrawdown:calcMaxDrawdown(closes),
      sharpe:     calcSharpe(closes, 0.053, 252 * 78), // 5-min US session
    };
    const extendedMove = session !== "OPEN" ? { price: y.price, changePct, change } : null;
    const quality = assessQuality({
      flagged: cleaning.hampelFlagged,
      capped: cleaning.winsorised,
      zeroVolFilled: cleaning.zeroVolFilled,
      lastFetched: Date.now(),
      session,
    });

    return {
      symbol,
      price: y.price, change, changePct, prevClose: y.prevClose,
      high52: y.high52, low52: y.low52,
      dayHigh: y.dayHigh, dayLow: y.dayLow,
      volume: y.volume ?? volumes[volumes.length-1],
      currency: y.currency,
      ...live, quant,
      session, extendedMove,
      quality, cleaning,
      marketState: "LIVE", lastFetched: Date.now(), isMock: false,
      source: "YAHOO",
    };
  }

  if (!finnhubKey) {
    const m = generateMockQuote(symbol);
    const cleaned = cleanAndRecompute(m, session, { last: m.price });
    const quality = assessQuality({
      flagged: cleaned.cleaning.hampelFlagged,
      capped: cleaned.cleaning.winsorised,
      zeroVolFilled: cleaned.cleaning.zeroVolFilled,
      lastFetched: Date.now(),
      session,
    });
    return { ...m, ...cleaned, session, quality };
  }
  try {
    const [quoteRes, metricRes] = await Promise.all([
      fetch(FH_QUOTE(symbol, finnhubKey), { signal: AbortSignal.timeout(8000) }),
      fetch(FH_METRIC(symbol, finnhubKey), { signal: AbortSignal.timeout(8000) }),
    ]);
    if (!quoteRes.ok) {
      const m = generateMockQuote(symbol);
      return { ...m, session, quality: "suspect" };
    }
    const quote = await quoteRes.json();
    const metric = metricRes.ok ? await metricRes.json() : null;
    const price = quote.c;
    const prevClose = quote.pc;
    if (!price || !prevClose) {
      const m = generateMockQuote(symbol);
      return { ...m, session, quality: "suspect" };
    }

    const change = price - prevClose;
    const changePct = (change / prevClose) * 100;
    const high52 = metric?.metric?.["52WeekHigh"] ?? quote.h;
    const low52  = metric?.metric?.["52WeekLow"]  ?? quote.l;

    // Fetch REAL intraday 5m bars from Yahoo (cached 3min per symbol). If that
    // succeeds, indicators are computed on actual market bars. If it fails
    // for any reason (CORS proxy down, rate-limited, etc.), fall back to the
    // synthetic walker and mark the resulting quote quality as "suspect" so
    // the user knows they're back on simulated bar data.
    const realBars = await fetchIntradayBars(symbol);
    const usedRealBars = realBars != null;
    const rawLive = usedRealBars
      ? realBars
      : generateLiveIndicators(symbol, price, prevClose);
    // Anchor last close to Finnhub's real-time price regardless of source —
    // Yahoo's final 5m bar can lag by up to 60 seconds. Skip cleaning on
    // synthetic fallback — running Hampel/winsorise on a random walk is
    // pure waste and causes false-positive "suspect" flags.
    const live = cleanAndRecompute(rawLive, session, { last: price }, { skipCleaning: !usedRealBars });
    const { closes, highs, lows, volumes, cleaning } = live;

    const quant = {
      adx:        calcADX(highs, lows, closes),
      williamsR:  calcWilliamsR(highs, lows, closes),
      stochastic: calcStochastic(highs, lows, closes),
      roc:        calcROC(closes),
      zScore:     calcZScore(closes),
      cmf:        calcCMF(highs, lows, closes, volumes),
      maxDrawdown:calcMaxDrawdown(closes),
      sharpe:     calcSharpe(closes, 0.053, 252 * 78), // 5-min US session
    };

    // When the market is NOT open, the Finnhub quote.c reflects extended-hours
    // price on supported symbols (or the last regular close if extended-hours
    // data isn't available). Compute the extended-hours move against prevClose
    // so we still have something meaningful when the bell isn't ringing.
    const extendedMove = session !== "OPEN" ? { price, changePct, change } : null;

    const baseQuality = assessQuality({
      flagged: cleaning.hampelFlagged,
      capped: cleaning.winsorised,
      zeroVolFilled: cleaning.zeroVolFilled,
      lastFetched: Date.now(),
      session,
    });
    // Force "suspect" when we fell back to synthetic bars — the price is
    // real but the indicators are computed on Math.random() bars, and the
    // user deserves to see that flagged.
    const quality = usedRealBars ? baseQuality : "suspect";

    return {
      symbol, price, change, changePct, prevClose,
      high52, low52,
      dayHigh: quote.h, dayLow: quote.l,
      open: quote.o,
      volume: quote.v ?? volumes[volumes.length-1],
      ...live, quant,
      session, extendedMove,
      quality, cleaning,
      barsSource: usedRealBars ? "yahoo-5m" : "synthetic",
      marketState: "LIVE", lastFetched: Date.now(), isMock: false,
      source: "FINNHUB",
    };
  } catch {
    const m = generateMockQuote(symbol);
    return { ...m, session, quality: "suspect" };
  }
}

function Sparkline({ data, up, width=80, height=28 }) {
  if (!data||data.length<2) return null;
  const min=Math.min(...data), max=Math.max(...data), range=max-min||1;
  const pts = data.map((v,i)=>`${(i/(data.length-1))*width},${height-((v-min)/range)*height}`).join(" ");
  return (
    <svg width={width} height={height} style={{display:"block"}}>
      <polyline points={pts} fill="none" stroke={up?"#2ECC71":"#E74C3C"} strokeWidth="1.5" opacity="0.85"/>
    </svg>
  );
}

function RSIBar({ rsi }) {
  if (rsi==null) return <span style={{color:"#555",fontSize:10}}>—</span>;
  const color = rsi>70?"#E74C3C":rsi<30?"#2ECC71":"#C9A84C";
  return (
    <div style={{display:"flex",alignItems:"center",gap:4}}>
      <div style={{width:44,height:5,background:"#222",borderRadius:2,overflow:"hidden"}}>
        <div style={{width:`${rsi}%`,height:"100%",background:color}}/>
      </div>
      <span style={{fontSize:9,color,letterSpacing:1}}>{rsi.toFixed(0)} {rsi>70?"OB":rsi<30?"OS":"NEU"}</span>
    </div>
  );
}

function ApiKeyModal({ onSave }) {
  const [ak, setAk] = useState(()=>localStorage.getItem("anthropic_key")||"");
  const [fk, setFk] = useState(()=>localStorage.getItem("finnhub_key")||"");
  const [pk, setPk] = useState(()=>localStorage.getItem("polygon_key")||"");
  const valid = ak.startsWith("sk-") && fk.length > 5;  // Polygon is optional
  function save() {
    if (!valid) return;
    localStorage.setItem("anthropic_key", ak);
    localStorage.setItem("finnhub_key", fk);
    if (pk) localStorage.setItem("polygon_key", pk);
    else    localStorage.removeItem("polygon_key");
    onSave(ak, fk, pk);
  }
  return (
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.88)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:999}}>
      <div style={{background:"#0F0F0F",border:"1px solid #C9A84C",padding:28,width:420}}>
        <div style={{fontSize:14,fontWeight:900,color:"#C9A84C",letterSpacing:3,marginBottom:8}}>◈ API KEYS</div>
        <div style={{fontSize:10,color:"#666",marginBottom:16,lineHeight:1.7}}>
          Keys are stored locally in your browser only. Polygon is optional — without it, backtests fall back to Yahoo and are capped at ~7 days of 5-min history.
        </div>
        <div style={{fontSize:9,color:"#C9A84C",letterSpacing:2,marginBottom:4}}>ANTHROPIC KEY (AI analysis) — REQUIRED</div>
        <input value={ak} onChange={e=>setAk(e.target.value)} placeholder="sk-ant-..."
          style={{width:"100%",boxSizing:"border-box",background:"#080808",border:"1px solid #2A2A2A",
            color:"#D8D0C0",fontFamily:"'Courier New',monospace",fontSize:12,padding:"9px 12px",
            outline:"none",marginBottom:14}}/>
        <div style={{fontSize:9,color:"#C9A84C",letterSpacing:2,marginBottom:4}}>FINNHUB KEY (live US prices) — REQUIRED</div>
        <input value={fk} onChange={e=>setFk(e.target.value)} placeholder="your finnhub key..."
          style={{width:"100%",boxSizing:"border-box",background:"#080808",border:"1px solid #2A2A2A",
            color:"#D8D0C0",fontFamily:"'Courier New',monospace",fontSize:12,padding:"9px 12px",
            outline:"none",marginBottom:14}}/>
        <div style={{fontSize:9,color:"#7FD8A6",letterSpacing:2,marginBottom:4}}>POLYGON KEY (long-horizon backtest data) — OPTIONAL</div>
        <input value={pk} onChange={e=>setPk(e.target.value)} placeholder="leave blank to use Yahoo (7d cap)..."
          onKeyDown={e=>e.key==="Enter"&&save()}
          style={{width:"100%",boxSizing:"border-box",background:"#080808",border:"1px solid #1A3A2A",
            color:"#D8D0C0",fontFamily:"'Courier New',monospace",fontSize:12,padding:"9px 12px",
            outline:"none",marginBottom:14}}/>
        <button onClick={save} disabled={!valid}
          style={{width:"100%",background:valid?"#C9A84C":"#1A1A1A",color:valid?"#000":"#444",
            border:"none",fontFamily:"'Courier New',monospace",fontWeight:900,fontSize:11,
            letterSpacing:2,padding:10,cursor:valid?"pointer":"not-allowed"}}>
          SAVE &amp; CONNECT
        </button>
        <div style={{fontSize:9,color:"#444",marginTop:10,lineHeight:1.6}}>
          Anthropic: console.anthropic.com · Finnhub: finnhub.io · Not financial advice.
        </div>
      </div>
    </div>
  );
}

// Rank all loaded quotes by model edge and return the top N
function rankOpportunities(quotes, topN = 3, context = {}) {
  // context.earningsMap is { symbol → earnings[] } — per-symbol PEAD data
  // that needs per-symbol context building. macro/calendar are shared.
  // context.fundingMap: Map<symbol, fundingRecords[]> for crypto funding-z.
  const { earningsMap = {}, fundingMap = new Map(), ...sharedCtx } = context;
  const isCrypto = isCryptoUniverse(sharedCtx.universe);
  return Object.values(quotes)
    .filter(q => q && q.price)
    .map(q => {
      const pead = earningsMap[q.symbol] ? computePeadFeatures(earningsMap[q.symbol]) : null;
      // Per-symbol crypto context: swap in this symbol's own tsMom + its
      // XS rank within the active universe + its own funding-z. Without
      // per-symbol switching, all candidates would share the selected
      // symbol's crypto features and the ranking would misrepresent each
      // candidate.
      const perSymbolCtx = isCrypto && q.closes ? {
        ...sharedCtx,
        macro: sharedCtx.macro ? {
          ...sharedCtx.macro,
          cryptoContext: {
            ...(sharedCtx.macro.cryptoContext || {}),
            tsMom: timeSeriesMomentum(q.closes),
            xsMomRank: sharedCtx.universe === "btc" ? 0 : xsMomRankLive(q.symbol, quotes),
            fundingZ: fundingZLive(fundingMap.get(q.symbol)),
            rvRatio: rvRatioLive(q.closes, 5, 30),
            dowSin: dayOfWeekSinAt(Math.floor(Date.now() / 1000)),
          },
        } : null,
      } : sharedCtx;
      const m = scoreSetup(q, { ...perSymbolCtx, pead });
      const edge = Math.abs(parseFloat(m.lrProb) - 50); // 0–50, higher = more conviction
      const treeBoost = m.treeSignal === "STRONG_BUY" || m.treeSignal === "STRONG_SELL" ? 8 : 0;
      const volBoost = (q.volRatio ?? 1) > 1.3 ? 4 : 0;
      const score = edge + treeBoost + volBoost;
      return { q, m, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);
}

function buildBestOpportunityContext(quotes, news = [], context = {}) {
  const top = rankOpportunities(quotes, 3, context);
  const blocks = top.map(({ q, m }, i) => {
    const pct52 = q.high52 && q.low52 ? ((q.price - q.low52) / (q.high52 - q.low52) * 100).toFixed(0) : "?";
    return `
--- CANDIDATE ${i + 1}: ${q.symbol} ---
Price: $${q.price?.toFixed(2)}  Change: ${q.changePct >= 0 ? "+" : ""}${q.changePct?.toFixed(2)}%  52W pct: ${pct52}th
RSI: ${q.rsi?.toFixed(1) || "N/A"}  MACD: ${q.macd != null ? (q.macd > 0 ? "BULL" : "BEAR") : "N/A"}  ATR: $${q.atr?.toFixed(2) || "N/A"}
EMA9/20/50: $${q.ema9?.toFixed(2) || "?"} / $${q.ema20?.toFixed(2) || "?"} / $${q.ema50?.toFixed(2) || "?"}
BB: ${q.bb ? (q.bb.pos * 100).toFixed(0) + "% of range" : "N/A"}  VWAP: $${q.vwap?.toFixed(2) || "N/A"}  Vol: ${q.volRatio?.toFixed(1) || "N/A"}x
ADX: ${q.quant?.adx?.adx?.toFixed(1) || "N/A"}  Williams%R: ${q.quant?.williamsR?.toFixed(1) || "N/A"}  CMF: ${q.quant?.cmf?.toFixed(3) || "N/A"}
LR Score: ${m.lrProb}% (${m.direction})  Tree: ${m.treeSignal} — ${m.treeReason}
Crisis Match: ${m.crisis?.name || "N/A"} (${m.crisis ? (m.crisis.similarity * 100).toFixed(0) : 0}%)
Suggested LONG: entry $${q.price?.toFixed(2)}, stop $${m.stopLong || "?"}, target $${m.tgt3Long || "?"}
Suggested SHORT: entry $${q.price?.toFixed(2)}, stop $${m.stopShort || "?"}, target $${m.tgt3Short || "?"}`;
  }).join("\n");

  const headlineBlurb = news.length > 0
    ? `\n=== TOP HEADLINES (last 100 days) ===\n${news.slice(0, 10).map(n => `[${n.source || ""}] ${n.title}`).join("\n")}`
    : "";

  return `=== BEST OPPORTUNITY SCAN — ${new Date().toLocaleTimeString()} ===
The system has ranked ALL ${Object.keys(quotes).length} watchlist symbols by model conviction. The top 3 candidates are presented below. You MUST choose exactly ONE and provide a full actionable trade.
${blocks}
${headlineBlurb}

INSTRUCTION: Review all three candidates. Choose the ONE with the cleanest, highest-conviction setup right now. State the chosen symbol on the first line as "PICK: {SYMBOL}". Then follow the IN-DEPTH format below.
${DEEP_DIRECTIVE}`;
}

function buildContext(quotes, selected, news = [], context = {}) {
  const q = quotes[selected];
  if (!q) return `[No data for ${selected}]`;
  const pct52 = q.high52&&q.low52 ? ((q.price-q.low52)/(q.high52-q.low52)*100).toFixed(0) : "?";
  const model = scoreSetup(q, context);
  const session = q.session || getMarketSession(selected);
  const sessLine = session === "OPEN"
    ? `Market session: OPEN`
    : `Market session: ${sessionLabel(session)} — extended-hours price $${q.price?.toFixed(2)} vs prev close $${q.prevClose?.toFixed(2)} (${q.changePct>=0?"+":""}${q.changePct?.toFixed(2)}%). Regular bell has NOT rung; treat intraday indicators (VWAP, volume ratio, 5-bar momentum) as stale from the prior session.`;
  const qualityLine = `Data quality: ${cleaningSummary(q.cleaning, q.quality || "unknown")}${q.quality === "suspect" ? " — treat technicals with reduced confidence." : q.quality === "stale" ? " — quote hasn't updated recently; act only if independently confirmed." : ""}`;
  const snapshot = Object.values(quotes).map(d=>
    `${d.symbol.padEnd(5)} $${d.price?.toFixed(2).padStart(8)} ${(d.changePct>=0?"+":"")+d.changePct?.toFixed(2)}%  RSI:${d.rsi?.toFixed(0)||"?"}  Vol:${d.volRatio?.toFixed(1)||"?"}x  [${sessionLabel(d.session||getMarketSession(d.symbol))}${d.quality && d.quality !== "clean" ? " " + d.quality.toUpperCase() : ""}]`
  ).join("\n");
  return `=== LIVE DATA ${q.isMock?"(SIMULATED)":""} — ${new Date().toLocaleTimeString()} ===
${sessLine}
${qualityLine}
SYMBOL: ${selected}  Price: $${q.price?.toFixed(2)}  Change: ${q.changePct>=0?"+":""}${q.changePct?.toFixed(2)}%
Day Range: $${q.dayLow?.toFixed(2)||"?"}–$${q.dayHigh?.toFixed(2)||"?"}
52W Range: $${q.low52?.toFixed(2)||"?"}–$${q.high52?.toFixed(2)||"?"} (${pct52}th pct)
5-bar momentum: ${q.momentum5!=null?(q.momentum5>=0?"+":"")+q.momentum5.toFixed(2)+"%":"N/A"}
Volume: ${q.volRatio?.toFixed(1)||"N/A"}x avg

TECHNICALS:
RSI(14): ${q.rsi?.toFixed(1)||"N/A"} ${q.rsi>70?"⚠ OVERBOUGHT":q.rsi<30?"✓ OVERSOLD":""}
MACD: ${q.macd?.toFixed(3)||"N/A"} (${q.macd!=null?(q.macd>0?"BULLISH":"BEARISH"):"N/A"})
EMA9/20/50: $${q.ema9?.toFixed(2)||"?"} / $${q.ema20?.toFixed(2)||"?"} / $${q.ema50?.toFixed(2)||"?"}
EMA Trend: ${q.ema9&&q.ema20?(q.ema9>q.ema20?"SHORT-TERM BULL":"SHORT-TERM BEAR"):"N/A"} | ${q.ema20&&q.ema50?(q.ema20>q.ema50?"MED-TERM BULL":"MED-TERM BEAR"):"N/A"}
VWAP: $${q.vwap?.toFixed(2)||"N/A"} → ${q.vwap?(q.price>q.vwap?"ABOVE (intraday strength)":"BELOW (intraday weakness)"):"N/A"}
ATR(14): $${q.atr?.toFixed(2)||"N/A"}
Bollinger: ${q.bb?(q.bb.pos*100).toFixed(0)+"% of range (upper=$"+q.bb.upper.toFixed(2)+", lower=$"+q.bb.lower.toFixed(2)+")":"N/A"}

=== QUANT ANALYSIS (Institutional Grade) ===
ADX(14): ${q.quant?.adx?.adx?.toFixed(1)||"N/A"} ${q.quant?.adx?.adx>25?"(TRENDING)":"(WEAK TREND)"}  DI+: ${q.quant?.adx?.diPlus?.toFixed(1)||"?"}  DI-: ${q.quant?.adx?.diMinus?.toFixed(1)||"?"}
Williams %R: ${q.quant?.williamsR?.toFixed(1)||"N/A"} ${q.quant?.williamsR<-80?"OVERSOLD":q.quant?.williamsR>-20?"OVERBOUGHT":"NEUTRAL"}
Stochastic: K=${q.quant?.stochastic?.k?.toFixed(1)||"?"}  D=${q.quant?.stochastic?.d?.toFixed(1)||"?"}
ROC(10): ${q.quant?.roc!=null?(q.quant.roc>=0?"+":"")+q.quant.roc.toFixed(2)+"%":"N/A"}
Z-Score(20): ${q.quant?.zScore?.toFixed(2)||"N/A"} ${Math.abs(q.quant?.zScore||0)>2?"⚠ EXTENDED":"within normal range"}
Chaikin Money Flow: ${q.quant?.cmf?.toFixed(3)||"N/A"} ${q.quant?.cmf>0.1?"BUYING PRESSURE":q.quant?.cmf<-0.1?"SELLING PRESSURE":"NEUTRAL"}
Max Drawdown: ${q.quant?.maxDrawdown?.toFixed(2)||"N/A"}%
Sharpe (ann.): ${q.quant?.sharpe?.toFixed(2)||"N/A"}

=== FEATURE ENGINEERING ===
${(()=>{ const f=engineerFeatures(q, quotes); return [
  `EMA Alignment: ${f.emaScore}/3 — ${f.emaLabel}`,
  `Momentum Composite: ${f.momentumComposite}% — ${f.momentumLabel}`,
  `Volatility Regime: ${f.volRegime}  ATR%: ${f.atrPct}%`,
  `BB State: ${f.bbState||"N/A"}  Bandwidth: ${f.bbBandwidth||"?"}%`,
  `Relative Strength vs SPY: ${f.relStrVsSpy!=null?(f.relStrVsSpy>=0?"+":"")+f.relStrVsSpy+"%":"N/A"}`,
  `VWAP Deviation: ${f.vwapDev!=null?(f.vwapDev>=0?"+":"")+f.vwapDev+"%":"N/A"}`,
  `Volume Trend: ${f.volTrendLabel||"N/A"} (${f.volTrend||"?"}x recent vs prior)`,
].join("\n"); })()}

=== PRE-TRAINED MODEL OUTPUT ===
LR Probability (bullish): ${model.lrProb}%
Direction: ${model.direction} | Confidence: ${model.confidence}%
Decision Tree: ${model.treeSignal} — ${model.treeReason}
Nearest Crisis Analogue: ${model.crisis?.name||"N/A"} (${model.crisis?(model.crisis.similarity*100).toFixed(0)+"% similarity":"N/A"})
Crisis Note: ${model.crisis?.note||"N/A"}
Suggested LONG: entry $${q.price?.toFixed(2)}, stop $${model.stopLong||"?"}, target $${model.tgt3Long||"?"}
Suggested SHORT: entry $${q.price?.toFixed(2)}, stop $${model.stopShort||"?"}, target $${model.tgt3Short||"?"}

Last 10 closes: ${q.closes?.slice(-10).map(c=>"$"+c?.toFixed(2)).join(", ")||"N/A"}

=== LIVE NEWS — GEOPOLITICAL & MARKET CONTEXT (last 100 days) ===
${news.length>0 ? news.slice(0,20).map(n=>`[${n.date.toLocaleDateString()}][${n.source||"News"}] ${n.title}`).join("\n") : "[No news loaded — fetch from NEWS tab]"}

=== WATCHLIST SNAPSHOT ===
${snapshot}`;
}

// ─── Verdict parser — tolerates both QUICK and IN-DEPTH output formats ─────
// QUICK:  "⚡ NVDA — BUY @ $875.40"   "SL $853 | TP $942 | R/R 3:1 | HIGH"
// DEEP:   "⚡ BUY @ $875.40 | SL $853 | TP $942 | R/R 3:1 | HIGH"
function parseTradeData(reply, q, symbol, context = {}) {
  if (!q) return null;
  const verdictMatch = reply.match(/[⚡]\s*(?:[A-Z.]{1,6}\s+[—–-]\s+)?(BUY|SELL|AVOID)\b/i)
                    || reply.match(/FINAL VERDICT:\s*(BUY|SELL|AVOID)/i)
                    || reply.match(/\b(BUY|SELL|AVOID)\b\s+@\s*\$/i);
  if (!verdictMatch) return null;
  // If the caller passed earningsMap, resolve per-symbol PEAD for this
  // specific trade. Otherwise context.pead (if set for the selected symbol)
  // passes through unchanged.
  const { earningsMap, ...sharedCtx } = context;
  const perSymCtx = earningsMap?.[symbol]
    ? { ...sharedCtx, pead: computePeadFeatures(earningsMap[symbol]) }
    : context;
  const model = scoreSetup(q, perSymCtx);
  const stop   = parseFloat(reply.match(/(?:Stop|SL):\s*\$?([\d.]+)/i)?.[1]) || null;
  const target = parseFloat(reply.match(/(?:Target|TP):\s*\$?([\d.]+)/i)?.[1]) || null;
  const rr     = parseFloat(reply.match(/R\/R:?\s*([\d.]+)/i)?.[1]) || null;
  const confidence = reply.match(/\b(HIGH|MEDIUM|MED|LOW)\b/i)?.[1]?.toUpperCase() || null;
  return {
    symbol, entryPrice: q.price,
    verdict: verdictMatch[1].toUpperCase(),
    stop, target, rr,
    confidence: confidence === "MED" ? "MEDIUM" : confidence,
    modelScore: { direction: model.direction, confidence: model.confidence, treeSignal: model.treeSignal },
    features: model.features,
  };
}

export default function App() {
  const [quotes, setQuotes] = useState({});
  const [selected, setSelected] = useState("SPY");
  const [messages, setMessages] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [tab, setTab] = useState("chat");
  const [apiKey, setApiKey] = useState(()=>localStorage.getItem("anthropic_key")||"");
  const [finnhubKey, setFinnhubKey] = useState(()=>localStorage.getItem("finnhub_key")||"");
  // Polygon key is OPTIONAL — only used by the backtest for long-horizon bars.
  // Its presence/absence has zero effect on the live quote loop (Finnhub + Yahoo
  // fallback), the cleaning pipeline, or the LR/NN models.
  const [polygonKey, setPolygonKey] = useState(()=>localStorage.getItem("polygon_key")||"");
  const [decisionLog, setDecisionLog] = useState(()=>getLog());
  const [loggedMsgIds, setLoggedMsgIds] = useState(new Set());
  const [news, setNews] = useState([]);
  const [newsLoading, setNewsLoading] = useState(false);
  // Macro snapshot (VIX, VIX term, DXY/TNX/Oil/Gold momentum, credit spread)
  // refreshed every 2 minutes in parallel with quotes. scoreSetup consumes
  // this to compute the 7 new macro+calendar features alongside the 7 legacy
  // technicals. `null` until the first fetch completes; all-zero feature
  // contributions in that gap (safe default).
  const [macro, setMacro] = useState(null);
  // Earnings history per symbol — Finnhub /stock/earnings results. Cached
  // 1 week in localStorage; refreshed lazily on mount. Feeds the two PEAD
  // features (daysSinceEarnings + surpriseDecayed). Missing data = zero
  // PEAD contribution, safe default.
  const [earningsMap, setEarningsMap] = useState({});
  const [simState, setSimState] = useState({ running: false, phase: null, symbol: null, done: 0, total: 0 });
  const [simResult, setSimResult] = useState(null);
  // Max-hold (timeout) for the simulator. Stop and target still exit early
  // on the first bar that touches them — this only governs how long the trade
  // is held when NEITHER stop nor target has been hit.
  const [maxHoldHours, setMaxHoldHours] = useState(3);
  // Horizon mode: "5m" (intraday 5-min bars, hold 1-24h) vs "1d" (daily
  // bars, hold 1-20 days). The daily horizon is where published retail
  // effects like PEAD and factor momentum live; intraday is mostly noise
  // net of costs. Changing this resets maxHoldHours to a sensible default.
  const [simInterval, setSimInterval] = useState("5m");
  // Universe: "equities" (default, the existing US stocks + UK pipeline) or
  // "crypto" (BTC/ETH/alts via Yahoo). Switches the watchlist, session
  // handling, cost defaults, and which model guards fire (e.g. PEAD
  // disabled on crypto since there's no earnings concept).
  const [universe, setUniverse] = useState("equities");
  // Runtime-resolved watchlists. Use these throughout the live loop +
  // sim dispatch rather than the compile-time WATCHLIST constant, so that
  // switching universe actually changes which symbols are fetched.
  const activeWatchlist = watchlistFor(universe);
  const activeBacktestSymbols = backtestSymbolsFor(universe);
  const activeHighVolSymbols = highVolSymbolsFor(universe);
  // Diagnostic: temporarily exclude high-vol speculative names (IONQ, RGTI)
  // from the sim. Their 5-6% per-bar vol dominates run-to-run variance.
  // Toggle on to test whether the underlying signal is stable once the
  // noise sources are removed. Live dashboard / live ANALYZE unaffected.
  const [excludeHighVol, setExcludeHighVol] = useState(false);
  // Round-trip transaction cost applied to every simulated trade's P&L.
  const [costBps, setCostBps] = useState(15);
  // How far back the backtester fetches bars. Capped at 7 on Yahoo, unlimited
  // with Polygon. Set higher for more training data (and more regime variety).
  const [simDaysAgo, setSimDaysAgo] = useState(7);
  const [wfResult, setWfResult] = useState(null);
  const [wfRunning, setWfRunning] = useState(false);
  const [multiSimResult, setMultiSimResult] = useState(null);
  const [multiSimRunning, setMultiSimRunning] = useState(false);
  const [multiSimState, setMultiSimState] = useState({ phase: null, run: 0, total: 0 });
  // Number of sims in the multi-sim batch. Default 20 — enough for a tight
  // 95% CI on the mean AUC (~±0.02) without running for 30 minutes.
  // 10 = quick exploratory, 20 = standard, 50 = thorough.
  const [multiSimNRuns, setMultiSimNRuns] = useState(20);
  const [trainResult, setTrainResult] = useState(null);
  const [training, setTraining] = useState(false);
  const [ablationResult, setAblationResult] = useState(null);
  const [ablationRunning, setAblationRunning] = useState(false);
  const [ablationProgress, setAblationProgress] = useState({ idx: 0, total: 0, current: "" });
  // Phase 3d step 2: per-symbol perp funding-rate history, fetched in bulk
  // on refresh when universe is crypto. Map<symbol, records[]> where each
  // record is { time, rate }. Consumed by scoreSetup call sites via
  // fundingZLive(). Empty Map in equity mode.
  const [fundingMap, setFundingMap] = useState(new Map());
  // Phase 4.5: top-150 crypto snapshot for btc-universe broad-market context.
  // One CoinGecko call per refresh (cached in-module), populates XS rank +
  // breadth + raw dominance for live scoring of BTC without needing to pull
  // 150 bar series in the browser. Backtest uses the historical-bar pipeline.
  const [broadMarketSnapshot, setBroadMarketSnapshot] = useState(null);
  const chatRef = useRef(null);
  const importInputRef = useRef(null);

  // Cross-device sync: trigger the hidden file picker, parse the dropped JSON,
  // restore log + LR + NN weights, then refresh React state from localStorage
  // so the UI immediately reflects the imported state.
  function handleImportFile(file, mode = "merge") {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const payload = JSON.parse(reader.result);
        const restored = importState(payload, { mode });
        setDecisionLog(getLog());
        setLoggedMsgIds(new Set());
        alert(
          `Import OK.\n` +
          `Decision log: ${mode === "merge" ? `+${restored.log} new entries` : `${restored.log} entries restored`}\n` +
          `LR weights: ${restored.lrWeights ? "restored" : "not present in file"}\n` +
          `NN weights: ${restored.nnWeights ? "restored" : "not present in file"}`
        );
      } catch (err) {
        alert(`Import failed: ${err.message}`);
      }
    };
    reader.readAsText(file);
  }

  // refreshAll MUST depend on finnhubKey — otherwise the closure captures the
  // initial (possibly empty) key and quotes stay SIMULATED forever even after
  // the user saves a real key.
  //
  // Resilience rule: if a fresh fetch returned mock data (rate-limited, network
  // blip, Finnhub free tier doesn't support this symbol, etc.) but we already
  // had LIVE data for that symbol, keep the live data and flag it stale rather
  // than flipping the whole dashboard back to SIMULATED.
  const refreshAll = useCallback(async (silent=false) => {
    if (!silent) setRefreshing(true);
    // Fan out quotes + macro in parallel. Macro has its own 2-min cache so
    // this call is nearly free after the first refresh. When universe is
    // crypto, also fetch BTC dominance in parallel — feeds the dominanceZ
    // feature slot via macro.cryptoContext downstream.
    const isCryptoUni = isCryptoUniverse(universe);
    const [results, macroSnap, btcDom, funding, broadSnap] = await Promise.all([
      Promise.all(activeWatchlist.map(s=>fetchQuote(s, finnhubKey))),
      fetchMacroSnapshot().catch(() => null),
      // BTC dominance from CoinGecko /global — kept for crypto universe
      // (approximated feed). On btc we compute REAL dominance from the
      // top-150 snapshot below.
      universe === "crypto" ? fetchBTCDominance().catch(() => null) : Promise.resolve(null),
      // Funding rates: fetched for both "crypto" (40 symbols) and "btc"
      // (just BTC-USD). Session-cached inside funding.js.
      isCryptoUni
        ? fetchFundingForUniverse(activeWatchlist, { concurrency: 10 }).catch(() => new Map())
        : Promise.resolve(new Map()),
      // Phase 4.5: btc-only broad-market snapshot. CoinGecko is used here
      // specifically because it's the best free source for market-cap +
      // cross-sectional return metadata that Polygon doesn't provide. Not a
      // Polygon downgrade — different data layer entirely.
      universe === "btc"
        ? fetchTopCryptoSnapshot(150).catch(() => null)
        : Promise.resolve(null),
    ]);
    if (broadSnap) setBroadMarketSnapshot(broadSnap);
    if (funding && funding.size) setFundingMap(funding);
    setQuotes(prev => {
      const map = { ...prev };
      results.forEach((r, i) => {
        if (!r) return;
        const sym = activeWatchlist[i];
        const existing = map[sym];
        if (r.isMock && existing && !existing.isMock) {
          // Preserve prior live quote, downgrade quality tag
          map[sym] = { ...existing, quality: "stale" };
        } else {
          map[sym] = r;
        }
      });
      return map;
    });
    // Attach crypto context to macro when relevant. The cryptoContext field
    // is consumed by extractFeatures in crypto mode (slots 7-9) and ignored
    // in equity mode. tsMom is per-symbol so it's computed downstream in
    // sendToAI / rankOpportunities from each quote's own closes.
    const mergedMacro = macroSnap || btcDom ? {
      ...(macroSnap || {}),
      cryptoContext: btcDom ? {
        dominanceZ: btcDom.dominanceZ,
        dominance:  btcDom.dominance,
        // tsMom + xsMomRank filled in per-symbol at scoreSetup time
      } : undefined,
    } : null;
    if (mergedMacro) setMacro(mergedMacro);
    setLastRefresh(Date.now());
    if (!silent) setRefreshing(false);
    // universe must be a dep: switching it changes activeWatchlist, and
    // without the dep refreshAll would keep fetching the previous universe's
    // tickers. activeWatchlist itself is derived from universe (always a new
    // array ref per render) so adding it directly would rebind every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finnhubKey, universe]);

  // Re-run (and reset the interval) whenever refreshAll changes, i.e. when the
  // finnhub key changes. Also re-fetch when the tab becomes visible again after
  // being hidden, so you don't come back to stale prices.
  useEffect(() => {
    if (!finnhubKey) return;
    refreshAll();
    const id = setInterval(()=>refreshAll(true), 30000);
    const onVis = () => { if (document.visibilityState === "visible") refreshAll(true); };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      clearInterval(id);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [refreshAll, finnhubKey]);

  // News: refresh every 5 minutes so the feed stays current without the user
  // having to click REFRESH.
  useEffect(() => {
    const loadNews = () => {
      setNewsLoading(true);
      fetchAllNews()
        .then(articles => setNews(articles))
        .catch(()=>{})
        .finally(()=>setNewsLoading(false));
    };
    loadNews();
    const id = setInterval(loadNews, 5 * 60 * 1000);
    return () => clearInterval(id);
  }, []);

  // Earnings fetch — once per session. Caches 1 week in localStorage so the
  // Finnhub calls don't happen on every reload. PEAD features depend on this
  // map being populated; if the fetch fails (no key, symbol uncovered), the
  // feature just contributes zero — safe degradation.
  useEffect(() => {
    if (!finnhubKey) return;
    // Crypto has no earnings concept — don't waste 10 Finnhub calls per
    // session on BTC-USD etc (they'd fail silently anyway). When universe
    // is crypto, PEAD features stay at zero.
    if (isCryptoUniverse(universe)) {
      setEarningsMap({});
      return;
    }
    getEarningsBatch(activeBacktestSymbols, finnhubKey).then(setEarningsMap).catch(() => {});
    // Only refetch when finnhubKey or universe actually change — not on
    // every render (activeBacktestSymbols is a new array reference each time).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finnhubKey, universe]);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages, thinking]);

  async function sendToAI(userText, mode = "deep") {
    setThinking(true);
    // For crypto, augment macro with this symbol's time-series momentum —
    // computed from its own bars with no external API. Feeds the TS_mom_z
    // feature slot (which equity mode uses for VIX_term).
    const selectedQuote = quotes[selected];
    const isCryptoUni = isCryptoUniverse(universe);
    // Phase 4.5: btc universe uses top-150 snapshot for XS rank + breadth
    // + (raw) dominance live. Crypto universe keeps the in-watchlist
    // 40-coin rank; equity unchanged.
    const btcSnapEntry = universe === "btc" && broadMarketSnapshot
      ? broadMarketSnapshot.find(s => s.symbol === "BTC") : null;
    const cryptoCtx = (isCryptoUni && selectedQuote?.closes) ? {
      ...(macro?.cryptoContext || {}),
      tsMom: timeSeriesMomentum(selectedQuote.closes),
      // XS rank: btc → 150-coin snapshot; crypto → 40-coin live.
      xsMomRank: universe === "btc"
        ? (btcSnapEntry && broadMarketSnapshot
            ? xsRankLiveBroad(btcSnapEntry.ret14dPct, broadMarketSnapshot)
            : 0)
        : xsMomRankLive(selected, quotes),
      fundingZ: fundingZLive(fundingMap.get(selected)),
      rvRatio: rvRatioLive(selectedQuote.closes, 5, 30),
      // DOW on crypto universe; breadth on btc (replaced DOW).
      dowSin: dayOfWeekSinAt(Math.floor(Date.now() / 1000)),
      breadth: universe === "btc" && broadMarketSnapshot
        ? breadthLive(broadMarketSnapshot) : 0,
    } : macro?.cryptoContext;
    const modelCtx = {
      macro: macro ? { ...macro, cryptoContext: cryptoCtx } : null,
      calendar: calendarFeatures(),
      // No earnings on any crypto asset (multi or btc).
      pead: isCryptoUni ? null : computePeadFeatures(earningsMap[selected]),
      universe,
    };
    const context = buildContext(quotes, selected, news, modelCtx);
    const directive = mode === "quick" ? QUICK_DIRECTIVE : DEEP_DIRECTIVE;
    // Horizon preamble tells Claude the intended hold period, which
    // SUGGESTED LEVELS row applies, AND which of the five traders are
    // most historically relevant to this horizon. Without this the model
    // was forcing every persona into intraday scalping, which misuses
    // Livermore/Jones/Dennis (all primarily swing/position traders in
    // their actual careers) and makes Williams (a daytrader) the de-facto
    // voice. Horizon-aware weighting fixes this.
    const isSwing = simInterval === "1d";
    const selQForPreamble = quotes[selected];
    const dailyAtrEstimate = selQForPreamble?.atr ? selQForPreamble.atr * Math.sqrt(78) : null;
    // Universe prefix tells Claude which panel is answering. System prompt
    // already routed (equity SYSTEM_PROMPT vs CRYPTO_SYSTEM_PROMPT); this
    // preamble reinforces + adds horizon-specific persona weighting.
    const universePreamble = universe === "btc"
      ? `\n═══ ASSET UNIVERSE: BTC (single-asset) ═══\nThis session is a BTC-focused analysis. No cross-asset rotation to reason about; every frame is about Bitcoin specifically. The five analysts are Pal / Woo / Brandt / Hayes / Cowen. Reference: funding rates, halving-cycle phase, realized-cap cohort behavior, DVOL/IV, macro liquidity. BTC dominance is not useful as a signal here (BTC IS the reference). Do NOT compare to altcoin performance unless specifically asked.\n`
      : universe === "crypto"
      ? `\n═══ ASSET UNIVERSE: CRYPTO ═══\n${selected} is a cryptocurrency traded 24/7. No earnings concept, no FOMC calendar, no equity VIX. The five analysts are Pal / Woo / Brandt / Hayes / Cowen — NONE of the equity legends. Reference: BTC dominance, funding rates, halving-cycle phase, on-chain cohort positioning, DXY as macro backdrop (but not equity-centric technicals like VIX term structure).\n`
      : "";
    const horizonPreamble = isSwing
      ? (isCryptoUniverse(universe)
        ? `\n═══ INTENDED HORIZON: SWING (1-5 DAYS) ═══
Use SWING level set (2× daily-ATR stop, 6× daily-ATR target${dailyAtrEstimate ? ` — roughly $${(dailyAtrEstimate * 2).toFixed(2)} away for stop` : ""}). Weekly/daily chart structure, not 5-min chop. Entry timing can still be intraday but the trade is multi-day.

PERSONA WEIGHTING AT THIS HORIZON:
  BRANDT — DOMINANT. Weekly charts are his native timeframe. Speak to pattern completion, 50%-retracement, confirmed breakouts on daily/weekly.
  COWEN — DOMINANT. Risk-band position on daily close drives swing entries. Speak to current risk metric, log-regression band, ETH/BTC context.
  WOO — DOMINANT. On-chain positioning resolves over days-to-weeks. Speak to realized-cap levels, HODL cohort behavior, accumulation/distribution signals.
  HAYES — SECONDARY. Funding + basis matter at this horizon as regime context but his native timeframe is tighter. One line: is funding supportive or contrarian?
  PAL — SECONDARY. Macro liquidity backdrop (M2, DXY, RRP). Long horizon. Compress to one line: is the cycle-phase tailwind or headwind?
`
        : `\n═══ INTENDED HORIZON: SWING (1-5 DAYS) ═══
Use the SWING level set from SUGGESTED LEVELS (2× daily-ATR stop, 6× daily-ATR target${dailyAtrEstimate ? ` — roughly $${(dailyAtrEstimate * 2).toFixed(2)} away for stop` : ""}). Tape reading focuses on DAILY structure — 50-day MA, recent daily swings, multi-day setups — NOT 5-minute chop. Entry timing can still be intraday ("wait for a pullback to VWAP this session") but the trade is multi-day. Do NOT quote intraday levels as the primary SL/TP.

PERSONA WEIGHTING AT THIS HORIZON:
  LIVERMORE — DOMINANT. He held shorts for weeks in 1929 ("It was never my thinking, it was my sitting"). Speak to his real style: pivot points on daily charts, position building into confirmed breakouts, riding trends.
  TUDOR JONES — DOMINANT. Macro swing trader. 200-day MA is his daily-chart line of demarcation. 5:1 R/R works on multi-day holds. Speak to daily structure + macro.
  DENNIS — DOMINANT. Turtles were 20-day breakout traders holding weeks-to-months. Swing is his native horizon. Speak to daily breakouts, ATR-based sizing, pyramiding.
  SIMONS — SECONDARY. Adaptable. Speak to statistical divergences at the daily scale.
  WILLIAMS — SECONDARY. A daytrader by specialty. Compress his short-term ideas to "is the crowd wrong at this daily inflection?"
`)
      : (isCryptoUniverse(universe)
        ? `\n═══ INTENDED HORIZON: INTRADAY (1-3 HOURS) ═══
Use INTRADAY level set (1.5× 5-min ATR stop). Session-scale action — recent 4h structure, hourly levels, funding flips, liquidation zones.

PERSONA WEIGHTING AT THIS HORIZON:
  HAYES — DOMINANT. Derivatives desk at BitMEX — intraday perp dynamics are his bread and butter. Speak to funding rate, basis, OI moves, liquidation cascades.
  BRANDT — SECONDARY. Classical chart patterns scale down but he'd normally pass on intraday. Compress to "what's the hourly chart saying?"
  WOO — SECONDARY. On-chain is slow-moving; at intraday scale mostly stale. One line: is holder cohort behavior supportive this week?
  COWEN — SECONDARY. Risk metric is daily-close; intraday wobble doesn't change it. One line of cycle context only.
  PAL — SECONDARY. Macro is slow. One line of liquidity regime context.
`
        : `\n═══ INTENDED HORIZON: INTRADAY (1-3 HOURS) ═══
Use the INTRADAY level set from SUGGESTED LEVELS (1.5× 5-min ATR stop). Tape reading focuses on this session's structure — VWAP, day's range, opening auction, volume bursts. The trade closes before the bell.

PERSONA WEIGHTING AT THIS HORIZON:
  WILLIAMS — DOMINANT. World Cup daytrader. His %R, volume exhaustion, and COT signals are intraday-native. Speak confidently.
  SIMONS — DOMINANT. Short-horizon statistical arbitrage. Speak to divergences inside this session.
  LIVERMORE — SECONDARY. His pivot-point concept works intraday but compress to session levels, not multi-week. Don't pretend he's a scalper.
  TUDOR JONES — SECONDARY. Note the macro backdrop (Fed, dollar) but don't pretend this is his natural horizon — he'd hold multi-day. One line of macro context.
  DENNIS — SECONDARY. Turtles don't day-trade. If his rule applies it's because the daily breakout happens to coincide with today's session; otherwise note he'd pass.
`);
    const fullContent = `${context}${universePreamble}${horizonPreamble}\n${directive}\n\nUSER: ${userText}`;
    const newHistory = [...chatHistory, { role:"user", content:fullContent }];
    setChatHistory(newHistory);
    setMessages(prev=>[...prev, { type:"user", text:userText }]);
    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": apiKey,
          "anthropic-version": "2023-06-01",
          "anthropic-dangerous-direct-browser-access": "true",
        },
        body: JSON.stringify({
          model: "claude-opus-4-5",
          max_tokens: mode === "quick" ? 400 : 1800,
          // Per-universe system prompt. Crypto uses the crypto-native five
          // (Pal/Woo/Brandt/Hayes/Cowen); equity keeps the original legends.
          // Both share the same output format so parseTradeData regexes work
          // unchanged.
          system: isCryptoUniverse(universe) ? CRYPTO_SYSTEM_PROMPT : SYSTEM_PROMPT,
          messages: newHistory,
        }),
      });
      const data = await res.json();
      if (!res.ok||data.error) {
        setMessages(prev=>[...prev,{type:"bot",text:`⚠️ API error: ${data.error?.message||`HTTP ${res.status}`}`}]);
        setThinking(false); return;
      }
      const reply = data.content?.filter(b=>b.type==="text").map(b=>b.text).join("\n")||"No response.";
      setChatHistory(prev=>[...prev,{role:"assistant",content:reply}]);
      const q = quotes[selected];
      const tradeData = parseTradeData(reply, q, selected, modelCtx);
      const msgId = Date.now();
      setMessages(prev=>[...prev,{id: msgId, type:"bot", text:reply, tradeData}]);
    } catch(err) {
      setMessages(prev=>[...prev,{type:"bot",text:`⚠️ Connection failed: ${err.message}`}]);
    }
    setThinking(false);
  }

  function handleSend() {
    if (!input.trim()||thinking) return;
    const txt = input.trim(); setInput(""); sendToAI(txt);
  }

  async function sendBestOpportunity() {
    if (thinking || Object.keys(quotes).length < 3) return;
    setThinking(true);
    setTab("chat");
    const modelCtx = { macro, calendar: calendarFeatures(), earningsMap, universe, fundingMap };
    const context = buildBestOpportunityContext(quotes, news, modelCtx);
    // Horizon preamble — same shape as sendToAI including persona weighting.
    // Best-opportunity scans are most at risk of getting horizon-wrong
    // because the user isn't picking the symbol, so explicit guidance
    // matters even more.
    const isSwing = simInterval === "1d";
    const horizonPreamble = isSwing
      ? `\n═══ INTENDED HORIZON: SWING (1-5 DAYS) ═══
Rank and pick for a 1-5 day hold. Use SWING levels (2× daily-ATR stop, 6× daily-ATR target) for the FINAL VERDICT's SL/TP. Reject setups whose best edge is intraday-only.
Persona weighting: LIVERMORE / TUDOR JONES / DENNIS are DOMINANT (all historically swing/position traders). SIMONS / WILLIAMS are SECONDARY. Do NOT pretend Dennis day-trades or Williams holds for weeks.
`
      : `\n═══ INTENDED HORIZON: INTRADAY (1-3 HOURS) ═══
Rank and pick for a 1-3 hour hold. Use INTRADAY levels (1.5× 5-min ATR stop). Trade closes before bell.
Persona weighting: WILLIAMS / SIMONS are DOMINANT (intraday-native). LIVERMORE / TUDOR JONES / DENNIS are SECONDARY — compress their multi-day thinking to session-scale observations, don't force them into scalp framings.
`;
    setMessages(prev=>[...prev,{type:"user",text:"⚡ BEST OPPORTUNITY SCAN — rank all stocks and pick ONE trade now."}]);
    const newHistory = [...chatHistory, { role:"user", content: context + horizonPreamble }];
    setChatHistory(newHistory);
    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method:"POST",
        headers:{
          "Content-Type":"application/json",
          "x-api-key":apiKey,
          "anthropic-version":"2023-06-01",
          "anthropic-dangerous-direct-browser-access":"true",
        },
        body: JSON.stringify({
          model:"claude-opus-4-5",
          max_tokens:4096,
          system: isCryptoUniverse(universe) ? CRYPTO_SYSTEM_PROMPT : SYSTEM_PROMPT,
          messages:newHistory,
        }),
      });
      const data = await res.json();
      if (!res.ok||data.error) {
        setMessages(prev=>[...prev,{type:"bot",text:`⚠️ API error: ${data.error?.message||`HTTP ${res.status}`}`}]);
        setThinking(false); return;
      }
      const reply = data.content?.filter(b=>b.type==="text").map(b=>b.text).join("\n")||"No response.";
      setChatHistory(prev=>[...prev,{role:"assistant",content:reply}]);
      // Detect which symbol was picked (new "PICK: SYM" format, or fall back to
      // matching any watchlist symbol appearing early in the reply)
      const pickMatch = reply.match(/PICK:\s*([A-Z.]{2,6})/i);
      const symMatch = pickMatch || reply.match(/(?:picking|chosen?|trade on|go with)\s+([A-Z.]{2,6})/i);
      const pickedSym = symMatch ? activeWatchlist.find(s => s === symMatch[1].toUpperCase()) : null;
      const logSym = pickedSym || selected;
      const q = quotes[logSym];
      const tradeData = parseTradeData(reply, q, logSym, modelCtx);
      if (tradeData && pickedSym) setSelected(pickedSym);
      const msgId = Date.now();
      // Single setMessages call — previously this fired twice (once without
      // tradeData, once with), producing a duplicate bot message in the chat.
      setMessages(prev=>[...prev,{id: msgId, type:"bot", text:reply, tradeData}]);
    } catch(err) {
      setMessages(prev=>[...prev,{type:"bot",text:`⚠️ Connection failed: ${err.message}`}]);
    }
    setThinking(false);
  }

  // Buffett's verdict is SEPARATE from the trader system. His horizon (decades)
  // and framework (value, moats, intrinsic value) would contaminate a short-term
  // BUY/SELL signal, so we fire a second, independent Claude call with his own
  // system prompt and render it as a distinct message.
  async function sendToBuffett(userText) {
    if (thinking) return;
    setThinking(true);
    setTab("chat");
    const q = quotes[selected];
    const session = q?.session || getMarketSession(selected);
    const buffettCtx = buildBuffettContext(q, selected, session);
    const prompt = `${buffettCtx}\n\nUSER ASKS: ${userText || `Warren, what's your honest take on ${selected} right now?`}`;
    setMessages(prev=>[...prev,{type:"user",text:`🏛 BUFFETT: ${userText || `Your take on ${selected}?`}`}]);
    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": apiKey,
          "anthropic-version": "2023-06-01",
          "anthropic-dangerous-direct-browser-access": "true",
        },
        body: JSON.stringify({
          model: "claude-opus-4-5",
          max_tokens: 500,
          system: BUFFETT_SYSTEM_PROMPT,
          // Intentionally NOT using chatHistory — Buffett runs in his own
          // conversation so his framework doesn't bleed into the trader chat.
          messages: [{ role: "user", content: prompt }],
        }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        setMessages(prev=>[...prev,{type:"bot",persona:"buffett",text:`⚠️ API error: ${data.error?.message||`HTTP ${res.status}`}`}]);
        setThinking(false); return;
      }
      const reply = data.content?.filter(b=>b.type==="text").map(b=>b.text).join("\n") || "No response.";
      setMessages(prev=>[...prev,{type:"bot",persona:"buffett",text:reply}]);
    } catch(err) {
      setMessages(prev=>[...prev,{type:"bot",persona:"buffett",text:`⚠️ Connection failed: ${err.message}`}]);
    }
    setThinking(false);
  }

  // ─── Simulate & train — runs the backtest, feeds NN training ───────────
  // ─── Simulate ONLY — produce labelled trades + metrics, no training ───────
  // Splitting this from training (per the user's diagnosis) means you can
  // (a) inspect P&L / profit factor / equity curve before deciding to train,
  // (b) re-run sims at different timeouts without retraining each time,
  // (c) train repeatedly on the same simulated set if you like.
  async function runSimulation() {
    if (simState.running) return;
    const expectedSyms = excludeHighVol
      ? activeBacktestSymbols.filter(s => !activeHighVolSymbols.includes(s))
      : activeBacktestSymbols;
    setSimState({ running: true, phase: "starting", symbol: null, done: 0, total: expectedSyms.length });
    setSimResult(null);
    setTrainResult(null);
    try {
      // Sample count per symbol — needs to be tuned for the actual entry
      // range available per mode, otherwise pickEntries can't fit the
      // requested N and produces fewer (or none) trades.
      //
      // Daily mode: ~250 trading days/year, warmup eats 50, forward eats
      // the hold period. At 180d daysAgo we get ~70 valid entry bars.
      // Asking for >40 here leaves no room for non-clustered sampling.
      //
      // Intraday (5-min) mode: ~78 bars/day, so 7d = ~550 bars. Much more
      // room, can safely ask for more samples.
      const isDaily = simInterval === "1d";
      // Sample-size calibration MUST consider universe cardinality.
      // Total trades/run = samplesPerSymbol × numSymbols. Walk-forward
      // needs all.length ≥ folds*8 AND trainGBM needs ≥20 samples per
      // fold. At 5 folds with even distribution, that's ~100 trades
      // minimum for any fold to evaluate. Multi-symbol crypto hits
      // that trivially (40 × 18 = 720); single-asset BTC with the old
      // formula produced 1 × 18 = 18, which fails every fold silently
      // and shows as "0/N runs produced valid OOS results".
      // Target ≥100 trades/run regardless of symbol count.
      const rawSyms = excludeHighVol
        ? activeBacktestSymbols.filter(s => !activeHighVolSymbols.includes(s))
        : activeBacktestSymbols;
      const nSyms = Math.max(1, rawSyms.length);
      const baseSamples = isDaily
        ? Math.min(30, Math.max(5, Math.floor(simDaysAgo / 10)))
        : (simDaysAgo <= 7 ? 10 : simDaysAgo <= 30 ? 20 : 40);
      // Scale up for small universes so total trades/run ≥ 100.
      // pickEntries caps at 80% of the feasible entry range so this
      // multiplier can't over-sample beyond what the bars support —
      // it'll just clip naturally.
      const TARGET_TRADES_PER_RUN = 120;
      const samples = Math.max(
        baseSamples,
        Math.ceil(TARGET_TRADES_PER_RUN / nSyms)
      );
      const symsForRun = rawSyms;
      const res = await runBacktest(symsForRun, {
        interval: simInterval,
        daysAgo: simDaysAgo,
        holdHours: maxHoldHours,    // max-hold = timeout; in daily mode this is days, not hours
        samplesPerSymbol: samples,
        costBps,                    // round-trip costs baked in
        polygonKey: polygonKey || null,
        earningsMap,                // PEAD features applied per-entry, point-in-time
        universe,                   // routes scoreSetup to the correct per-universe trained models
        onProgress: (p) => setSimState(prev => ({ ...prev, ...p, running: true })),
      });
      const metrics = computeSimMetrics(res.trades);
      // Explicit error surface when NO trades were produced. Without this
      // the metrics block is gated on metrics != null and the UI shows a
      // completely blank sim result — "runs and does nothing". The most
      // common cause is pickEntries not fitting N samples in the available
      // range; second-most is all symbols failing to fetch.
      if (!res.trades.length) {
        const symbolsFetched = res.barsSource != null ? "yes" : "no";
        const reason = res.errors?.length === activeBacktestSymbols.length
          ? `All ${activeBacktestSymbols.length} symbols failed to fetch. Check rate limits / proxy availability.`
          : symbolsFetched === "no"
            ? "No bars were fetched. Check network / API keys."
            : `0 trades generated despite bars fetched OK. Likely the sample count (${samples}/symbol) doesn't fit the available entry range at ${simDaysAgo}d × ${maxHoldHours}${isDaily?"d":"h"} hold. Try a longer DAYS AGO or shorter MAX HOLD.`;
        setSimResult({ ...res, error: reason, metrics: null, holdHours: maxHoldHours, costBps, interval: simInterval });
        setSimState({ running: false, phase: "done", done: 0, total: 0 });
        return;
      }
      setSimResult({ ...res, metrics, holdHours: maxHoldHours, costBps, interval: simInterval });
      setWfResult(null); // stale — forces user to re-run WF on the new sim
    } catch (err) {
      setSimResult({ error: err.message || String(err) });
    }
    setSimState({ running: false, phase: "done", done: 0, total: 0 });
  }

  // ─── Walk-forward cross-validation ────────────────────────────────────────
  // Evaluates whether the NN has learned anything real by training on past
  // folds and testing on future folds, strictly chronologically. Does NOT
  // touch the production NN weights — each fold trains an isolated copy.
  function runWF() {
    if (wfRunning || !simResult?.trades?.length) return;
    setWfRunning(true);
    setWfResult(null);
    // runWalkForward is now async (yields between folds so UI can repaint).
    // Setting state then awaiting lets the "running..." label render before
    // the compute-heavy loop starts.
    (async () => {
      try {
        await new Promise(r => setTimeout(r, 30));  // let "running" paint
        const out = await runWalkForward(simResult.trades, {
          folds: 5, epochs: 80,
          modelKind: isCryptoUniverse(universe) ? "gbm" : "nn",
        });
        setWfResult(out);
      } catch (err) {
        setWfResult({ error: err.message || String(err) });
      }
      setWfRunning(false);
    })();
  }

  // ─── Multi-sim averaging ──────────────────────────────────────────────────
  // Runs N independent simulations + walk-forward each, then computes a
  // trimmed-mean AUC (drop the highest and lowest, average the middle).
  // This is the right defence against multiple-testing self-deception:
  // ONE sim can hit AUC 0.62 by luck; the trimmed mean of 5 sims is much
  // more honest. Useful for deciding whether to actually train.
  async function runMultiSim() {
    if (multiSimRunning) return;
    setMultiSimRunning(true);
    setMultiSimResult(null);
    // N configurable from the UI. Session-scoped bars cache makes each
    // additional sim nearly free after the first, so going from 5 → 20
    // runs only increases total time by the WF computation (fractions of
    // a second per run for our sample sizes). With 20 runs the std-error
    // on the mean AUC tightens ~2× — enough to distinguish a real 0.53
    // edge from noise.
    const N_RUNS = multiSimNRuns;
    const aucs = [];
    const accs = [];
    const losses = [];
    const tradeCounts = [];
    // Conviction-stratified arrays — per-run AUC/log-loss at top-50/30/10
    // percentile of |yHat − 0.5|. If top-10% AUC materially exceeds the
    // overall AUC, the model HAS edge on high-conviction setups — the
    // user should trade only those, not every signal.
    const aucsTop50 = [], aucsTop30 = [], aucsTop10 = [];
    const lossesTop50 = [], lossesTop30 = [], lossesTop10 = [];
    const thrTop50 = [], thrTop30 = [], thrTop10 = [];
    // Meta-labeling arrays (AFML Ch.4). Per-run: meta AUC, and primary
    // AUC/log-loss on the subset where meta ≥ threshold (0.50/0.55/0.60).
    // If gated AUC substantially exceeds ungated, the meta model IS the
    // edge — trading filter becomes "take the trade only if meta ≥ X".
    const metaAUCs = [], metaLosses = [];
    const gatedAUC50 = [], gatedAUC55 = [], gatedAUC60 = [];
    const gatedLoss50 = [], gatedLoss55 = [], gatedLoss60 = [];
    const gatedKept50 = [], gatedKept55 = [], gatedKept60 = [];
    // Pool trades across all runs for ablation. Previously only runSimulation
    // populated simResult, so users who only did RUN 20-SIM AVERAGE couldn't
    // use the ABLATE button — it was silently disabled with no feedback.
    // Multi-sim now aggregates all trades into a single pool and stores that
    // as simResult.trades so downstream features (training + ablation) work
    // immediately without requiring a separate single-sim pass.
    const pooledTrades = [];
    let lastValidRes = null;
    // Per-symbol aggregated stats across all runs for the summary table.
    const perSymbolStats = {};  // symbol → { total: n, wins: n, pnl: sum }
    let firstRunFetchLog = null;  // captured from run #1 — bars are cached
                                  // after that, so run #1 shows real fetch sources
    try {
      // Sample-size calibration mirrors runSimulation — must scale up on
      // small universes. Without this, 1-asset BTC produces 18 trades/run,
      // below the 20-sample trainGBM floor AND below walk-forward's 5-fold
      // viability threshold. See "Only 0/N runs produced valid OOS" bug.
      const rawSymsMS = excludeHighVol
        ? activeBacktestSymbols.filter(s => !activeHighVolSymbols.includes(s))
        : activeBacktestSymbols;
      const nSymsMS = Math.max(1, rawSymsMS.length);
      for (let i = 0; i < N_RUNS; i++) {
        setMultiSimState({ phase: "sim", run: i + 1, total: N_RUNS });
        const baseSamples = simInterval === "1d"
          ? Math.min(30, Math.max(5, Math.floor(simDaysAgo / 10)))
          : (simDaysAgo <= 7 ? 10 : simDaysAgo <= 30 ? 20 : 40);
        const samples = Math.max(baseSamples, Math.ceil(120 / nSymsMS));
        const symsForRun = rawSymsMS;
      const res = await runBacktest(symsForRun, {
          interval: simInterval,
          daysAgo: simDaysAgo,
          holdHours: maxHoldHours,
          samplesPerSymbol: samples,
          costBps,
          polygonKey: polygonKey || null,
          earningsMap,
          universe,
          onProgress: () => {},
        });
        // Capture the fetchLog from the FIRST run — that's where real
        // fetch attempts happen. Later runs hit the bars cache so their
        // fetchLog is less informative about data-source issues.
        if (i === 0 && res.fetchLog) firstRunFetchLog = res.fetchLog;
        if (!res.trades.length) continue;
        tradeCounts.push(res.trades.length);
        pooledTrades.push(...res.trades);
        lastValidRes = res;
        setMultiSimState({ phase: "wf", run: i + 1, total: N_RUNS });
        // Brief yield so UI repaints between heavy WF runs.
        await new Promise(r => setTimeout(r, 30));
        const wf = await runWalkForward(res.trades, {
          folds: 5, epochs: 60,
          modelKind: isCryptoUniverse(universe) ? "gbm" : "nn",
        });
        if (wf.overall?.oosAUC != null) {
          aucs.push(wf.overall.oosAUC);
          accs.push(wf.overall.oosAccuracy);
          losses.push(wf.overall.oosLogLoss);
          // Capture conviction-stratified metrics so we can aggregate across
          // runs. If a bucket is absent (edge case: too few OOS preds) the
          // push is skipped and means are computed over available runs.
          const bc = wf.overall.byConviction;
          if (bc?.top50?.auc != null) { aucsTop50.push(bc.top50.auc); lossesTop50.push(bc.top50.logLoss); thrTop50.push(bc.top50.threshold); }
          if (bc?.top30?.auc != null) { aucsTop30.push(bc.top30.auc); lossesTop30.push(bc.top30.logLoss); thrTop30.push(bc.top30.threshold); }
          if (bc?.top10?.auc != null) { aucsTop10.push(bc.top10.auc); lossesTop10.push(bc.top10.logLoss); thrTop10.push(bc.top10.threshold); }
          // Meta-labeling aggregation
          const meta = wf.overall.meta;
          if (meta?.metaAUC != null) {
            metaAUCs.push(meta.metaAUC);
            metaLosses.push(meta.metaLogLoss);
            if (meta.gated?.t50?.auc != null) { gatedAUC50.push(meta.gated.t50.auc); gatedLoss50.push(meta.gated.t50.logLoss); gatedKept50.push(meta.gated.t50.kept); }
            if (meta.gated?.t55?.auc != null) { gatedAUC55.push(meta.gated.t55.auc); gatedLoss55.push(meta.gated.t55.logLoss); gatedKept55.push(meta.gated.t55.kept); }
            if (meta.gated?.t60?.auc != null) { gatedAUC60.push(meta.gated.t60.auc); gatedLoss60.push(meta.gated.t60.logLoss); gatedKept60.push(meta.gated.t60.kept); }
          }
        }
        // Aggregate per-symbol trade stats across this run for the summary.
        for (const t of res.trades) {
          const ps = perSymbolStats[t.symbol] ||= { total: 0, wins: 0, pnl: 0 };
          ps.total++;
          if (t.outcome === "WIN") ps.wins++;
          ps.pnl += t.pnlPct || 0;
        }
      }

      if (aucs.length < 3) {
        const totalSimTrades = tradeCounts.reduce((a, b) => a + b, 0);
        const avgPerRun = totalSimTrades / N_RUNS;
        // Specific diagnosis for the failure category instead of the old
        // vague "try different settings". Three possible causes here and
        // the message should distinguish them so the user doesn't chase
        // the wrong fix.
        let reason;
        if (totalSimTrades === 0) {
          reason = `All ${N_RUNS} runs produced 0 labelled trades — the model is neutral (|prob − 0.5| < 0.02) on every entry. Expected for an untrained model on an empty feature vector. TRAIN models first (RUN SIMULATION once → TRAIN GBM ON SIM), then re-run this diagnostic.`;
        } else if (avgPerRun < 100) {
          // Structural — sample count below walk-forward viability.
          reason = `Sample-size shortfall: ${totalSimTrades} trades across ${N_RUNS} runs ≈ ${avgPerRun.toFixed(0)}/run. Walk-forward with 5 folds needs ≈100 trades/run minimum (GBM trains on ≥12 samples per fold × 5 folds + embargo purge margin). Cause: single-symbol universe on a short DAYS AGO window. Fix: increase DAYS AGO (try 365 on btc universe) or add more symbols to the backtest set. ${nSymsMS === 1 ? "You're on a 1-symbol universe — try 180+ DAYS AGO or switch to multi-symbol CRYPTO for structural sample count." : ""}`;
        } else {
          reason = `Only ${aucs.length}/${N_RUNS} runs produced valid OOS results (total sim trades: ${totalSimTrades}, avg ${avgPerRun.toFixed(0)}/run). Trades exist but walk-forward couldn't score them — possibly a labelling issue or per-fold class imbalance. Check the fetch diagnostics below.`;
        }
        setMultiSimResult({ error: reason, fetchLog: firstRunFetchLog });
        setMultiSimRunning(false);
        return;
      }

      // ─── Statistics ───────────────────────────────────────────────────
      // For N_RUNS sims we care about three things:
      //  1. Central tendency: trimmed mean (drops top + bottom, robust to
      //     one lucky and one unlucky outlier)
      //  2. Uncertainty: 95% CI via normal approximation to the mean
      //     (se = std / sqrt(n), CI = mean ± 1.96*se). At n=20 this is
      //     well-approximated by normal; at n<10 the CI widens materially
      //     because student-t kicks in.
      //  3. Shape: histogram of per-run AUCs, min/max, IQR.
      const sortedAUC = [...aucs].sort((a, b) => a - b);
      // Trim ~10% from each tail for robust mean (at N=20, drop 2 each side)
      const trimPct = 0.10;
      const trimCount = Math.max(1, Math.floor(sortedAUC.length * trimPct));
      const trimmed = sortedAUC.slice(trimCount, -trimCount);
      const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
      const std  = arr => {
        const m = mean(arr);
        return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
      };
      const muAUC = mean(aucs);
      const sigmaAUC = std(aucs);
      const seAUC = sigmaAUC / Math.sqrt(aucs.length);
      // Percentile helper (used for IQR)
      const pctile = (arr, p) => {
        const sorted = [...arr].sort((a, b) => a - b);
        const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor(p * sorted.length)));
        return sorted[idx];
      };

      // Per-symbol aggregated breakdown sorted best→worst by win rate.
      const symbolBreakdown = Object.entries(perSymbolStats)
        .map(([symbol, s]) => ({
          symbol,
          total: s.total,
          wins: s.wins,
          winRate: s.wins / s.total,
          avgPnl: s.pnl / s.total,
          totalPnl: s.pnl,
        }))
        .sort((a, b) => b.winRate - a.winRate);

      // Conviction-stratified summary: mean AUC + mean log-loss + mean
      // threshold at each quantile. The threshold is the |yHat − 0.5|
      // floor that reproduces the bucket in production — "only trade
      // when compositeProb > 0.5 + threshold or < 0.5 − threshold".
      const meanOrNull = arr => arr.length ? mean(arr) : null;
      const bucket = (aucArr, lossArr, thrArr) => ({
        meanAUC: meanOrNull(aucArr),
        meanLogLoss: meanOrNull(lossArr),
        meanThreshold: meanOrNull(thrArr),
        runs: aucArr.length,
      });
      // Populate simResult with the pooled trades from ALL runs so TRAIN
      // and ABLATE are available immediately after multi-sim — no need for
      // a separate single-sim pass. Pooling ~N × per-run trades gives
      // ablation the richest dataset possible (2400 trades at 120/run × 20).
      if (lastValidRes && pooledTrades.length > 0) {
        setSimResult({
          ...lastValidRes,
          trades: pooledTrades,
          metrics: null,  // multi-sim metrics are the headline; skip recomputing
          holdHours: maxHoldHours,
          costBps,
          interval: simInterval,
          source: "multi-sim",  // marker so downstream can distinguish if needed
        });
      }
      setMultiSimResult({
        runs: aucs.length,
        aucs,
        accs,
        losses,
        tradeCounts,
        meanAUC: muAUC,
        trimmedMeanAUC: mean(trimmed),
        stdAUC: sigmaAUC,
        seAUC,
        ci95Low:  muAUC - 1.96 * seAUC,
        ci95High: muAUC + 1.96 * seAUC,
        iqrLow:   pctile(aucs, 0.25),
        iqrHigh:  pctile(aucs, 0.75),
        minAUC: Math.min(...aucs),
        maxAUC: Math.max(...aucs),
        meanAccuracy: mean(accs),
        meanLogLoss: mean(losses),
        conviction: {
          all:   { meanAUC: muAUC, meanLogLoss: mean(losses), meanThreshold: 0, runs: aucs.length },
          top50: bucket(aucsTop50, lossesTop50, thrTop50),
          top30: bucket(aucsTop30, lossesTop30, thrTop30),
          top10: bucket(aucsTop10, lossesTop10, thrTop10),
        },
        meta: metaAUCs.length >= 3 ? {
          runs: metaAUCs.length,
          meanMetaAUC: meanOrNull(metaAUCs),
          meanMetaLogLoss: meanOrNull(metaLosses),
          gated: {
            t50: { meanAUC: meanOrNull(gatedAUC50), meanLogLoss: meanOrNull(gatedLoss50), meanKept: meanOrNull(gatedKept50), threshold: 0.50 },
            t55: { meanAUC: meanOrNull(gatedAUC55), meanLogLoss: meanOrNull(gatedLoss55), meanKept: meanOrNull(gatedKept55), threshold: 0.55 },
            t60: { meanAUC: meanOrNull(gatedAUC60), meanLogLoss: meanOrNull(gatedLoss60), meanKept: meanOrNull(gatedKept60), threshold: 0.60 },
          },
        } : null,
        symbolBreakdown,
        fetchLog: firstRunFetchLog,
      });
    } catch (err) {
      setMultiSimResult({ error: err.message || String(err) });
    }
    setMultiSimRunning(false);
    setMultiSimState({ phase: null, run: 0, total: 0 });
  }

  // Ablation study — per-feature leave-one-out AUC delta. Runs on the
  // trades from the most recent sim (single-sim result preferred; falls
  // back to triggering a fresh sim if no trades are cached). N+1 walk-
  // forwards total; ~10-30s depending on fold count.
  async function runAblation() {
    if (ablationRunning) return;
    // Need sim trades to run walk-forwards against. Prefer whatever the
    // single-sim produced; if the user hasn't run one yet, bail with a
    // helpful message rather than surprising them with a long fetch.
    const tradesSource = simResult?.trades;
    if (!tradesSource || tradesSource.length < 50) {
      setAblationResult({ error: "Run SIM first (with a daily-horizon BTC or crypto window) so there are ≥50 labelled trades to ablate against." });
      return;
    }
    setAblationRunning(true);
    setAblationResult(null);
    // Per-universe ablation targets. The crypto and btc universes share
    // most slots (slots 0-13) but differ on [14] and [15] after Phase 5
    // Commit C (btc: Parkinson + Breakout; crypto retains DOW + TopLS).
    // Labels reflect the ACTUAL feature in each slot so the ablation
    // table doesn't mislabel what it's testing.
    const btcTargets = [
      { slot: 7,  name: "BTC dominance z (150-coin real)" },
      { slot: 8,  name: "TS momentum z" },
      { slot: 9,  name: "XS momentum rank (150-coin)" },
      { slot: 10, name: "Perp funding z" },
      { slot: 11, name: "DVOL-RV spread z" },
      { slot: 12, name: "BTC OI Δlog z" },
      { slot: 13, name: "RV 5d/30d ratio (close-to-close)" },
      { slot: 14, name: "Parkinson 5d/30d ratio (hi-lo vol)" },
      { slot: 15, name: "Breakout flag (20d Donchian)" },
    ];
    const cryptoTargets = [
      { slot: 7,  name: "BTC dominance z" },
      { slot: 8,  name: "TS momentum z" },
      { slot: 9,  name: "XS momentum rank" },
      { slot: 10, name: "Perp funding z" },
      { slot: 11, name: "DVOL-RV spread z" },
      { slot: 12, name: "BTC OI Δlog z" },
      { slot: 13, name: "RV 5d/30d ratio" },
      { slot: 14, name: "DOW sin" },
      { slot: 15, name: "Top L/S contrarian z" },
    ];
    const targets = universe === "btc" ? btcTargets
                  : universe === "crypto" ? cryptoTargets
                  : [];
    if (!targets.length) {
      setAblationResult({ error: "Ablation targets defined only for crypto/btc universes in this build." });
      setAblationRunning(false);
      return;
    }
    try {
      setAblationProgress({ idx: 0, total: targets.length + 1, current: "baseline" });
      await new Promise(r => setTimeout(r, 30));
      const out = await runAblationStudy(tradesSource, {
        folds: 5,
        epochs: 60,
        modelKind: isCryptoUniverse(universe) ? "gbm" : "nn",
      }, targets, (ev) => {
        setAblationProgress({
          idx: ev.idx,
          total: ev.total,
          current: ev.phase === "baseline_done" ? "baseline" : ev.name,
        });
      });
      setAblationResult(out);
    } catch (err) {
      setAblationResult({ error: err.message || String(err) });
    }
    setAblationRunning(false);
    setAblationProgress({ idx: 0, total: 0, current: "" });
  }

  // ─── Train NN on the most recent sim batch ────────────────────────────────
  function trainOnSim() {
    if (training || !simResult?.trades?.length) return;
    setTraining(true);
    setTrainResult(null);
    // Defer to next tick so the "training..." UI state can paint first —
    // trainNNFromSim runs synchronously and would otherwise lock the UI.
    setTimeout(() => {
      try {
        // Train both the NN (flexible) and the LR bag (calibrated + uncertainty)
        // on the same sim batch. Running them together means the UI's
        // uncertainty band is always aligned with the NN's current state.
        // Train all three model types on the same sim batch:
        //   - NN: small dense net (16→16→8→1), captures smooth nonlinearities
        //   - GBM: gradient-boosted trees, captures feature interactions
        //   - LR Bag: 30 bootstrap LRs, gives ensemble uncertainty
        // The composite ensemble in scoreSetup picks them up automatically
        // on the next refresh (each is loaded from localStorage).
        const nnOut     = trainNNFromSim(simResult.trades, universe);
        const gbmOut    = trainGBMFromSim(simResult.trades, universe);
        const bagOut    = trainBagFromSim(simResult.trades, universe);
        // Regime-conditional GBMs need ≥60 trades (30 each side); skip
        // silently with a status if there aren't enough samples on each
        // side of the VIX-z midpoint.
        const regimeOut = trainRegimeFromSim(simResult.trades, universe);
        setTrainResult({ ...nnOut, gbm: gbmOut, bag: bagOut, regime: regimeOut });
      } catch (err) {
        setTrainResult({ error: err.message || String(err) });
      }
      setTraining(false);
    }, 50);
  }

  const selQ = quotes[selected];
  const marketUp = selQ ? selQ.changePct>=0 : true;
  const loadedCount = Object.keys(quotes).length;
  const mockCount = Object.values(quotes).filter(q=>q.isMock).length;

  if (!apiKey||!finnhubKey) return <ApiKeyModal onSave={(ak,fk,pk)=>{setApiKey(ak);setFinnhubKey(fk);setPolygonKey(pk||"");}}/>;

  return (
    <div style={{display:"flex",flexDirection:"column",height:"100vh",background:"#080808",
      color:"#D8D0C0",fontFamily:"'Courier New',monospace",overflow:"hidden"}}>

      {/* Header */}
      <div style={{background:"#0F0F0F",borderBottom:"1px solid #1E1E1E",padding:"8px 14px",
        display:"flex",alignItems:"center",gap:12,flexShrink:0}}>
        <div style={{fontSize:18,fontWeight:900,color:"#C9A84C",letterSpacing:4}}>◈ THE TRADER</div>
        {/* Global horizon indicator — set by the HORIZON dropdown in the SIM
            card, but drives MORE than just sim behaviour: the suggested
            levels card highlights the matching row, the AI analysis adapts
            its reasoning, and the position sizing uses the matching stop
            distance. One knob, everything aligned. */}
        {/* Universe toggle — swaps the entire asset class. Switching auto-
            adjusts the horizon default (crypto → daily) and the cost model
            (crypto → 30bps), resets the selected symbol to the first of the
            new watchlist, and disables equity-only features (PEAD, crisis
            analogue, Buffett persona) for crypto. One knob propagates. */}
        <button onClick={()=>{
          // 3-way cycle: equities → crypto → btc → equities. BTC mode is
          // single-asset focus for the Phase 4 pivot after the 40-coin
          // crypto universe landed three diagnostics at coin flip.
          const cycle = { equities: "crypto", crypto: "btc", btc: "equities" };
          const nextU = cycle[universe] || "equities";
          setUniverse(nextU);
          const newList = watchlistFor(nextU);
          if (!newList.includes(selected)) setSelected(newList[0]);
          // Per-universe defaults:
          //   equities → 5m + 15 bps (US-session, tight stops, tuned)
          //   crypto   → 1d + 30 bps (24/7, research-Tier-A evidence)
          //   btc      → 1d + 18 bps (single-asset, top-tier venue costs)
          if (nextU === "crypto") {
            setSimInterval("1d");
            setCostBps(30);
            if (simDaysAgo < 90) setSimDaysAgo(polygonKey ? 180 : 7);
            if (maxHoldHours < 24) setMaxHoldHours(5);
          } else if (nextU === "btc") {
            setSimInterval("1d");
            setCostBps(18);
            if (simDaysAgo < 90) setSimDaysAgo(polygonKey ? 180 : 7);
            if (maxHoldHours < 24) setMaxHoldHours(5);
          } else {
            setSimInterval("5m");
            setCostBps(15);
          }
        }}
          title="Cycle asset universe: equities → crypto (40 symbols) → btc (single-asset focus) → equities. BTC mode isolates training on Bitcoin only — fresh storage keys, xsMomRank disabled (structurally undefined on n=1), dominance proxy disabled (redundant with TS momentum on self)."
          style={{background: universe === "btc" ? "#14141F" : universe === "crypto" ? "#1A140F" : "#0F1A0F",
            border:`1px solid ${universe === "btc" ? "#6A6AC9" : universe === "crypto" ? "#C97A2A" : "#2A6A4A"}`,
            color: universe === "btc" ? "#B0B0FF" : universe === "crypto" ? "#FFB07A" : "#7FD8A6",
            fontFamily:"inherit",fontSize:9,padding:"3px 10px",cursor:"pointer",letterSpacing:2,fontWeight:700,marginRight:6}}>
          UNIVERSE: {universe === "btc" ? "BTC (1-asset)" : universe === "crypto" ? "CRYPTO (40)" : "EQUITIES"}
        </button>
        <button onClick={()=>setSimInterval(simInterval === "1d" ? "5m" : "1d")}
          title="Global horizon — switches the entire app between intraday (tight stops, short holds) and swing (wider stops, multi-day holds). Set here or in the SIM card's HORIZON dropdown; they share state."
          style={{background:simInterval === "1d" ? "#0F1F18" : "#0A0F14",
            border:`1px solid ${simInterval === "1d" ? "#2A6A4F" : "#2A6A9A"}`,
            color:simInterval === "1d" ? "#7FD8A6" : "#5AACDF",
            fontFamily:"inherit",fontSize:9,padding:"3px 10px",cursor:"pointer",letterSpacing:2,fontWeight:700}}>
          HORIZON: {simInterval === "1d" ? "SWING (1-5d)" : "INTRADAY (1-3h)"}
        </button>
        <div style={{flex:1,fontSize:9,color:"#555",letterSpacing:2,marginLeft:12}}>
          {isCryptoUniverse(universe)
            ? "Raoul Pal · Willy Woo · Peter Brandt · Arthur Hayes · Benjamin Cowen"
            : "Livermore · Tudor Jones · Dennis · Simons · Williams"}
        </div>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          {mockCount>0&&<span style={{fontSize:9,color:"#C9A84C",letterSpacing:1}}>SIM {mockCount}/{loadedCount}</span>}
          <div style={{width:6,height:6,borderRadius:"50%",
            background:refreshing?"#C9A84C":loadedCount>0?"#2ECC71":"#555",animation:"pulse 2s infinite"}}/>
          <span style={{fontSize:9,color:"#555",letterSpacing:1}}>
            {lastRefresh?`${loadedCount}/${activeWatchlist.length} · ${new Date(lastRefresh).toLocaleTimeString()}`:"CONNECTING..."}
          </span>
          <button onClick={sendBestOpportunity} disabled={thinking||loadedCount<3}
            style={{background:thinking||loadedCount<3?"#1A1500":"#C9A84C",
              color:thinking||loadedCount<3?"#555":"#000",border:"none",fontFamily:"'Courier New',monospace",
              fontWeight:900,fontSize:9,letterSpacing:1,padding:"4px 10px",
              cursor:thinking||loadedCount<3?"not-allowed":"pointer"}}>
            ★ BEST TRADE
          </button>
          <button onClick={()=>refreshAll()} style={{background:"#1A1A1A",border:"1px solid #2A2A2A",
            color:"#888",fontSize:9,padding:"3px 8px",cursor:"pointer",letterSpacing:1,fontFamily:"inherit"}}>
            ↻ REFRESH
          </button>
          <button onClick={()=>setApiKey("")} style={{background:"#1A1A1A",border:"1px solid #2A2A2A",
            color:"#555",fontSize:9,padding:"3px 8px",cursor:"pointer",letterSpacing:1,fontFamily:"inherit"}}>
            KEY
          </button>
        </div>
      </div>

      {/* Ticker */}
      <div style={{background:"#0C0C0C",borderBottom:"1px solid #1E1E1E",overflow:"hidden",height:24,flexShrink:0}}>
        <div style={{display:"flex",whiteSpace:"nowrap",height:"100%",alignItems:"center",
          animation:"ticker 35s linear infinite",fontSize:10,letterSpacing:"0.05em"}}>
          {[...activeWatchlist,...activeWatchlist].map((sym,i)=>{
            const q=quotes[sym]; const up=q?q.changePct>=0:true;
            return <span key={i} style={{marginRight:32,color:up?"#2ECC71":"#E74C3C",fontWeight:600}}>
              {up?"▲":"▼"} {sym} {q?`$${q.price?.toFixed(2)} (${q.changePct>=0?"+":""}${q.changePct?.toFixed(2)}%)`:"..."}
            </span>;
          })}
        </div>
      </div>

      <div style={{flex:1,display:"flex",overflow:"hidden"}}>
        {/* Watchlist */}
        <div style={{width:200,background:"#0C0C0C",borderRight:"1px solid #1A1A1A",overflowY:"auto",flexShrink:0}}>
          <div style={{padding:"8px 10px",fontSize:8,color:"#444",letterSpacing:2,borderBottom:"1px solid #1A1A1A"}}>
            {universe === "btc" ? "BTC" : universe === "crypto" ? "CRYPTO" : "EQUITIES"} · {loadedCount}/{activeWatchlist.length}
          </div>
          {activeWatchlist.map(sym=>{
            const q=quotes[sym]; const up=q?q.changePct>=0:null; const isSel=sym===selected;
            return (
              <div key={sym} onClick={()=>setSelected(sym)} style={{padding:"8px 10px",cursor:"pointer",
                background:isSel?"#1A1500":"transparent",
                borderLeft:isSel?"2px solid #C9A84C":"2px solid transparent",borderBottom:"1px solid #111"}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                  <span style={{fontSize:12,fontWeight:700,color:isSel?"#C9A84C":"#CCC",letterSpacing:1}}>{sym}</span>
                  {q&&<span style={{fontSize:10,color:up?"#2ECC71":"#E74C3C",fontWeight:600}}>
                    {up?"+":""}{q.changePct?.toFixed(2)}%
                  </span>}
                </div>
                {q&&<div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginTop:2}}>
                  <span style={{fontSize:11,color:"#888"}}>${q.price?.toFixed(2)}</span>
                  <RSIBar rsi={q.rsi}/>
                </div>}
                {q&&<Sparkline data={q.sparkline} up={up} width={180} height={18}/>}
                {!q&&<div style={{fontSize:9,color:"#333",marginTop:4}}>loading...</div>}
                {q?.isMock&&<div style={{fontSize:8,color:"#555",letterSpacing:1}}>SIMULATED</div>}
              </div>
            );
          })}
        </div>

        {/* Main panel */}
        <div style={{flex:1,display:"flex",flexDirection:"column",overflow:"hidden"}}>
          {selQ&&(
            <div style={{background:"#0F0F0F",borderBottom:"1px solid #1A1A1A",padding:"10px 14px",
              display:"flex",alignItems:"center",gap:16,flexShrink:0,flexWrap:"wrap"}}>
              <div>
                <span style={{fontSize:20,fontWeight:900,color:"#FFF",letterSpacing:2}}>{selected}</span>
                <span style={{fontSize:22,marginLeft:10,color:"#FFF"}}>${selQ.price?.toFixed(2)}</span>
                <span style={{fontSize:14,marginLeft:8,color:marketUp?"#2ECC71":"#E74C3C",fontWeight:700}}>
                  {marketUp?"▲":"▼"} {selQ.changePct>=0?"+":""}{selQ.changePct?.toFixed(2)}%
                </span>
                {(()=>{ const s = selQ.session || getMarketSession(selected);
                  return <span style={{fontSize:9,color:sessionColor(s),marginLeft:8,letterSpacing:1,
                    padding:"2px 6px",border:`1px solid ${sessionColor(s)}44`}}>● {sessionLabel(s)}</span>;
                })()}
                {(()=>{ const qc = selQ.quality || "clean";
                  const qColor = qc === "clean" ? "#2ECC71" : qc === "suspect" ? "#C9A84C" : qc === "stale" ? "#888" : "#555";
                  const title = cleaningSummary(selQ.cleaning, qc);
                  return <span title={title} style={{fontSize:9,color:qColor,marginLeft:6,letterSpacing:1,
                    padding:"2px 6px",border:`1px solid ${qColor}44`,cursor:"help"}}>◆ {qc.toUpperCase()}</span>;
                })()}
                {(()=>{ const bs = selQ.barsSource;
                  if (!bs) return null;
                  const isReal = bs === "yahoo-5m" || bs === "yahoo-1m";
                  const color = isReal ? "#7FD8A6" : "#E74C3C";
                  const label = isReal ? "REAL BARS" : "SYNTHETIC BARS";
                  const tip = isReal
                    ? "Indicators computed on real 5-min bars from Yahoo (cached 3min, last close anchored to Finnhub real-time price)."
                    : "Bar fetch failed — indicators are being computed on a random-walk synthetic series. Trade decisions on this symbol are NOT backed by real intraday history.";
                  return <span title={tip} style={{fontSize:9,color,marginLeft:6,letterSpacing:1,
                    padding:"2px 6px",border:`1px solid ${color}44`,cursor:"help"}}>◈ {label}</span>;
                })()}
                {selQ.extendedMove && (selQ.session==="PREMARKET"||selQ.session==="AFTERHOURS") && (
                  <span style={{fontSize:10,color:"#888",marginLeft:8,letterSpacing:1}}>
                    {selQ.session==="PREMARKET"?"PRE":"POST"} {selQ.changePct>=0?"+":""}{selQ.changePct?.toFixed(2)}% vs prev close
                  </span>
                )}
                {selQ.isMock&&<span style={{fontSize:9,color:"#C9A84C",marginLeft:8,letterSpacing:1}}>SIMULATED</span>}
              </div>
              <div style={{display:"flex",gap:14,marginLeft:10,flexWrap:"wrap"}}>
                {[
                  ["RSI",selQ.rsi!=null?`${selQ.rsi.toFixed(0)}${selQ.rsi>70?" ⚠":selQ.rsi<30?" ✓":""}`:"-"],
                  ["MACD",selQ.macd!=null?(selQ.macd>0?"▲ BULL":"▼ BEAR"):"-"],
                  ["VWAP",selQ.vwap?`$${selQ.vwap.toFixed(2)}`:"-"],
                  ["ATR",selQ.atr?`$${selQ.atr.toFixed(2)}`:"-"],
                  ["VOL",selQ.volRatio?`${selQ.volRatio.toFixed(1)}x`:"-"],
                  ["BB",selQ.bb?`${(selQ.bb.pos*100).toFixed(0)}%`:"-"],
                ].map(([label,val])=>(
                  <div key={label} style={{textAlign:"center"}}>
                    <div style={{fontSize:8,color:"#555",letterSpacing:1}}>{label}</div>
                    <div style={{fontSize:11,color:"#C9A84C",fontWeight:700}}>{val}</div>
                  </div>
                ))}
              </div>
              <div style={{marginLeft:"auto",display:"flex",gap:6}}>
                <button onClick={()=>{setTab("chat");sendToAI(`Quick call on ${selected}.`, "quick");}}
                  disabled={thinking} title="4-line answer: price, verdict, stop/target, confidence"
                  style={{
                    background:thinking?"#1A1A1A":"#C9A84C",color:thinking?"#444":"#000",
                    border:"none",fontFamily:"'Courier New',monospace",fontWeight:900,
                    fontSize:11,letterSpacing:2,padding:"7px 12px",cursor:thinking?"not-allowed":"pointer"}}>
                  ⚡ QUICK
                </button>
                <button onClick={()=>{setTab("chat");sendToAI(`Full assessment on ${selected}.`, "deep");}}
                  disabled={thinking} title="Compact structured analysis: tape, model, 5 traders, verdict"
                  style={{
                    background:thinking?"#1A1A1A":"#1A1500",color:thinking?"#444":"#C9A84C",
                    border:"1px solid #C9A84C",fontFamily:"'Courier New',monospace",fontWeight:900,
                    fontSize:11,letterSpacing:2,padding:"7px 12px",cursor:thinking?"not-allowed":"pointer"}}>
                  📊 IN-DEPTH
                </button>
                {/* Buffett called crypto "rat poison squared" and has never
                    bought any. Rather than force-generate a fake valuation
                    persona on crypto that would be incoherent with his public
                    record, hide the button entirely in crypto mode. Phase 3b
                    could replace with a value-investor-on-crypto persona if
                    there's appetite. */}
                {universe !== "crypto" && (
                  <button onClick={()=>sendToBuffett("")} disabled={thinking} title="Warren Buffett's separate, long-horizon take — NOT part of the BUY/SELL verdict"
                    style={{
                      background:thinking?"#1A1A1A":"#0D2A1F",color:thinking?"#444":"#7FD8A6",
                      border:"1px solid #2A7A4F",fontFamily:"'Courier New',monospace",fontWeight:900,
                      fontSize:11,letterSpacing:2,padding:"7px 12px",cursor:thinking?"not-allowed":"pointer"}}>
                    🏛 BUFFETT
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Tabs */}
          <div style={{display:"flex",borderBottom:"1px solid #1A1A1A",background:"#0C0C0C",flexShrink:0}}>
            {[["chat","💬 ANALYSIS"],["data","📊 RAW DATA"],["log","📋 LOG"],["model","🤖 MODEL"],["quant","📐 QUANT"],["news","📰 NEWS"]].map(([t,label])=>(
              <button key={t} onClick={()=>setTab(t)} style={{
                padding:"8px 14px",fontSize:9,letterSpacing:2,textTransform:"uppercase",
                background:tab===t?"#1A1A1A":"transparent",border:"none",
                borderBottom:tab===t?"2px solid #C9A84C":"2px solid transparent",
                color:tab===t?"#C9A84C":"#555",cursor:"pointer",fontFamily:"inherit"}}>
                {label}
              </button>
            ))}
          </div>

          {tab==="data"&&(
            <div style={{flex:1,overflowY:"auto",padding:14,fontSize:11,color:"#888"}}>
              <pre style={{whiteSpace:"pre-wrap",fontFamily:"'Courier New',monospace",fontSize:11,margin:0}}>
                {buildContext(quotes,selected,news)}
              </pre>
            </div>
          )}

          {tab==="model"&&(()=>{
            const q = quotes[selected];
            const btcSnapEntryM = universe === "btc" && broadMarketSnapshot
              ? broadMarketSnapshot.find(s => s.symbol === "BTC") : null;
            const cryptoCtx = (isCryptoUniverse(universe) && q?.closes) ? {
              ...(macro?.cryptoContext || {}),
              tsMom: timeSeriesMomentum(q.closes),
              xsMomRank: universe === "btc"
                ? (btcSnapEntryM && broadMarketSnapshot
                    ? xsRankLiveBroad(btcSnapEntryM.ret14dPct, broadMarketSnapshot)
                    : 0)
                : xsMomRankLive(selected, quotes),
              fundingZ: fundingZLive(fundingMap.get(selected)),
              rvRatio: rvRatioLive(q.closes, 5, 30),
              dowSin: dayOfWeekSinAt(Math.floor(Date.now() / 1000)),
              breadth: universe === "btc" && broadMarketSnapshot
                ? breadthLive(broadMarketSnapshot) : 0,
            } : macro?.cryptoContext;
            const m = q ? scoreSetup(q, {
              macro: macro ? { ...macro, cryptoContext: cryptoCtx } : null,
              calendar: calendarFeatures(),
              pead: isCryptoUniverse(universe) ? null : computePeadFeatures(earningsMap[selected]),
              universe,
            }) : null;
            return (
              <div style={{flex:1,overflowY:"auto",padding:14,fontSize:11,color:"#888"}}>
                {!m ? <div>No data</div> : (
                  <div style={{display:"flex",flexDirection:"column",gap:12}}>
                    <div style={{color:"#C9A84C",fontSize:13,fontWeight:900,letterSpacing:2}}>{selected} — MODEL READOUT</div>

                    {/* ═══ COMPOSITE ═══ */}
                    <div style={{background:"#0E1512",border:"1px solid #2A7A4F",padding:12}}>
                      <div style={{fontSize:9,color:"#7FD8A6",letterSpacing:2,marginBottom:8}}>◆ COMPOSITE ENSEMBLE</div>
                      <div style={{fontSize:20,fontWeight:900,color:m.compositeProb>58?"#2ECC71":m.compositeProb<42?"#E74C3C":"#C9A84C"}}>
                        {m.direction} — {m.compositeProb}% bullish
                      </div>
                      <div style={{marginTop:6,fontSize:10,color:"#888"}}>
                        Confidence: <b style={{color:"#FFF"}}>{m.confidence}%</b>
                        &nbsp;·&nbsp; Agreement: <b style={{color:m.agreement.count>=m.agreement.total?"#2ECC71":m.agreement.count>=(m.agreement.total-1)?"#C9A84C":"#E74C3C"}}>{m.agreement.count}/{m.agreement.total} {m.agreement.total===1?"(NN untrained)":"models"}</b>
                        &nbsp;·&nbsp; Weights: LR {(m.weights.lr*100).toFixed(0)}% · NN {(m.weights.nn*100).toFixed(0)}% · GBM {((m.weights.gbm||0)*100).toFixed(0)}% · Tree {(m.weights.tree*100).toFixed(0)}%
                        {m.gbmSource && m.gbmSource !== "universal" && (
                          <> &nbsp;·&nbsp; <span style={{color:"#D87FD8"}}>GBM regime: <b>{m.gbmSource.toUpperCase()}</b></span></>
                        )}
                      </div>
                      <div style={{marginTop:8,width:"100%",height:8,background:"#0A0A0A",borderRadius:4,overflow:"hidden"}}>
                        <div style={{width:`${m.compositeProb}%`,height:"100%",background:m.compositeProb>58?"#2ECC71":m.compositeProb<42?"#E74C3C":"#C9A84C"}}/>
                      </div>
                      {/* Bagged ensemble uncertainty band — only shown if bag is trained */}
                      {(() => {
                        const bag = loadBag(universe);
                        if (!bag || !m.features) return null;
                        const bp = predictBag(bag, m.features);
                        if (!bp) return null;
                        const meanPct = (bp.mean * 100).toFixed(1);
                        const stdPct = (bp.std * 100).toFixed(1);
                        const widePct = bp.std > 0.12;
                        return (
                          <div style={{marginTop:10,paddingTop:8,borderTop:"1px solid #1A3A2A"}}>
                            <div style={{fontSize:8,color:"#7FD8A6",letterSpacing:2,marginBottom:4}}>
                              BAGGED-LR UNCERTAINTY ({bp.nBags} bootstrap models)
                            </div>
                            <div style={{fontSize:11,color:"#D8D0C0"}}>
                              Mean <b style={{color:"#7FD8A6"}}>{meanPct}%</b> ± <b style={{color:widePct?"#E74C3C":"#C9A84C"}}>{stdPct}%</b>
                              &nbsp;·&nbsp; band <span style={{color:"#888"}}>{(bp.min*100).toFixed(0)}–{(bp.max*100).toFixed(0)}%</span>
                              {widePct && <span style={{color:"#E74C3C",marginLeft:6}}> ⚠ wide — low ensemble agreement</span>}
                            </div>
                          </div>
                        );
                      })()}
                    </div>

                    {/* ═══ LR ═══ */}
                    <div style={{background:"#111",border:"1px solid #1E1E1E",padding:12}}>
                      <div style={{fontSize:9,color:"#555",letterSpacing:2,marginBottom:8}}>LOGISTIC REGRESSION (1-layer)</div>
                      <div style={{fontSize:18,fontWeight:900,color:m.lrProb>58?"#2ECC71":m.lrProb<42?"#E74C3C":"#C9A84C"}}>
                        {m.lrProb}% bullish
                      </div>
                      <div style={{marginTop:4,width:"100%",height:6,background:"#222",borderRadius:3}}>
                        <div style={{width:`${m.lrProb}%`,height:"100%",background:m.lrProb>58?"#2ECC71":m.lrProb<42?"#E74C3C":"#C9A84C",borderRadius:3}}/>
                      </div>
                    </div>

                    {/* ═══ NN ═══ */}
                    <div style={{background:"#111",border:"1px solid #1E1E1E",padding:12}}>
                      <div style={{fontSize:9,color:"#555",letterSpacing:2,marginBottom:8}}>NEURAL NET (16→16→8→1, backprop)</div>
                      {m.nnProb == null ? (
                        <div style={{fontSize:11,color:"#666"}}>Untrained — run SIMULATE then TRAIN, or 🧠 LEARN with ≥8 reviewed trades.</div>
                      ) : (
                        <>
                          <div style={{fontSize:18,fontWeight:900,color:m.nnProb>58?"#2ECC71":m.nnProb<42?"#E74C3C":"#C9A84C"}}>
                            {m.nnProb}% bullish
                          </div>
                          <div style={{marginTop:4,width:"100%",height:6,background:"#222",borderRadius:3}}>
                            <div style={{width:`${m.nnProb}%`,height:"100%",background:m.nnProb>58?"#2ECC71":m.nnProb<42?"#E74C3C":"#C9A84C",borderRadius:3}}/>
                          </div>
                          <div style={{marginTop:6,fontSize:9,color:"#666"}}>
                            Trained on {m.nnInfo.trainedOn} samples · {m.nnInfo.epochs} epochs total · final loss {m.nnInfo.finalLoss?.toFixed(4) ?? "?"}
                          </div>
                        </>
                      )}
                    </div>

                    {/* ═══ GBM ═══ */}
                    <div style={{background:"#111",border:"1px solid #1E1E1E",padding:12,borderLeft:"3px solid #2A6A9A"}}>
                      <div style={{fontSize:9,color:"#5AACDF",letterSpacing:2,marginBottom:8}}>GRADIENT-BOOSTED TREES (depth-4, 100 rounds, second-order Newton)</div>
                      {m.gbmProb == null ? (
                        <div style={{fontSize:11,color:"#666"}}>Untrained — run SIMULATE then TRAIN with ≥20 sim trades, or 🧠 LEARN with ≥20 reviewed trades. GBM is the most likely to find genuine edge in tabular features.</div>
                      ) : (
                        <>
                          <div style={{fontSize:18,fontWeight:900,color:m.gbmProb>58?"#2ECC71":m.gbmProb<42?"#E74C3C":"#C9A84C"}}>
                            {m.gbmProb}% bullish
                          </div>
                          <div style={{marginTop:4,width:"100%",height:6,background:"#222",borderRadius:3}}>
                            <div style={{width:`${m.gbmProb}%`,height:"100%",background:m.gbmProb>58?"#2ECC71":m.gbmProb<42?"#E74C3C":"#C9A84C",borderRadius:3}}/>
                          </div>
                          <div style={{marginTop:6,fontSize:9,color:"#666"}}>
                            Trained on {m.gbmInfo.trainedOn} samples · {m.gbmInfo.rounds} boosting rounds · final loss {m.gbmInfo.finalLoss?.toFixed(4) ?? "?"}
                          </div>
                        </>
                      )}
                    </div>
                    <div style={{background:"#111",border:"1px solid #1E1E1E",padding:12}}>
                      <div style={{fontSize:9,color:"#555",letterSpacing:2,marginBottom:8}}>DECISION TREE (rule-based, 7-factor scan)</div>
                      <div style={{fontSize:14,fontWeight:700,color:
                        m.treeSignal==="STRONG_BUY"?"#2ECC71":
                        m.treeSignal==="BUY"?"#2ECC71":
                        m.treeSignal==="STRONG_SELL"?"#E74C3C":
                        m.treeSignal==="SELL"?"#E74C3C":"#C9A84C"
                      }}>{m.treeSignal} <span style={{fontSize:10,color:"#666",marginLeft:6}}>strength {((m.treeStrength||0)*100).toFixed(0)}%</span></div>
                      <div style={{fontSize:10,color:"#666",marginTop:4}}>{m.treeReason}</div>
                    </div>
                    <div style={{background:"#111",border:"1px solid #1E1E1E",padding:12}}>
                      <div style={{fontSize:9,color:"#555",letterSpacing:2,marginBottom:8}}>NEAREST HISTORICAL ANALOGUE (of 55 regimes)</div>
                      <div style={{fontSize:13,fontWeight:700,color:"#C9A84C"}}>{m.crisis?.name}
                        {m.crisis?.regime && <span style={{fontSize:9,color:"#666",marginLeft:8,textTransform:"uppercase",letterSpacing:1}}>[{m.crisis.regime}]</span>}
                      </div>
                      <div style={{fontSize:10,color:"#888",marginTop:4}}>{m.crisis?.note}</div>
                      <div style={{marginTop:8,fontSize:10,color:"#555"}}>Similarity: {m.crisis?(m.crisis.similarity*100).toFixed(0):0}%</div>
                      <div style={{marginTop:4,width:"100%",height:4,background:"#222",borderRadius:2}}>
                        <div style={{width:`${m.crisis?(m.crisis.similarity*100):0}%`,height:"100%",background:"#C9A84C",borderRadius:2}}/>
                      </div>
                    </div>

                    {/* ═══ SIMULATE ═══ (data + metrics, no training) */}
                    <div style={{background:"#0A0F14",border:"1px solid #2A6A9A",padding:12}}>
                      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8,flexWrap:"wrap",gap:6}}>
                        <div style={{fontSize:9,color:"#5AACDF",letterSpacing:2}}>🎲 SIMULATE (real candles → labelled trades)</div>
                        <div style={{display:"flex",alignItems:"center",gap:12,flexWrap:"wrap"}}>
                          <div style={{display:"flex",alignItems:"center",gap:6}}
                            title="Intraday (5-min) = short-term, noisy, costs eat edge. Daily = swing-trade horizon where published retail effects (PEAD, momentum) actually live. Daily is the recommended mode for finding genuine edge.">
                            <span style={{fontSize:8,color:"#666",letterSpacing:1,cursor:"help"}}>HORIZON</span>
                            <select value={simInterval} onChange={e=>{
                              const v = e.target.value;
                              setSimInterval(v);
                              // Reset hold to a sensible default for the new mode
                              if (v === "1d" && maxHoldHours < 24) setMaxHoldHours(5);
                              if (v === "5m" && maxHoldHours >= 24) setMaxHoldHours(3);
                              // Daily mode needs more days to be useful
                              if (v === "1d" && simDaysAgo < 90) setSimDaysAgo(polygonKey ? 180 : 7);
                            }}
                              disabled={simState.running}
                              style={{background:"#080808",border:"1px solid #2A2A2A",color:simInterval==="1d"?"#D87FD8":"#5AACDF",fontFamily:"inherit",fontSize:9,padding:"2px 6px"}}>
                              <option value="5m">intraday (5-min)</option>
                              <option value="1d">daily (1-20d swing)</option>
                            </select>
                          </div>
                          <div style={{display:"flex",alignItems:"center",gap:6}} title={polygonKey ? "Polygon key detected — longer histories available." : "No Polygon key — capped at 7 days by Yahoo (5-min) / unlimited for daily. Enter a key via KEY button."}>
                            <span style={{fontSize:8,color:"#666",letterSpacing:1,cursor:"help"}}>DAYS AGO</span>
                            <select value={simDaysAgo} onChange={e=>setSimDaysAgo(Number(e.target.value))}
                              disabled={simState.running}
                              style={{background:"#080808",border:"1px solid #2A2A2A",color:polygonKey?"#7FD8A6":"#5AACDF",fontFamily:"inherit",fontSize:9,padding:"2px 6px"}}>
                              {simInterval === "5m" ? <>
                                <option value={7}>7d (Yahoo)</option>
                                <option value={30} disabled={!polygonKey}>30d {polygonKey?"":"(Polygon)"}</option>
                                <option value={90} disabled={!polygonKey}>90d {polygonKey?"":"(Polygon)"}</option>
                                <option value={180} disabled={!polygonKey}>180d {polygonKey?"":"(Polygon)"}</option>
                                <option value={365} disabled={!polygonKey}>1y {polygonKey?"":"(Polygon)"}</option>
                              </> : <>
                                <option value={90}>90d (~60 trades)</option>
                                <option value={180}>180d (~130 trades)</option>
                                <option value={365}>1y (~250 trades)</option>
                                <option value={730} disabled={!polygonKey}>2y {polygonKey?"":"(Polygon)"}</option>
                                <option value={1095} disabled={!polygonKey}>3y {polygonKey?"":"(Polygon)"}</option>
                                <option value={1825} disabled={!polygonKey}>5y {polygonKey?"":"(Polygon)"}</option>
                              </>}
                            </select>
                          </div>
                          <div style={{display:"flex",alignItems:"center",gap:6}}>
                            <span style={{fontSize:8,color:"#666",letterSpacing:1}}>MAX HOLD (timeout)</span>
                            <select value={maxHoldHours} onChange={e=>setMaxHoldHours(Number(e.target.value))}
                              disabled={simState.running}
                              style={{background:"#080808",border:"1px solid #2A2A2A",color:"#5AACDF",fontFamily:"inherit",fontSize:9,padding:"2px 6px"}}>
                              {simInterval === "5m" ? <>
                                <option value={1}>1h</option>
                                <option value={3}>3h (default)</option>
                                <option value={6}>6h</option>
                                <option value={24}>1d</option>
                              </> : <>
                                <option value={1}>1 day</option>
                                <option value={3}>3 days</option>
                                <option value={5}>5 days (default)</option>
                                <option value={10}>10 days</option>
                                <option value={20}>20 days</option>
                              </>}
                            </select>
                          </div>
                          <div style={{display:"flex",alignItems:"center",gap:6}} title="Round-trip cost: commission + spread + slippage, deducted from every trade's P&L before outcome labelling. 15 bps = 0.15% (realistic retail default on liquid US equities). 0 bps = gross / pre-cost.">
                            <span style={{fontSize:8,color:"#666",letterSpacing:1,cursor:"help"}}>COST MODEL</span>
                            <select value={costBps} onChange={e=>setCostBps(Number(e.target.value))}
                              disabled={simState.running}
                              style={{background:"#080808",border:"1px solid #2A2A2A",color:"#5AACDF",fontFamily:"inherit",fontSize:9,padding:"2px 6px"}}>
                              <option value={0}>0 bps (gross)</option>
                              <option value={5}>5 bps (best case)</option>
                              <option value={15}>15 bps (default)</option>
                              <option value={30}>30 bps (illiquid)</option>
                              <option value={50}>50 bps (worst)</option>
                            </select>
                          </div>
                          {/* Diagnostic toggle — exclude high-vol speculative names from
                              the sim. When the multi-sim shows high std dev, this is the
                              first thing to test: do IONQ/RGTI variance bombs disappear?
                              If std dev tightens, they're the source. */}
                          <label style={{display:"flex",alignItems:"center",gap:6,cursor:simState.running?"not-allowed":"pointer"}}
                            title="Diagnostic. Excludes IONQ + RGTI (5-6% per-bar vol) from the sim/training pipeline. Use to test whether high per-run AUC variance is driven by these names. Live dashboard unaffected.">
                            <input type="checkbox" checked={excludeHighVol}
                              disabled={simState.running}
                              onChange={e=>setExcludeHighVol(e.target.checked)}
                              style={{accentColor:"#E74C3C"}}/>
                            <span style={{fontSize:8,color:excludeHighVol?"#E74C3C":"#666",letterSpacing:1}}>
                              EXCLUDE HIGH-VOL ({activeHighVolSymbols.join("/")})
                            </span>
                          </label>
                        </div>
                      </div>
                      <div style={{fontSize:10,color:"#888",lineHeight:1.6,marginBottom:10}}>
                        Fetches real {simInterval === "1d" ? "daily" : "5-min"} candles for {activeBacktestSymbols.length} {universe === "btc" ? "BTC bars (single-asset focus)" : universe === "crypto" ? "crypto pairs (24/7 markets)" : "US-session symbols (non-US skipped)"},
                        samples random entries per symbol, runs the current model verdict at each, then walks
                        forward bar-by-bar. <b style={{color:"#5AACDF"}}>Stop or target exits the trade IMMEDIATELY</b> on
                        the first bar that touches them — max-hold is only the <i>timeout</i> for trades that
                        hit neither. Round-trip cost deducted <b style={{color:"#C9A84C"}}>before</b> outcome labelling,
                        so win rate and edge are <b>net of costs</b>.
                        {simInterval === "1d" && <> <span style={{color:"#D87FD8"}}>Daily mode</span> uses a 1-to-20-day swing horizon where published retail effects actually live — generally higher AUC ceiling than intraday.</>}
                      </div>
                      <button onClick={runSimulation} disabled={simState.running}
                        style={{background:simState.running?"#111":"#0A1A2A",
                          border:`1px solid ${simState.running?"#2A2A2A":"#2A6A9A"}`,
                          color:simState.running?"#666":"#5AACDF",
                          fontSize:11,padding:"8px 14px",cursor:simState.running?"not-allowed":"pointer",
                          fontFamily:"inherit",letterSpacing:2,fontWeight:700,width:"100%"}}>
                        {simState.running
                          ? `${simState.phase || "running"}... ${simState.symbol ? `[${simState.symbol}]` : ""} ${simState.total ? `${simState.done}/${simState.total}` : ""}`
                          : "▶ RUN SIMULATION"}
                      </button>

                      {simResult?.error && (
                        <div style={{marginTop:10,fontSize:10,color:"#E74C3C"}}>⚠ {simResult.error}</div>
                      )}

                      {simResult && !simResult.error && simResult.metrics && (() => {
                        const M = simResult.metrics;
                        const edge = summariseEdge(M);
                        const exitTotal = M.exitReasons.stopHits + M.exitReasons.targetHits + M.exitReasons.timedOut;
                        return (
                          <div style={{marginTop:12}}>
                            {/* Edge headline */}
                            <div style={{padding:"8px 10px",background:"#080808",border:`1px solid ${edge.color}55`,borderLeft:`3px solid ${edge.color}`,marginBottom:10}}>
                              <div style={{fontSize:9,color:"#555",letterSpacing:2}}>EDGE — NET OF COSTS</div>
                              <div style={{fontSize:13,color:edge.color,fontWeight:900,marginTop:2}}>{edge.label}</div>
                              <div style={{fontSize:9,color:"#666",marginTop:2}}>
                                {M.n} trades · {(M.winRate*100).toFixed(0)}% wins · {simResult.daysAgo}d of {simResult.barsSource || "?"} bars · max-hold {simResult.holdHours}h ·
                                <span style={{color:simResult.costBps>0?"#C9A84C":"#666"}}>
                                  {" "}costs {simResult.costBps ?? 0} bps/round-trip
                                </span>
                              </div>
                            </div>

                            {/* P&L grid */}
                            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:6,marginBottom:10}}>
                              {[
                                ["TOTAL P&L",     `${M.totalPnl>=0?"+":""}${M.totalPnl.toFixed(2)}%`, M.totalPnl>=0?"#2ECC71":"#E74C3C"],
                                ["PROFIT FACTOR", isFinite(M.profitFactor)?M.profitFactor.toFixed(2):"∞",  M.profitFactor>=1.5?"#2ECC71":M.profitFactor>=1?"#C9A84C":"#E74C3C"],
                                ["EXPECTANCY",    `${M.expectancy>=0?"+":""}${M.expectancy.toFixed(3)}%/trade`, M.expectancy>=0?"#2ECC71":"#E74C3C"],
                                ["AVG WIN",       `+${M.avgWin.toFixed(2)}%`, "#2ECC71"],
                                ["AVG LOSS",      `−${M.avgLoss.toFixed(2)}%`, "#E74C3C"],
                                ["REALISED R:R",  `${M.realisedRR.toFixed(2)}:1`, M.realisedRR>=2?"#2ECC71":M.realisedRR>=1?"#C9A84C":"#E74C3C"],
                                ["SHARPE (per-trade)", M.sharpe.toFixed(2), M.sharpe>=0.3?"#2ECC71":M.sharpe>=0?"#C9A84C":"#E74C3C"],
                                ["MAX DRAWDOWN",  `−${M.maxDD.toFixed(2)}%`, M.maxDD<5?"#2ECC71":M.maxDD<15?"#C9A84C":"#E74C3C"],
                                ["MAX CONSEC LOSSES", `${M.maxConsLoss}`, M.maxConsLoss<=4?"#888":M.maxConsLoss<=8?"#C9A84C":"#E74C3C"],
                              ].map(([k,v,c])=>(
                                <div key={k} style={{padding:"6px 8px",background:"#080808",border:"1px solid #1A1A1A"}}>
                                  <div style={{fontSize:8,color:"#555",letterSpacing:1}}>{k}</div>
                                  <div style={{fontSize:12,color:c,fontWeight:700,marginTop:2}}>{v}</div>
                                </div>
                              ))}
                            </div>

                            {/* Equity curve */}
                            {M.equity.length > 1 && (
                              <div style={{padding:"8px 10px",background:"#080808",border:"1px solid #1A1A1A",marginBottom:10}}>
                                <div style={{fontSize:9,color:"#555",letterSpacing:2,marginBottom:4}}>EQUITY CURVE (cumulative %, in trade order)</div>
                                <svg width="100%" height="50" viewBox="0 0 200 50" preserveAspectRatio="none">
                                  {(() => {
                                    const a = M.equity;
                                    const min = Math.min(...a, 0);
                                    const max = Math.max(...a, 0);
                                    const rng = (max-min)||1;
                                    const pts = a.map((v,i)=>`${(i/(a.length-1))*200},${50-((v-min)/rng)*50}`).join(" ");
                                    const zeroY = 50-((0-min)/rng)*50;
                                    return <>
                                      <line x1="0" y1={zeroY} x2="200" y2={zeroY} stroke="#333" strokeDasharray="2,2"/>
                                      <polyline points={pts} fill="none"
                                        stroke={a[a.length-1]>=0?"#2ECC71":"#E74C3C"} strokeWidth="1.2"/>
                                    </>;
                                  })()}
                                </svg>
                              </div>
                            )}

                            {/* Exit reason breakdown */}
                            <div style={{padding:"8px 10px",background:"#080808",border:"1px solid #1A1A1A",marginBottom:10,fontSize:10,color:"#888"}}>
                              <div style={{fontSize:9,color:"#555",letterSpacing:2,marginBottom:4}}>EXIT REASONS</div>
                              <div style={{display:"flex",gap:14,flexWrap:"wrap"}}>
                                <span>🎯 Target: <b style={{color:"#2ECC71"}}>{M.exitReasons.targetHits}</b> ({((M.exitReasons.targetHits/exitTotal)*100).toFixed(0)}%)</span>
                                <span>🛑 Stop: <b style={{color:"#E74C3C"}}>{M.exitReasons.stopHits}</b> ({((M.exitReasons.stopHits/exitTotal)*100).toFixed(0)}%)</span>
                                <span>⏱ Timed out: <b style={{color:"#C9A84C"}}>{M.exitReasons.timedOut}</b> ({((M.exitReasons.timedOut/exitTotal)*100).toFixed(0)}%)</span>
                              </div>
                            </div>

                            {/* Per-symbol breakdown — top 3 best, top 3 worst */}
                            {M.symbolRows.length > 0 && (
                              <div style={{padding:"8px 10px",background:"#080808",border:"1px solid #1A1A1A",marginBottom:10}}>
                                <div style={{fontSize:9,color:"#555",letterSpacing:2,marginBottom:6}}>PER-SYMBOL P&L (best → worst)</div>
                                {M.symbolRows.map(s=>(
                                  <div key={s.symbol} style={{display:"flex",justifyContent:"space-between",fontSize:10,padding:"2px 0",borderBottom:"1px solid #111"}}>
                                    <span style={{color:"#CCC",letterSpacing:1,minWidth:60}}>{s.symbol}</span>
                                    <span style={{color:"#666"}}>{s.n} trades</span>
                                    <span style={{color:"#888"}}>{(s.winRate*100).toFixed(0)}% wins</span>
                                    <span style={{color:s.pnl>=0?"#2ECC71":"#E74C3C",fontWeight:700,minWidth:60,textAlign:"right"}}>
                                      {s.pnl>=0?"+":""}{s.pnl.toFixed(2)}%
                                    </span>
                                  </div>
                                ))}
                              </div>
                            )}

                            {simResult.errors?.length > 0 && (
                              <div style={{color:"#C9A84C",fontSize:9,marginTop:4}}>
                                ⚠ {simResult.errors.length} symbol(s) failed to fetch: {simResult.errors.map(e=>e.symbol).join(", ")}
                              </div>
                            )}

                            {/* Per-symbol fetch diagnostics — same as multi-sim
                                panel. Shows source + trade count in-UI so iPad
                                users don't need devtools to see silent drops. */}
                            {simResult.fetchLog?.length > 0 && (
                              <div style={{padding:"6px 10px",background:"#080808",border:"1px solid #1A1A1A",marginTop:10}}>
                                <div style={{fontSize:8,color:"#555",letterSpacing:2,marginBottom:6}}>
                                  🔍 FETCH DIAGNOSTICS
                                </div>
                                <div style={{display:"grid",gridTemplateColumns:"1.2fr 0.8fr 0.8fr 0.8fr 1.4fr",gap:4,fontSize:9}}>
                                  <div style={{color:"#555"}}>SYMBOL</div>
                                  <div style={{color:"#555"}}>SOURCE</div>
                                  <div style={{color:"#555"}}>BARS</div>
                                  <div style={{color:"#555"}}>TRADES</div>
                                  <div style={{color:"#555"}}>STATUS</div>
                                  {simResult.fetchLog.map(f => (
                                    <React.Fragment key={f.symbol}>
                                      <div style={{color:"#CCC",letterSpacing:1}}>{f.symbol}</div>
                                      <div style={{color:f.source === "polygon" ? "#7FD8A6" : f.source === "yahoo" ? "#5AACDF" : "#E74C3C"}}>
                                        {f.source || "—"}
                                      </div>
                                      <div style={{color:"#888"}}>{f.bars || 0}</div>
                                      <div style={{color:f.trades > 0 ? "#888" : "#C9A84C"}}>{f.trades || 0}</div>
                                      <div style={{color:f.reason === "fetch_failed" ? "#E74C3C" : f.trades === 0 ? "#C9A84C" : "#2ECC71"}}>
                                        {f.reason === "fetch_failed" ? "FETCH FAILED"
                                          : f.trades === 0 ? "0 trades — all skipped as neutral (train models)"
                                          : "OK"}
                                      </div>
                                    </React.Fragment>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })()}
                    </div>

                    {/* ═══ TRAIN NN ON SIM ═══ (separate, sim-only training) */}
                    <div style={{background:"#0A140E",border:"1px solid #2A6A4F",padding:12}}>
                      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8,gap:6,flexWrap:"wrap"}}>
                        <div style={{fontSize:9,color:"#7FD8A6",letterSpacing:2}}>🧠 TRAIN NN / GBM ON SIMULATED TRADES</div>
                        {/* Universe-aware reset buttons. Previously the NN
                            reset button was hidden because getNNInfo() was
                            called without a universe arg, defaulting to
                            equities — so on btc universe the check read
                            the wrong NN state and hid the button forever.
                            Now always-visible per-universe reset controls
                            so the user can clear any model state without
                            trying to guess which tier they're on. */}
                        <div style={{display:"flex",gap:4}}>
                          {(() => {
                            const nnInfo = getNNInfo(universe);
                            const gbmInfo = getGBMInfo(universe);
                            const nnTrained = (nnInfo?.trainedOn || 0) > 0;
                            const gbmTrained = (gbmInfo?.trainedOn || 0) > 0;
                            const resetAll = () => {
                              if (!confirm(`Reset ALL trained models for ${universe.toUpperCase()} universe? (NN + GBM + regime). LR weights stay.`)) return;
                              resetNN(universe);
                              resetGBM(universe);
                              resetRegime(universe);
                              setTrainResult(null);
                            };
                            return (
                              <>
                                <button onClick={()=>{resetNN(universe);setTrainResult(null);}}
                                  disabled={!nnTrained}
                                  title={nnTrained ? `Reset NN weights for ${universe}` : `NN not trained on ${universe}`}
                                  style={{background:nnTrained?"#1A0808":"#0A0A0A",border:`1px solid ${nnTrained?"#4A1A1A":"#222"}`,color:nnTrained?"#E74C3C":"#555",fontSize:8,padding:"3px 8px",cursor:nnTrained?"pointer":"not-allowed",fontFamily:"inherit",letterSpacing:1}}>
                                  RESET NN{nnTrained ? "" : " (untrained)"}
                                </button>
                                <button onClick={()=>{resetGBM(universe);setTrainResult(null);}}
                                  disabled={!gbmTrained}
                                  title={gbmTrained ? `Reset GBM weights for ${universe}` : `GBM not trained on ${universe}`}
                                  style={{background:gbmTrained?"#1A0808":"#0A0A0A",border:`1px solid ${gbmTrained?"#4A1A1A":"#222"}`,color:gbmTrained?"#E74C3C":"#555",fontSize:8,padding:"3px 8px",cursor:gbmTrained?"pointer":"not-allowed",fontFamily:"inherit",letterSpacing:1}}>
                                  RESET GBM{gbmTrained ? "" : " (untrained)"}
                                </button>
                                <button onClick={resetAll}
                                  disabled={!nnTrained && !gbmTrained}
                                  title={`Reset NN + GBM + regime for ${universe}`}
                                  style={{background:(nnTrained||gbmTrained)?"#2A0A0A":"#0A0A0A",border:`1px solid ${(nnTrained||gbmTrained)?"#6A1A1A":"#222"}`,color:(nnTrained||gbmTrained)?"#FF8A7A":"#555",fontSize:8,padding:"3px 8px",cursor:(nnTrained||gbmTrained)?"pointer":"not-allowed",fontFamily:"inherit",letterSpacing:1,fontWeight:700}}>
                                  RESET ALL
                                </button>
                              </>
                            );
                          })()}
                        </div>
                      </div>
                      <div style={{fontSize:10,color:"#888",lineHeight:1.6,marginBottom:10}}>
                        Backprops the neural network on the {simResult?.trades?.length || 0} sim-labelled trades above
                        (L2 regularisation, time-decay weighting, early stopping). Run a sim first, then train.
                        Training is independent of the user-reviewed log — that's a separate training source.
                      </div>
                      <button onClick={trainOnSim} disabled={training || !simResult?.trades?.length || simResult?.trades?.length < 8}
                        style={{background:training||!simResult?.trades?.length?"#111":"#0A1A14",
                          border:`1px solid ${training||!simResult?.trades?.length?"#2A2A2A":"#2A6A4F"}`,
                          color:training||!simResult?.trades?.length?"#666":"#7FD8A6",
                          fontSize:11,padding:"8px 14px",cursor:training||!simResult?.trades?.length?"not-allowed":"pointer",
                          fontFamily:"inherit",letterSpacing:2,fontWeight:700,width:"100%"}}>
                        {training
                          ? "training..."
                          : !simResult?.trades?.length
                            ? "▷ TRAIN NN ON SIM (run a simulation first)"
                            : simResult.trades.length < 8
                              ? `▷ TRAIN NN ON SIM (need ≥8 trades, have ${simResult.trades.length})`
                              : `▶ TRAIN NN ON ${simResult.trades.length} SIM TRADES`}
                      </button>

                      {trainResult && !trainResult.error && (
                        <div style={{marginTop:10,fontSize:10,color:"#888",lineHeight:1.7}}>
                          <div>✓ NN trained on <b style={{color:"#FFF"}}>{trainResult.trained}</b> samples
                            for <b style={{color:"#FFF"}}>{trainResult.epochs}</b> epochs
                            {trainResult.loss != null && <> (final loss <b style={{color:"#FFF"}}>{trainResult.loss.toFixed(4)}</b>)</>}.
                            Stopped: {trainResult.reason}</div>
                          {trainResult.gbm?.trees && (
                            <div>✓ GBM trained on <b style={{color:"#FFF"}}>{trainResult.gbm.trainedOn}</b> samples for <b style={{color:"#FFF"}}>{trainResult.gbm.rounds}</b> boosting rounds (final loss <b style={{color:"#FFF"}}>{trainResult.gbm.finalLoss?.toFixed(4)}</b>). Stopped: {trainResult.gbm.reason}</div>
                          )}
                          {trainResult.gbm && !trainResult.gbm.trees && (
                            <div style={{color:"#C9A84C"}}>⚠ GBM: {trainResult.gbm.reason}</div>
                          )}
                          {trainResult.bag?.ok && (
                            <div>✓ LR bag (<b style={{color:"#FFF"}}>{trainResult.bag.nBags}</b> bootstrap models) trained on <b style={{color:"#FFF"}}>{trainResult.bag.trainedOn}</b> samples — ensemble ready for uncertainty-aware predictions.</div>
                          )}
                          {trainResult.regime?.ok && (
                            <div>✓ Regime-switching GBMs:
                              {trainResult.regime.highTrained && <> high-VIX trained on <b style={{color:"#FFF"}}>{trainResult.regime.counts.high}</b> samples ({trainResult.regime.highRounds} rounds);</>}
                              {trainResult.regime.lowTrained && <> low-VIX trained on <b style={{color:"#FFF"}}>{trainResult.regime.counts.low}</b> samples ({trainResult.regime.lowRounds} rounds).</>}
                              {(!trainResult.regime.highTrained || !trainResult.regime.lowTrained) && <> The other regime had too few samples — falls back to universal GBM in that regime.</>}
                            </div>
                          )}
                          {trainResult.regime?.error && (
                            <div style={{color:"#888"}}>· Regime models: {trainResult.regime.error} (universal GBM still active)</div>
                          )}
                          {trainResult.bag?.error && (
                            <div style={{color:"#C9A84C"}}>⚠ LR bag: {trainResult.bag.error}</div>
                          )}
                          {trainResult.history?.length > 0 && (
                            <svg width="100%" height="40" style={{marginTop:6,background:"#080808"}} viewBox="0 0 200 40" preserveAspectRatio="none">
                              <polyline
                                points={trainResult.history.map((l,i,a) => {
                                  const maxL = Math.max(...a);
                                  const minL = Math.min(...a);
                                  const x = (i/(a.length-1))*200;
                                  const y = 40 - ((l-minL)/((maxL-minL)||1))*40;
                                  return `${x},${y}`;
                                }).join(" ")}
                                fill="none" stroke="#7FD8A6" strokeWidth="1.2" />
                            </svg>
                          )}
                        </div>
                      )}
                      {trainResult?.error && (
                        <div style={{marginTop:10,fontSize:10,color:"#E74C3C"}}>⚠ {trainResult.error}</div>
                      )}
                    </div>

                    {/* ═══ WALK-FORWARD VALIDATION ═══ (honest out-of-sample test) */}
                    <div style={{background:"#140A14",border:"1px solid #6A2A6A",padding:12}}>
                      <div style={{fontSize:9,color:"#D87FD8",letterSpacing:2,marginBottom:8}}>🧪 WALK-FORWARD VALIDATION (honest out-of-sample)</div>
                      <div style={{fontSize:10,color:"#888",lineHeight:1.6,marginBottom:10}}>
                        Splits sim trades chronologically into 5 folds. For each fold, trains a <b>fresh, isolated</b> NN
                        on all earlier folds and evaluates on the held-out fold. The production NN is NOT touched.
                        Out-of-sample metrics are the honest read — if test loss {"≫"} train loss, the model is
                        overfitting and the edge you see in training is fiction.
                      </div>
                      <button onClick={runWF} disabled={wfRunning || !simResult?.trades?.length || simResult?.trades?.length < 40}
                        style={{background:wfRunning||!simResult?.trades?.length?"#111":"#1A0A1A",
                          border:`1px solid ${wfRunning||!simResult?.trades?.length?"#2A2A2A":"#6A2A6A"}`,
                          color:wfRunning||!simResult?.trades?.length?"#666":"#D87FD8",
                          fontSize:11,padding:"8px 14px",cursor:wfRunning||!simResult?.trades?.length?"not-allowed":"pointer",
                          fontFamily:"inherit",letterSpacing:2,fontWeight:700,width:"100%"}}>
                        {wfRunning
                          ? "validating..."
                          : !simResult?.trades?.length
                            ? "▷ WALK-FORWARD (run a simulation first)"
                            : simResult.trades.length < 40
                              ? `▷ WALK-FORWARD (need ≥40 trades, have ${simResult.trades.length})`
                              : `▶ RUN 5-FOLD WALK-FORWARD on ${simResult.trades.length} trades`}
                      </button>

                      {wfResult?.error && (
                        <div style={{marginTop:10,fontSize:10,color:"#E74C3C"}}>⚠ {wfResult.error}</div>
                      )}

                      {wfResult && !wfResult.error && wfResult.overall && (() => {
                        const O = wfResult.overall;
                        const v = interpretWF(O);
                        const overfit = (O.avgTestLoss - O.avgTrainLoss) > 0.1;
                        return (
                          <div style={{marginTop:12}}>
                            <div style={{padding:"8px 10px",background:"#080808",border:`1px solid ${v.color}55`,borderLeft:`3px solid ${v.color}`,marginBottom:10}}>
                              <div style={{fontSize:9,color:"#555",letterSpacing:2}}>VERDICT — OUT-OF-SAMPLE</div>
                              <div style={{fontSize:13,color:v.color,fontWeight:900,marginTop:2}}>{v.label}</div>
                              <div style={{fontSize:9,color:"#666",marginTop:2}}>
                                {O.oosSamples} OOS predictions across {wfResult.folds.length} folds
                              </div>
                            </div>

                            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6,marginBottom:10}}>
                              {[
                                ["OOS AUC",       O.oosAUC?.toFixed(3) ?? "—",       O.oosAUC>=0.6?"#2ECC71":O.oosAUC>=0.55?"#C9A84C":O.oosAUC>=0.5?"#888":"#E74C3C"],
                                ["OOS ACCURACY",  `${((O.oosAccuracy||0)*100).toFixed(1)}%`, O.oosAccuracy>=0.55?"#2ECC71":O.oosAccuracy>=0.5?"#C9A84C":"#E74C3C"],
                                ["OOS LOG-LOSS",  O.oosLogLoss?.toFixed(4) ?? "—",   O.oosLogLoss<0.65?"#2ECC71":O.oosLogLoss<0.70?"#C9A84C":"#E74C3C"],
                                ["OOS BRIER",     O.oosBrier?.toFixed(4) ?? "—",     O.oosBrier<0.22?"#2ECC71":O.oosBrier<0.25?"#C9A84C":"#E74C3C"],
                                ["AVG TRAIN LOSS", O.avgTrainLoss?.toFixed(4) ?? "—", "#888"],
                                ["AVG TEST LOSS",  O.avgTestLoss?.toFixed(4) ?? "—",  overfit?"#E74C3C":"#888"],
                              ].map(([k,val,c])=>(
                                <div key={k} style={{padding:"6px 8px",background:"#080808",border:"1px solid #1A1A1A"}}>
                                  <div style={{fontSize:8,color:"#555",letterSpacing:1}}>{k}</div>
                                  <div style={{fontSize:12,color:c,fontWeight:700,marginTop:2}}>{val}</div>
                                </div>
                              ))}
                            </div>

                            {overfit && (
                              <div style={{padding:"6px 10px",background:"#1A0808",border:"1px solid #4A1A1A",color:"#E74C3C",fontSize:10,marginBottom:10}}>
                                ⚠ Train/test loss gap = {(O.avgTestLoss - O.avgTrainLoss).toFixed(4)}. The NN is memorising training folds
                                but not generalising to future ones. Get more data or simplify the model.
                              </div>
                            )}

                            <div style={{padding:"8px 10px",background:"#080808",border:"1px solid #1A1A1A"}}>
                              <div style={{fontSize:9,color:"#555",letterSpacing:2,marginBottom:6}}>PER-FOLD BREAKDOWN</div>
                              <div style={{display:"grid",gridTemplateColumns:"30px 1fr 1fr 1fr 1fr",gap:4,fontSize:9}}>
                                <div style={{color:"#555"}}>FOLD</div>
                                <div style={{color:"#555"}}>TRAIN/TEST</div>
                                <div style={{color:"#555"}}>TEST LOSS</div>
                                <div style={{color:"#555"}}>ACC</div>
                                <div style={{color:"#555"}}>AUC</div>
                                {wfResult.folds.map(f => (
                                  <React.Fragment key={f.fold}>
                                    <div style={{color:"#888"}}>{f.fold}</div>
                                    <div style={{color:"#888"}}>{f.trainSize}/{f.testSize}</div>
                                    <div style={{color:"#CCC"}}>{f.testLoss?.toFixed(3) ?? "—"}</div>
                                    <div style={{color:"#CCC"}}>{((f.testAccuracy||0)*100).toFixed(0)}%</div>
                                    <div style={{color:f.testAUC>=0.6?"#2ECC71":f.testAUC>=0.5?"#C9A84C":"#E74C3C"}}>{f.testAUC?.toFixed(2) ?? "—"}</div>
                                  </React.Fragment>
                                ))}
                              </div>
                            </div>
                          </div>
                        );
                      })()}
                    </div>

                    {/* ═══ MULTI-SIM AVERAGING ═══ */}
                    <div style={{background:"#0A140F",border:"1px solid #2A6A4F",padding:12}}>
                      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8,flexWrap:"wrap",gap:6}}>
                        <div style={{fontSize:9,color:"#7FD8A6",letterSpacing:2}}>📊 MULTI-SIM AVERAGING — statistical edge test</div>
                        <div style={{display:"flex",alignItems:"center",gap:6}}
                          title="Number of independent sims to average. More runs = tighter confidence interval but longer wall-clock. With the session bars cache, each additional run after the first is nearly free (just re-samples entry points from already-fetched bars). 20 runs at tight SE gives a statistically meaningful answer; 50 for definitive.">
                          <span style={{fontSize:8,color:"#666",letterSpacing:1,cursor:"help"}}>RUNS</span>
                          <select value={multiSimNRuns} onChange={e=>setMultiSimNRuns(Number(e.target.value))}
                            disabled={multiSimRunning}
                            style={{background:"#080808",border:"1px solid #2A2A2A",color:"#7FD8A6",fontFamily:"inherit",fontSize:9,padding:"2px 6px"}}>
                            <option value={5}>5 (quick, weak SE)</option>
                            <option value={10}>10 (exploratory)</option>
                            <option value={20}>20 (standard)</option>
                            <option value={30}>30 (thorough)</option>
                            <option value={50}>50 (definitive)</option>
                          </select>
                        </div>
                      </div>
                      <div style={{fontSize:10,color:"#888",lineHeight:1.6,marginBottom:10}}>
                        Runs <b>{multiSimNRuns} independent</b> sims + walk-forwards, caches bars across runs so the
                        network cost is paid once. Reports <b style={{color:"#7FD8A6"}}>95% CI</b> on the mean AUC
                        (the honest answer to "could this be luck?"), trimmed-mean, std dev, and per-symbol
                        breakdown. One sim at AUC 0.62 is statistically meaningless — the {multiSimNRuns}-run
                        trimmed mean with a tight CI is not.
                      </div>
                      <button onClick={runMultiSim} disabled={multiSimRunning}
                        style={{background:multiSimRunning?"#111":"#0F1F18",
                          border:`1px solid ${multiSimRunning?"#2A2A2A":"#2A6A4F"}`,
                          color:multiSimRunning?"#666":"#7FD8A6",
                          fontSize:11,padding:"8px 14px",cursor:multiSimRunning?"not-allowed":"pointer",
                          fontFamily:"inherit",letterSpacing:2,fontWeight:700,width:"100%"}}>
                        {multiSimRunning
                          ? `${multiSimState.phase || "starting"}... run ${multiSimState.run}/${multiSimState.total}`
                          : `▶ RUN ${multiSimNRuns}-SIM AVERAGE`}
                      </button>

                      {/* ─── ABLATION STUDY (Phase 4 Commit 7) ─────────────
                          Per-feature leave-one-out AUC delta on the trades
                          from the LAST single-sim. Cheap compared to multi-
                          sim (~N+1 walk-forwards), objective measurement
                          of which features actually carry signal. */}
                      {isCryptoUniverse(universe) && (() => {
                        const hasTrades = simResult?.trades?.length >= 50;
                        const disabled = ablationRunning || !hasTrades;
                        const label = ablationRunning
                          ? `ABLATING ${ablationProgress.current || "…"} (${ablationProgress.idx}/${ablationProgress.total})`
                          : !hasTrades
                            ? `🧪 ABLATE FEATURES — NEEDS ≥50 TRADES FIRST (have ${simResult?.trades?.length ?? 0})`
                            : `🧪 ABLATE FEATURES — uses ${simResult.trades.length} pooled trades`;
                        return (
                        <div style={{marginTop:10}}>
                          <button onClick={runAblation} disabled={disabled}
                            style={{background: ablationRunning ? "#111" : !hasTrades ? "#1A0F0F" : "#14141F",
                              border:`1px solid ${ablationRunning?"#2A2A2A":!hasTrades?"#6A2A2A":"#6A6AC9"}`,
                              color: ablationRunning ? "#666" : !hasTrades ? "#C97A7A" : "#B0B0FF",
                              fontSize:10,padding:"6px 12px",cursor: disabled ? "not-allowed" : "pointer",
                              fontFamily:"inherit",letterSpacing:1.5,fontWeight:700,width:"100%"}}
                            title={hasTrades
                              ? "Leave-one-out feature ablation. Runs walk-forward once per feature slot with that slot zeroed, measures AUC delta vs baseline."
                              : "Run RUN SIMULATION or RUN 20-SIM AVERAGE first. Ablation needs ≥50 trades in simResult to work."}>
                            {label}
                          </button>
                          {ablationResult?.error && (
                            <div style={{marginTop:6,fontSize:9,color:"#E74C3C"}}>⚠ {ablationResult.error}</div>
                          )}
                          {ablationResult && !ablationResult.error && (
                            <div style={{marginTop:8,padding:"6px 10px",background:"#080810",border:"1px solid #1A1A28",fontSize:9,color:"#888"}}>
                              <div style={{fontSize:8,color:"#6A6AC9",letterSpacing:2,marginBottom:6}}>
                                🧪 ABLATION — per-feature AUC delta vs baseline ({ablationResult.baselineAUC?.toFixed(3)})
                              </div>
                              <div style={{display:"grid",gridTemplateColumns:"0.6fr 2fr 1fr 1fr",gap:4,fontSize:9,marginBottom:4}}>
                                <div style={{color:"#555"}}>SLOT</div>
                                <div style={{color:"#555"}}>FEATURE</div>
                                <div style={{color:"#555"}}>AUC w/o</div>
                                <div style={{color:"#555"}}>Δ vs base</div>
                                {ablationResult.results.map(r => {
                                  const d = r.delta;
                                  const col = d == null ? "#666"
                                            : d > 0.005 ? "#2ECC71"    // feature helps
                                            : d < -0.005 ? "#E74C3C"    // feature hurts
                                            : "#888";                   // dead / neutral
                                  return (
                                    <React.Fragment key={r.slot}>
                                      <div style={{color:"#888"}}>[{r.slot}]</div>
                                      <div style={{color:"#CCC"}}>{r.name}</div>
                                      <div style={{color:"#888"}}>{r.aucWithout?.toFixed(3) ?? "—"}</div>
                                      <div style={{color:col,fontWeight:700}}>
                                        {d == null ? "—" : (d >= 0 ? "+" : "") + d.toFixed(4)}
                                      </div>
                                    </React.Fragment>
                                  );
                                })}
                              </div>
                              <div style={{fontSize:8,color:"#555",marginTop:6,lineHeight:1.5}}>
                                Δ &gt; +0.005 (green) = feature carries signal; removing it hurts.
                                Δ ≈ 0 = feature is dead or redundant with others.
                                Δ &lt; -0.005 (red) = feature is actively contributing noise; consider dropping.
                                Ordering: top = most important. Δ values below ±0.005 are within CI
                                noise at our sample size; repeat ablation on different sim windows
                                to confirm before acting on small deltas.
                              </div>
                            </div>
                          )}
                        </div>
                        );
                      })()}
                      {multiSimResult?.error && (
                        <>
                          <div style={{marginTop:10,fontSize:10,color:"#E74C3C",lineHeight:1.6}}>⚠ {multiSimResult.error}</div>
                          {/* Render fetch diagnostics even in the error case —
                              often the failure reason is data-related (symbol
                              dropped silently) or skip-neutral, and the panel
                              tells the user exactly what happened per-symbol. */}
                          {multiSimResult.fetchLog?.length > 0 && (() => {
                            const okCount = multiSimResult.fetchLog.filter(f => f.source).length;
                            const failCount = multiSimResult.fetchLog.filter(f => !f.source).length;
                            return (
                              <div style={{padding:"6px 10px",background:"#080808",border:"1px solid #1A1A1A",marginTop:10}}>
                                <div style={{fontSize:8,color:"#555",letterSpacing:2,marginBottom:6}}>
                                  🔍 FETCH DIAGNOSTICS — {okCount} fetched / {failCount} failed ({multiSimResult.fetchLog.length} attempted)
                                </div>
                                <div style={{display:"grid",gridTemplateColumns:"1.2fr 0.8fr 0.8fr 0.8fr 1.4fr",gap:4,fontSize:9}}>
                                  <div style={{color:"#555"}}>SYMBOL</div>
                                  <div style={{color:"#555"}}>SOURCE</div>
                                  <div style={{color:"#555"}}>BARS</div>
                                  <div style={{color:"#555"}}>TRADES</div>
                                  <div style={{color:"#555"}}>STATUS</div>
                                  {multiSimResult.fetchLog.map(f => (
                                    <React.Fragment key={f.symbol}>
                                      <div style={{color:"#CCC",letterSpacing:1}}>{f.symbol}</div>
                                      <div style={{color:f.source === "polygon" ? "#7FD8A6" : f.source === "yahoo" ? "#5AACDF" : "#E74C3C"}}>
                                        {f.source || "—"}
                                      </div>
                                      <div style={{color:"#888"}}>{f.bars || 0}</div>
                                      <div style={{color:f.trades > 0 ? "#888" : "#C9A84C"}}>{f.trades || 0}</div>
                                      <div style={{color:f.reason === "fetch_failed" ? "#E74C3C" : f.trades === 0 ? "#C9A84C" : "#2ECC71",fontSize:8}}>
                                        {f.reason === "fetch_failed" ? "FETCH FAILED"
                                          : f.trades === 0 ? "0 trades (neutral skip)"
                                          : "OK"}
                                      </div>
                                    </React.Fragment>
                                  ))}
                                </div>
                              </div>
                            );
                          })()}
                        </>
                      )}
                      {multiSimResult && !multiSimResult.error && (() => {
                        const r = multiSimResult;
                        // ─── POOLED AUC is the honest headline ────────────
                        // Trimmed-mean-AUC averages 20 walk-forwards each on
                        // ~N/20 trades per run → at single-asset BTC that's
                        // ~22 samples per fold, which is sample-size-starved
                        // and produces high-variance AUCs in the 0.30-0.60
                        // range. Averaging them looks like coin-flip even
                        // when the trained model HAS signal.
                        //
                        // Ablation baseline, when available, runs ONE walk-
                        // forward on ALL pooled trades (~440/fold at 2200
                        // total). That's the honest trained-model AUC
                        // measurement. Use it as the primary verdict when
                        // the user has run ablation; fall back to trimmed
                        // mean otherwise.
                        const pooledAUC = ablationResult && !ablationResult.error
                          ? ablationResult.baselineAUC
                          : null;
                        const hasPooled = pooledAUC != null;
                        const HIGH_VARIANCE = r.stdAUC > 0.10;
                        const MED_VARIANCE  = r.stdAUC > 0.06;
                        // Verdict prefers pooled AUC when available; only
                        // falls back to trimmed mean if ablation hasn't run.
                        let verdict;
                        if (hasPooled) {
                          // Pooled AUC is the trained-model truth. At
                          // N≈2200 trades SE is ~0.011, so thresholds can
                          // be tighter than the trimmed-mean versions.
                          if (pooledAUC >= 0.56) {
                            verdict = { label: "Real signal on pooled data — train and deploy", color: "#2ECC71" };
                          } else if (pooledAUC >= 0.53) {
                            verdict = { label: "Marginal edge on pooled data — train, watch variance", color: "#C9A84C" };
                          } else if (pooledAUC >= 0.50) {
                            verdict = { label: "Near-coin-flip even on pooled data — feature set needs work", color: "#C9A84C" };
                          } else {
                            verdict = { label: "Inverted — pooled AUC below 0.50, features anti-signal", color: "#E74C3C" };
                          }
                        } else if (HIGH_VARIANCE) {
                          verdict = { label: "Unstable — fragile / regime-dependent, don't train", color: "#E74C3C" };
                        } else if (r.trimmedMeanAUC >= 0.55 && !MED_VARIANCE) {
                          verdict = { label: "Genuine signal — train it", color: "#2ECC71" };
                        } else if (r.trimmedMeanAUC >= 0.55 && MED_VARIANCE) {
                          verdict = { label: "Edge present but inconsistent — investigate variance first", color: "#C9A84C" };
                        } else if (r.trimmedMeanAUC >= 0.52) {
                          verdict = { label: "Marginal — borderline", color: "#C9A84C" };
                        } else {
                          verdict = { label: "Coin flip — don't train", color: "#E74C3C" };
                        }
                        return (
                          <div style={{marginTop:12}}>
                            {/* POOLED AUC headline panel — only renders
                                once ablation has been run. This is the
                                honest trained-model measurement (single
                                WF on all ~2200 pooled trades) vs the
                                per-run trimmed mean (20 WFs on ~110 each
                                which is sample-size starved). */}
                            {hasPooled && (
                              <div style={{padding:"10px 12px",background:"#080810",border:`2px solid ${verdict.color}`,marginBottom:10}}>
                                <div style={{fontSize:9,color:"#555",letterSpacing:2,marginBottom:4}}>
                                  🎯 POOLED AUC — trained-model measurement ({ablationResult.baselineSamples ?? "?"} trades in walk-forward)
                                </div>
                                <div style={{display:"flex",alignItems:"baseline",gap:12,marginBottom:4}}>
                                  <div style={{fontSize:32,color:verdict.color,fontWeight:900,fontFamily:"inherit"}}>
                                    {pooledAUC.toFixed(3)}
                                  </div>
                                  <div style={{fontSize:11,color:verdict.color,fontWeight:700}}>
                                    {verdict.label}
                                  </div>
                                </div>
                                <div style={{fontSize:8,color:"#666",lineHeight:1.5}}>
                                  This is the honest trained-model AUC. The per-run trimmed
                                  mean below is a VARIANCE DIAGNOSTIC — it measures how the
                                  same model performs on sample-starved 110-trade folds.
                                  Per-run variance &gt; 0.06 is expected at single-asset
                                  sample sizes and is NOT a verdict against the model.
                                </div>
                              </div>
                            )}
                            <div style={{padding:"8px 10px",background:"#080808",border:`1px solid ${verdict.color}55`,borderLeft:`3px solid ${verdict.color}`,marginBottom:10}}>
                              <div style={{fontSize:9,color:"#555",letterSpacing:2}}>
                                {hasPooled ? `PER-RUN STABILITY (${r.runs} runs × ~110 trades each)` : `VERDICT — ${r.runs}-RUN TRIMMED MEAN`}
                              </div>
                              <div style={{fontSize:13,color:verdict.color,fontWeight:900,marginTop:2}}>
                                {hasPooled ? `Trimmed mean ${r.trimmedMeanAUC.toFixed(3)} — diagnostic only, not the headline` : verdict.label}
                              </div>
                            </div>
                            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:6,marginBottom:10}}>
                              {[
                                ["TRIMMED MEAN AUC", r.trimmedMeanAUC.toFixed(3), r.trimmedMeanAUC>=0.55?"#2ECC71":r.trimmedMeanAUC>=0.52?"#C9A84C":"#E74C3C"],
                                ["RAW MEAN AUC",     r.meanAUC.toFixed(3), "#888"],
                                ["95% CI ON MEAN",   `[${r.ci95Low.toFixed(3)}, ${r.ci95High.toFixed(3)}]`, (r.ci95Low>0.50?"#2ECC71":r.ci95High<0.50?"#E74C3C":"#C9A84C")],
                                ["AUC STD DEV",      r.stdAUC.toFixed(3), r.stdAUC<0.03?"#2ECC71":r.stdAUC<0.06?"#C9A84C":"#E74C3C"],
                                ["IQR",              `[${r.iqrLow.toFixed(3)}, ${r.iqrHigh.toFixed(3)}]`, "#888"],
                                ["RANGE",            `${r.minAUC.toFixed(3)} – ${r.maxAUC.toFixed(3)}`, "#888"],
                                ["MEAN OOS ACC",     `${(r.meanAccuracy*100).toFixed(1)}%`, "#888"],
                                ["MEAN LOG-LOSS",    r.meanLogLoss.toFixed(4), r.meanLogLoss<0.69?"#2ECC71":r.meanLogLoss<0.75?"#C9A84C":"#E74C3C"],
                                ["SE(MEAN)",         r.seAUC.toFixed(4), "#888"],
                              ].map(([k,v,c])=>(
                                <div key={k} style={{padding:"6px 8px",background:"#080808",border:"1px solid #1A1A1A"}}>
                                  <div style={{fontSize:8,color:"#555",letterSpacing:1}}>{k}</div>
                                  <div style={{fontSize:11,color:c,fontWeight:700,marginTop:2}}>{v}</div>
                                </div>
                              ))}
                            </div>

                            {/* Histogram of per-run AUCs — visual sanity check
                                on the distribution shape. If the histogram is
                                bimodal or has a long tail, the point-estimate
                                metrics are misleading. */}
                            <div style={{padding:"6px 10px",background:"#080808",border:"1px solid #1A1A1A",marginBottom:10}}>
                              <div style={{fontSize:8,color:"#555",letterSpacing:2,marginBottom:6}}>AUC DISTRIBUTION HISTOGRAM ({r.runs} runs, bins of 0.025)</div>
                              {(() => {
                                // 16 bins covering AUC 0.30 to 0.70 (0.025 each)
                                const bins = new Array(16).fill(0);
                                for (const a of r.aucs) {
                                  const idx = Math.max(0, Math.min(15, Math.floor((a - 0.30) / 0.025)));
                                  bins[idx]++;
                                }
                                const maxCount = Math.max(1, ...bins);
                                return (
                                  <div style={{display:"flex",alignItems:"flex-end",height:50,gap:2}}>
                                    {bins.map((c, i) => {
                                      const center = 0.30 + (i + 0.5) * 0.025;
                                      const isAt50 = center >= 0.4875 && center < 0.5125;
                                      const barColor = isAt50 ? "#C9A84C"
                                        : center >= 0.55 ? "#2ECC71"
                                        : center < 0.50 ? "#E74C3C"
                                        : "#888";
                                      return (
                                        <div key={i} title={`${center.toFixed(3)}±0.0125: ${c} run${c!==1?"s":""}`}
                                          style={{
                                            flex:1,
                                            height:`${(c/maxCount)*100}%`,
                                            background:c>0?barColor:"#1A1A1A",
                                            minHeight:c>0?2:0,
                                          }}/>
                                      );
                                    })}
                                  </div>
                                );
                              })()}
                              <div style={{display:"flex",justifyContent:"space-between",fontSize:7,color:"#555",marginTop:2}}>
                                <span>0.30</span><span>0.40</span><span style={{color:"#C9A84C"}}>0.50 (coin)</span><span>0.60</span><span>0.70</span>
                              </div>
                            </div>

                            {/* Per-run AUC list for transparency */}
                            <div style={{padding:"6px 10px",background:"#080808",border:"1px solid #1A1A1A",fontSize:9,color:"#888",marginBottom:10}}>
                              <div style={{fontSize:8,color:"#555",letterSpacing:2,marginBottom:4}}>PER-RUN AUCs (sorted; outer 10% trimmed)</div>
                              <div style={{lineHeight:1.8}}>
                                {[...r.aucs].sort((a,b)=>a-b).map((auc,i,a) => {
                                  const trimN = Math.max(1, Math.floor(a.length * 0.10));
                                  const isTrimmed = i < trimN || i >= a.length - trimN;
                                  return (
                                    <span key={i} style={{marginRight:8,color:isTrimmed?"#555":auc>0.50?"#7FD8A6":"#C9A84C"}}>
                                      {isTrimmed ? "⊘" : "·"}{auc.toFixed(3)}
                                    </span>
                                  );
                                })}
                              </div>
                            </div>

                            {/* CONVICTION-STRATIFIED METRICS — the answer to
                                "is there edge hiding in high-conviction
                                setups?" (meta-labeling, López de Prado
                                AFML Ch.3). If the overall AUC is 0.50 but
                                TOP-10 is 0.60+ with tight CI, the policy
                                "only trade when |prob − 0.5| > threshold"
                                IS the edge. Shown to green when AUC ≥ 0.55,
                                amber 0.52-0.55, red below 0.52. */}
                            {r.conviction && (
                              <div style={{padding:"6px 10px",background:"#080808",border:"1px solid #1A1A1A",fontSize:9,color:"#888",marginBottom:10}}>
                                <div style={{fontSize:8,color:"#555",letterSpacing:2,marginBottom:6}}>
                                  📈 CONVICTION-STRATIFIED — where the edge actually lives (meta-labeling)
                                </div>
                                <div style={{display:"grid",gridTemplateColumns:"1.2fr 1fr 1fr 1.2fr 1.2fr",gap:4,fontSize:9,marginBottom:4}}>
                                  <div style={{color:"#555"}}>SUBSET</div>
                                  <div style={{color:"#555"}}>MEAN AUC</div>
                                  <div style={{color:"#555"}}>LOG-LOSS</div>
                                  <div style={{color:"#555"}}>THRESHOLD |p−0.5|</div>
                                  <div style={{color:"#555"}}>POLICY</div>
                                  {["all","top50","top30","top10"].map(key => {
                                    const b = r.conviction[key];
                                    if (!b || b.meanAUC == null) return null;
                                    const col = b.meanAUC >= 0.55 ? "#2ECC71" : b.meanAUC >= 0.52 ? "#C9A84C" : "#E74C3C";
                                    const label = key === "all" ? "ALL trades" :
                                                  key === "top50" ? "Top 50%" :
                                                  key === "top30" ? "Top 30%" : "Top 10%";
                                    const thr = b.meanThreshold != null ? b.meanThreshold.toFixed(3) : "—";
                                    const policy = key === "all"
                                      ? "baseline (every setup)"
                                      : `trade only if |prob−0.5| ≥ ${thr}`;
                                    return (
                                      <React.Fragment key={key}>
                                        <div style={{color:"#CCC"}}>{label}</div>
                                        <div style={{color:col,fontWeight:700}}>{b.meanAUC.toFixed(3)}</div>
                                        <div style={{color: b.meanLogLoss != null && b.meanLogLoss < 0.69 ? "#2ECC71" : "#888"}}>
                                          {b.meanLogLoss != null ? b.meanLogLoss.toFixed(3) : "—"}
                                        </div>
                                        <div style={{color:"#888"}}>{thr}</div>
                                        <div style={{color:"#666",fontSize:8}}>{policy}</div>
                                      </React.Fragment>
                                    );
                                  })}
                                </div>
                                <div style={{fontSize:8,color:"#555",marginTop:6,lineHeight:1.5}}>
                                  If TOP-10% AUC ≫ ALL AUC with tight CI, the model has real edge on
                                  high-conviction setups. The THRESHOLD column gives the |p−0.5| floor
                                  you'd use as a trading filter — e.g. 0.08 means "only trade when
                                  composite ≥ 0.58 or ≤ 0.42". If all rows land at 0.50, no amount of
                                  selectivity rescues the model — features genuinely carry no signal.
                                </div>
                              </div>
                            )}

                            {/* META-LABELING (López de Prado AFML Ch. 4).
                                A SECOND GBM trained on each fold learns
                                "when is the primary model right?" from
                                the same features. If meta-AUC clearly
                                exceeds 0.5, the meta has found feature-
                                patterns that predict primary-correctness.
                                The gated rows show primary AUC on just
                                the samples where meta ≥ threshold —
                                "only trade when meta endorses". */}
                            {r.meta && (
                              <div style={{padding:"6px 10px",background:"#080808",border:"1px solid #1A1A1A",fontSize:9,color:"#888",marginBottom:10}}>
                                <div style={{fontSize:8,color:"#555",letterSpacing:2,marginBottom:6}}>
                                  🎯 META-LABELING — secondary model predicts primary-correctness (AFML Ch.4)
                                </div>
                                <div style={{fontSize:9,color:"#CCC",marginBottom:6}}>
                                  META AUC: <span style={{color: r.meta.meanMetaAUC > 0.55 ? "#2ECC71" : r.meta.meanMetaAUC > 0.52 ? "#C9A84C" : "#E74C3C", fontWeight:700}}>
                                    {r.meta.meanMetaAUC?.toFixed(3) ?? "—"}
                                  </span>
                                  {" "}— can the meta model predict "was primary right?" ({r.meta.runs} runs)
                                </div>
                                <div style={{display:"grid",gridTemplateColumns:"1.3fr 0.9fr 0.9fr 0.9fr 1.3fr",gap:4,fontSize:9,marginBottom:4}}>
                                  <div style={{color:"#555"}}>META GATE</div>
                                  <div style={{color:"#555"}}>PRIMARY AUC</div>
                                  <div style={{color:"#555"}}>LOG-LOSS</div>
                                  <div style={{color:"#555"}}>% KEPT</div>
                                  <div style={{color:"#555"}}>POLICY</div>
                                  {["t50","t55","t60"].map(key => {
                                    const b = r.meta.gated[key];
                                    if (!b || b.meanAUC == null) return null;
                                    const col = b.meanAUC >= 0.55 ? "#2ECC71" : b.meanAUC >= 0.52 ? "#C9A84C" : "#E74C3C";
                                    const label = key === "t50" ? "meta ≥ 0.50" : key === "t55" ? "meta ≥ 0.55" : "meta ≥ 0.60";
                                    return (
                                      <React.Fragment key={key}>
                                        <div style={{color:"#CCC"}}>{label}</div>
                                        <div style={{color:col,fontWeight:700}}>{b.meanAUC.toFixed(3)}</div>
                                        <div style={{color: b.meanLogLoss != null && b.meanLogLoss < 0.69 ? "#2ECC71" : "#888"}}>
                                          {b.meanLogLoss != null ? b.meanLogLoss.toFixed(3) : "—"}
                                        </div>
                                        <div style={{color:"#888"}}>{b.meanKept != null ? (b.meanKept * 100).toFixed(0)+"%" : "—"}</div>
                                        <div style={{color:"#666",fontSize:8}}>trade if meta ≥ {b.threshold.toFixed(2)}</div>
                                      </React.Fragment>
                                    );
                                  })}
                                </div>
                                <div style={{fontSize:8,color:"#555",marginTop:6,lineHeight:1.5}}>
                                  Meta-labeling differs from conviction-stratified: conviction trusts
                                  |prob−0.5| as confidence. Meta LEARNS from features when the primary
                                  is trustworthy — can find regime-specific signal (e.g. "primary
                                  reliable when funding-z &gt; 0 AND TS-mom &gt; 0, noise elsewhere").
                                  If META AUC &gt; 0.55 AND gated primary AUC &gt; ungated, the meta
                                  filter IS the edge. If META AUC ≈ 0.50, primary errors are
                                  unstructured noise with no feature-regime to learn from.
                                </div>
                              </div>
                            )}

                            {/* FETCH DIAGNOSTICS — which data source served
                                each symbol on the first run. Essential on
                                iPad where devtools console isn't accessible.
                                Shows silent drops (BNB etc) in-UI. */}
                            {r.fetchLog?.length > 0 && (() => {
                              const okCount = r.fetchLog.filter(f => f.source).length;
                              const failCount = r.fetchLog.filter(f => !f.source).length;
                              return (
                                <div style={{padding:"6px 10px",background:"#080808",border:"1px solid #1A1A1A",marginBottom:10}}>
                                  <div style={{fontSize:8,color:"#555",letterSpacing:2,marginBottom:6}}>
                                    🔍 FETCH DIAGNOSTICS — {okCount} OK / {failCount} failed ({r.fetchLog.length} attempted)
                                  </div>
                                  <div style={{display:"grid",gridTemplateColumns:"1.2fr 0.8fr 0.8fr 0.8fr 1.4fr",gap:4,fontSize:9}}>
                                    <div style={{color:"#555"}}>SYMBOL</div>
                                    <div style={{color:"#555"}}>SOURCE</div>
                                    <div style={{color:"#555"}}>BARS</div>
                                    <div style={{color:"#555"}}>TRADES</div>
                                    <div style={{color:"#555"}}>STATUS</div>
                                    {r.fetchLog.map(f => (
                                      <React.Fragment key={f.symbol}>
                                        <div style={{color:"#CCC",letterSpacing:1}}>{f.symbol}</div>
                                        <div style={{color:f.source === "polygon" ? "#7FD8A6" : f.source === "yahoo" ? "#5AACDF" : "#E74C3C"}}>
                                          {f.source || "—"}
                                        </div>
                                        <div style={{color:"#888"}}>{f.bars || 0}</div>
                                        <div style={{color:f.trades > 0 ? "#888" : "#C9A84C"}}>{f.trades || 0}</div>
                                        <div style={{color:f.reason === "fetch_failed" ? "#E74C3C" : f.trades === 0 ? "#C9A84C" : "#2ECC71"}}>
                                          {f.reason === "fetch_failed" ? "FETCH FAILED"
                                            : f.trades === 0 ? "0 trades (all skipped as neutral)"
                                            : "OK"}
                                        </div>
                                      </React.Fragment>
                                    ))}
                                  </div>
                                  <div style={{fontSize:8,color:"#555",marginTop:6,lineHeight:1.5}}>
                                    Source: <span style={{color:"#7FD8A6"}}>polygon</span> = paid tier served it ·
                                    <span style={{color:"#5AACDF"}}> yahoo</span> = fallback ·
                                    <span style={{color:"#E74C3C"}}> —</span> = both sources failed.
                                    0 trades with a valid source means all entries hit the neutral-trade skip
                                    (untrained model → no conviction → no labelled trades). Train models first.
                                  </div>
                                </div>
                              );
                            })()}

                            {/* Per-symbol aggregated breakdown — which names
                                are driving the result. Useful for deciding
                                which symbols to drop from the watchlist. */}
                            {r.symbolBreakdown?.length > 0 && (
                              <div style={{padding:"6px 10px",background:"#080808",border:"1px solid #1A1A1A"}}>
                                <div style={{fontSize:8,color:"#555",letterSpacing:2,marginBottom:6}}>PER-SYMBOL AGGREGATED ({r.symbolBreakdown.reduce((s,x)=>s+x.total,0)} total trades)</div>
                                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:4,fontSize:9}}>
                                  <div style={{color:"#555"}}>SYMBOL</div>
                                  <div style={{color:"#555"}}>TRADES</div>
                                  <div style={{color:"#555"}}>WIN-RATE</div>
                                  <div style={{color:"#555"}}>AVG P&amp;L</div>
                                  {r.symbolBreakdown.map(s => (
                                    <React.Fragment key={s.symbol}>
                                      <div style={{color:"#CCC",letterSpacing:1}}>{s.symbol}</div>
                                      <div style={{color:"#888"}}>{s.total}</div>
                                      <div style={{color:s.winRate>=0.55?"#2ECC71":s.winRate>=0.45?"#888":"#E74C3C"}}>
                                        {(s.winRate*100).toFixed(0)}% ({s.wins}/{s.total})
                                      </div>
                                      <div style={{color:s.avgPnl>0?"#2ECC71":"#E74C3C"}}>
                                        {s.avgPnl>=0?"+":""}{s.avgPnl.toFixed(2)}%
                                      </div>
                                    </React.Fragment>
                                  ))}
                                </div>
                                <div style={{fontSize:8,color:"#555",marginTop:6,lineHeight:1.5}}>
                                  Symbols with &lt;45% win-rate across {r.runs} runs are likely dragging the model.
                                  Consider excluding them from BACKTEST_SYMBOLS or investigating why features don't fit.
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })()}
                    </div>

                    {/* ═══ SUGGESTED LEVELS — horizon-aware, two rows visible ═══ */}
                    {/* Active row (matching global HORIZON) is bright and bordered; the
                        other row is dimmed for reference. Still shown because sometimes
                        you want to sanity-check both (e.g. "if I got stopped at the
                        intraday level I'd still be OK on the swing target"). */}
                    {(() => {
                      const isSwing = simInterval === "1d";
                      return (
                        <div style={{background:"#111",border:"1px solid #1E1E1E",padding:12}}>
                          <div style={{fontSize:9,color:"#C9A84C",letterSpacing:2,marginBottom:10}}>
                            SUGGESTED LEVELS — active row matches current HORIZON (<span style={{color:isSwing?"#7FD8A6":"#5AACDF"}}>{isSwing?"swing":"intraday"}</span>)
                          </div>

                          {/* Intraday row */}
                          <div style={{padding:"8px 10px",background:"#0A0A0A",
                            border:!isSwing?"1px solid #5AACDF":"1px solid #1A1A1A",
                            borderLeft:`3px solid ${!isSwing?"#5AACDF":"#2A3A4A"}`,
                            marginBottom:8,opacity:isSwing?0.45:1}}>
                            <div style={{fontSize:9,color:"#5AACDF",letterSpacing:2,marginBottom:6}}>
                              INTRADAY (1-3h hold) — 1.5×ATR stop, 4.5×ATR target — 3:1 R/R
                              {!isSwing && <span style={{marginLeft:6,color:"#7FD8A6"}}>◀ ACTIVE</span>}
                            </div>
                            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:6,fontSize:10}}>
                              {[["LONG entry",`$${q.price?.toFixed(2)}`],["stop",`$${m.stopLong||"?"}`],["target",`$${m.tgt3Long||"?"}`],
                                ["SHORT entry",`$${q.price?.toFixed(2)}`],["stop",`$${m.stopShort||"?"}`],["target",`$${m.tgt3Short||"?"}`],
                              ].map(([l,v],i)=>(
                                <div key={i}>
                                  <div style={{fontSize:8,color:"#444",letterSpacing:1,textTransform:"uppercase"}}>{l}</div>
                                  <div style={{fontSize:11,color:"#C9A84C",fontWeight:700}}>{v}</div>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Swing row */}
                          <div style={{padding:"8px 10px",background:"#0A0A0A",
                            border:isSwing?"1px solid #7FD8A6":"1px solid #1A1A1A",
                            borderLeft:`3px solid ${isSwing?"#7FD8A6":"#2A4A3A"}`,
                            opacity:!isSwing?0.45:1}}>
                            <div style={{fontSize:9,color:"#7FD8A6",letterSpacing:2,marginBottom:6}}>
                              SWING (1-5d hold) — 2×daily-ATR stop, 6×daily-ATR target — 3:1 R/R
                              {m.dailyAtrEst > 0 && <span style={{color:"#666"}}> · daily-ATR est ${m.dailyAtrEst.toFixed(2)}</span>}
                              {isSwing && <span style={{marginLeft:6,color:"#C9A84C"}}>◀ ACTIVE</span>}
                            </div>
                            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:6,fontSize:10}}>
                              {[["LONG entry",`$${q.price?.toFixed(2)}`],["stop",`$${m.swingStopLong||"?"}`],["target",`$${m.swingTargetLong||"?"}`],
                                ["SHORT entry",`$${q.price?.toFixed(2)}`],["stop",`$${m.swingStopShort||"?"}`],["target",`$${m.swingTargetShort||"?"}`],
                              ].map(([l,v],i)=>(
                                <div key={i}>
                                  <div style={{fontSize:8,color:"#444",letterSpacing:1,textTransform:"uppercase"}}>{l}</div>
                                  <div style={{fontSize:11,color:"#7FD8A6",fontWeight:700}}>{v}</div>
                                </div>
                              ))}
                            </div>
                          </div>
                          <div style={{fontSize:9,color:"#555",marginTop:8,lineHeight:1.5}}>
                            Change horizon with the HORIZON toggle at the top of the app. Active row
                            is the one Claude uses when you ask for analysis, and the one position
                            sizing is calibrated against.
                          </div>
                        </div>
                      );
                    })()}

                    {/* ═══ POSITION SIZING ═══ */}
                    {(() => {
                      const sz = recommendSize(q, m);
                      if (!sz.explanation) return null;
                      const pctOfAccount = (sz.sizePct * 100).toFixed(2);
                      const notional = sz.notionalPct?.toFixed(1);
                      return (
                        <div style={{background:"#111",border:"1px solid #1E1E1E",padding:12}}>
                          <div style={{fontSize:9,color:"#555",letterSpacing:2,marginBottom:8}}>
                            POSITION SIZING — vol-targeted × fractional Kelly × 2% cap
                          </div>
                          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:6,marginBottom:8}}>
                            <div>
                              <div style={{fontSize:8,color:"#444",letterSpacing:1}}>ACCOUNT RISK</div>
                              <div style={{fontSize:14,color:"#7FD8A6",fontWeight:900}}>{pctOfAccount}%</div>
                              <div style={{fontSize:8,color:"#555"}}>of account on this trade</div>
                            </div>
                            <div>
                              <div style={{fontSize:8,color:"#444",letterSpacing:1}}>NOTIONAL SIZE</div>
                              <div style={{fontSize:14,color:"#C9A84C",fontWeight:900}}>{notional}%</div>
                              <div style={{fontSize:8,color:"#555"}}>of account as position</div>
                            </div>
                            <div>
                              <div style={{fontSize:8,color:"#444",letterSpacing:1}}>BINDING CONSTRAINT</div>
                              <div style={{fontSize:10,color:"#D87FD8",fontWeight:700,marginTop:3,textTransform:"uppercase",letterSpacing:1}}>{sz.explanation.binding}</div>
                            </div>
                          </div>
                          <div style={{fontSize:8,color:"#666",lineHeight:1.6,borderTop:"1px solid #1A1A1A",paddingTop:6}}>
                            ANN VOL {((sz.explanation.annualisedVol||0)*100).toFixed(1)}% ·
                            VOL-TARGET×{sz.explanation.volMultiplier?.toFixed(2)} ·
                            KELLY {(sz.explanation.kelly*100).toFixed(2)}% ·
                            MODEL EDGE {(sz.explanation.edge*100).toFixed(0)}%
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            );
          })()}

          {tab==="log"&&(()=>{
            const stats = getPerformanceStats();
            const reviewed = decisionLog.filter(d=>d.reviewed && d.features);
            const wts = getCurrentWeights(universe);
            // FEATURE_NAMES is now imported from model.js (14 entries, kept
            // in sync with the feature vector). The local 7-entry copy was
            // silently truncating the display of the 7 new macro+calendar
            // LR weights.
            return (
              <div style={{flex:1,overflowY:"auto",padding:14}}>
                {/* Header row */}
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10,flexWrap:"wrap",gap:8}}>
                  <div style={{color:"#C9A84C",fontSize:11,letterSpacing:2}}>DECISION LOG — {decisionLog.length} entries</div>
                  <div style={{display:"flex",gap:8,alignItems:"center",flexWrap:"wrap"}}>
                    {stats&&<div style={{fontSize:10,color:"#888"}}>
                      Win: <span style={{color:stats.winRate>50?"#2ECC71":"#E74C3C",fontWeight:700}}>{stats.winRate}%</span>
                      &nbsp;·&nbsp;<span style={{color:stats.avgPnl>0?"#2ECC71":"#E74C3C",fontWeight:700}}>{stats.avgPnl>0?"+":""}{stats.avgPnl}%</span>
                      &nbsp;·&nbsp;{stats.wins}W/{stats.losses}L
                    </div>}
                    <button
                      disabled={reviewed.length < 2}
                      onClick={()=>{
                        const reviewedSet = decisionLog.filter(d=>d.reviewed);
                        const lrRes  = adaptWeights(reviewedSet, 0.08, 40, universe);
                        const nnRes  = reviewedSet.length >= 8  ? trainNNFromLog(reviewedSet, universe)  : null;
                        const gbmRes = reviewedSet.length >= 20 ? trainGBMFromLog(reviewedSet, universe) : null;
                        alert(
                          `LR (logistic regression):\n` +
                          `  Updated from ${lrRes.trained} reviewed trades (40 epochs).\n\n` +
                          `NN (neural network):\n` +
                          (nnRes
                            ? `  Trained on ${nnRes.trained} samples for ${nnRes.epochs} epochs.\n  Stopped: ${nnRes.reason}`
                            : `  Skipped — needs ≥8 reviewed trades, have ${reviewedSet.length}.`) +
                          `\n\nGBM (gradient-boosted trees):\n` +
                          (gbmRes
                            ? `  Trained on ${gbmRes.trainedOn} samples for ${gbmRes.rounds} rounds.\n  Stopped: ${gbmRes.reason}`
                            : `  Skipped — needs ≥20 reviewed trades, have ${reviewedSet.length}.`)
                        );
                      }}
                      style={{background:reviewed.length>=2?"#0A1A2A":"#111",
                        border:`1px solid ${reviewed.length>=2?"#2A6A9A":"#2A2A2A"}`,
                        color:reviewed.length>=2?"#5AACDF":"#444",
                        fontSize:9,padding:"4px 10px",cursor:reviewed.length>=2?"pointer":"not-allowed",
                        fontFamily:"inherit",letterSpacing:1}}>
                      🧠 LEARN ({reviewed.length} trades)
                    </button>
                    <button
                      onClick={()=>{resetWeights(universe);alert(`LR weights reset for ${universe}.`);}}
                      style={{background:"#111",border:"1px solid #2A2A2A",color:"#444",
                        fontSize:9,padding:"4px 8px",cursor:"pointer",fontFamily:"inherit",letterSpacing:1}}>
                      RESET MODEL
                    </button>
                    <span style={{width:1,height:14,background:"#222",margin:"0 2px"}}/>
                    <button
                      onClick={()=>{
                        const p = downloadExport();
                        const lr = p.lrWeights ? "LR✓" : "LR—";
                        const nn = p.nnWeights ? `NN✓ (${p.nnWeights.trainedOn||0} trades, ${p.nnWeights.epochs||0} epochs)` : "NN—";
                        // No alert needed; the browser triggers the download. Keep it quiet.
                        void lr; void nn;
                      }}
                      title="Download log + LR + NN weights as a JSON file. Move to another machine and IMPORT to sync."
                      style={{background:"#0A1A0A",border:"1px solid #2A6A2A",color:"#7FD8A6",
                        fontSize:9,padding:"4px 10px",cursor:"pointer",fontFamily:"inherit",letterSpacing:1}}>
                      ⬇ EXPORT
                    </button>
                    <button
                      onClick={()=>importInputRef.current?.click()}
                      title="Restore log + weights from a previously-exported JSON file. By default, NEW log entries are merged in (existing entries kept)."
                      style={{background:"#1A1500",border:"1px solid #C9A84C",color:"#C9A84C",
                        fontSize:9,padding:"4px 10px",cursor:"pointer",fontFamily:"inherit",letterSpacing:1}}>
                      ⬆ IMPORT
                    </button>
                    <input
                      ref={importInputRef}
                      type="file"
                      accept="application/json,.json"
                      style={{display:"none"}}
                      onChange={e=>{
                        const f = e.target.files?.[0];
                        const mode = window.confirm(
                          "Merge with existing decision log? (OK = merge, Cancel = replace)"
                        ) ? "merge" : "replace";
                        handleImportFile(f, mode);
                        e.target.value = ""; // allow re-importing the same file
                      }}
                    />
                  </div>
                </div>

                {/* Current weights display */}
                <div style={{background:"#0A0A0A",border:"1px solid #1A1A1A",padding:"8px 10px",marginBottom:12,fontSize:9,color:"#444"}}>
                  <span style={{color:"#555",letterSpacing:1}}>LR WEIGHTS: </span>
                  {FEATURE_NAMES.map((n,i)=>(
                    <span key={n} style={{marginRight:10,color:wts.weights[i]>0?"#2ECC71":"#E74C3C"}}>
                      {n}:{wts.weights[i]?.toFixed(2)}
                    </span>
                  ))}
                </div>

                {decisionLog.length===0&&<div style={{color:"#333",fontSize:11,lineHeight:1.7}}>
                  No decisions logged yet. Every AI verdict (BUY, SELL, or AVOID) is now logged automatically.
                </div>}

                {decisionLog.map(d=>{
                  const q = quotes[d.symbol];
                  const currentPrice = q?.price;
                  const isAvoid = d.verdict === "AVOID";
                  const livePnl = currentPrice && !d.reviewed && !isAvoid ? (d.verdict==="BUY"
                    ? ((currentPrice-d.entryPrice)/d.entryPrice*100)
                    : ((d.entryPrice-currentPrice)/d.entryPrice*100)
                  ).toFixed(2) : null;
                  const pnl = d.reviewed ? d.pnlPct : livePnl;
                  const verdictColor = d.verdict==="BUY"?"#2ECC71":d.verdict==="SELL"?"#E74C3C":"#666";
                  const outcomeColor = d.outcome==="WIN"?"#2ECC71":d.outcome==="LOSS"?"#E74C3C":"#C9A84C";
                  return (
                    <div key={d.id} style={{background:"#0C0C0C",border:"1px solid #1A1A1A",
                      borderLeft:`3px solid ${verdictColor}`,padding:10,marginBottom:6,
                      opacity:isAvoid?0.6:1}}>
                      <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start"}}>
                        <div>
                          <span style={{fontWeight:900,color:"#FFF",fontSize:13,letterSpacing:1}}>{d.symbol}</span>
                          <span style={{marginLeft:8,fontWeight:900,color:verdictColor,fontSize:12}}>{d.verdict}</span>
                          {d.confidence&&<span style={{marginLeft:6,fontSize:9,color:"#555",letterSpacing:1}}>{d.confidence}</span>}
                        </div>
                        <span style={{fontSize:9,color:"#333"}}>{new Date(d.timestamp).toLocaleString()}</span>
                      </div>
                      <div style={{display:"flex",gap:12,marginTop:6,fontSize:10,color:"#888",flexWrap:"wrap"}}>
                        <span>@ <b style={{color:"#FFF"}}>${d.entryPrice?.toFixed(2)}</b></span>
                        {d.stop&&<span>SL:<b style={{color:"#E74C3C"}}>${d.stop}</b></span>}
                        {d.target&&<span>TP:<b style={{color:"#2ECC71"}}>${d.target}</b></span>}
                        {d.rr&&<span>R/R:<b style={{color:"#C9A84C"}}>{d.rr}:1</b></span>}
                        {currentPrice&&!d.reviewed&&!isAvoid&&<span>Now:<b style={{color:"#FFF"}}>${currentPrice?.toFixed(2)}</b></span>}
                        {pnl!=null&&<span>P&L:<b style={{color:pnl>0?"#2ECC71":"#E74C3C"}}>{pnl>0?"+":""}{pnl}%</b></span>}
                        {d.reviewed&&<span style={{color:outcomeColor,fontWeight:900}}>{d.outcome}</span>}
                      </div>
                      <div style={{fontSize:8,color:"#2A2A2A",marginTop:4}}>
                        {d.modelScore?.direction} · {d.modelScore?.treeSignal} · {d.modelScore?.confidence}%
                      </div>
                      {!d.reviewed&&!isAvoid&&currentPrice&&(
                        <button onClick={()=>setDecisionLog(reviewDecision(d.id, currentPrice))}
                          style={{marginTop:6,background:"#1A1500",border:"1px solid #C9A84C",color:"#C9A84C",
                            fontSize:9,padding:"3px 8px",cursor:"pointer",fontFamily:"inherit",letterSpacing:1}}>
                          ✓ REVIEW (${currentPrice?.toFixed(2)})
                        </button>
                      )}
                    </div>
                  );
                })}
                {decisionLog.length>0&&(
                  <button onClick={()=>{localStorage.removeItem("trader_decision_log");setDecisionLog([]);}}
                    style={{marginTop:8,background:"#1A0808",border:"1px solid #4A1A1A",color:"#E74C3C",
                      fontSize:9,padding:"4px 10px",cursor:"pointer",fontFamily:"inherit",letterSpacing:1}}>
                    CLEAR LOG
                  </button>
                )}
              </div>
            );
          })()}

          {tab==="quant"&&(()=>{
            const q = quotes[selected];
            if (!q) return <div style={{padding:14,color:"#333"}}>No data</div>;
            const qt = q.quant||{};
            const f = engineerFeatures(q, quotes);
            const beta = calcBeta(q.closes||[], quotes.SPY?.closes||[]);
            const rows = [
              ["ADX(14)", qt.adx?.adx?.toFixed(1)||"—", qt.adx?.adx>25?"TRENDING":"WEAK", qt.adx?.adx>50?"#E74C3C":qt.adx?.adx>25?"#C9A84C":"#555"],
              ["DI+ / DI−", `${qt.adx?.diPlus?.toFixed(1)||"?"}  /  ${qt.adx?.diMinus?.toFixed(1)||"?"}`, qt.adx?.diPlus>qt.adx?.diMinus?"BULL DI":"BEAR DI", qt.adx?.diPlus>qt.adx?.diMinus?"#2ECC71":"#E74C3C"],
              ["Williams %R", qt.williamsR?.toFixed(1)||"—", qt.williamsR<-80?"OVERSOLD":qt.williamsR>-20?"OVERBOUGHT":"NEUTRAL", qt.williamsR<-80?"#2ECC71":qt.williamsR>-20?"#E74C3C":"#888"],
              ["Stoch K/D", `${qt.stochastic?.k?.toFixed(1)||"?"}  /  ${qt.stochastic?.d?.toFixed(1)||"?"}`, qt.stochastic?.k>80?"OVERBOUGHT":qt.stochastic?.k<20?"OVERSOLD":"NEUTRAL", qt.stochastic?.k>80?"#E74C3C":qt.stochastic?.k<20?"#2ECC71":"#888"],
              ["ROC(10)", qt.roc!=null?(qt.roc>=0?"+":"")+qt.roc.toFixed(2)+"%":"—", qt.roc>0?"POSITIVE":"NEGATIVE", qt.roc>0?"#2ECC71":"#E74C3C"],
              ["Z-Score(20)", qt.zScore?.toFixed(2)||"—", Math.abs(qt.zScore||0)>2?"EXTENDED":Math.abs(qt.zScore||0)>1?"ELEVATED":"NORMAL", Math.abs(qt.zScore||0)>2?"#E74C3C":Math.abs(qt.zScore||0)>1?"#C9A84C":"#888"],
              ["CMF(20)", qt.cmf?.toFixed(3)||"—", qt.cmf>0.1?"BUYING":qt.cmf<-0.1?"SELLING":"NEUTRAL", qt.cmf>0.1?"#2ECC71":qt.cmf<-0.1?"#E74C3C":"#888"],
              ["Max Drawdown", qt.maxDrawdown?.toFixed(2)+"%"||"—", "from peak", qt.maxDrawdown>20?"#E74C3C":"#888"],
              ["Sharpe (ann.)", qt.sharpe?.toFixed(2)||"—", qt.sharpe>1?"GOOD":qt.sharpe>0?"POSITIVE":qt.sharpe!=null?"NEGATIVE":"—", qt.sharpe>1?"#2ECC71":qt.sharpe>0?"#C9A84C":"#E74C3C"],
              ["Beta vs SPY", beta?.toFixed(2)||"—", beta>1.5?"HIGH BETA":beta<0.5?"LOW BETA":"MODERATE", beta>1.5?"#E74C3C":"#888"],
            ];
            return (
              <div style={{flex:1,overflowY:"auto",padding:14}}>
                <div style={{color:"#C9A84C",fontSize:11,letterSpacing:2,marginBottom:12}}>INSTITUTIONAL QUANT — {selected}</div>
                <div style={{marginBottom:16}}>
                  {rows.map(([label,val,note,color])=>(
                    <div key={label} style={{display:"flex",justifyContent:"space-between",alignItems:"center",
                      padding:"7px 10px",borderBottom:"1px solid #111",background:"#0C0C0C"}}>
                      <span style={{fontSize:10,color:"#555",letterSpacing:1,width:120}}>{label}</span>
                      <span style={{fontSize:12,color:"#FFF",fontWeight:700,width:100,textAlign:"center"}}>{val}</span>
                      <span style={{fontSize:9,color,letterSpacing:1,width:120,textAlign:"right"}}>{note}</span>
                    </div>
                  ))}
                </div>
                <div style={{color:"#C9A84C",fontSize:11,letterSpacing:2,marginBottom:8}}>FEATURE ENGINEERING</div>
                {[
                  ["EMA Alignment",`${f.emaScore}/3 — ${f.emaLabel}`],
                  ["Momentum Composite",`${f.momentumComposite}% — ${f.momentumLabel}`],
                  ["Volatility Regime",f.volRegime],
                  ["BB State",`${f.bbState||"N/A"}  bw=${f.bbBandwidth||"?"}%`],
                  ["Rel. Strength vs SPY",`${f.relStrVsSpy!=null?(f.relStrVsSpy>=0?"+":"")+f.relStrVsSpy+"%":"N/A"}`],
                  ["VWAP Deviation",`${f.vwapDev!=null?(f.vwapDev>=0?"+":"")+f.vwapDev+"%":"N/A"}`],
                  ["Volume Trend",`${f.volTrendLabel||"N/A"} (${f.volTrend||"?"}x)`],
                ].map(([l,v])=>(
                  <div key={l} style={{display:"flex",justifyContent:"space-between",padding:"6px 10px",borderBottom:"1px solid #0F0F0F"}}>
                    <span style={{fontSize:10,color:"#555"}}>{l}</span>
                    <span style={{fontSize:10,color:"#C9A84C",fontWeight:600}}>{v}</span>
                  </div>
                ))}
              </div>
            );
          })()}

          {tab==="news"&&(
            <div style={{flex:1,overflowY:"auto",padding:14}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
                <div>
                  <div style={{color:"#C9A84C",fontSize:11,letterSpacing:2}}>LIVE NEWS FEED</div>
                  <div style={{fontSize:9,color:"#444",marginTop:2}}>Al Jazeera · Reuters · MarketWatch · CNBC · BBC · Yahoo Finance — {news.length} articles</div>
                </div>
                <button onClick={()=>{setNewsLoading(true);fetchAllNews().then(a=>setNews(a)).finally(()=>setNewsLoading(false));}}
                  style={{background:"#1A1A1A",border:"1px solid #2A2A2A",color:"#888",fontSize:9,
                    padding:"3px 8px",cursor:"pointer",letterSpacing:1,fontFamily:"inherit"}}>
                  ↻ REFRESH
                </button>
              </div>
              {newsLoading&&<div style={{color:"#555",fontSize:10,letterSpacing:1,padding:"20px 0"}}>FETCHING FROM 9 SOURCES...</div>}
              {!newsLoading&&news.length===0&&(
                <div style={{color:"#333",fontSize:11,lineHeight:1.7}}>
                  No articles loaded. RSS proxies may be blocked.<br/>Click REFRESH to retry.
                </div>
              )}
              {news.map((n,i)=>(
                <div key={i} style={{padding:"8px 0",borderBottom:"1px solid #0F0F0F",display:"flex",gap:10}}>
                  <div style={{flexShrink:0,width:72,paddingTop:2}}>
                    <div style={{fontSize:8,color:"#C9A84C",letterSpacing:1,fontWeight:700,textTransform:"uppercase",
                      overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{n.source||"News"}</div>
                    <div style={{fontSize:8,color:"#333",marginTop:2}}>{n.date.toLocaleDateString()}</div>
                  </div>
                  <div style={{flex:1}}>
                    <div style={{fontSize:11,color:"#D8D0C0",lineHeight:1.5,marginBottom:2}}>{n.title}</div>
                    {n.desc&&<div style={{fontSize:9,color:"#444",lineHeight:1.4}}>{n.desc}</div>}
                  </div>
                </div>
              ))}
            </div>
          )}

          {tab==="chat"&&(
            <>
              <div ref={chatRef} style={{flex:1,overflowY:"auto",padding:14,display:"flex",flexDirection:"column",gap:12}}>
                {messages.length===0&&(
                  <div style={{textAlign:"center",padding:"40px 20px",color:"#333"}}>
                    <div style={{fontSize:28,marginBottom:10}}>📈</div>
                    <div style={{fontSize:12,color:"#555",letterSpacing:2,textTransform:"uppercase"}}>
                      Pick a symbol → ⚡ ANALYZE NOW<br/>or ask a question
                    </div>
                    <div style={{marginTop:20,display:"flex",gap:8,flexWrap:"wrap",justifyContent:"center"}}>
                      {["What's the strongest setup right now?",`Is ${selected} worth trading today?`,
                        "Which symbol has the best R/R?","Any short setups?","Where would Livermore enter SPY?"]
                        .map((c,i)=>(
                          <button key={i} onClick={()=>setInput(c)} style={{fontSize:10,padding:"5px 10px",
                            background:"#111",border:"1px solid #222",color:"#555",cursor:"pointer",fontFamily:"inherit"}}>
                            {c}
                          </button>
                        ))}
                    </div>
                  </div>
                )}
                {messages.map((m,i)=>{
                  const td = m.tradeData;
                  const isLogged = m.id && loggedMsgIds.has(m.id);
                  const vColor = td?.verdict==="BUY"?"#2ECC71":td?.verdict==="SELL"?"#E74C3C":"#888";
                  const isBuffett = m.persona === "buffett";
                  const botAvatar = isBuffett ? "🏛" : "📈";
                  const botBorderColor = isBuffett ? "#2A7A4F" : "#C9A84C";
                  const botBgColor = isBuffett ? "#0D1A14" : "#1A1500";
                  return (
                    <div key={i} style={{display:"flex",gap:10,alignSelf:m.type==="user"?"flex-end":"flex-start",
                      maxWidth:"95%",flexDirection:m.type==="user"?"row-reverse":"row"}}>
                      <div style={{width:28,height:28,flexShrink:0,alignSelf:"flex-start",
                        background:m.type==="user"?"#0d1117":botBgColor,
                        border:`1px solid ${m.type==="user"?"#222":botBorderColor}`,
                        display:"flex",alignItems:"center",justifyContent:"center",fontSize:13}}>
                        {m.type==="user"?"👤":botAvatar}
                      </div>
                      <div style={{display:"flex",flexDirection:"column",gap:6,flex:1,minWidth:0}}>
                        {isBuffett && (
                          <div style={{fontSize:9,color:"#7FD8A6",letterSpacing:2,fontWeight:700}}>
                            🏛 BUFFETT — SEPARATE LONG-HORIZON VIEW (not part of the trader verdict)
                          </div>
                        )}
                        <div style={{background:m.type==="user"?"#0d1117":isBuffett?"#0A140E":"#111",border:"1px solid #1E1E1E",
                          borderLeft:m.type==="bot"?`3px solid ${botBorderColor}`:"1px solid #1E1E1E",
                          borderRight:m.type==="user"?"3px solid #333":"1px solid #1E1E1E",
                          padding:"10px 14px",fontSize:12,lineHeight:1.8,
                          color:m.type==="user"?"#888":"#D8D0C0",whiteSpace:"pre-wrap"}}>
                          {m.text}
                        </div>
                        {td && !isLogged && (
                          <div style={{display:"flex",alignItems:"center",gap:10,padding:"6px 10px",
                            background:"#0A0A0A",border:`1px solid ${vColor}33`,borderLeft:`3px solid ${vColor}`}}>
                            <span style={{fontSize:10,color:vColor,fontWeight:900,letterSpacing:1}}>{td.symbol} {td.verdict}</span>
                            <span style={{fontSize:9,color:"#555"}}>@ ${td.entryPrice?.toFixed(2)}</span>
                            {td.stop&&<span style={{fontSize:9,color:"#666"}}>SL ${td.stop}</span>}
                            {td.target&&<span style={{fontSize:9,color:"#666"}}>TP ${td.target}</span>}
                            {td.rr&&<span style={{fontSize:9,color:"#C9A84C"}}>{td.rr}:1</span>}
                            <button onClick={()=>{
                              logDecision(td);
                              setDecisionLog(getLog());
                              setLoggedMsgIds(prev=>new Set([...prev, m.id]));
                              setTab("log");
                            }} style={{marginLeft:"auto",background:"#0A1A0A",border:`1px solid ${vColor}`,
                              color:vColor,fontSize:9,padding:"3px 10px",cursor:"pointer",
                              fontFamily:"inherit",letterSpacing:1,fontWeight:700}}>
                              📋 LOG THIS TRADE
                            </button>
                          </div>
                        )}
                        {td && isLogged && (
                          <div style={{fontSize:9,color:"#2ECC71",padding:"3px 10px",letterSpacing:1}}>
                            ✓ Logged to decision log
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
                {thinking&&(
                  <div style={{display:"flex",gap:10}}>
                    <div style={{width:28,height:28,background:"#1A1500",border:"1px solid #C9A84C",
                      display:"flex",alignItems:"center",justifyContent:"center",fontSize:13}}>📈</div>
                    <div style={{background:"#111",border:"1px solid #1E1E1E",borderLeft:"3px solid #C9A84C",
                      padding:"12px 14px",display:"flex",gap:6,alignItems:"center"}}>
                      {[0,1,2].map(d=>(
                        <div key={d} style={{width:7,height:7,borderRadius:"50%",background:"#C9A84C",
                          animation:`dots 1.2s ease-in-out ${d*0.2}s infinite`}}/>
                      ))}
                      <span style={{fontSize:10,color:"#555",marginLeft:6,letterSpacing:1}}>READING THE TAPE...</span>
                    </div>
                  </div>
                )}
              </div>
              <div style={{background:"#0C0C0C",borderTop:"1px solid #1A1A1A",
                padding:"10px 14px",display:"flex",gap:8,flexShrink:0}}>
                <input value={input} onChange={e=>setInput(e.target.value)}
                  onKeyDown={e=>e.key==="Enter"&&handleSend()}
                  placeholder={`Ask about ${selected}...`} disabled={thinking}
                  style={{flex:1,background:"#080808",border:"1px solid #1E1E1E",color:"#D8D0C0",
                    fontFamily:"'Courier New',monospace",fontSize:12,padding:"9px 12px",outline:"none"}}/>
                <button onClick={handleSend} disabled={thinking||!input.trim()} style={{
                  background:thinking||!input.trim()?"#1A1A1A":"#C9A84C",
                  color:thinking||!input.trim()?"#444":"#000",border:"none",
                  fontFamily:"'Courier New',monospace",fontWeight:900,fontSize:11,letterSpacing:2,
                  padding:"0 16px",cursor:thinking||!input.trim()?"not-allowed":"pointer",textTransform:"uppercase"}}>
                  SEND
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      <style>{`
        @keyframes ticker{from{transform:translateX(0)}to{transform:translateX(-50%)}}
        @keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}
        @keyframes dots{0%,100%{opacity:.2;transform:scale(.8)}50%{opacity:1;transform:scale(1.2)}}

        /* Visible, click-and-drag friendly scrollbar. Replaces the near-invisible
           3px default. Firefox gets scrollbar-color; webkit browsers get the
           explicit track/thumb rules below. */
        * { scrollbar-width: auto; scrollbar-color: #4A4A4A #0F0F0F; }
        ::-webkit-scrollbar { width: 12px; height: 12px; }
        ::-webkit-scrollbar-track { background: #0F0F0F; border-left: 1px solid #1A1A1A; }
        ::-webkit-scrollbar-thumb { background: #3A3A3A; border: 2px solid #0F0F0F; border-radius: 6px; }
        ::-webkit-scrollbar-thumb:hover { background: #C9A84C; }
        ::-webkit-scrollbar-corner { background: #0F0F0F; }
      `}</style>
    </div>
  );
}
