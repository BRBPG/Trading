// Per-symbol daily-vol estimates used by the synthetic candle fallback when
// live fetch fails. Only consulted by generateMockQuote — has no effect on
// real Yahoo/Finnhub/Polygon data. Missing entries fall through to 0.015
// (see generateCandles). Values are rough intraday ranges; exact numbers
// don't matter because this path is only a fallback, not primary data.
const VOL_PROFILES = {
  // Broad indices
  SPY:  { vol: 0.008 }, QQQ:  { vol: 0.011 },
  // Mega-cap tech
  AAPL: { vol: 0.013 }, MSFT: { vol: 0.012 }, AMZN: { vol: 0.016 },
  // Semiconductors
  NVDA: { vol: 0.028 }, AMD:  { vol: 0.025 }, TSM:  { vol: 0.018 },
  // EV / autos
  TSLA: { vol: 0.035 },
  // Quantum — extremely high vol, small-cap behaviour
  IONQ: { vol: 0.055 }, RGTI: { vol: 0.060 },
  // Airline
  UAL:  { vol: 0.022 },
  // Commodities
  USO:  { vol: 0.020 }, BNO:  { vol: 0.022 }, GLD:  { vol: 0.009 },
  // LSE
  "TW.L": { vol: 0.018 },
  // Crypto — ~3-5× equity vol. Daily ranges frequently 3-6% on majors,
  // 5-10% on small-caps. Per-bar vol estimates here are for the random-walk
  // fallback only (real data comes from Yahoo/Polygon). Ordered roughly by
  // market cap / historical volatility.
  "BTC-USD":   { vol: 0.025 },
  "ETH-USD":   { vol: 0.032 },
  "SOL-USD":   { vol: 0.045 },
  "BNB-USD":   { vol: 0.028 },
  "XRP-USD":   { vol: 0.038 },
  "ADA-USD":   { vol: 0.042 },
  "AVAX-USD":  { vol: 0.050 },
  "LINK-USD":  { vol: 0.040 },
  "DOGE-USD":  { vol: 0.055 },
  "MATIC-USD": { vol: 0.045 },
};

// Synthetic-fallback starting prices. Rough ballpark — doesn't need to match
// the live price exactly, just needs to be in the right order of magnitude
// so the random walk doesn't produce nonsense during a full Finnhub + Yahoo
// outage.
const FALLBACK_PRICES = {
  SPY: 530, QQQ: 455,
  AAPL: 213, MSFT: 415, AMZN: 195,
  NVDA: 875, AMD: 158, TSM: 185,
  TSLA: 248,
  IONQ: 32, RGTI: 12,
  UAL: 68,
  USO: 78, BNO: 20, GLD: 224,
  "TW.L": 112,
  // Crypto (Apr 2026 ballpark). Off-market fallback only, real data via Yahoo.
  "BTC-USD":   95000,
  "ETH-USD":   3300,
  "SOL-USD":   175,
  "BNB-USD":   620,
  "XRP-USD":   2.20,
  "ADA-USD":   0.85,
  "AVAX-USD":  40,
  "LINK-USD":  18,
  "DOGE-USD":  0.38,
  "MATIC-USD": 0.85,
};

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const SESSION_SEED = Date.now() & 0xFFFFFF;

// Generate candle history anchored to a real current price
function generateCandles(symbol, anchorPrice) {
  const vol = VOL_PROFILES[symbol]?.vol ?? 0.015;
  const rand = mulberry32(SESSION_SEED ^ symbol.charCodeAt(0) * 7919);
  const candles = 390;

  // Work backwards from the anchor price
  const closes = new Array(candles);
  closes[candles - 1] = anchorPrice;
  const drift = (rand() - 0.48) * 0.0002;

  for (let i = candles - 2; i >= 0; i--) {
    const next = closes[i + 1];
    const change = next * (drift + (rand() - 0.5) * vol * 0.3);
    closes[i] = Math.max(next * 0.85, next - change);
  }

  const highs = closes.map(c => {
    const range = c * vol * (0.3 + rand() * 0.7);
    return c + range * rand();
  });
  const lows = closes.map(c => {
    const range = c * vol * (0.3 + rand() * 0.7);
    return c - range * rand();
  });
  const volumes = closes.map(() => Math.floor(100000 + rand() * 600000));

  return { closes, highs, lows, volumes };
}

export function computeIndicators(closes, highs, lows, volumes) {
  const k = (period) => {
    if (closes.length < period) return null;
    const kf = 2 / (period + 1);
    let ema = closes.slice(0, period).reduce((a, b) => a + b, 0) / period;
    for (let i = period; i < closes.length; i++) ema = closes[i] * kf + ema * (1 - kf);
    return ema;
  };

  const ema9 = k(9), ema20 = k(20), ema50 = k(50);
  const ema12 = k(12), ema26 = k(26);
  const macd = ema12 != null && ema26 != null ? ema12 - ema26 : null;

  let gains = 0, losses = 0;
  for (let i = closes.length - 14; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1];
    if (diff > 0) gains += diff; else losses -= diff;
  }
  const rsi = 100 - 100 / (1 + gains / (losses || 0.001));

  const trs = [];
  for (let i = 1; i < highs.length; i++)
    trs.push(Math.max(highs[i]-lows[i], Math.abs(highs[i]-closes[i-1]), Math.abs(lows[i]-closes[i-1])));
  const atr = trs.slice(-14).reduce((a, b) => a + b, 0) / 14;

  const slice78c = closes.slice(-78), slice78v = volumes.slice(-78);
  let pvSum = 0, vSum = 0;
  for (let i = 0; i < slice78c.length; i++) { pvSum += slice78c[i]*slice78v[i]; vSum += slice78v[i]; }
  const vwap = vSum > 0 ? pvSum / vSum : null;

  const bbSlice = closes.slice(-20);
  const bbMean = bbSlice.reduce((a,b)=>a+b,0)/20;
  const bbSd = Math.sqrt(bbSlice.reduce((a,b)=>a+(b-bbMean)**2,0)/20);
  const bbUpper = bbMean+2*bbSd, bbLower = bbMean-2*bbSd;
  const price = closes[closes.length-1];
  const bb = { pos:(price-bbLower)/(bbUpper-bbLower), upper:bbUpper, lower:bbLower, mean:bbMean };

  const avgVol = volumes.slice(-20).reduce((a,b)=>a+b,0)/20;
  const curVol = volumes[volumes.length-1];
  const volRatio = avgVol ? curVol/avgVol : null;

  const recent5 = closes.slice(-5);
  const momentum5 = recent5.length===5 ? ((recent5[4]-recent5[0])/recent5[0])*100 : null;

  return { rsi, macd, ema9, ema20, ema50, atr, vwap, volRatio, bb, momentum5 };
}

export function generateMockQuote(symbol) {
  const anchorPrice = FALLBACK_PRICES[symbol] ?? 100;
  const { closes, highs, lows, volumes } = generateCandles(symbol, anchorPrice);
  const price = closes[closes.length-1];
  const prevClose = closes[0];
  const change = price - prevClose;
  const changePct = (change/prevClose)*100;
  const indicators = computeIndicators(closes, highs, lows, volumes);
  const rand = mulberry32(SESSION_SEED ^ symbol.charCodeAt(0) * 1234);
  const high52 = Math.max(...closes)*(1+rand()*0.08);
  const low52  = Math.min(...closes)*(1-rand()*0.08);

  return {
    symbol, price, change, changePct, prevClose,
    high52, low52,
    dayHigh: Math.max(...highs.slice(-78)),
    dayLow: Math.min(...lows.slice(-78)),
    volume: volumes[volumes.length-1],
    ...indicators,
    sparkline: closes.slice(-30), closes, highs, lows, volumes,
    marketState: "SIMULATED", lastFetched: Date.now(), isMock: true,
  };
}

// Called when we have real price/prevClose from Finnhub
export function generateLiveIndicators(symbol, realPrice, realPrevClose) {
  const { closes, highs, lows, volumes } = generateCandles(symbol, realPrice);
  // Anchor the last candle to the real current price. Do NOT also anchor
  // closes[0] to realPrevClose — the random walker already starts somewhere
  // close to realPrice and overriding closes[0] creates an artificial
  // discontinuity vs closes[1] that the winsoriser flags as an outlier on
  // every refresh. This was particularly visible on low-vol assets like GLD
  // and XAUUSD (vol=0.009) where the anchor jump dwarfs the natural ~0.13%
  // bar moves. realPrevClose is still used for the displayed change/changePct
  // (computed from quote.pc, not from closes[0]).
  closes[closes.length-1] = realPrice;
  // realPrevClose is intentionally not used to mutate closes[0]
  void realPrevClose;
  const indicators = computeIndicators(closes, highs, lows, volumes);
  return { closes, highs, lows, volumes, ...indicators,
    sparkline: closes.slice(-30),
    dayHigh: Math.max(...highs.slice(-78)),
    dayLow: Math.min(...lows.slice(-78)),
  };
}
