// Pure indicator computations used by both the live quote pipeline (App.jsx)
// and the backtest engine (backtest.js). Extracted from the old mockData.js
// because it's the only piece worth keeping — everything else in mockData.js
// generated synthetic random-walker prices, which has been removed from the
// live path entirely (Coinbase real-money integration is coming; fabricated
// prices on the live path are unacceptable).

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
