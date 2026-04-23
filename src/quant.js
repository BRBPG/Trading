// ─── ADX (Average Directional Index) ────────────────────────────────────────
export function calcADX(highs, lows, closes, period = 14) {
  if (highs.length < period * 2 + 1) return null;
  const tr = [], plusDM = [], minusDM = [];
  for (let i = 1; i < highs.length; i++) {
    const upMove = highs[i] - highs[i-1];
    const downMove = lows[i-1] - lows[i];
    plusDM.push(upMove > downMove && upMove > 0 ? upMove : 0);
    minusDM.push(downMove > upMove && downMove > 0 ? downMove : 0);
    tr.push(Math.max(highs[i]-lows[i], Math.abs(highs[i]-closes[i-1]), Math.abs(lows[i]-closes[i-1])));
  }
  const wilder = (arr, p) => {
    let s = arr.slice(0, p).reduce((a,b)=>a+b, 0);
    const out = [s];
    for (let i = p; i < arr.length; i++) { s = s - s/p + arr[i]; out.push(s); }
    return out;
  };
  const sTR = wilder(tr, period);
  const sPDM = wilder(plusDM, period);
  const sMDM = wilder(minusDM, period);
  const diP = sPDM.map((v,i) => sTR[i]>0 ? (v/sTR[i])*100 : 0);
  const diM = sMDM.map((v,i) => sTR[i]>0 ? (v/sTR[i])*100 : 0);
  const dx  = diP.map((v,i) => { const s=v+diM[i]; return s>0?(Math.abs(v-diM[i])/s)*100:0; });
  const adx = wilder(dx, period);
  return { adx: adx[adx.length-1], diPlus: diP[diP.length-1], diMinus: diM[diM.length-1] };
}

// ─── Williams %R ─────────────────────────────────────────────────────────────
export function calcWilliamsR(highs, lows, closes, period = 14) {
  if (closes.length < period) return null;
  const hh = Math.max(...highs.slice(-period));
  const ll = Math.min(...lows.slice(-period));
  return hh === ll ? -50 : ((hh - closes[closes.length-1]) / (hh - ll)) * -100;
}

// ─── Stochastic Oscillator ───────────────────────────────────────────────────
export function calcStochastic(highs, lows, closes, k = 14, d = 3) {
  if (closes.length < k + d) return null;
  const kVals = [];
  for (let i = k-1; i < closes.length; i++) {
    const hh = Math.max(...highs.slice(i-k+1, i+1));
    const ll = Math.min(...lows.slice(i-k+1, i+1));
    kVals.push(hh===ll ? 50 : ((closes[i]-ll)/(hh-ll))*100);
  }
  return { k: kVals[kVals.length-1], d: kVals.slice(-d).reduce((a,b)=>a+b,0)/d };
}

// ─── Rate of Change ──────────────────────────────────────────────────────────
export function calcROC(closes, period = 10) {
  if (closes.length < period+1) return null;
  const prev = closes[closes.length-1-period];
  return ((closes[closes.length-1]-prev)/prev)*100;
}

// ─── Z-Score (standard deviations from rolling mean) ─────────────────────────
export function calcZScore(closes, period = 20) {
  if (closes.length < period) return null;
  const sl = closes.slice(-period);
  const mean = sl.reduce((a,b)=>a+b,0)/period;
  const sd = Math.sqrt(sl.reduce((a,b)=>a+(b-mean)**2,0)/period);
  return sd > 0 ? (closes[closes.length-1]-mean)/sd : 0;
}

// ─── Chaikin Money Flow ───────────────────────────────────────────────────────
export function calcCMF(highs, lows, closes, volumes, period = 20) {
  if (closes.length < period) return null;
  let mfv = 0, vol = 0;
  for (let i = closes.length-period; i < closes.length; i++) {
    const hl = highs[i]-lows[i];
    const mf = hl > 0 ? ((closes[i]-lows[i]-(highs[i]-closes[i]))/hl) : 0;
    mfv += mf*(volumes[i]||0);
    vol += volumes[i]||0;
  }
  return vol > 0 ? mfv/vol : 0;
}

// ─── Max Drawdown ─────────────────────────────────────────────────────────────
export function calcMaxDrawdown(closes) {
  if (closes.length < 2) return null;
  let peak = closes[0], maxDD = 0;
  for (const c of closes) {
    if (c > peak) peak = c;
    const dd = (peak-c)/peak;
    if (dd > maxDD) maxDD = dd;
  }
  return maxDD*100;
}

// ─── Sharpe Ratio (annualised, bar-frequency-aware) ─────────────────────────
// Default barsPerYear=252 assumes DAILY bars. For intraday the caller must
// pass the correct value — a US equity session is ~78 five-min bars, so
// 252 × 78 = 19,656. Previously this function was unconditionally applying
// the daily factor to 5-min return series, inflating reported Sharpe by a
// factor of √78 ≈ 8.8×.
export function calcSharpe(closes, rfRate = 0.053, barsPerYear = 252) {
  if (closes.length < 10) return null;
  const ret = [];
  for (let i = 1; i < closes.length; i++) ret.push((closes[i]-closes[i-1])/closes[i-1]);
  const mean = ret.reduce((a,b)=>a+b,0)/ret.length;
  const sd = Math.sqrt(ret.reduce((a,b)=>a+(b-mean)**2,0)/ret.length);
  const annMean = mean * barsPerYear;
  const annSD   = sd   * Math.sqrt(barsPerYear);
  return annSD > 0 ? (annMean-rfRate)/annSD : 0;
}

// ─── Beta vs SPY ─────────────────────────────────────────────────────────────
export function calcBeta(closes, spyCloses) {
  const n = Math.min(closes.length, spyCloses?.length||0, 30);
  if (n < 5) return null;
  const ret = (arr, i) => (arr[arr.length-1-i]-arr[arr.length-2-i])/arr[arr.length-2-i];
  const xy = [], xx = [];
  for (let i = 0; i < n-1; i++) {
    const x = ret(spyCloses, i), y = ret(closes, i);
    if (isFinite(x) && isFinite(y)) { xy.push(x*y); xx.push(x*x); }
  }
  const cov = xy.reduce((a,b)=>a+b,0)/xy.length;
  const varX = xx.reduce((a,b)=>a+b,0)/xx.length;
  return varX > 0 ? cov/varX : null;
}

// ─── Feature Engineering ─────────────────────────────────────────────────────
export function engineerFeatures(q, allQuotes) {
  const f = {};

  // EMA alignment score (0-3): price>EMA9, EMA9>EMA20, EMA20>EMA50
  f.emaScore = [
    q.price&&q.ema9 && q.price>q.ema9,
    q.ema9&&q.ema20 && q.ema9>q.ema20,
    q.ema20&&q.ema50 && q.ema20>q.ema50,
  ].filter(Boolean).length;
  f.emaLabel = f.emaScore===3?"FULLY BULLISH":f.emaScore===2?"LEANING BULL":f.emaScore===1?"LEANING BEAR":"FULLY BEARISH";

  // Momentum composite: RSI + MACD + 5-bar momentum → -1 to +1
  const r = q.rsi!=null?(q.rsi-50)/50:0;
  const m = q.macd!=null?Math.sign(q.macd):0;
  const p = q.momentum5!=null?Math.max(-1,Math.min(1,q.momentum5/3)):0;
  f.momentumComposite = ((r+m+p)/3*100).toFixed(0);
  f.momentumLabel = f.momentumComposite>25?"STRONG BULL":f.momentumComposite>5?"WEAK BULL":f.momentumComposite<-25?"STRONG BEAR":f.momentumComposite<-5?"WEAK BEAR":"NEUTRAL";

  // Volatility regime
  const atrPct = q.atr&&q.price ? (q.atr/q.price)*100 : 0;
  f.atrPct = atrPct.toFixed(2);
  f.volRegime = atrPct<0.5?"LOW — potential compression":atrPct<1.5?"NORMAL":"HIGH — elevated risk";

  // BB squeeze / expansion
  if (q.bb) {
    const bw = ((q.bb.upper-q.bb.lower)/q.bb.mean)*100;
    f.bbBandwidth = bw.toFixed(2);
    f.bbState = bw<3?"SQUEEZE (breakout imminent)":bw>8?"EXPANDED (post-move)":"NORMAL";
  }

  // Relative strength vs SPY
  const spy = allQuotes?.SPY;
  if (spy&&q.changePct!=null&&spy.changePct!=null)
    f.relStrVsSpy = (q.changePct-spy.changePct).toFixed(2);

  // VWAP deviation
  if (q.price&&q.vwap)
    f.vwapDev = ((q.price-q.vwap)/q.vwap*100).toFixed(2);

  // Volume trend: is volume expanding or contracting?
  if (q.volumes&&q.volumes.length>10) {
    const recent5 = q.volumes.slice(-5).reduce((a,b)=>a+b,0)/5;
    const prev5   = q.volumes.slice(-10,-5).reduce((a,b)=>a+b,0)/5;
    f.volTrend = prev5>0 ? (recent5/prev5).toFixed(2) : null;
    f.volTrendLabel = f.volTrend>1.2?"EXPANDING (conviction)":f.volTrend<0.8?"CONTRACTING (fading)":"STABLE";
  }

  return f;
}

// ─── Multi-source news aggregation ──────────────────────────────────────────
const PROXIES = [
  u => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
  u => `https://corsproxy.io/?${encodeURIComponent(u)}`,
];

const NEWS_FEEDS = [
  { source: "Al Jazeera",  url: "https://www.aljazeera.com/xml/rss/all.xml" },
  { source: "Reuters",     url: "https://feeds.reuters.com/reuters/businessNews" },
  { source: "Reuters Mkt", url: "https://feeds.reuters.com/news/wealth" },
  { source: "MarketWatch", url: "https://feeds.content.dowjones.io/public/rss/mw_topstories" },
  { source: "MarketWatch Real-time", url: "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines" },
  { source: "CNBC Top",    url: "https://www.cnbc.com/id/100003114/device/rss/rss.html" },
  { source: "CNBC Mkt",    url: "https://www.cnbc.com/id/15839135/device/rss/rss.html" },
  { source: "BBC Business",url: "https://feeds.bbci.co.uk/news/business/rss.xml" },
  { source: "Yahoo Fin",   url: "https://finance.yahoo.com/news/rssindex" },
];

async function fetchFeed(source, url, cutoff) {
  for (const proxy of PROXIES) {
    try {
      const res = await fetch(proxy(url), {signal: AbortSignal.timeout(10000)});
      if (!res.ok) continue;
      const text = await res.text();
      const doc = new DOMParser().parseFromString(text, "text/xml");
      const items = [...doc.querySelectorAll("item")];
      return items.map(el => ({
        source,
        title: el.querySelector("title")?.textContent?.trim()||"",
        date: new Date(el.querySelector("pubDate")?.textContent||0),
        link: el.querySelector("link")?.textContent?.trim()||"",
        desc: (el.querySelector("description")?.textContent||"").replace(/<[^>]*>/g,"").slice(0,220),
      })).filter(n => n.date.getTime() > cutoff && n.title);
    } catch { continue; }
  }
  return [];
}

// Fetch all news sources in parallel and merge, sorted by date desc
export async function fetchAllNews() {
  const cutoff = Date.now() - 100*24*60*60*1000;
  const results = await Promise.all(
    NEWS_FEEDS.map(f => fetchFeed(f.source, f.url, cutoff).catch(()=>[]))
  );
  const merged = results.flat()
    .sort((a,b) => b.date.getTime() - a.date.getTime())
    .slice(0, 500);
  return merged;
}

// Kept for backwards compatibility
export async function fetchAlJazeeraNews() {
  return fetchAllNews();
}
