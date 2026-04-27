// ─── Simulation metrics ─────────────────────────────────────────────────────
// Win-rate alone is a weak signal — a 43% win rate with 3:1 R/R is profitable;
// a 70% win rate with 1:5 R/R loses money. This module turns a list of
// labelled sim trades into the metrics a real trading desk would actually
// look at.
//
// Input trade shape (from backtest.js runBacktest):
//   { symbol, timestamp, verdict, entryPrice, exitPrice, pnlPct, outcome,
//     hitStop, hitTarget, ageDays, features }
//
// Notes:
//   • All P&L figures are in PERCENT, not currency. Treat each trade as risking
//     the same fixed unit — the equity curve is then unitless and comparable.
//   • Sharpe is computed across the trade-return series, not annualised.
//     Annualising is misleading at these sample sizes (≤200 trades).

export function computeSimMetrics(trades) {
  if (!trades || trades.length === 0) return null;

  const wins = trades.filter(t => t.outcome === "WIN");
  const losses = trades.filter(t => t.outcome === "LOSS");

  const winRate = wins.length / trades.length;
  const lossRate = losses.length / trades.length;

  // Average win / average loss in % (loss expressed as POSITIVE magnitude)
  const avgWin = wins.length
    ? wins.reduce((s, t) => s + t.pnlPct, 0) / wins.length
    : 0;
  const avgLoss = losses.length
    ? Math.abs(losses.reduce((s, t) => s + t.pnlPct, 0) / losses.length)
    : 0;

  // Profit factor = gross wins / gross losses. >1 profitable, <1 losing.
  // The single most-quoted backtest metric on real prop desks.
  const grossWin = wins.reduce((s, t) => s + t.pnlPct, 0);
  const grossLoss = Math.abs(losses.reduce((s, t) => s + t.pnlPct, 0));
  const profitFactor = grossLoss > 0 ? grossWin / grossLoss : (grossWin > 0 ? Infinity : 0);

  // Expectancy per trade in % — what you "expect" to make on the next trade.
  // = (winRate × avgWin) − (lossRate × avgLoss). Drives long-run equity growth.
  const expectancy = (winRate * avgWin) - (lossRate * avgLoss);

  // Total P&L = sum of trade returns (each trade = 1 unit risk)
  const totalPnl = trades.reduce((s, t) => s + t.pnlPct, 0);

  // Equity curve (cumulative %), and its peak/drawdown
  const sortedByTime = [...trades].sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
  const equity = [];
  let cum = 0, peak = 0, maxDD = 0;
  for (const t of sortedByTime) {
    cum += t.pnlPct;
    equity.push(cum);
    if (cum > peak) peak = cum;
    const dd = peak - cum;
    if (dd > maxDD) maxDD = dd;
  }

  // Max consecutive losses — useful for stop-out psychology + sizing
  let maxConsLoss = 0, curConsLoss = 0;
  for (const t of sortedByTime) {
    if (t.outcome === "LOSS") { curConsLoss++; if (curConsLoss > maxConsLoss) maxConsLoss = curConsLoss; }
    else curConsLoss = 0;
  }

  // Sharpe ratio over the trade-return series (NOT annualised — sample too small)
  const meanRet = totalPnl / trades.length;
  const variance = trades.reduce((s, t) => s + (t.pnlPct - meanRet) ** 2, 0) / trades.length;
  const stdRet = Math.sqrt(variance);
  const sharpe = stdRet > 0 ? meanRet / stdRet : 0;

  // Reward-to-risk realised = avgWin / avgLoss. If <2, you need a high win rate.
  const realisedRR = avgLoss > 0 ? avgWin / avgLoss : 0;

  // Exit-reason breakdown: were we mostly hitting stops, targets, or timing out?
  const stopHits = trades.filter(t => t.hitStop).length;
  const targetHits = trades.filter(t => t.hitTarget).length;
  const timedOut = trades.length - stopHits - targetHits;

  // Per-symbol breakdown
  const bySymbol = {};
  for (const t of trades) {
    bySymbol[t.symbol] ||= { n: 0, wins: 0, pnl: 0 };
    bySymbol[t.symbol].n++;
    if (t.outcome === "WIN") bySymbol[t.symbol].wins++;
    bySymbol[t.symbol].pnl += t.pnlPct;
  }
  const symbolRows = Object.entries(bySymbol)
    .map(([sym, s]) => ({
      symbol: sym, n: s.n, wins: s.wins,
      winRate: s.wins / s.n,
      pnl: s.pnl,
      avgPnl: s.pnl / s.n,
    }))
    .sort((a, b) => b.pnl - a.pnl);

  return {
    n: trades.length,
    wins: wins.length,
    losses: losses.length,
    winRate,
    avgWin, avgLoss,
    realisedRR,
    profitFactor,
    expectancy,
    totalPnl,
    sharpe,
    maxDD,
    maxConsLoss,
    equity,
    exitReasons: { stopHits, targetHits, timedOut },
    symbolRows,
  };
}

// One-line verdict for the top of the metrics card.
//   "Profitable, edge confirmed"     — profitFactor >= 1.5 AND expectancy > 0
//   "Marginally profitable"          — profitFactor 1.0-1.5
//   "Break-even / coin flip"         — profitFactor 0.9-1.1
//   "Loses money — invert the model" — profitFactor < 0.9
export function summariseEdge(m) {
  if (!m) return { label: "—", color: "#888" };
  if (m.profitFactor >= 1.5 && m.expectancy > 0)
    return { label: "Profitable — edge confirmed", color: "#2ECC71" };
  if (m.profitFactor >= 1.0)
    return { label: "Marginally profitable", color: "#7FD8A6" };
  if (m.profitFactor >= 0.9)
    return { label: "Break-even / coin flip", color: "#C9A84C" };
  return { label: "Loses money — consider inverting", color: "#E74C3C" };
}
