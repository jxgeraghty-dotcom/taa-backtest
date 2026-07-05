# Tactical Overlay Research Note (template)

Two pages, no more. This is the deliverable that actually gets read. Fill every
section. The limitations section is not boilerplate; it is where you show you know
what you built.

## 1. Question and thesis (3 to 4 sentences)
What tactical question are you asking, and what is the economic reason to expect an
edge? State the thesis before any result.

## 2. Setup
- Universe and the ETF/index proxies used, with start dates and the sample window.
- Strategic policy portfolio (weights) and why it is the right benchmark.
- Signal(s), defined precisely, committed to before testing. List any you tried and
  discarded. Selective reporting is the tell of an amateur backtest.
- Construction: tilt cap, funding sleeve, rebalance frequency.
- Costs: assumption and the sensitivity grid.

## 3. Results (lead with the overlay, not the beta)
- Information ratio and tracking error versus the policy, net of costs. This is the
  headline. Total return versus cash or equities is mostly policy beta; do not lead
  with it.
- Summary table (CAGR, vol, Sharpe, Sortino, max drawdown, Calmar, turnover) for
  context.
- Cost sensitivity: how much of the edge survives at 5, 10, 15 bps.
- Regime table: where the edge lives and where it disappears.

## 4. Robustness (the real content)
- Parameter sensitivity: plateau or knife-edge?
- Bootstrap CI on the IR. State plainly whether zero is inside it.
- Deflated Sharpe with the honest number of trials you ran.
- Random-tilt null: does a turnover-matched random overlay match your IR?

## 5. Honest limitations
- Sample length and regime concentration (roughly 220 monthly points from ~2007).
- Proxy and survivorship issues, macro release timing, no true vintages.
- What would change your mind, and what data would let you actually test the claim.

## 6. Current positioning read (1 paragraph)
Given the framework and its limits, what tilt would the signal suggest today, and
how much confidence does the evidence actually support? Be modest. "Here is the
small, fragile edge that survives" beats "I beat the market" every time.
