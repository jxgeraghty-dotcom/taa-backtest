# Tactical Overlay on a Multi-Asset Policy Portfolio — Research Note

*Data pulled early July 2026 (ETF total returns + FRED yields + yfinance dividends).
Reproduce with `python scripts/run_backtest.py --real`. All figures are net of costs
and after turnover limits unless stated. The offline `run_backtest.py` uses synthetic
data and proves the machinery only.*

## 1. Question and thesis
Can a disciplined tactical overlay, driven by pre-committed, economically motivated
signals, add information *over* a fixed strategic policy portfolio net of realistic
costs? The prior is modest: cross-asset momentum, value, carry, and a macro nowcast
are all documented, but on a short single-cycle ETF sample they should produce a
**small, regime-dependent** edge that is hard to distinguish from zero. That prior
turns out to be the honest result — and, after fixing a data artifact, an even more
sober one than a first pass suggested.

## 2. Setup
- **Universe / proxies (monthly total return):** EQ_US=VTI, EQ_INTL=VEA, GOVT=IEF,
  CREDIT=HYG, TIPS=TIP, CMDTY=DBC, CASH=BIL. Sample **2007-01 to 2026-07, 235 months**;
  the overlay trades after a 12-month warmup (~2008–2026).
- **Strategic policy (the benchmark):** EQ_US 35, EQ_INTL 15, GOVT 20, CREDIT 10,
  TIPS 5, CMDTY 5, CASH 10 (%). The overlay is judged on information ratio *over this
  policy*, run through the identical loop and cost model — not on total return.
- **Signals (committed before testing, four documented families):**
  *momentum* = 12m trailing return (Moskowitz-Ooi-Pedersen 2012); *value* = negative
  5-year return, 12m skip (long-horizon reversal, Asness-Moskowitz-Pedersen 2013);
  *carry* = yield/carry level per sleeve — FRED yields for GOVT/TIPS/CASH, Moody's Baa
  corporate yield for CREDIT, **trailing dividend yield for equities**, and a
  **roll-yield proxy for commodities** (DBC total return over the S&P GSCI benchmark
  net of cash collateral); *macro nowcast* = trailing change in the yield-curve slope
  and Baa credit spread mapped to a **fixed, pre-committed** risk-on/off asset-response
  vector. *composite* = equal-weight z-blend of the four (no weight tuning).
- **Construction:** bounded tilt around policy, ±10% per-sleeve cap, CASH funding
  sleeve, monthly rebalance. **Turnover limits:** 0.20 one-way monthly cap + 0.02
  no-trade band. **Costs:** 10 bps one-way headline, on a 0/5/10/15 grid. Point-in-time
  throughout; a no-look-ahead test poisons all future prices/carry/macro and asserts
  scores don't move.

## 3. Results (composite overlay, net of costs)
- **Headline: IR vs policy 0.04, tracking error 1.2%/yr.** CAGR 3.73% vs policy 3.66%;
  vol 6.00% vs 6.28%. The overlay adds ~7 bps/yr at slightly lower vol — negligible,
  and (Section 4) statistically indistinguishable from zero.
- **By signal (IR vs policy):** macro +0.08, momentum +0.01, value −0.01,
  **carry −0.19**, composite +0.04. *No family delivers a reliable positive IR net of
  costs on this sample.* Carry, holding the high-yield sleeves, actively hurt: in an
  equity-led, low-yield era the low-carry winner (US equity, 1.2% dividend yield) beat
  the high-carry sleeves, and static carry is fragile in the 2008/2020 shocks.
- **A data-artifact correction, disclosed:** an earlier cut showed carry IR **+0.57**.
  Tracing it, the ICE BofA high-yield series on FRED are restricted to a rolling ~3-year
  window, so the credit-carry leg was silently blank before 2023 and the "edge" was
  three years of curve-fit. Switching to Moody's Baa yield (full history) gives the
  honest −0.19. Finding and killing that is the point of the exercise.
- **Cost sensitivity (composite IR):** 0.105 / 0.072 / 0.038 / 0.005 at 0 / 5 / 10 / 15
  bps. Costs eat most of it; by 15 bps it is zero. The overlay is highly cost-fragile.
- **Turnover-cap sensitivity (a genuine finding):** uncapped IR **−0.03** (turnover
  1.14), cap 0.30 → −0.01, cap 0.20 → **+0.04**, cap 0.10 → **+0.07**. Tighter caps
  *raise* the IR — the multi-signal turnover is largely uncompensated churn, so the
  turnover limit is net-additive, not just risk control.
- **Regime table (composite IR vs policy):** inflation 2021–22 **+1.58** (the only real
  positive — leaning to commodities/away from duration paid), GFC +0.24, COVID-2020
  −1.06, the 2010s QE decade −0.16, 2023–2026 −0.50. The edge lives in one inflationary
  regime and is flat-to-negative everywhere else.

## 4. Robustness (the actual result)
- **Bootstrap 95% CI on the composite IR: [−0.54, +0.58]**, point 0.04. **Zero sits
  squarely in the middle.** This is the honest verdict.
- **Deflated Sharpe = 0.15** (5 trials): after paying for the signals searched, the
  probability the true Sharpe exceeds zero is ~15%. We cannot reject "no edge."
- **Random-tilt null:** a turnover-matched random overlay beats the composite 17% of
  the time — so the signal is only mildly better than random tilting.
- **Simple tilt vs Black-Litterman:** BL IR **−0.14** vs simple tilt +0.04, the expected
  result — BL translates views into constrained weights (the job), but its estimated
  covariance and six parameters add estimation error, not alpha, out of the box.
- **Walk-forward (momentum lookback family):** best in-sample IR 0.38, OOS 0.30 — the
  most impressive number in the deck and the one I trust least. The lookback surface is
  a **knife-edge**: +0.38 (3m) → +0.01 (12m) → −0.19 (18m), a sign flip across the
  plausible range, with the winning short lookbacks carrying the highest turnover. That
  is overfitting, not a durable edge.

## 5. Honest limitations
- **Sample:** ~19 years, one credit cycle, dominated by the GFC, a QE decade, 2020, and
  2022. Far too short and peculiar for strong claims; a 0.04 IR is noise here.
- **Data quality:** the ICE BofA truncation above is a live reminder that free vintages
  are treacherous. Credit carry now uses Moody's **Baa (investment grade)** as a
  full-history stand-in for the **HY** sleeve — a rating mismatch that co-moves but is
  not the same bet. Commodity carry is a **roll proxy** (two broad baskets differ, so
  basket tracking error leaks in); equity carry is dividend yield, not earnings yield.
  The macro nowcast uses FRED release timing, not true ALFRED vintages.
- **What would change my mind:** a longer spliced sample where the composite CI clears
  zero out of sample; HY-specific full-history carry replacing the Baa proxy; and the
  momentum knife-edge resolving into a plateau. Absent that, this is a methods showcase.

## 6. Current positioning read (as of 2026-07-31)
The composite leans **defensive**. Active tilts vs policy: **GOVT +5.9%, CREDIT +4.3%,
CMDTY +3.0%, CASH 0%, TIPS −0.4%, EQ_INTL −4.9%, EQ_US −8.0%.** The driver is a
**risk-off macro nowcast** (flat curve / wider Baa spread → duration, cash, TIPS; away
from equity and credit) plus **carry** favoring high-yielding GOVT/CREDIT over
low-dividend US equity, with **momentum** contributing the commodity overweight. In
plain terms: overweight duration and credit, a modest commodity tilt, funded by
underweighting equities — US most of all.

**Confidence: low, and sized accordingly.** The composite IR is ~0.04 with a confidence
interval that straddles zero and a deflated Sharpe of 0.15, so this is a *small lean*,
not a call — which is exactly why the ±10% cap and (net-additive) turnover limit exist.
The honest summary is worth saying out loud: *a disciplined multi-signal overlay on this
sample produces a modest, regime-dependent tilt whose edge cannot be distinguished from
zero, so the right size is small and the right posture is humility.* That, and the fact
that the process caught and corrected a data artifact that would have flattered the
result, is the actual deliverable — not a Sharpe ratio.
