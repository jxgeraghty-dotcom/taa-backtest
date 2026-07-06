# Tactical Overlay on a Multi-Asset Policy Portfolio — Research Note

*Data vintage 2026-07-05 (ETF total returns + FRED yields + yfinance dividends),
frozen as a snapshot for reproducibility. Regenerate with
`python scripts/run_backtest.py --snapshot-load data/snapshot_2026-07` (or `--real`
to pull live). All figures are net of costs and after turnover limits unless stated.*

## 1. Question and thesis
Can a disciplined tactical overlay, driven by pre-committed, economically motivated
signals, add information *over* a fixed strategic policy portfolio net of realistic
costs? The prior is modest: cross-asset momentum, value, carry, and a macro nowcast
are documented, but on a short single-cycle ETF sample they should produce a small,
regime-dependent edge hard to distinguish from zero. That is what the evidence shows.

## 2. Setup
- **Universe / proxies (monthly total return):** EQ_US=VTI, EQ_INTL=VEA, GOVT=IEF,
  CREDIT=HYG, TIPS=TIP, CMDTY=DBC, CASH=BIL. Sample **2007-01 to 2026-07, 235 months**;
  overlay trades after a 12-month warmup (~2008–2026).
- **Strategic policy (the benchmark):** EQ_US 35, EQ_INTL 15, GOVT 20, CREDIT 10,
  TIPS 5, CMDTY 5, CASH 10 (%). Judged on information ratio *over policy*, run through
  the identical loop and cost model — not total return.
- **Signals (committed before testing, four documented families):** *momentum* = 12m
  trailing return (Moskowitz-Ooi-Pedersen 2012); *value* = negative 5y return, 12m skip
  (long-horizon reversal, Asness-Moskowitz-Pedersen 2013); *carry* = yield/carry level
  per sleeve — FRED yields (govt/TIPS/cash), Moody's Baa for credit, trailing dividend
  yield for equities, a roll-yield proxy for commodities; *macro nowcast* = trailing
  change in the curve slope and Baa credit spread mapped to a fixed, pre-committed
  risk-on/off asset-response vector. *composite* = equal-weight z-blend (no weight tuning).
- **Construction:** bounded tilt around policy, ±10% per-sleeve cap, CASH funding
  sleeve, monthly rebalance. **Turnover limits:** 0.20 one-way monthly cap + 0.02
  no-trade band. **Costs:** 10 bps one-way headline, on a 0/5/10/15 grid. Point-in-time
  throughout; a no-look-ahead test poisons all future prices/carry/macro.

## 3. Results (composite overlay, net of costs)
- **Headline: IR vs policy 0.04, tracking error 1.2%/yr.** CAGR 3.73% vs policy 3.66%;
  vol 6.00% vs 6.28%. A negligible add, and (Section 4) not separable from zero.
- **By signal (IR vs policy):** momentum **+0.09**, macro +0.08, composite +0.04,
  carry **−0.09**, value **−0.18**. Momentum is the only clearly positive family;
  value (long-horizon reversal) actively hurt in this momentum-led sample; carry, now
  spanning all seven sleeves, is mildly negative. The equal-weight composite averages
  a positive and a negative pair down to ~zero.

![Cumulative active return of the composite overlay vs policy](figures/composite_active_return.png)

  The curve wanders around zero: negative through 2010–2014, a spike in the 2022
  inflation shock, and a fade since. Net cumulative active return over ~18 years is
  under +1%.
- **Cost sensitivity (composite IR):** 0.105 / 0.072 / 0.038 / 0.005 at 0 / 5 / 10 / 15
  bps — highly cost-fragile.
- **Per-sleeve vs flat costs:** under a per-sleeve schedule (HY 15, commodities 12,
  Treasuries/cash 1–3 bps) the IR is **0.06** vs 0.04 at a flat 10 bps — the flat
  assumption is conservative, because the overlay's turnover sits mostly in
  cheap-to-trade sleeves.
- **Turnover cap and rebalance frequency (a genuine finding):** tighter turnover caps
  *raise* IR (uncapped −0.03 → 0.20-cap 0.04 → 0.10-cap 0.07), and **quarterly
  rebalancing lifts IR to 0.10 while halving turnover** (0.54 vs 1.04/yr). Much of the
  monthly turnover is uncompensated churn; trading less is the single most reliable
  improvement here.
- **Regime table (composite IR vs policy):** inflation 2021–22 **+1.58** (the only real
  positive), GFC +0.24, COVID-2020 −1.06, the 2010s QE decade −0.16, 2023–2026 −0.50.
  The edge lives in one inflationary regime.

## 4. Robustness (the actual result)
- **Bootstrap 95% CI on the composite IR: [−0.54, +0.58]**, point 0.04. **Zero is
  central.** With ~215 monthly active observations, this is the honest verdict.
- **Deflated Sharpe = 0.15** (and the true trial count is higher once the config search
  — caps, frequency, cost schedule, vol scaling, lookbacks — is counted): we cannot
  reject "no edge."
- **Random-tilt null:** a turnover-matched random overlay beats the composite 18% of
  the time — only mildly better than random tilting.

![Rolling 24-month information ratio of the composite overlay](figures/composite_rolling_ir.png)

  The rolling IR swings from about −1.8 to +1.6 and crosses zero repeatedly — there is
  no stable edge, only regimes.
- **Vol-scaled tilts** (risk-equalized instead of weight-equalized) IR **−0.17**: an
  available construction that did not help here, shown so the choice is explicit.
- **Simple tilt vs Black-Litterman:** BL IR **−0.14** vs simple tilt +0.04 — BL
  translates views into constrained weights (the job) but adds estimation error, not
  alpha, out of the box.
- **Walk-forward (momentum lookback family):** best in-sample IR 0.42, OOS 0.31 — the
  most impressive number, and the one I trust least. **White's Reality Check p-value for
  the best lookback (7 tried) is 0.13**, so even that OOS figure is explainable by the
  search; the lookback surface is a knife-edge (sign flips across the grid).

## 5. Honest limitations
- **Sample:** ~19 years, one credit cycle (GFC, QE decade, 2020, 2022). Too short and
  peculiar for strong claims; a 0.04 IR is noise here.
- **Two artifacts caught and disclosed.** (a) An earlier cut showed carry IR +0.57
  until traced to FRED restricting the ICE BofA HY series to a rolling ~3-year window;
  credit now uses Moody's **Baa** (investment grade — a documented mismatch for the HY
  sleeve). (b) The per-signal numbers changed once a weight bug was fixed: sleeves with
  no early history (VEA pre-2007, or carry before its trailing window fills) returned a
  NaN score that was silently dropping the sleeve and over-trading; keeping them at
  policy weight corrected momentum, value, and carry. The composite was never affected.
- **Proxies:** commodity carry is a two-basket roll proxy; equity carry is dividend, not
  earnings, yield; macro uses FRED release timing, not true ALFRED vintages.
- **What would change my mind:** a longer spliced sample where the composite CI clears
  zero out of sample, HY-specific full-history carry, and the momentum knife-edge
  resolving into a plateau. Absent that, this is a methods showcase.

## 6. Current positioning read (as of 2026-07-31)
The composite leans **defensive**. Active tilts vs policy: **GOVT +6.0%, CREDIT +4.4%,
CMDTY +2.7%, CASH 0%, TIPS −0.4%, EQ_INTL −4.8%, EQ_US −8.0%.** The driver is a risk-off
macro nowcast (flat curve / wider Baa spread → duration, cash, TIPS; away from equity)
plus carry favoring high-yielding GOVT/CREDIT over low-dividend US equity, with momentum
adding the commodity overweight. In plain terms: overweight duration and credit, a
modest commodity tilt, funded by underweighting equities — US most of all.

**Confidence: low, and sized accordingly.** The composite IR is ~0.04 with a confidence
interval straddling zero and a deflated Sharpe of 0.15, so this is a *small lean*, not a
call — which is why the ±10% cap and the (net-additive) turnover limit exist, and why
the most defensible configuration found is the lowest-turnover one (quarterly, tightly
capped). The honest summary: *a disciplined multi-signal overlay on this sample produces
a modest, regime-dependent tilt whose edge cannot be distinguished from zero, so the
right size is small and the right posture is humility.* Two data/code artifacts that
would have flattered the result were caught and corrected in the process — which is the
actual deliverable, not a Sharpe ratio.
