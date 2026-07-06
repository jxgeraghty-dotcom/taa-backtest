# Multi-Asset TAA Backtesting Framework

A methods scaffold for researching a tactical asset allocation overlay on a
strategic multi-asset policy portfolio. Built to test honestly, not to win.

> Read this first. A clean trend or dual-momentum backtest with a 1.2 Sharpe is the
> most common vanity project in this corner of the industry, and anyone on a TAA
> desk has seen hundreds. The framework itself impresses no one. What reads as
> credible is the rigor scaffolding around it and the willingness to conclude that
> the overlay adds something modest and regime-dependent net of costs. Build for
> that conclusion. If your writeup says "I beat the market," you invite a takedown.

## Quick start (offline, no network)

```bash
pip install -r requirements.txt
pytest -q                       # the no-look-ahead harness runs first
python scripts/run_backtest.py  # end-to-end on synthetic data
```

The synthetic path proves the machinery only. It is not a market simulation and
must never back a claim about a real strategy.

## Real data

Uncomment `yfinance` and `pandas-datareader` in `requirements.txt`, then:

```bash
python scripts/run_backtest.py --real
```

Honest data constraint: liquid ETF proxies mostly begin around 2003 to 2007
(TIP 2003, HYG 2007, DBC 2006). A clean all-ETF backtest realistically starts
circa 2007, roughly 220 monthly points dominated by the GFC, the bond bull, 2020,
and 2022. That sample is too short and peculiar for strong claims. Treat this as a
methods showcase and say so. If you want to claim the strategy works, you need
spliced index total-return history, which is hard to get cleanly for free.

## How look-ahead is prevented

Signals receive a `DataBundle` (prices, and optionally a carry panel and a macro
series) and read each source only through its `history(as_of)`, which returns rows
dated on or before `as_of`. The engine builds weights from data through `t` and
earns the NEXT period's return on them, so a trade never sees the return it earns.
`tests/test_no_lookahead.py` poisons every future row of **all three** sources with
absurd values and asserts signal output does not move. Every new signal you add must
be listed there. No exceptions: that test is the discipline the whole framework
protects. (A bare `PriceHistory` is still accepted anywhere a bundle is — it is
treated as a bundle whose only source is prices — so price-only call sites are
unchanged.)

## Layout

```
taa/
  data/         point-in-time containers (panel.py: PriceHistory/MacroHistory/DataBundle),
                offline synthetic bundle, real ETF+FRED loaders
  signals/      the pre-committed set: momentum, value (5y reversal), carry, macro
                nowcast, and an equal-weight composite (do not tune after seeing results)
  construction/ strategic policy, simple bounded tilt, and Black-Litterman
  backtest/     rebalance loop with drift, explicit linear costs, turnover limits,
                pluggable constructor
  evaluation/   metrics (IR over policy leads), regime table, robustness protocol
tests/          no-look-ahead harness (first, all sources), engine/turnover/signal checks
scripts/        run_backtest.py end-to-end
config/         default.yaml, every default is a decision you must defend
notes/          research_note.md (finished, real-data deliverable) + the template
```

## Design decisions worth defending in an interview

- The benchmark is the strategic policy, not SPY or cash. The overlay is judged on
  information ratio over policy, net of costs. Total return is mostly policy beta.
- The tilt is a transparent bounded overlay, net-zero active weight, capped per
  sleeve. A real desk uses a constrained optimizer (mean-variance or
  Black-Litterman); this stand-in is honest about being one.
- Costs are modeled explicitly and reported across a grid. Monthly signals can
  generate enough turnover to eat the entire edge.
- The overlay is driven by four pre-committed, documented signal families — momentum,
  value (5y reversal), carry, and a macro nowcast — plus an equal-weight composite.
  They are compared head-to-head on IR net of costs; the weights are not tuned.
- Turnover is governed, not just costed: a per-rebalance turnover cap (partial
  rebalance toward target) and a no-trade band sit between the constructor and the
  book. Realized turnover/cost are reported after the limits bind, and the run shows
  IR across a grid of caps so you can see where the limit starts to cost you edge.
- Realism knobs, all reported as comparisons rather than baked into one number:
  configurable rebalance frequency (monthly vs quarterly), per-sleeve transaction
  costs (HY/commodities dearer than Treasuries/cash), and optional vol-scaled tilts
  (risk-equalized rather than weight-equalized).
- Multiple testing is priced: a White Reality-Check p-value puts an honest number on
  the walk-forward's best-lookback, since a grid search inflates naive significance.
- Reproducibility: `--snapshot-save/--snapshot-load` freeze a dated data vintage to
  CSV so a note's numbers regenerate exactly, instead of drifting with each live pull.
- Robustness (bootstrap CI, deflated Sharpe, random-tilt null, parameter
  sensitivity) is treated as the main result, because with ~220 points an IR gap of
  0.2 is not statistically distinguishable.

## Construction: simple tilt vs Black-Litterman

Two constructors implement the same interface, `weights(scores, policy, prices, as_of)`,
so the engine swaps between them with a `constructor=` argument.

- `SimpleTilt`: transparent, bounded, self-funding tilt around policy. No estimation.
- `BlackLitterman`: reverse-optimizes the policy into equilibrium returns, turns the
  signal into absolute views, forms posterior returns, and solves a constrained
  mean-variance problem (long-only, policy +/- max_tilt). With a flat signal it
  returns the policy exactly, which is the property that makes it a proper overlay.

Be clear-eyed about what BL is for. It is the institutionally standard way to turn
views into mandate-constrained weights, and it is the right thing to show for a role
that asks you to translate research into portfolios. It is not a source of alpha. It
adds an estimated covariance and six free parameters (tau, delta, view_scale,
view_conf, cov_lookback, shrink), all of which are estimation error and tuning
surface. On the offline demo it underperforms the simple tilt, which is the expected
and honest result. Judge it out of sample, never on the in-sample number.

## Walk-forward, out of sample

`walk_forward_select` does expanding-window parameter selection: at each boundary it
picks the candidate that maximized the objective on realized returns so far, deploys
it for `step` months, then re-selects. It reports a single stitched backtest that
pays the real cost of switching parameters at each boundary, and metrics are computed
from the first out-of-sample date onward. Use it to show the gap between the best
in-sample IR (optimistic) and the OOS IR (what you would actually have earned). That
gap is the point. With ~220 monthly points and a 12-month step you get only a handful
of OOS blocks, so the OOS estimate is itself noisy; this reduces overfitting bias, it
does not make a short sample long.

## The finished deliverable

`notes/research_note.md` is the two-page note written against a real-data run
(2007–2026 ETF proxies + FRED yields), with two figures (`notes/figures/`) and a dated
positioning recommendation. Regenerate the figures with
`python scripts/make_figures.py --snapshot data/snapshot_2026-07`. The headline is the
intended one: a disciplined multi-signal overlay adds a small, regime-dependent tilt
whose IR cannot be statistically distinguished from zero on this sample, so it is sized
as a lean, not a call — and two data/code artifacts that would have flattered the
result were caught and disclosed along the way.

## Still left for you

Vintage macro data (FRED/ALFRED) is the right answer to release timing; the fixed
`release_lag` in `MacroHistory` is a placeholder. Carry now covers all seven sleeves —
FRED yields (govt/TIPS/cash), Moody's Baa yield for credit, trailing dividend yield for
equities, and a roll-yield proxy for commodities — but each has a wart the note is
explicit about: the ICE BofA HY series on FRED are restricted to a rolling ~3-year
window (so credit falls back to investment-grade Baa), the commodity leg is a
two-basket roll proxy, and equity carry is dividend rather than earnings yield. A true
HY-yield vintage and a cleaner per-contract roll are the obvious upgrades. The BL view
model here is absolute per-asset views; relative views (top vs bottom sleeves, Idzorek
style) are a natural extension for expressing cross-sectional rather than directional
bets.
