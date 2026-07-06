# Project Handoff — Multi-Asset TAA Backtesting Framework

A status/context document for picking this work back up. The technical readme is
[README.md](README.md); the deliverable is [notes/research_note.md](notes/research_note.md).

## Goal
Build the flagship: a multi-asset **tactical asset allocation (TAA) backtester** — a
strategic policy portfolio across asset-class ETF proxies, with tactical tilts driven by
documented signals (**valuation, cross-asset momentum, carry, macro nowcast**). The
differentiator is **discipline**: point-in-time data, realistic transaction costs,
turnover limits, and walk-forward out-of-sample testing. Ship the code plus a two-page
research note that ends in an **actual positioning recommendation with honest
limitations** — a methods showcase, not a vanity Sharpe.

## Current state
- **Framework complete and tested.** 68 tests pass (offline, no network). Runs
  end-to-end on synthetic data and on real ETF/FRED data.
- **CRITICAL FIX (2026-07-05, second session): the original real-data results were
  computed on a scrambled universe.** `load_etf_prices` assigned sleeve names to
  yfinance columns positionally, but yfinance returns columns in alphabetical ticker
  order — "EQ_US" held BIL, "CASH" held VTI, etc. Caught via a vol/inception-date
  sanity check on the snapshot; confirmed by 1.000 per-sleeve correlation between the
  old snapshot (relabeled) and a fresh pull. Fixed with a by-ticker rename
  (`rename_ticker_columns`, raises on missing tickers) + regression test; the snapshot
  was re-pulled and the note rewritten. A one-month look-ahead at walk-forward
  selection boundaries was also found and fixed (boundary rebalance was one month
  early), plus `--snapshot-load` runs now use REAL_REGIMES and the correct data label.
- **Methodology hardening (2026-07-06):** turnover governors now apply to the policy
  benchmark too (a benchmark forced to churn drift corrections the strategy may skip
  was flattering the overlay — this alone trimmed composite IR 0.08 → 0.07); the
  mandate cap now binds the funding sleeve as well; per-sleeve cost schedules raise on
  a missing sleeve instead of silently filling the mean; the engine raises on NaN
  next-period returns instead of silently dropping the sleeve from the book.
- **Corrected real-data result (vintage 2026-07-05, frozen snapshot):** composite
  overlay **IR vs policy ≈ 0.07, 95% CI [−0.46, +0.60]** (zero central), deflated
  Sharpe 0.18 — an edge still indistinguishable from zero. Per-family IR: macro +0.22,
  momentum +0.18, carry +0.02, value −0.52. Edge is regime-concentrated: GFC +0.68 and
  2021–22 inflation +0.83, paid back in COVID's rebound (−2.01). Random-tilt null:
  only 3.5% of random overlays beat it. Current positioning leans defensive: OW
  GOVT +8.0% / CREDIT +2.8%, UW EQ_US −8.7%.
- **Research note written** with two figures ([notes/figures/](notes/figures/)) and a
  dated positioning read.
- **Git:** local repo at `taa-backtest/`, branch `main`, **2 commits**, working tree
  clean. Identity set locally to `jxgeraghty <jxgeraghty@gmail.com>`.
- **Push in progress (blocked on auth).** GitHub repo created empty at
  `https://github.com/jxgeraghty-dotcom/taa-backtest`. No remote/token/`gh`/SSH key on
  this machine yet — waiting on a fine-grained PAT (Contents: write) to push `main`.

## Key decisions (and why)
- **Judge on IR over policy, net of costs** — not total return, which is mostly policy
  beta. The overlay's job is active return per unit of active risk.
- **Four pre-committed signal families + equal-weight composite, no weight tuning** —
  with ~215 monthly points you can't estimate an optimal blend without overfitting.
- **Turnover is governed, not just costed** (cap + no-trade band), and the governors
  apply to strategy and policy benchmark alike. Empirically tighter caps and
  **quarterly rebalancing raise IR** (monthly turnover is largely uncompensated
  churn) — reported as comparisons, not baked into one number.
- **Credit carry uses Moody's Baa (BAA / BAA10Y)**, because FRED restricts the ICE BofA
  HY series to a rolling ~3-year window (a silent-truncation artifact that was caught and
  disclosed). Documented IG-vs-HY mismatch.
- **Fixed a NaN-score weight bug** so a "no view" sleeve (missing early history) stays at
  policy weight instead of being dropped and over-traded. This corrected the per-signal
  numbers; the composite was never affected (it `fillna(0)`s each family).
- **Columns are renamed by ticker, never positionally** (see CRITICAL FIX above) — the
  regression test in `tests/test_loaders.py` shuffles column order to keep it that way.
- **Dated CSV snapshots for reproducibility**; raw vendor data is git-ignored (`/data/`).
- **Black-Litterman shown as constrained-MV construction, not alpha** (it underperforms
  the simple tilt OOS) — the honest, expected result.
- **Deliberately modest framing** throughout: a small regime-dependent lean sized small,
  plus disclosure of the two data/code artifacts caught along the way.

## Open questions / unresolved
- **Push:** provide a fine-grained PAT (repo `taa-backtest`, Contents: Read and write) in
  `gh_token.txt`, then push `main`. Decide **repo public vs private**.
- **Commit author name:** currently `jxgeraghty` (derived from email) — keep or change to
  a real name (would need `--amend --reset-author`).
- **Data length:** the ~2007–2026 ETF sample is too short/peculiar for strong claims;
  a longer spliced index history is needed to test whether the edge clears zero OOS.
- **Carry proxies:** HY-specific full-history carry (currently Baa IG stand-in); a cleaner
  per-contract commodity roll (currently a two-basket proxy); equity carry is dividend,
  not earnings, yield.
- **Macro vintages:** true ALFRED point-in-time vintages instead of the fixed
  `release_lag` placeholder in `MacroHistory`.
- **BL extension:** relative (Idzorek-style) views for cross-sectional rather than
  directional bets.

## Files / artifacts
- [README.md](README.md) — technical readme (design decisions, how look-ahead is prevented).
- [notes/research_note.md](notes/research_note.md) — the two-page deliverable; figures in
  [notes/figures/](notes/figures/). Original template:
  [notes/research_note_template.md](notes/research_note_template.md).
- **`taa/` package:**
  - `data/` — `panel.py` (PriceHistory/MacroHistory/**DataBundle**, PIT containers),
    `synthetic.py` (offline bundle), `loaders.py` (real ETF+FRED, carry legs),
    `snapshot.py` (dated CSV vintages).
  - `signals/` — `momentum.py`, `value.py`, `carry.py`, `macro.py`, `composite.py`,
    `base.py`, `registry.py`.
  - `construction/` — `policy.py`, `tilt.py` (SimpleTilt: NaN-safe, optional vol-scale),
    `black_litterman.py`.
  - `backtest/` — `engine.py` (rebalance loop, turnover limits, rebalance_every),
    `costs.py` (scalar or per-sleeve).
  - `evaluation/` — `metrics.py`, `regimes.py`, `robustness.py` (bootstrap CI, deflated
    Sharpe, random-tilt null, **reality_check**, walk-forward).
- [scripts/run_backtest.py](scripts/run_backtest.py) — end-to-end
  (`--real`, `--snapshot-load/-save`). [scripts/make_figures.py](scripts/make_figures.py).
- [config/default.yaml](config/default.yaml) — every default is a defensible decision.
- `tests/` — 66 tests; [tests/test_no_lookahead.py](tests/test_no_lookahead.py) is the
  core discipline (poisons all sources across every signal);
  [tests/test_loaders.py](tests/test_loaders.py) pins the by-ticker column rename.
- [pyproject.toml](pyproject.toml), [requirements.txt](requirements.txt),
  [conftest.py](conftest.py) (session fixtures).
- `data/snapshot_2026-07/` — frozen real-data vintage (git-ignored; regenerate with
  `--snapshot-save`).

## Quick commands
```bash
pip install -r requirements.txt
pytest -q                                                   # 68 tests
python scripts/run_backtest.py                              # offline (synthetic)
python scripts/run_backtest.py --snapshot-load data/snapshot_2026-07   # reproduce the note
python scripts/run_backtest.py --real --snapshot-save data/snapshot_YYYY-MM   # fresh pull
python scripts/make_figures.py --snapshot data/snapshot_2026-07
```
