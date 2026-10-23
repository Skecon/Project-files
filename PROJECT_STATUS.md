# Project status and handoff (as of 20 Oct 2026)

## How to help me (read first)
I am a student learning to build an ML project for the first time. **Guide me; don't do it for me.**
Give one task at a time with the goal, the reason and hints. Let me write the code. Review my code
when I share it. Teach the concept and show code only when I'm stuck or ask. Explain in plain language.

---

## 1. The project
**Course:** HST-105 Analytics for Business and Society, IIT Roorkee. I'm working on it alone for now.

**Research question:** Can machine learning reconstruct the implied volatility (IV) surface of SPY
options more accurately than the industry-standard SVI model, and does any gain hold up in crisis periods?

**Task design (reconstruction, not forecasting):** on each test day, hide some of that day's observed
quotes, let each model predict them from the visible quotes, and score **only the hidden quotes**.

**Data:** Kaggle "SPY Option Chains Q1 2020 – Q4 2022" (end-of-day, one snapshot per trading day),
`spy_2020_2022.csv`, 1.28 GB, 3,589,079 rows × 33 columns. Each row = one strike/expiry with call
and put columns side by side (`C_IV`, `P_IV`, bid, ask, volume, Greeks, sizes "a x b" = bid size x ask size).

**Repo:** github.com/Skecon/Project-files (private). Local folder: `Main project/Project files/`.
```
data/raw/           spy_2020_2022.csv (git-ignored)
data/processed/     parquet outputs of each stage (git-ignored)
notebooks/          01_Cleaning, 02_Filters, 03_Features (04_evaluation next)
src/, src/models/   reusable code (empty so far, starting Stage 4)
results/            model outputs (git-ignored)
report/figures/     Phase I figures
config.yaml         shared settings; requirements.txt; README.md
```
Note: no virtual environment (.venv) has been set up yet; system Python 3.12 + pandas 3 is used.

---

## 2. Done so far

### Stage 1: Setup ✅
Git and GitHub repo, folder structure, .gitignore (data never committed), README, config.yaml.

### Stage 2: Cleaning ✅ (`01_Cleaning.ipynb`, `02_Filters.ipynb`)
1. Column names cleaned (`[C_IV]` → `c_iv`); text → numbers and dates.
2. `c_size`/`p_size` ("955 x 596") split into bid size and ask size (verified: first number = bid).
3. **Dropped 9 holiday snapshots in 2022** (2022-01-17, 02-21, 04-15, 05-30, 06-20, 07-04, 09-05,
   11-24, 12-26): near-copies of the previous day. Result: **749 trading days**.
   (Also missing from the data: 2020-06-18, 2020-11-27, 2020-12-24, 2022-05-11, 2022-06-17,
   2022-06-24, 2022-12-16; noted as a limitation.)
4. Reshaped to **one row per option** (call/put stacked, `type` column): 7,099,618 rows →
   `options_long.parquet`.
5. Quality filters, with a removal log:

| Filter | Removed | Left |
|---|---|---|
| IV not missing | 382,881 | 6,716,737 |
| volume > 0 | 2,714,730 | 4,002,007 |
| bid > 0 | 176,261 | 3,825,746 |
| ask > bid | 57,027 | 3,768,719 |
| relative spread ≤ 0.5 | 134,575 | 3,634,144 |
| 0.01 ≤ IV ≤ 2 | 61,006 | 3,573,138 |
| DTE ≥ 7 | 181,257 | **3,391,881** (47.8% kept, 749 days) |

   Added columns `mid`, `spread`, `relative_spread`. Saved `options_filtered.parquet`.

### Stage 3: Features ✅ (code tested, being finalised in `03_Features.ipynb`)
1. Paired calls and puts at the same (date, expiry, strike): 1,082,530 pairs.
2. **Forward price F** per (date, expiry) from put-call parity: fit `C_mid − P_mid = D·F − D·K` over
   near-the-money pairs (|K/S − 1| ≤ 0.05, at least 3 pairs) with `np.polyfit`; F = intercept / D.
   - F is good: F/S < 1 for long maturities in 2020–21 (dividends > rates), F/S > 1 in 2022 (rates rose).
   - **D (discount factor) is unreliable** (median 1.002, 64% of groups > 1). Likely the American
     early-exercise effect plus a narrow strike range. D is ignored; only F is needed.
   - 2.1% of rows (72,166, mostly long-dated) had no forward → **dropped**.
3. Features: `T = dte/365`, `k = ln(K/F)`, `w = iv² · T`.
4. **Out-of-the-money only:** puts with k < 0, calls with k ≥ 0; plus |k| ≤ 1.
   Result: **2,017,522 quotes, 749 days, about 2,750/day, no duplicate (date, expiry, strike)**.
   Saved to `options_features.parquet`.
5. Smile plots look right (skew, smooth) in 2021.

### Key assumption and findings about American options
- SPY options are **American**. **Assumption:** the provider's IVs are correct American IVs
  (equivalent to the industry practice of "de-Americanizing" prices with a binomial tree).
- Test: put IV − call IV at the **same strike** near the money is ≈ 0 for 2020 Q2 – 2022 Q2 ✅,
  but **+0.3 to +0.5 vol points in 2022 Q3–Q4** (high rates; up to about 2 points on some days) and
  **−1.9 in 2020 Q1** (COVID). This causes a small step at k = 0 on some late-2022 days.
  → Report as a limitation; a possible later fix is recomputing IV from mid prices using our forward.
- Literature: almost all papers use European SPX options; only the 2026 SPY paper uses SPY (American)
  and doesn't discuss it. ML work on American-option surfaces found online is almost all neural nets;
  **no XGBoost or Gaussian-process studies found**, which is a possible "what's new" angle.

---

## 3. How the papers test models (what we copy)
Within-day hide-and-reconstruct (Deep Smoothing NeurIPS 2020; 2026 SPY paper arXiv 2605.24031):
- score only hidden points; random **and** wing (extrapolation) masks; sparsity sweep 10–90% hidden
- time-ordered split, no future data; error by moneyness × maturity region
- spread across days (quantiles); arbitrage violation rate and size
- crisis periods reported separately; several random seeds for neural nets

Our additions: hedging test, Diebold–Mariano significance test, price-space error.

Two kinds of models:
- **Per-day calibration** (SVI, SSVI, GP, linear interpolation): use today's visible quotes only.
- **Learned** (XGBoost, NN): trained on past days, and also given today's visible info (ATM level,
  skew, yesterday's surface). Never see today's hidden quotes or any future day.

---

## 4. Still to do

### Stage 4: Testing system (NEXT) → `notebooks/04_evaluation.ipynb` + `src/`
- 4.1 Make `src/` importable (`sys.path.append('..')`, `%autoreload`).
- 4.2 `src/masking.py`: `random_mask(day, frac=0.2, seed)`, `wing_mask(day)` (outer 10% per side
  per expiry), `expiry_mask(day, seed)` (one whole interior expiry). Seed from the date so every
  model gets the same mask.
- 4.3 `src/metrics.py`: rmse, mae, max_error in vol points; `regional_rmse` with k buckets
  (<−0.2, −0.2..−0.05, −0.05..0.05, 0.05..0.2, >0.2) × T buckets (≤0.25, 0.25..1, >1). Test with asserts.
- 4.4 `src/models/base.py`: `BaseModel` with `fit(visible, history)` and `predict(k, T)`;
  `src/models/dummy.py`: `MeanModel` (predicts the mean visible IV).
- 4.5 `history_for(test_day)`: all days before the first day of the test month (walk-forward).
- 4.6 `evaluate(model_class, data, test_days, mask_name)`: loop over days → one results row per day.
- 4.7 Run MeanModel on July 2020 with all 3 masks; save `results/dummy_july2020.csv`.

### Stage 5: Models → `src/models/`
B0 linear interpolation in (k, w) → B1 **SVI per expiry** with Gatheral–Jacquier constraints (main
benchmark) → B2 SSVI → M1 XGBoost on IV → M2 XGBoost on SVI residuals → M3 Gaussian process (per day)
→ M4 neural net with no-arbitrage penalty (with/without penalty, 3 seeds). Add the arbitrage checks
(calendar: w increasing in T; butterfly) here.

### Stage 6: Experiments
All models × masks × test days (from July 2020), sparsity sweep, hedging test, Diebold–Mariano +
bootstrap confidence intervals vs SVI.

### Stage 7: Analysis
Main table by regime (COVID crash 20 Feb – 30 Apr 2020, recovery rest of 2020, calm 2021, bear 2022);
error heatmaps; sparsity curve; crisis-day surface plots; accuracy vs arbitrage; hedging distributions.

### Stage 8: Report and slides
Reproducible repo with README.

---

## 5. Open questions I still need to answer (for the report)
1. Which filter removed the most rows and why?
2. Does filter order change the final result? The log?
3. What share was DTE < 7? Keep as a separate group later?
4. Why do more puts than calls survive the filters?
5. Why should the raw CSV never be edited?
6. Is F above or below S in 2020–21, and what does it say about dividends vs rates?
7. How does the implied rate behave in 2022 and why? (Note: D turned out unreliable.)
8. What did the call-vs-put IV check show about the American-IV assumption?
9. Why is there a step at k = 0 in late 2022?
10. Why must the mask depend on the date?
11. Why may a model never see days after its test month?
12. Which mask will be hardest for models, and why?

## 6. Habits to keep
Restart → Run All before saving; check row counts after every step; assign pandas results back
(`df = df.drop(...)`); watch for silent type mismatches (text vs dates/numbers); save each stage
to a new parquet file; commit after each working step.
