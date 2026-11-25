# CLAUDE.md — Final report for "Machine Learning for Implied Volatility Surface Reconstruction"

You are helping write the **Phase 2 final report** of a university team project (HST-105 Analytics
for Business and Society, IIT Roorkee; team: Aryan, Shachi, Shreekrushna). All experiments are
finished. Your job is to turn the results in this folder into a deep, accurate report.
**Every number you write must come from a file in `results/` or from this document.** Never invent
numbers, citations or team contributions.

---

## 0. READ FIRST — what the professor will grade

The professor told the team he will check:
1. **How each model was implemented** (the mechanics, not just the name).
2. **Why each model was chosen** (what weakness of the benchmark it addresses).
3. **How we tackled the problem stated in Phase 1** (see `context/Phase1_presentation.pdf`).

So the report must:
- **Go deep into every model**: inputs, formula or algorithm, what is fitted, what is fixed,
  how it predicts a hidden point, and its strengths/weaknesses. Section 5 of this file has all details;
  the code is in `code/src/models/`.
- **Explain every step of the process**: cleaning → features → masking → walk-forward evaluation →
  metrics → statistics. Section 3–4 below.
- **Make the masking design and evaluation thresholds the centrepiece.** The team considers this
  its strongest point: three masking schemes, a sparsity sweep, regional error grids, arbitrage
  checks, regime splits, significance tests. Explain each one, why it exists, and what it revealed.
- **Explain why this problem is NOT a standard ML problem** and therefore needs a special evaluation
  design (Section 2). This argument should appear early in the report and be referred back to.
- **Explain the results thoroughly** — not just "model X is best", but where, when, by how much,
  how sure we are, and why (Section 6 has interpretations).
- Follow the assignment format in `context/assignment_guidelines.pdf` (motivation + topic, empirical
  strategy, literature appendix with one page per paper in the given format, code/readme, statement
  of contribution).

A complete **draft** already exists: `context/Final_Report_DRAFT.docx` (22 pages, built by
`code/report/make_report.py`). It is a good starting point but is too brief on models and on the
evaluation design. **Expand it** following the instructions above; keep its correct numbers.

The deadline for submission was 25 November 2026 (penalty 4 marks per hour late) — work fast and
check with the user what format they want (.docx is accepted).

---

## 1. The project in one page

**Phase 1 problem (from the Phase 1 presentation):** on any day only some strike/expiry
combinations of SPY options trade. A trading desk needs an implied volatility (IV) at every point to
price, hedge and measure risk. Missing points must be estimated from observed ones. Industry uses
**SVI**, a parametric formula fitted per expiry (Gatheral & Jacquier 2014). It is fast and well
understood but imposes a fixed shape.

**Research question:** *Can machine learning reconstruct the implied volatility surface more
accurately than SVI, and does any gain hold up during crisis periods?*

**Approach (as promised in Phase 1):** hide IVs that we actually know, ask each model to predict
them, compare with the truth. SVI is the benchmark; ML must show a *meaningful* improvement to
justify its complexity. Test across crisis (COVID crash), calm (2021) and bear (2022) markets.
Also check no-arbitrage (consistency) and hedging.

**Answer in one paragraph:** Yes where it matters most — beyond the observed strikes (the wings).
XGBoost trained to correct SVI cut wing error from 3.53 to 1.25 vol points (−65%, 95% CI −61% to
−68%, Diebold–Mariano p < 0.001, better on 99% of days). Inside the observed range, simple linear
interpolation already beats SVI by 48% and ML adds little. Gains shrink in the March 2020 crash
(learned models had only two calm months of history). Flexible models break no-arbitrage more
often; SSVI is arbitrage-free but least accurate; the neural net with a no-arbitrage penalty is the
best compromise. Better IVs do not measurably improve one-day delta hedges.

---

## 2. Why this is not a standard ML problem (explain this in the report)

1. **No labels for the thing we want to predict.** The points we ultimately need (untraded strikes)
   have no true value. We therefore use **self-supervised "hide-and-reconstruct"**: hide quotes that
   *do* trade, predict them, score them. This is the method of the reconstruction literature
   (Deep Smoothing, NeurIPS 2020; 2026 SPY study arXiv 2605.24031).
2. **The unit of observation is a whole surface (one day), not a row.** About 2,700 quotes per day
   are strongly linked (neighbouring strikes, the same expiry). A random row-level train/test split
   would put a hidden quote's immediate neighbours in training → leakage. Hiding happens *within*
   each test day, and the model sees only that day's visible quotes (plus past days).
3. **Different kinds of "missing" need different tests.** Real gaps are of three types: scattered
   untraded strikes inside the surface, entire missing maturities, and strikes beyond the last quote
   (the wings). One random hold-out would only test the easiest type. Hence **three masking schemes**
   plus a **sparsity sweep** (Section 4).
4. **Time dependence and regime shifts.** Volatility today resembles yesterday. Random k-fold would
   let a model learn from the future. Learned models use **walk-forward (expanding-window)
   validation**: retrained each month on all earlier days only (34 monthly folds). Results are
   reported **by market regime** because a model can be good on average but fail in a crash.
5. **Accuracy is not enough — economic consistency matters.** A surface can fit well and still
   allow riskless profit (static arbitrage). We measure **butterfly** and **calendar** violations
   for every fitted surface.
6. **Errors are heavy-tailed and autocorrelated across days.** A few crisis days dominate. So we use
   per-day errors, quantiles (5%/50%/95%), the **Diebold–Mariano test with Newey–West variance**
   and a **moving-block bootstrap** instead of naive t-tests.
7. **Errors differ by region.** Wings and short maturities are much harder. We report a
   **regional error grid** (moneyness × maturity).
8. **Two families of models need two protocols.** Per-day "calibration" models (SVI, GP, NN) are
   fitted fresh on each day's visible quotes; "learned" models (XGBoost) are trained on past days and
   then use today's visible quotes as input. The evaluation framework supports both without leakage.
9. **Units.** Errors are in **volatility points** (1 point = 0.01 IV, e.g. 21% instead of 20%).

---

## 3. Data pipeline (our main run) — every step

Source: Kaggle "SPY Option Chains Q1 2020 – Q4 2022" (end-of-day; one snapshot per trading day).
Raw: 3,589,079 rows × 33 columns, 1.28 GB; each row = one strike/expiry with call and put columns.

| Step | What | Result |
|---|---|---|
| 1 | Clean column names, convert text to numbers/dates | — |
| 2 | Split "bid size x ask size" (e.g. "955 x 596") into numbers (verified: first = bid; when bid = 0 the first number is 0 in 97.9% of rows) | 4 new columns |
| 3 | **Drop 9 holiday snapshots in 2022** (Jan 17, Feb 21, Apr 15, May 30, Jun 20, Jul 4, Sep 5, Nov 24, Dec 26): prices within cents of the previous close → stale copies | **749 trading days** (758 → 749) |
| 4 | One row per option (call and put stacked, `type` column) | 7,099,618 rows |
| 5 | IV not missing | −382,881 → 6,716,737 |
| 6 | Volume > 0 (actually traded that day; untraded prices are provider-filled) | −2,714,730 → 4,002,007 |
| 7 | Bid > 0 | −176,261 → 3,825,746 |
| 8 | Ask > bid (no crossed quotes) | −57,027 → 3,768,719 |
| 9 | Relative spread (ask − bid)/mid ≤ 0.5 | −134,575 → 3,634,144 |
| 10 | 0.01 ≤ IV ≤ 2.0 | −61,006 → 3,573,138 |
| 11 | Days to expiry ≥ 7 | −181,257 → **3,391,881 (47.8% kept)** |
| 12 | **Forward price F** per (date, expiry) from put–call parity: fit `C_mid − P_mid = D·F − D·K` (`np.polyfit`) on near-the-money pairs (|K/S − 1| ≤ 5%, ≥ 3 pairs); F = intercept / D | F/S behaves as theory predicts: < 1 for long maturities in 2020–21 (dividends > rates), > 1 in 2022 (rates rose). D itself was unreliable (median 1.002) and is not used |
| 13 | Drop quotes whose (date, expiry) had no reliable forward | −72,166 (2.1%, mostly long maturities) |
| 14 | Features: `T = dte/365`, `k = ln(K/F)` (log-moneyness), `w = IV²·T` (total variance) | — |
| 15 | **Out-of-the-money only**: puts with k < 0, calls with k ≥ 0; and \|k\| ≤ 1 | **2,017,522 quotes, 749 days, ~2,700 per day (1,251–3,916)** |

**American options:** SPY options are American; SVI theory is European. Assumption: the provider's
IVs are American-model IVs (≈ industry "de-Americanization", Carr & Wu 2010). Test: put−call IV at
the same near-the-money strike ≈ 0 for 2020 Q2–2022 Q2, but +0.3 to +0.5 vol points in 2022 Q3–Q4
(high rates) and −1.9 in 2020 Q1. OTM-only selection minimises the effect; report as a limitation.
Also missing from the raw data: 2020-06-18, 2020-11-27, 2020-12-24, 2022-05-11, 2022-06-17,
2022-06-24, 2022-12-16.

---

## 4. Evaluation design — every component and threshold (the team's strong point)

### 4.1 Test days
Every trading day from **2 March 2020 to 30 December 2022 = 709 test days**. Jan–Feb 2020 serve
only as initial history for the learned models.

### 4.2 Masking schemes (`code/src/masking.py`)
Each mask takes one day's quotes and marks some as hidden. Masks are **seeded by the date**
(`seed = int(YYYYMMDD) + 0`) so **every model sees exactly the same hidden quotes** (verified:
837,112 hidden predictions for every model).

| Mask | Rule | Real-world gap it imitates | Hidden / visible per day (avg) |
|---|---|---|---|
| **Random** | In every expiry, hide round(20% × n) quotes at random; always keep ≥ 5 visible per expiry | scattered untraded strikes inside the surface (interpolation) | 544 / 2,176 |
| **Wing** | In every expiry, sort by k and hide the lowest round(10% × n) and highest round(10% × n); keep ≥ 5 visible | far out-of-the-money strikes with no quote (extrapolation — hardest, most risk-relevant) | 544 / 2,176 |
| **Expiry** | Hide one entire expiry chosen at random among the **interior** ones (never the shortest or longest; day needs ≥ 3 expiries) | a whole maturity with no trades (interpolation across T) | 93 / 2,628 |

709 days × 3 masks = **2,127 out-of-sample tests per model**. Figure `figures/res_masks.png` shows
the three masks on 15 June 2021.

**Sparsity sweep:** the random mask with 10%, 30%, 50%, 70%, 90% hidden, on every 10th test day
(71 days), for linear, SVI, SSVI, XGB-on-SVI, GP, NN (`results/sparsity.csv`). Shows how each model
degrades as data become scarce (the 2026 SPY paper reports SVI breaking down beyond ~70% hidden).
Results in Section 6.

### 4.3 Walk-forward protocol (`code/src/evaluate.py`)
- Per-day models (mean, linear, SVI, SSVI, GP, NN): fitted from scratch on each test day's visible
  quotes; never see other days.
- Learned models (3 × XGBoost): trained **once per month** on all days **before the first day of the
  test month** (`history_for`), then used for every day of that month with today's visible quotes as
  input. 34 monthly retrainings. In March 2020 the history was only Jan–Feb 2020 (41 days).
- Learned-model training rows come from **past days masked the same way** (same three mask types but
  a different seed, 1) → 874,994 training examples (random 403,551; wing 403,432; expiry 68,011).
  Features are computed **only from visible quotes**, never hidden ones.

### 4.4 Metrics (`code/src/metrics.py`)
- **RMSE, MAE, max error** on hidden quotes only, in vol points (×100), computed **per day** then
  averaged over days (also 5%/50%/95% quantiles across days).
- **Regional RMSE**: rows = standardised moneyness z = k/√T in buckets (≤ −0.4, −0.4…−0.1,
  −0.1…0.1, 0.1…0.4, > 0.4) [negative = OTM puts, positive = OTM calls]; columns = maturity
  T ≤ 0.25, 0.25–1, > 1 years.

### 4.5 No-arbitrage checks (`code/src/arbitrage.py`)
For every model on every test day, the fitted surface is evaluated on a grid: 41 log-moneyness
points between the 1st and 99th percentile of that day's k × every maturity of that day.
- **Butterfly**: Gatheral–Jacquier g(k) = (1 − k·w′/(2w))² − (w′²/4)(1/w + 1/4) + w″/2 ≥ 0
  (finite differences in k); violation if g < −1e−6. Ensures a non-negative risk-neutral density.
- **Calendar**: total variance w must not decrease from one maturity to the next at fixed k;
  violation if Δw < −1e−6.
- Reported: violation **rate** (share of grid points) and mean violation **size**.

### 4.6 Statistics (`code/src/stats.py`)
- **Diebold–Mariano test** on daily losses (daily RMSE², i.e. daily MSE) of model vs SVI, with
  Newey–West variance (5 lags) for autocorrelation; two-sided p-value.
- **Moving-block bootstrap** (10-day blocks, 1,000 resamples) for the % improvement in mean RMSE
  over SVI → 95% confidence interval.
- **Win rate**: share of days on which the model beat SVI.

### 4.7 Regimes (`code/src/regimes.py`)
COVID crash 20 Feb – 30 Apr 2020; recovery 1 May – 31 Dec 2020; calm 2021; bear market 2022.

### 4.8 Hedging test (`code/src/hedging.py`)
For each hidden option: Black delta on the forward using the model's IV (and, for comparison, the
market IV); hold the option against −delta SPY for one trading day; error = Δmid − δ·ΔS (same
contract next trading day). Reported: RMS hedging error in $ per option.

### 4.9 Why not k-fold cross-validation?
Answer for the report: walk-forward (expanding-window) out-of-sample evaluation **is** the
time-series form of cross-validation; random k-fold would leak future information across strongly
dependent days. Hyperparameters of XGBoost and the NN were set by hand (sensible defaults), **not
tuned** on a validation set — state this as a limitation; nested walk-forward tuning is future work.

---

## 5. The models — how and why (explain each in depth)

All models share one interface (`code/src/models/base.py`): `fit(visible, history)` then
`predict(k, T)` → IV. Names in results files in brackets.

### Mean — floor [`mean`]
Predicts the day's average visible IV everywhere. Sanity floor: any real model must beat it.

### B0 Linear interpolation [`linear`] — `interp.py`
Delaunay-triangulation linear interpolation of **total variance w** in (k, T) from the visible quotes
(`scipy.griddata`), nearest-neighbour outside the convex hull; IV = √(w/T). *Why:* the simplest
non-parametric baseline — if ML cannot beat this, its complexity is not justified. Interpolating w
(not IV) respects that w grows with T. Weakness: cannot extrapolate (flat beyond the last quote).

### B1 SVI — the industry benchmark [`svi`] — `svi.py`
Raw SVI per expiry: w(k) = a + b·(ρ(k − m) + √((k − m)² + σ²)) — five parameters (level a,
wing slope b, skew ρ, shift m, curvature σ). Fitted by SLSQP least squares on w from 3 starting
points, with bounds and constraints: minimum variance ≥ 0, Roger Lee wing bounds b(1 ± ρ) ≤ 2.
Needs ≥ 5 quotes per expiry. Between expiries: linear interpolation of w in T; outside the maturity
range: flat IV. *Why:* what desks actually use (Gatheral & Jacquier 2014); the Phase 1 benchmark.
Weakness: fixed shape per expiry; expiries fitted independently → slices can cross (calendar
arbitrage, 14% of grid points).

### B2 SSVI [`ssvi`] — `ssvi.py`
Surface SVI: one formula for the whole day, w(k, θ) = θ/2·(1 + ρφk + √((φk + ρ)² + 1 − ρ²)),
φ(θ) = η / (θ^γ (1+θ)^(1−γ)). Shared ρ, η, γ; one ATM total variance θ per expiry, parameterised as
**cumulative positive increments so θ always rises with T** (no calendar arbitrage). Gatheral–
Jacquier butterfly conditions θφ(1+|ρ|) ≤ 4 and θφ²(1+|ρ|) ≤ 4 added as penalty residuals.
Fitted by `least_squares` on IV errors. *Why:* the arbitrage-free version of the benchmark — tests
the accuracy cost of imposing full no-arbitrage. Result: 0% violations but the least accurate model.

### M1a XGBoost direct [`xgb`], M1b XGBoost on linear errors [`xgb_lin`], M2 XGBoost on SVI errors [`xgb_svi`] — `xgb.py`, `ml_features.py`
Gradient-boosted trees (500 trees, depth 6, learning rate 0.05, subsample 0.8, colsample 0.8,
min_child_weight 20, `tree_method='hist'`), **trained monthly on past days** (walk-forward).
Targets: M1a = IV itself; M1b = IV − linear-interpolation estimate; M2 = IV − SVI estimate
(residual learning, as in Audrino & Colangelo 2010). Prediction = baseline + predicted correction
(floored at 0.01).
**14 features, all computed from the day's visible quotes only:** k, T, √T, z = k/√T; ATM IV at the
point's maturity (`atm_T`); ATM IV at 30 days (`atm_30d`, level of the day); term slope (1y − 30d
ATM IV); skew (slope of IV on z for |z| < 1); distance in T to the nearest visible expiry;
whether that expiry is visible; distance in k to the nearest visible quote; how far the point lies
below/above the visible strike range (`extrap_lo`, `extrap_hi`); the linear-interpolation estimate;
(+ SVI estimate for M2). Feature importance of M2 (`results/xgb_svi_feature_importance.csv`,
share of total gain, model trained on all history before Dec 2022): SVI estimate 19%, distance beyond
the highest visible strike 13%, z 12%, linear estimate 10%, distance below the lowest visible strike
8%, distance to nearest visible quote 7%, √T 6%, skew 5%, ATM level 30d 5%, ATM at T 4%, term slope
4%. → The trees mainly learn *how far and in which direction a point lies outside the visible
quotes*, i.e. how smiles continue into the wings — which explains why XGBoost wins the wing test.
*Why:* learns from hundreds of past surfaces how smiles typically continue beyond the last quote —
information a one-day fit does not have. Residual targets let trees focus on the systematic errors of
a good baseline. Weakness: piecewise-constant predictions → bumpy surfaces (16–25% butterfly
violations); needs relevant history (weak in March 2020).

### M3 Gaussian process [`gp`] — `gp.py`
IV = trend + GP residual. Trend: least squares on [1, z, z², √T, z√T] (z = k/√T). GP on the residual
with inputs (k, √T), kernel Constant × Matérn(ν = 5/2, separate length scales) + White noise;
hyperparameters by maximum marginal likelihood each day; at most 800 visible quotes (random subsample,
seeded by date) for speed. Also gives a ±2σ uncertainty band. *Why:* the classic Bayesian
non-parametric smoother for volatility surfaces (De Spiegeleer et al. 2018; Chataigner et al. 2021);
flexible inside the data and honest about uncertainty. Weakness: outside the data it falls back to the
trend → poor wings.

### M4 Neural net + no-arbitrage penalty [`nn`], M4′ without penalty [`nn_nopen`] — `nn.py`
Deep-Smoothing style (Ackerer et al. 2020), fitted per day: **w(k,T) = w_SSVI(k,T) · exp(f_θ(k,T))**,
where w_SSVI is that day's SSVI fit (prior) and f_θ is an MLP (inputs k/0.3 and √T; 2 hidden layers
× 32 tanh units; last layer initialised at zero so training starts exactly at SSVI). Loss = mean
squared IV error in vol points on visible quotes + λ·(butterfly penalty + 10·calendar penalty) on a
grid of 30 strikes × (visible maturities and their midpoints); penalties = mean ReLU(−g) and
mean ReLU(−∂w/∂T) with derivatives by autograd. λ = 1 (M4) or 0 (M4′). Adam, lr 0.005, 300 steps,
float64, seed from date. *Why:* combines a financially sound prior with a flexible correction and
enforces no-arbitrage directly in training. Result: much better wings than SVI, the lowest violation
rates of the flexible models; the penalty barely changes accuracy but lowers violations.

Runtime of the full run (16 parallel workers on a 10-core laptop): mean 11 s, linear 22 s, SVI
32 min, SSVI 81 s, XGB 3.4 min, XGB-lin 3.4 min, XGB-SVI 44 min, GP 54 min, NN 38 min,
NN no penalty 5 min.

---

## 6. Results (main run) — numbers and interpretation

All in `results/tables/`. Mean daily RMSE, vol points (lower = better):

| Model | Random | Expiry | Wing |
|---|---|---|---|
| Mean (floor) | 11.67 | 10.96 | 20.37 |
| B0 Linear | **0.345** | **0.461** | 5.16 |
| B1 SVI | 0.670 | 0.856 | 3.53 |
| B2 SSVI | 1.65 | 1.57 | 4.12 |
| M1a XGB direct | 0.682 | 0.709 | 1.60 |
| M1b XGB on linear | 0.369 | 0.498 | 1.51 |
| M2 XGB on SVI | 0.398 | 0.574 | **1.25** |
| M3 GP | 0.412 | 0.632 | 3.26 |
| M4 NN + penalty | 0.661 | 0.683 | 2.01 |
| M4′ NN no penalty | 0.659 | 0.685 | 1.97 |

**Significance vs SVI** (`significance_vs_svi.csv`): wings — M2 −65% [61, 68], p < 0.001, wins 99%
of days; M1b −57%; M1a −55% (p = 0.04); NN −43% [39, 47], p < 0.001, wins 96%; GP −8% (n.s.).
Random — linear −48%, M1b −45%, M2 −41%, GP −38% (all p < 0.001); NN ≈ SVI (−1%, n.s.).
Expiry — linear −46%, M1b −42%, M2 −33%, GP −26%, NN −20%: bootstrap CIs exclude zero but DM
p-values > 0.05 because a few crisis days dominate squared errors; win rates 60–88%.

**Regimes** (`regime_*.csv`), wing mask: COVID crash — M2 2.99 vs SVI 4.36 (best), NN 3.51,
M1a 7.39 (worse than SVI!); 2021 — M2 0.74 vs SVI 2.96; 2022 — M1a 1.35 best, M2 1.56, SVI 3.93.
Random mask in the crash: linear 0.59 best, GP 0.67, SVI 1.23, M2 1.35, M1a 4.31.
*Interpretation:* learned models trained on only two calm months failed to anticipate crash-shaped
surfaces in March 2020; per-day models adapt instantly. Recommend combining both or retraining more
often in turbulent markets.

**Regional** (`per_model/*/regional_wing.csv`, `figures/res_regional.png`): wing errors largest for
deep OTM calls (z > 0.4) at short/medium maturities (SVI 5.9–6.2) — XGB-on-SVI cuts these to 1.0–1.7.

**Arbitrage** (`arbitrage.csv`), random mask, share of grid points violating:
SSVI 0% / 0%; NN 0.6% butterfly / 3.8% calendar; NN no penalty 1.0% / 4.0%; SVI 1.1% / 14.3%;
GP 6.0% / 4.9%; linear 15.6% / 1.9%; M2 16.3% / 8.5%; M1b 21.5% / 9.4%; M1a 25.0% / 5.7%.
*Interpretation:* accuracy–consistency trade-off; trees are step-shaped; NN with penalty is the best
compromise among flexible models; SVI's calendar violations come from independent expiry fits.

**Hedging** (`hedging.csv`): wing options, RMS one-day hedge error ≈ $0.233–0.240 for every model vs
$0.233 using the market IV and $0.396 unhedged → every model is within ~3% of the true-IV hedge.
Delta is insensitive to small IV errors; accuracy gains matter for pricing far strikes and risk
measures, not one-day delta hedging.

**Quantiles across days** (`rmse_quantiles.csv`): e.g. wing median M2 1.00 vs SVI 3.29; 95th
percentile M2 2.53 vs SVI 5.58.

**Sparsity** (`tables/sparsity.csv`, random mask on every 10th test day), mean RMSE:

| Hidden | GP | Linear | NN | SSVI | SVI | XGB on SVI |
|---|---|---|---|---|---|---|
| 10% | 0.43 | **0.35** | 0.65 | 1.69 | 0.67 | 0.41 |
| 30% | 0.43 | **0.39** | 0.66 | 1.69 | 0.70 | 0.43 |
| 50% | **0.42** | 0.46 | 0.68 | 1.67 | 0.75 | 0.46 |
| 70% | **0.47** | 0.66 | 0.72 | 1.70 | 0.92 | 0.53 |
| 90% | **0.68** | 1.24 | 0.90 | 1.78 | 1.75 | 0.94 |

*Interpretation:* the ranking flips as data get scarce. With many quotes, linear interpolation wins;
from 50% hidden the **GP is best** and degrades least (0.43 → 0.68). SVI (0.67 → 1.75, ×2.6) and
linear (×3.5) degrade fastest; SSVI barely changes (rigid, so robust but always inaccurate); NN and
XGB-on-SVI stay in between. Practical message: the best method depends on how sparse the surface is —
for illiquid underlyings or thin days, flexible-but-regularised models (GP) are preferable to SVI.

**Overall answer:** ML (especially XGBoost correcting SVI, and the penalised NN) significantly beats
SVI where data are missing *beyond* the observed range; for gaps *inside* the observed range simple
interpolation is already better than SVI and ML adds little. The extra complexity of ML is justified
for extrapolation and when combined with a sound baseline (residual learning / SSVI prior).

---

## 7. The independent run (her run) — HOW IT DIFFERS (do not mix the numbers)

A team member independently ran the GP and NN (`context/independent_GP_NN_run_results.docx`).
**Same model code (gp.py, nn.py) but different data and different tests**, so its numbers are
**not comparable** with Section 6 and must be presented separately as a **robustness check**.

| Item | Main run (Section 6) | Independent run |
|---|---|---|
| Trading days | 749 (9 holiday snapshots removed) | 758 (holidays kept) |
| Usable quotes | 2,017,522 | 2,446,808 |
| Days to expiry | ≥ 7 (no upper limit; \|k\| ≤ 1) | 7 – 730 |
| IV range | 0.01 – 2.0 | 0.03 – 1.50 |
| Liquidity filters | volume > 0, bid > 0, ask > bid, spread ≤ 50% of mid | valid bids, spread ≤ half the mid, plus limits on very cheap and very far-out options (volume rule not stated) |
| Forward / moneyness | put–call-parity forward, OTM by k = ln(K/F) | not stated in her report |
| Random mask | 20% of quotes per expiry | 30% of quotes |
| Expiry mask | 1 whole interior expiry | 30% of expiry dates |
| Wing mask | outer 10% each side per expiry | outer 15% each side per expiry |
| Masking / base / SSVI code | original `masking.py`, `base.py`, `ssvi.py` | replacement files (she notes numbers would change with the originals) |
| Test period | Mar 2020 – Dec 2022 (709 days) | all 758 days incl. Jan–Feb 2020 |
| Extra metrics | arbitrage, significance, regimes, hedging | bias, pooled RMSE, SSVI-start and trend-start baselines, monthly table |

Her results (mean daily RMSE): Random — GP 0.30, NN 0.59, NN no penalty 0.59, SSVI 1.26;
Expiries — GP 0.89, NN 0.77, SSVI 1.41; Wings — GP 2.36, NN 1.66, SSVI 2.80. NN beat its SSVI
start on > 97% of days in every test; penalty changed accuracy by < 0.01; both models biased low in
the wings (NN −0.43, GP −0.65 points); NN's wing lead faded in late 2022.
**Same qualitative pattern as the main run** (GP best for scattered gaps, NN better at wings and
missing expiries, penalty ≈ no accuracy effect) under a harsher test → robustness evidence.

---

## 8. Things to correct/keep consistent with Phase 1

- Phase 1 said "757 trading days" and "thousands of past surfaces": final numbers are **749 days**
  after removing holiday snapshots and **709 test days**; learned models trained on up to ~700 days.
- Phase 1 literature slide 25 had a broken link; correct link:
  https://papers.nips.cc/paper/2020/hash/858e47701162578e5e627cd93ab0938a-Abstract.html
- Phase 1 planned XGBoost, GP and NN with no-arbitrage — all delivered, plus SSVI, linear and three
  XGBoost variants, hedging, significance tests. Phase 1 promised "results by period" — delivered.

## 9. Limitations to state
Provider-computed IVs and the late-2022 put/call IV gap (American exercise); one underlying over three
turbulent years; hand-set hyperparameters (no validation tuning); one random mask draw per day;
learned models had only 41 days of history in March 2020; arbitrage checked on a finite grid;
hedging test ignores discounting and transaction costs.

## 10. Files in this package

| Path | Content |
|---|---|
| `context/Phase1_presentation.pdf` | Phase 1 deck: problem statement, data, EDA, literature (10 papers) |
| `context/assignment_guidelines.pdf` | deliverables and literature-page format |
| `context/Final_Report_DRAFT.docx` | current draft report (expand it) |
| `context/independent_GP_NN_run_results.docx` | her independent GP/NN run |
| `results/tables/*.csv` | every table: main RMSE/MAE, quantiles, regimes, significance, arbitrage, hedging, sparsity, test design |
| `results/all_daily_scores.csv` | one row per model × mask × day (RMSE, MAE, max error, arbitrage rates, regime) — for any new plot |
| `results/per_model/<model>/` | summary by mask/regime and regional error grids per model |
| `results/predictions_key_days.csv` | every hidden quote's true and predicted IV, all models, on 6 key days (2020-03-12/16/17, 2021-06-15, 2022-06-15, 2022-10-14) |
| `results/quotes_key_days.csv` | all quotes (k, T, IV, prices) on those days — for smile/mask illustrations |
| `results/daily_market_summary.csv` | SPY close, quote count, expiries, 1-month ATM IV per day |
| `results/xgb_svi_feature_importance.csv` | XGBoost (M2) feature importance (gain share) |
| `results/sparsity.csv`, `results/run_log.txt` | sparsity sweep; run log with timings |
| `figures/` | Phase 1 EDA figures (`fig_*`) and result figures (`res_*`) |
| `code/` | full model and evaluation code (reference for explaining implementation), config, README |

To make new figures, use the CSVs above with pandas/matplotlib. The scripts in `code/report/` need
the full dataset and are included only as a reference for how the existing figures were made.

## 11. Report structure suggested
1. Summary · 2. Motivation and problem (why surfaces matter; sparsity; SVI) · 3. Why this is not a
standard ML problem (Section 2) · 4. Data and pipeline (Section 3 with filter log) · 5. Evaluation
design (Section 4 — masks with figure, walk-forward, metrics, arbitrage, statistics, regimes) ·
6. Models in depth (Section 5, one subsection each: how, why, strengths, weaknesses) · 7. Results
(accuracy, significance, regimes, regional, arbitrage, hedging, sparsity) · 8. Robustness: independent
run (Section 7) · 9. Answer to the research question, limitations, future work · 10. Code and
reproducibility · 11. Statement of contribution (**ask the team; never invent**) · References ·
Appendix A: literature (one page per paper, guideline format; content in the draft and Phase 1 deck) ·
Appendix B: model settings.
