# Machine Learning for Implied Volatility Surface Modeling

HST-105 Analytics for Business and Society, IIT Roorkee — Aryan, Shachi, Shreekrushna

**Research question:** can machine learning reconstruct the implied volatility surface more
accurately than the industry-standard SVI model, and does any gain hold up during crisis periods?

**Data:** SPY end-of-day option chains, January 2020 – December 2022. See [data/README.md](data/README.md).

## Approach

1. Clean the option chains and keep traded, out-of-the-money quotes.
2. Compute forward prices, log-moneyness `k`, maturity `T` and total variance `w = σ²T`.
3. On each test day, hide part of the observed quotes (random, wing and whole-expiry masks)
   and ask each model to reconstruct them.
4. Compare models with a walk-forward (time-ordered) split:
   - Baselines: linear interpolation, SVI (per expiry), SSVI
   - Machine learning: XGBoost (direct and on SVI residuals), Gaussian process, neural network
     with no-arbitrage penalty
5. Score on reconstruction error, no-arbitrage violations and hedging performance, by market regime.

## Repository layout

```
config.yaml        settings shared by all scripts (filters, masking, splits, seeds)
data/              raw and processed data (not committed)
notebooks/         exploration and figures
src/               cleaning, features, masking, models, evaluation
src/models/        one file per model, all with the same fit/predict interface
results/           model outputs (not committed)
report/            slides and figures
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then place the data file as described in [data/README.md](data/README.md).

## Reproducing every result

Run from this folder, in order:

| Step | Command | Output | Time (10-core laptop) |
|---|---|---|---|
| 1. Clean and filter | run `notebooks/01_Cleaning.ipynb`, then `02_Filters.ipynb` | `data/processed/options_filtered.parquet` | ~5 min |
| 2. Features | run `notebooks/03_Features.ipynb` | `data/processed/options_features.parquet` | ~2 min |
| 3. All models | `python -m src.run_all` | `results/<model>/`, `results/sparsity.csv` | ~5 h (resumable) |
| 4. Figures and tables | `python report/build_figures.py` | `report/figures/`, `report/tables/` | ~3 min |
| 5. Report | `python report/make_report.py` | `report/Final_Report.docx` | seconds |

`notebooks/04_Evaluation.ipynb` builds and tests the evaluation framework, `05_Models.ipynb`
demonstrates every model on the 16 March 2020 crash day, and `06_Analysis.ipynb` contains the full
analysis of the saved results.

Step 3 first builds `data/processed/ml_train_table.parquet` (masked training examples for the
XGBoost models, ~10 min) and skips any model whose results already exist.

## Code map

| File | Purpose |
|---|---|
| `src/masking.py` | random, wing and whole-expiry masks (seeded by date) |
| `src/evaluate.py` | walk-forward evaluation loop; monthly training of learned models |
| `src/experiment.py` | runs one model on all masks and saves results |
| `src/run_all.py` | runs every model and the sparsity sweep |
| `src/metrics.py` | RMSE, MAE, max error, regional RMSE (vol points) |
| `src/arbitrage.py` | butterfly and calendar no-arbitrage checks |
| `src/stats.py` | Diebold–Mariano test, block bootstrap, comparison against SVI |
| `src/hedging.py` | one-day delta-hedging test |
| `src/ml_features.py` | features for learned models, computed from visible quotes only |
| `src/regimes.py` | market regimes (COVID crash, recovery, 2021, 2022) |
| `src/models/` | `dummy.py` (mean), `interp.py` (linear), `svi.py`, `ssvi.py`, `xgb.py` (3 variants), `gp.py`, `nn.py` (with and without penalty) |

## Working together

- Create a branch for your work (`git checkout -b your-feature`), commit there, and open a pull
  request for someone else to review before merging into `main`.
- Keep reusable code in `src/`; notebooks should only import from it.
- Clear notebook outputs before committing.
