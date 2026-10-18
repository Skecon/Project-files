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

## Working together

- Create a branch for your work (`git checkout -b your-feature`), commit there, and open a pull
  request for someone else to review before merging into `main`.
- Keep reusable code in `src/`; notebooks should only import from it.
- Clear notebook outputs before committing.
