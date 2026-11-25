"""Run every model on every test day and mask, then the sparsity sweep.

Usage (from the project folder):   python -m src.run_all
Results go to results/<model>/. Models whose results already include the arbitrage
columns are skipped, so the script can be stopped and restarted.
"""
import os
import sys
import time
from functools import partial
import numpy as np
import pandas as pd

from src.experiment import run_experiment
from src.evaluate import evaluate, make_factories
from src.masking import random_mask
from src.ml_features import build_training_table
from src.models.dummy import MeanModel
from src.models.interp import LinearInterpModel
from src.models.svi import SVIModel
from src.models.ssvi import SSVIModel
from src.models import xgb as xgb_module
from src.models.xgb import XGBModel, XGBLinearModel, XGBSVIModel
from src.models.gp import GPModel
from src.models.nn import NNModel, NNNoPenaltyModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
FIRST_TEST_DAY = '2020-03-01'
MODELS = [MeanModel, LinearInterpModel, SVIModel, SSVIModel,
          XGBModel, XGBLinearModel, XGBSVIModel, GPModel, NNModel, NNNoPenaltyModel]
SPARSITY_MODELS = [LinearInterpModel, SVIModel, SSVIModel, XGBSVIModel, GPModel, NNModel]
SPARSITY_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(os.path.join(RESULTS, 'run_log.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def done(name):
    path = os.path.join(RESULTS, name, 'scores.csv')
    return os.path.exists(path) and 'butterfly_rate' in pd.read_csv(path, nrows=1).columns

def main():
    os.makedirs(RESULTS, exist_ok=True)
    df = pd.read_parquet(os.path.join(ROOT, 'data', 'processed', 'options_features.parquet'))
    days = np.sort(df.loc[df['quote_date'] >= FIRST_TEST_DAY, 'quote_date'].unique())
    log(f'data {df.shape}, {len(days)} test days')

    if not os.path.exists(xgb_module.TABLE_PATH):
        t = time.time()
        tab = build_training_table(df, xgb_module.TABLE_PATH)
        log(f'training table {tab.shape} built in {time.time() - t:.0f}s')

    for M in MODELS:
        if done(M.name):
            log(f'{M.name}: already done, skipped')
            continue
        t = time.time()
        out = run_experiment(M, df, days, out_dir=RESULTS)
        log(f"{M.name}: {time.time() - t:.0f}s  rmse by mask "
            f"{out['by_mask']['rmse'].round(2).to_dict()}")

    # sparsity sweep: random mask with more and more hidden, every 10th test day
    path = os.path.join(RESULTS, 'sparsity.csv')
    sweep = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    sub_days = days[::10]
    for M in SPARSITY_MODELS:
        if len(sweep) and (sweep['model'] == M.name).any():
            continue
        t = time.time()
        factories = make_factories(M, df, sub_days)
        for frac in SPARSITY_FRACS:
            s = evaluate(M, df, sub_days, partial(random_mask, frac=frac, seed=0),
                         f'random_{frac}', arbitrage=False, factories=factories)
            s['frac'] = frac
            sweep = pd.concat([sweep, s], ignore_index=True)
        sweep.to_csv(path, index=False)
        log(f'sparsity {M.name}: {time.time() - t:.0f}s')
    log('all done')

if __name__ == '__main__':
    main()
