import os
from functools import partial
import pandas as pd
from src.evaluate import evaluate
from src.metrics import regional_rmse, rmse
from src.masking import random_mask, wing_mask, expiry_mask

DEFAULT_MASKS = {
    'random': partial(random_mask, seed=0),
    'wing':   wing_mask,
    'expiry': partial(expiry_mask, seed=0),
}

def run_experiment(model_class, data, test_days, masks=DEFAULT_MASKS,
                   out_dir='../results', use_history=False, n_jobs=-1):
    """Run one model on every mask, save everything to out_dir/<model>/, return a summary."""
    name = model_class.name
    path = os.path.join(out_dir, name)
    os.makedirs(path, exist_ok=True)

    all_scores, all_preds = [], []
    for mask_name, fn in masks.items():
        s, p = evaluate(model_class, data, test_days, fn, mask_name,
                        n_jobs=n_jobs, use_history=use_history, return_preds=True)
        all_scores.append(s); all_preds.append(p)
    scores = pd.concat(all_scores, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    # summaries
    by_mask = scores.groupby('mask')[['rmse', 'mae', 'max_error']].mean()
    by_regime = scores.pivot_table(index='regime', columns='mask', values='rmse', aggfunc='mean')
    pooled = preds.groupby('mask').apply(lambda g: rmse(g['iv_true'], g['iv_pred']))
    regional = {m: regional_rmse(g)[0] for m, g in preds.groupby('mask')}

    # save
    scores.to_csv(os.path.join(path, 'scores.csv'), index=False)
    preds.to_parquet(os.path.join(path, 'preds.parquet'), index=False)
    by_mask.to_csv(os.path.join(path, 'summary_by_mask.csv'))
    by_regime.to_csv(os.path.join(path, 'summary_by_regime.csv'))
    for m, tab in regional.items():
        tab.to_csv(os.path.join(path, f'regional_{m}.csv'))

    return {'scores': scores, 'preds': preds, 'by_mask': by_mask,
            'by_regime': by_regime, 'pooled_rmse': pooled, 'regional': regional}