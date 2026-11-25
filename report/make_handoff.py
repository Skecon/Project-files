"""Assemble the report hand-off package (everything needed to write the report, no raw dataset).

Run from the project folder:  python report/make_handoff.py
Creates  ../Report_Handoff/  and  ../Report_Handoff.zip  (next to the Project files folder)."""
import os
import shutil
import sys
import zipfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.dirname(ROOT)
OUT = os.path.join(MAIN, 'Report_Handoff')
sys.path.append(ROOT)

MODELS = ['mean', 'linear', 'svi', 'ssvi', 'xgb', 'xgb_lin', 'xgb_svi', 'gp', 'nn', 'nn_nopen']
KEY_DAYS = ['2020-03-12', '2020-03-16', '2020-03-17', '2021-06-15', '2022-06-15', '2022-10-14']

def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)

# ---- context documents
ctx = os.path.join(OUT, 'context')
cp(os.path.join(MAIN, 'HST Assignment Phase I.pdf'), os.path.join(ctx, 'Phase1_presentation.pdf'))
cp(os.path.join(MAIN, 'open topics & detailed guidelines.pdf'), os.path.join(ctx, 'assignment_guidelines.pdf'))
drafts = [os.path.join(ROOT, 'report', f) for f in os.listdir(os.path.join(ROOT, 'report'))
          if f.startswith('Final_Report') and f.endswith('.docx')]
cp(max(drafts, key=os.path.getmtime), os.path.join(ctx, 'Final_Report_DRAFT.docx'))
friend = os.path.join(os.path.expanduser('~'), 'Downloads', 'GP_and_NN_Results.docx')
if os.path.exists(friend):
    cp(friend, os.path.join(ctx, 'independent_GP_NN_run_results.docx'))

# ---- results: tables, figures, per-model summaries, daily scores
res = os.path.join(OUT, 'results')
for f in os.listdir(os.path.join(ROOT, 'report', 'tables')):
    cp(os.path.join(ROOT, 'report', 'tables', f), os.path.join(res, 'tables', f))
for f in os.listdir(os.path.join(ROOT, 'report', 'figures')):
    cp(os.path.join(ROOT, 'report', 'figures', f), os.path.join(OUT, 'figures', f))
for m in MODELS:
    d = os.path.join(ROOT, 'results', m)
    for f in os.listdir(d):
        if f.endswith('.csv') and f != 'scores.csv':
            cp(os.path.join(d, f), os.path.join(res, 'per_model', m, f))
scores = pd.concat([pd.read_csv(os.path.join(ROOT, 'results', m, 'scores.csv')) for m in MODELS],
                   ignore_index=True)
scores.to_csv(os.path.join(res, 'all_daily_scores.csv'), index=False)
for f in ['sparsity.csv', 'run_log.txt']:
    if os.path.exists(os.path.join(ROOT, 'results', f)):
        cp(os.path.join(ROOT, 'results', f), os.path.join(res, f))

# ---- small plotting data: predictions and quotes on a few key days
days = [pd.Timestamp(d) for d in KEY_DAYS]
preds = pd.concat([pd.read_parquet(os.path.join(ROOT, 'results', m, 'preds.parquet'),
                                   filters=[('date', 'in', days)]) for m in MODELS], ignore_index=True)
preds.to_csv(os.path.join(res, 'predictions_key_days.csv'), index=False)
cols = ['quote_date', 'expire_date', 'strike', 'type', 'underlying_last', 'F', 'dte', 'T', 'k', 'w',
        'iv', 'bid', 'ask', 'mid', 'volume', 'relative_spread']
quotes = pd.read_parquet(os.path.join(ROOT, 'data', 'processed', 'options_features.parquet'),
                         columns=cols, filters=[('quote_date', 'in', days)])
quotes.to_csv(os.path.join(res, 'quotes_key_days.csv'), index=False)

# daily market summary (ATM IV level, number of quotes) for time-series plots
feat = pd.read_parquet(os.path.join(ROOT, 'data', 'processed', 'options_features.parquet'),
                       columns=['quote_date', 'k', 'T', 'iv', 'underlying_last'])
near = feat[(feat['k'].abs() < 0.02) & (feat['T'].between(20 / 365, 45 / 365))]
daily = pd.DataFrame({'spy_close': feat.groupby('quote_date')['underlying_last'].first(),
                      'n_quotes': feat.groupby('quote_date').size(),
                      'n_expiries': feat.groupby('quote_date')['T'].nunique(),
                      'atm_iv_1m': near.groupby('quote_date')['iv'].median()})
daily.to_csv(os.path.join(res, 'daily_market_summary.csv'))

# XGBoost feature importance (model trained on all history before Dec 2022)
from src.evaluate import history_for
from src.models.xgb import XGBSVIModel
full = pd.read_parquet(os.path.join(ROOT, 'data', 'processed', 'options_features.parquet'),
                       columns=['quote_date'])
booster = XGBSVIModel.train(history_for(full, '2022-12-01'))
imp = pd.Series(booster.get_booster().get_score(importance_type='gain'), name='gain')
(imp / imp.sum()).sort_values(ascending=False).to_csv(os.path.join(res, 'xgb_svi_feature_importance.csv'))

# ---- code (reference: how each model is implemented)
code = os.path.join(OUT, 'code')
for sub in ['src', os.path.join('src', 'models')]:
    for f in os.listdir(os.path.join(ROOT, sub)):
        if f.endswith('.py'):
            cp(os.path.join(ROOT, sub, f), os.path.join(code, sub, f))
for f in ['config.yaml', 'README.md', 'requirements.txt']:
    cp(os.path.join(ROOT, f), os.path.join(code, f))
for f in ['build_figures.py', 'make_report.py']:
    cp(os.path.join(ROOT, 'report', f), os.path.join(code, 'report', f))

# ---- CLAUDE.md
cp(os.path.join(ROOT, 'report', 'HANDOFF_CLAUDE.md'), os.path.join(OUT, 'CLAUDE.md'))

# ---- zip
zpath = OUT + '.zip'
if os.path.exists(zpath):
    os.remove(zpath)
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for dirpath, _, files in os.walk(OUT):
        for f in files:
            full_path = os.path.join(dirpath, f)
            z.write(full_path, os.path.join('Report_Handoff', os.path.relpath(full_path, OUT)))
total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs)
print(f'{OUT}: {total / 1e6:.1f} MB unzipped; zip {os.path.getsize(zpath) / 1e6:.1f} MB')
