"""Build every table (report/tables/*.csv) and figure (report/figures/*.png) used in the report
from the saved results. Run from the project folder:  python report/build_figures.py
Models that have not finished yet are simply left out, so the script can be re-run later."""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
warnings.filterwarnings('ignore')
from src.stats import compare_to_benchmark
from src.hedging import hedge_errors, hedge_summary

RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'report', 'figures')
TAB = os.path.join(ROOT, 'report', 'tables')
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

ORDER = ['mean', 'linear', 'svi', 'ssvi', 'xgb', 'xgb_lin', 'xgb_svi', 'gp', 'nn', 'nn_nopen']
LABEL = {'mean': 'Mean (floor)', 'linear': 'B0 Linear interp.', 'svi': 'B1 SVI', 'ssvi': 'B2 SSVI',
         'xgb': 'M1a XGB direct', 'xgb_lin': 'M1b XGB on linear', 'xgb_svi': 'M2 XGB on SVI',
         'gp': 'M3 Gaussian process', 'nn': 'M4 NN + penalty', 'nn_nopen': "M4' NN no penalty"}
MASKS = ['random', 'expiry', 'wing']
REGIMES = ['covid_crash', 'recovery', 'calm_2021', 'bear_2022']
MODELS = [m for m in ORDER if os.path.exists(os.path.join(RES, m, 'scores.csv'))]
COLORS = dict(zip(ORDER, plt.cm.tab10.colors))

scores = pd.concat([pd.read_csv(os.path.join(RES, m, 'scores.csv'), parse_dates=['date'])
                    for m in MODELS], ignore_index=True)
scores = scores[scores['date'] >= '2020-03-01']
print('models:', MODELS)

def save(df, name):
    df.to_csv(os.path.join(TAB, name + '.csv'))

# ---------------- tables ----------------
main = scores.pivot_table(index='model', columns='mask', values='rmse', aggfunc='mean').loc[MODELS, MASKS]
save(main, 'main_rmse')
mae = scores.pivot_table(index='model', columns='mask', values='mae', aggfunc='mean').loc[MODELS, MASKS]
save(mae, 'main_mae')
q = scores.groupby(['model', 'mask'])['rmse'].quantile([0.05, 0.5, 0.95]).unstack()
q.columns = ['q05', 'median', 'q95']
save(q.loc[MODELS], 'rmse_quantiles')
for mask in MASKS:
    t = scores[scores['mask'] == mask].pivot_table(index='model', columns='regime',
                                                   values='rmse', aggfunc='mean')
    save(t.loc[MODELS, REGIMES], f'regime_{mask}')
sig = compare_to_benchmark(scores, benchmark='svi', n_boot=1000)
save(sig.set_index(['mask', 'model']), 'significance_vs_svi')
arb = scores[scores['mask'] == 'random'].groupby('model')[
    ['rmse', 'butterfly_rate', 'calendar_rate', 'butterfly_size', 'calendar_size']].mean().loc[MODELS]
save(arb, 'arbitrage')

counts = scores.groupby('mask').agg(days=('date', 'nunique'))
counts['hidden_per_day'] = scores.groupby('mask')['n_hidden'].mean()
counts['visible_per_day'] = scores.groupby('mask')['n_visible'].mean()
save(counts, 'test_design')

# hedging (one-day delta hedge), all masks pooled per model
data = pd.read_parquet(os.path.join(ROOT, 'data', 'processed', 'options_features.parquet'),
                       columns=['quote_date', 'expire_date', 'strike', 'type', 'mid', 'F',
                                'underlying_last'])
rows = {}
for m in [m for m in ['linear', 'svi', 'ssvi', 'xgb_svi', 'gp', 'nn'] if m in MODELS]:
    p = pd.read_parquet(os.path.join(RES, m, 'preds.parquet'))
    rows[m] = hedge_summary(hedge_errors(p, data))
if rows:
    save(pd.concat(rows), 'hedging')

# ---------------- figures ----------------
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
shown = [m for m in MODELS if m != 'mean']

# 1. main bar chart
fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(MASKS)); wbar = 0.8 / len(shown)
for i, m in enumerate(shown):
    ax.bar(x + i * wbar - 0.4 + wbar / 2, main.loc[m], wbar, label=LABEL[m], color=COLORS[m])
ax.set_xticks(x); ax.set_xticklabels(['Random 20% hidden', 'Whole expiry hidden', 'Wings hidden'])
ax.set_ylabel('Mean daily RMSE (vol points)'); ax.legend(ncol=3, fontsize=8, frameon=False)
ax.set_title('Reconstruction error on hidden quotes, Mar 2020 – Dec 2022 (709 days)')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'res_main_rmse.png'), dpi=160); plt.close()

# 2. rolling error through time
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for ax, mask in zip(axes, ['random', 'wing']):
    for m in [m for m in ['linear', 'svi', 'xgb_svi', 'gp', 'nn'] if m in MODELS]:
        s = scores[(scores['model'] == m) & (scores['mask'] == mask)].set_index('date')['rmse']
        ax.plot(s.sort_index().rolling(21, min_periods=5).mean(), label=LABEL[m], color=COLORS[m], lw=1.3)
    ax.axvspan(pd.Timestamp('2020-02-20'), pd.Timestamp('2020-04-30'), color='grey', alpha=0.2)
    ax.set_title(f'{mask} mask — 21-day rolling mean RMSE (grey = COVID crash)')
    ax.set_ylabel('vol points')
axes[0].legend(ncol=3, fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'res_rolling.png'), dpi=160); plt.close()

# 3. regional heatmaps, wing mask
import seaborn as sns
pair = [m for m in ['svi', 'xgb_svi'] if m in MODELS]
fig, axes = plt.subplots(1, len(pair), figsize=(10, 3.8))
for ax, m in zip(np.atleast_1d(axes), pair):
    tab = pd.read_csv(os.path.join(RES, m, 'regional_wing.csv'), index_col=0)
    tab.columns = ['T ≤ 0.25', '0.25–1', 'T > 1']
    tab.index = ['z ≤ −0.4', '−0.4 to −0.1', '−0.1 to 0.1', '0.1 to 0.4', 'z > 0.4']
    sns.heatmap(tab, annot=True, fmt='.1f', cmap='Reds', vmin=0, vmax=10, ax=ax, cbar=False)
    ax.set_title(f'{LABEL[m]} — wing mask RMSE'); ax.set_ylabel('moneyness z = k/√T')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'res_regional.png'), dpi=160); plt.close()

# 4. accuracy vs arbitrage
a = arb.loc[shown].copy()
a['viol'] = 100 * (a['butterfly_rate'] + a['calendar_rate'])
fig, ax = plt.subplots(figsize=(7, 4.5))
for m, r in a.iterrows():
    ax.scatter(r['viol'], r['rmse'], color=COLORS[m], s=50)
    ax.annotate(LABEL[m], (r['viol'], r['rmse']), fontsize=8, xytext=(5, 3), textcoords='offset points')
ax.set_xlabel('Grid points violating butterfly or calendar no-arbitrage (%)')
ax.set_ylabel('Mean RMSE, random mask (vol points)')
ax.set_title('Accuracy vs. no-arbitrage (lower-left is better)')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'res_arbitrage.png'), dpi=160); plt.close()

# 5. crisis day close-up
DAY = pd.Timestamp('2020-03-16')
show = [m for m in ['svi', 'gp', 'xgb_svi', 'nn'] if m in MODELS]
pr = {m: pd.read_parquet(os.path.join(RES, m, 'preds.parquet'),
                         filters=[('date', '==', DAY), ('mask', '==', 'wing')]) for m in show}
p0 = pr[show[0]]
exp = np.sort(p0['expire_date'].unique())
picks = exp[np.linspace(0, len(exp) - 1, 3).astype(int)]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, e in zip(axes, picks):
    q0 = p0[p0['expire_date'] == e]
    ax.scatter(q0['k'], q0['iv_true'], c='black', s=18, marker='x', label='market (hidden)', zorder=5)
    for m in show:
        qq = pr[m][pr[m]['expire_date'] == e].sort_values('k')
        ax.plot(qq['k'], qq['iv_pred'], 'o', ms=3, label=LABEL[m], color=COLORS[m])
    ax.set_title(f'Expiry {pd.Timestamp(e).date()}', fontsize=9); ax.set_xlabel('log-moneyness k')
axes[0].set_ylabel('implied volatility'); axes[0].legend(fontsize=7, frameon=False)
fig.suptitle('16 March 2020 (SPY −11%): predictions for the hidden wing quotes', fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'res_crisis_day.png'), dpi=160); plt.close()

# 6. what each mask hides, on one real day
from src.masking import random_mask, wing_mask, expiry_mask
mday = pd.read_parquet(os.path.join(ROOT, 'data', 'processed', 'options_features.parquet'),
                       filters=[('quote_date', '==', pd.Timestamp('2021-06-15'))])
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, (title, m) in zip(axes, [('Random: 20% of quotes in every expiry', random_mask(mday, seed=0)),
                                 ('Wings: outer 10% each side of every expiry', wing_mask(mday)),
                                 ('Expiry: one whole interior expiry', expiry_mask(mday, seed=0))]):
    ax.scatter(mday.loc[~m, 'k'], mday.loc[~m, 'T'], s=3, c='#9aa5b1', label='visible (model input)')
    ax.scatter(mday.loc[m, 'k'], mday.loc[m, 'T'], s=6, c='#d62728', label='hidden (scored)')
    ax.set_title(title, fontsize=10); ax.set_xlabel('log-moneyness k = ln(K/F)')
axes[0].set_ylabel('maturity T (years)'); axes[0].legend(fontsize=8, markerscale=3, frameon=False)
fig.suptitle('The three masking schemes on 15 June 2021 (each dot = one option quote)', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'res_masks.png'), dpi=160); plt.close()

# 7. sparsity sweep (if finished)
sp_path = os.path.join(RES, 'sparsity.csv')
if os.path.exists(sp_path):
    sp = pd.read_csv(sp_path)
    curve = sp.groupby(['model', 'frac'])['rmse'].mean().unstack('model')
    save(curve, 'sparsity')
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in curve.columns:
        ax.plot(curve.index * 100, curve[m], 'o-', label=LABEL.get(m, m), color=COLORS.get(m))
    ax.set_yscale('log'); ax.set_xlabel('Share of quotes hidden (%)')
    ax.set_ylabel('Mean RMSE (vol points, log scale)'); ax.legend(fontsize=8, frameon=False)
    ax.set_title('Sparsity sweep (every 10th test day)')
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'res_sparsity.png'), dpi=160); plt.close()

print('tables:', sorted(os.listdir(TAB)))
print('figures:', sorted(os.listdir(FIG)))
