"""Build the final report (report/Final_Report.docx) from report/tables and report/figures.
Run after report/build_figures.py:   python report/make_report.py"""
import os
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, RGBColor, Cm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'report', 'tables')
FIG = os.path.join(ROOT, 'report', 'figures')
OUT = os.path.join(ROOT, 'report', 'Final_Report.docx')

LABEL = {'mean': 'Mean (floor)', 'linear': 'B0 Linear interpolation', 'svi': 'B1 SVI (benchmark)',
         'ssvi': 'B2 SSVI', 'xgb': 'M1a XGBoost, direct', 'xgb_lin': 'M1b XGBoost on linear errors',
         'xgb_svi': 'M2 XGBoost on SVI errors', 'gp': 'M3 Gaussian process',
         'nn': 'M4 Neural net + no-arbitrage penalty', 'nn_nopen': "M4' Neural net, no penalty"}
REGIME = {'covid_crash': 'COVID crash', 'recovery': 'Recovery 2020', 'calm_2021': 'Calm 2021',
          'bear_2022': 'Bear 2022'}
MASK = {'random': 'Random', 'expiry': 'Expiry', 'wing': 'Wings'}

def tab(name, **kw):
    return pd.read_csv(os.path.join(TAB, name + '.csv'), **kw)

main = tab('main_rmse', index_col=0)
HAS_NN = 'nn' in main.index
sig = tab('significance_vs_svi')
arb = tab('arbitrage', index_col=0)
design = tab('test_design', index_col=0)
s = lambda mask, model, col: sig[(sig['mask'] == mask) & (sig['model'] == model)][col].iloc[0]

# ------------------------------------------------------------------ document helpers
doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
    setattr(sec, side, Cm(2.2))

st = doc.styles['Normal']
st.font.name, st.font.size = 'Calibri', Pt(11)
st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Calibri')
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15
for lvl, size in [(1, 15), (2, 12.5), (3, 11.5)]:
    h = doc.styles[f'Heading {lvl}']
    h.font.name, h.font.size, h.font.bold = 'Calibri', Pt(size), True
    h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    h.element.rPr.rFonts.set(qn('w:eastAsia'), 'Calibri')
    h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True

def H(text, lvl=1):
    doc.add_heading(text, level=lvl)

def P(text='', bold_prefix=None, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix); r.bold = True
        if size: r.font.size = Pt(size)
    r = p.add_run(text); r.italic = italic
    if size: r.font.size = Pt(size)
    if align: p.alignment = align
    return p

def B(text, bold_prefix=None, style='List Bullet'):
    p = doc.add_paragraph(style=style)
    if bold_prefix:
        r = p.add_run(bold_prefix); r.bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(3)
    return p

_num = [0]
def restart():
    _num[0] = 0

def N(text, bold_prefix=None):
    """Numbered item; numbering restarts after restart()."""
    _num[0] += 1
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.first_line_indent = Cm(-0.6)
    p.paragraph_format.space_after = Pt(3)
    p.add_run(f'{_num[0]}.  ')
    if bold_prefix:
        r = p.add_run(bold_prefix); r.bold = True
    p.add_run(text)
    return p

def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hex_fill)
    tcPr.append(shd)

def T(header, rows, widths=None, font=9, bold_first_col=False, highlight=None):
    """header: list of str; rows: list of lists; widths in cm; highlight: {(r, c): hex}."""
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, htxt in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = ''
        r = c.paragraphs[0].add_run(str(htxt)); r.bold = True; r.font.size = Pt(font)
        shade(c, 'DCE6F0')
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = t.rows[i].cells[j]
            c.text = ''
            r = c.paragraphs[0].add_run(str(val)); r.font.size = Pt(font)
            if bold_first_col and j == 0:
                r.bold = True
            if j > 0:
                c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if highlight and (i, j) in highlight:
                shade(c, highlight[(i, j)])
    for row in t.rows:
        for c in row.cells:
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
            c.paragraphs[0].paragraph_format.line_spacing = 1.0
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

def caption(text):
    p = doc.add_paragraph()
    r = p.add_run(text); r.italic = True; r.font.size = Pt(9)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p

def F(name, cap, width=16):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        return
    doc.add_picture(path, width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(cap)

def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

f2 = lambda x: f'{x:.2f}'
best_cells = lambda df, cols: {(i + 1, j + 1): 'E2F0D9' for j, c in enumerate(cols)
                               for i, m in enumerate(df.index) if df.loc[m, c] == df[c].min()}

# ------------------------------------------------------------------ title page
for _ in range(5):
    doc.add_paragraph()
P('HST-105 Analytics for Business and Society — Phase 2 Final Report', size=11,
  align=WD_ALIGN_PARAGRAPH.CENTER)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run('Machine Learning for Implied Volatility Surface Reconstruction'); r.bold = True
r.font.size = Pt(22); r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
P('Can machine learning fill in the SPY options volatility surface more accurately than the '
  'industry-standard SVI model — including during market crises?', italic=True, size=12,
  align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
P('Theme: Risk and Stability of Markets', size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
P('Team: Aryan · Shachi · Shreekrushna', size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
P('IIT Roorkee · November 2026', size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
page_break()

# ------------------------------------------------------------------ abstract
H('Summary')
nn_txt = ''
if HAS_NN:
    nn_txt = (f' A neural network fitted on top of an SSVI prior reached {f2(main.loc["nn", "wing"])} in the '
              f'wings and {f2(main.loc["nn", "random"])} on scattered gaps.')
P('Trading desks need an implied volatility (IV) for every strike and maturity, but on any day only '
  'some contracts trade. We test whether machine learning (ML) can fill the gaps better than SVI, '
  'the parametric formula used in industry. Using 2.0 million end-of-day SPY option quotes '
  '(749 trading days, 2020–2022), we hide known quotes on every test day — scattered quotes, a whole '
  'expiry, or the far strikes ("wings") — ask each model to predict them from the rest, and score '
  'the predictions against the real values. Models trained on history only ever see the past '
  '(walk-forward, retrained monthly).')
P(f'The largest gain is in the wings, the hardest and most risk-relevant region: XGBoost trained '
  f'to correct SVI\'s errors cut the wing error from {f2(main.loc["svi", "wing"])} to '
  f'{f2(main.loc["xgb_svi", "wing"])} volatility points ({s("wing", "xgb_svi", "improvement_%"):.0f}% '
  f'lower; 95% CI {s("wing", "xgb_svi", "ci_low_%"):.0f}–{s("wing", "xgb_svi", "ci_high_%"):.0f}%; '
  f'Diebold–Mariano p < 0.001; better on {s("wing", "xgb_svi", "win_rate_%"):.0f}% of days). '
  f'For scattered gaps inside the surface, simple linear interpolation was already '
  f'{s("random", "linear", "improvement_%"):.0f}% better than SVI, and ML added little.' + nn_txt)
P('The gains shrink in the March 2020 crash, when learned models had only two months of history, and '
  'more flexible models break static no-arbitrage conditions more often. Better IVs did not '
  'translate into measurably better one-day delta hedges.')

# ------------------------------------------------------------------ 1 introduction
H('1. Motivation and topic')
P('An option\'s price can be turned into an implied volatility: the market\'s estimate of how much '
  'the underlying asset will move. Plotted across all strikes and maturities, these values form the '
  'implied volatility surface. Trading desks use it to price and hedge contracts, and it feeds '
  'risk limits and regulatory stress tests. Because it shows how the market prices risk in real '
  'time, it fits our theme, risk and stability of markets.')
P('The problem is that only some strike–maturity combinations trade on a given day. Figure 1 shows '
  'that the observed quotes cluster near the current price and along a few expiry dates, leaving '
  'most of the surface empty. Every empty point must be estimated. The industry standard fits SVI '
  '(stochastic volatility inspired; Gatheral & Jacquier, 2014), a five-parameter formula per expiry. '
  'It is fast and well understood but imposes a fixed shape. ML models can, in principle, learn '
  'richer patterns, and can learn from thousands of past surfaces.')
F('fig_sparsity.png', 'Figure 1. Observed quotes on one day: dense near the money and along a few '
  'expiries, sparse elsewhere.', 13)
H('Research question', 2)
P('Can machine learning reconstruct the implied volatility surface more accurately than SVI — and '
  'does any gain hold up during crisis periods, when getting risk right matters most?', italic=True)
P('Following the reconstruction literature (Ackerer et al., 2020; the 2026 SPY study), we hide '
  'volatility values that we actually know, ask each model to predict them, and compare predictions '
  'with the truth. SVI is the benchmark. Beyond accuracy we also check whether the reconstructed '
  'surfaces are free of static arbitrage, whether better surfaces give better hedges, and whether '
  'gains are statistically significant.')

# ------------------------------------------------------------------ 2 data
H('2. Data')
P('Source: "SPY Option Chains Q1 2020 – Q4 2022" (kylegraupe), a public Kaggle dataset '
  '(https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022) of end-of-day option chains '
  'for SPY, the exchange-traded fund tracking the S&P 500. One snapshot per trading day; each row '
  'gives a strike and expiry with call and put bid, ask, last price, volume, bid/ask size, implied '
  'volatility and Greeks, plus the SPY price. The raw file has 3,589,079 rows × 33 columns (1.28 GB).')
H('2.1 Cleaning', 2)
restart()
P('Each step was checked by row counts and saved as a separate file, so every stage can be '
  'reproduced (notebooks 01–03).')
N('Removed 9 snapshots dated on 2022 market holidays (e.g. 4 July, 26 December). Their prices were '
  'within cents of the previous close: stale copies, not trading days. 749 trading days remain.')
N('Split the "bid size x ask size" text fields into numbers and converted all fields to proper types.')
N('Reshaped to one row per option (call and put separately): 7,099,618 rows.')
N('Applied quality filters, logged in Table 1.')
T(['Filter (keep rows where…)', 'Removed', 'Remaining'],
  [['IV is present', '382,881', '6,716,737'],
   ['Volume > 0 (actually traded that day)', '2,714,730', '4,002,007'],
   ['Bid > 0', '176,261', '3,825,746'],
   ['Ask > bid (no crossed quotes)', '57,027', '3,768,719'],
   ['Relative spread ≤ 50% of mid price', '134,575', '3,634,144'],
   ['0.01 ≤ IV ≤ 2', '61,006', '3,573,138'],
   ['Days to expiry ≥ 7', '181,257', '3,391,881 (47.8%)']],
  widths=[9, 3, 4])
caption('Table 1. Filter log. About half of the data survives; untraded contracts are the largest loss.')
H('2.2 Features', 2)
B(' for each date and expiry, from put–call parity: a straight-line fit of (call mid − put mid) '
  'against strike over near-the-money pairs. The forward behaves as theory predicts: below spot for '
  'long maturities in 2020–21 (dividends above interest rates) and above spot in 2022 (rates rose).',
  'Forward price F')
B(' k = ln(K / F), maturity T = days/365, and total variance w = IV² × T.', 'Log-moneyness')
B(' kept only out-of-the-money options (puts below the forward, calls above) and |k| ≤ 1. '
  'Out-of-the-money options are the most liquid and are barely affected by early exercise.',
  'Selection:')
P('Final sample: 2,017,522 quotes on 749 days (about 2,700 per day). 2.1% of quotes, mostly long '
  'maturities, were dropped because no reliable forward could be fitted.')
F('fig_smile.png', 'Figure 2. The volatility smile/skew: out-of-the-money puts (left) carry higher IV.', 12)
H('2.3 American exercise', 2)
P('SPY options are American-style, while the theory behind SVI assumes European options. We assume '
  'the provider\'s IVs are computed with an American model, which is equivalent to the industry '
  'practice of "de-Americanizing" prices (Carr & Wu, 2010). We tested this: near the money, put and '
  'call IVs at the same strike differ by under 0.1 vol points in most quarters, but by +0.3 to +0.5 '
  'points in late 2022 (high interest rates) and −1.9 in early 2020. We therefore use only '
  'out-of-the-money options and report the remaining gap as a limitation.')

# ------------------------------------------------------------------ 3 strategy
H('3. Empirical strategy')
H('3.1 Test design: hide, predict, score', 2)
P('On each of the 709 test days (March 2020 – December 2022) we hide part of the day\'s quotes, '
  'fit each model on the visible quotes, predict the hidden ones, and score only the hidden ones. '
  'Masks are seeded by date, so every model is tested on exactly the same hidden quotes.')
T(['Mask', 'What is hidden', 'Tests', 'Hidden / visible per day'],
  [['Random', '20% of quotes in every expiry', 'Filling scattered gaps',
    f'{design.loc["random", "hidden_per_day"]:.0f} / {design.loc["random", "visible_per_day"]:.0f}'],
   ['Expiry', 'One whole expiry (never the first or last)', 'Interpolating across maturities',
    f'{design.loc["expiry", "hidden_per_day"]:.0f} / {design.loc["expiry", "visible_per_day"]:.0f}'],
   ['Wings', 'Outer 10% of strikes on each side of every expiry', 'Extrapolating to far strikes',
    f'{design.loc["wing", "hidden_per_day"]:.0f} / {design.loc["wing", "visible_per_day"]:.0f}']],
  widths=[2.2, 6, 4.5, 3.6])
caption('Table 2. The three masks. 709 days × 3 masks = 2,127 out-of-sample tests per model.')
P('Models trained on history use walk-forward validation: at the start of each test month they are '
  'retrained on all earlier days only (34 monthly folds). Random k-fold cross-validation would let '
  'models learn from the future, because neighbouring days are strongly related.')
H('3.2 Models', 2)
T(['Code', 'Model', 'How it works', 'Learns from'],
  [['—', 'Mean', 'Day\'s average IV everywhere (floor)', 'Today'],
   ['B0', 'Linear interpolation', 'Linear interpolation of total variance in (k, T)', 'Today'],
   ['B1', 'SVI (benchmark)', 'Raw SVI per expiry with Gatheral–Jacquier/Lee constraints; '
    'linear in T between expiries', 'Today'],
   ['B2', 'SSVI', 'One arbitrage-constrained surface per day; ATM variance rising with T', 'Today'],
   ['M1a', 'XGBoost, direct', 'Gradient-boosted trees predict IV from 14 features', 'Past days'],
   ['M1b', 'XGBoost on linear', 'Trees predict the error of linear interpolation', 'Past days'],
   ['M2', 'XGBoost on SVI', 'Trees predict the error of SVI (as in Audrino & Colangelo, 2010)',
    'Past days'],
   ['M3', 'Gaussian process', 'Smile trend + Matérn 5/2 GP on the residual; kernel by '
    'maximum likelihood each day', 'Today'],
   ['M4', 'Neural net + penalty', 'SSVI prior × neural correction; butterfly and calendar '
    'penalties via autograd (as in Ackerer et al., 2020)', 'Today'],
   ["M4'", 'Neural net, no penalty', 'Same network without the penalty (ablation)', 'Today']],
  widths=[1.2, 3.4, 9, 2.4], font=8.5)
caption('Table 3. Models compared. "Today" models are fitted on the visible quotes of each test day; '
        '"past days" models are trained monthly on earlier days and then use today\'s visible quotes '
        'as input.')
P('The XGBoost features are computed only from the visible quotes: log-moneyness, maturity, '
  'standardised moneyness z = k/√T, the day\'s at-the-money IV level, term-structure slope and skew, '
  'distance to the nearest visible quote and expiry, how far the point lies beyond the visible strikes, '
  'and the linear-interpolation estimate. Training rows come from past days masked the same way as '
  'test days, so the model learns to fill gaps from visible data only.')
H('3.3 Measures', 2)
B(' RMSE, MAE and maximum error on hidden quotes, in volatility points (1 point = 0.01 IV); also '
  'by region of the surface.', 'Accuracy:')
B(' Diebold–Mariano test on daily squared errors (Newey–West, 5 lags) and a 95% moving-block '
  'bootstrap interval (10-day blocks) for the % improvement over SVI.', 'Significance:')
B(' share of grid points where the fitted surface violates the butterfly condition (Gatheral–Jacquier '
  'g(k) ≥ 0) or the calendar condition (total variance non-decreasing in T).', 'No-arbitrage:')
B(' one-day delta hedge of each hidden option using the model\'s IV vs. the market\'s IV.', 'Hedging:')
B(' COVID crash (20 Feb – 30 Apr 2020), recovery (rest of 2020), calm 2021, bear market 2022.',
  'Regimes:')

# ------------------------------------------------------------------ 4 results
H('4. Results')
H('4.1 Accuracy', 2)
order = [m for m in LABEL if m in main.index]
mt = main.loc[order, ['random', 'expiry', 'wing']]
T(['Model', 'Random', 'Expiry', 'Wings'],
  [[LABEL[m]] + [f2(mt.loc[m, c]) for c in mt.columns] for m in order],
  widths=[7, 3, 3, 3], highlight=best_cells(mt, mt.columns))
caption('Table 4. Mean daily RMSE on hidden quotes (volatility points; lower is better; '
        'green = best). 709 test days.')
F('res_main_rmse.png', 'Figure 3. Reconstruction error by mask (mean daily RMSE).')
P('The main findings:')
restart()
N(f' XGBoost correcting SVI cut the wing error from {f2(main.loc["svi", "wing"])} to '
  f'{f2(main.loc["xgb_svi", "wing"])} points, the largest improvement in the study. All three XGBoost '
  f'variants beat SVI in the wings by more than half, while the Gaussian process barely did '
  f'({f2(main.loc["gp", "wing"])}). Trees that have seen hundreds of past surfaces know how smiles '
  f'typically continue beyond the last quoted strike; a model fitted to one day does not.',
  'ML wins clearly where data are missing beyond the observed range.')
N(f' For scattered gaps, linear interpolation ({f2(main.loc["linear", "random"])}) beat SVI '
  f'({f2(main.loc["svi", "random"])}) by half. With about 2,200 visible quotes per day, neighbouring '
  f'quotes already pin down the answer, and SVI\'s fixed shape costs more than it gains. The best ML '
  f'models (M1b {f2(main.loc["xgb_lin", "random"])}, M2 {f2(main.loc["xgb_svi", "random"])}, GP '
  f'{f2(main.loc["gp", "random"])}) land close to linear interpolation but do not beat it.',
  'Inside the surface, simple interpolation is hard to beat.')
N(f' SSVI ({f2(main.loc["ssvi", "random"])} random) is the least accurate structural model: one '
  f'surface for the whole day is too rigid. Its advantage is no-arbitrage (Section 4.4).',
  'Imposing more structure costs accuracy.')
if HAS_NN:
    N(f' The neural net on an SSVI prior scored {f2(main.loc["nn", "random"])} (random), '
      f'{f2(main.loc["nn", "expiry"])} (expiry) and {f2(main.loc["nn", "wing"])} (wings). The '
      f'version without the arbitrage penalty scored {f2(main.loc["nn_nopen", "wing"])} in the wings.',
      'Neural network.')

H('4.2 Statistical significance', 2)
rows, hl = [], {}
for mask in ['random', 'expiry', 'wing']:
    sub = sig[(sig['mask'] == mask) & (sig['model'] != 'mean')].sort_values('rmse_model')
    for _, r in sub.iterrows():
        p = '< 0.001' if r['p_value'] < 0.001 else f'{r["p_value"]:.3f}'
        rows.append([MASK[mask], LABEL[r['model']], f'{r["improvement_%"]:+.0f}%',
                     f'{r["ci_low_%"]:+.0f}% to {r["ci_high_%"]:+.0f}%', p, f'{r["win_rate_%"]:.0f}%'])
T(['Mask', 'Model', 'vs SVI', '95% CI', 'DM p-value', 'Days better'], rows,
  widths=[1.8, 6.2, 1.8, 3.6, 2, 2], font=8.5)
caption('Table 5. Improvement in mean RMSE over SVI (positive = better than SVI), with a moving-block '
        'bootstrap interval, Diebold–Mariano p-value on daily squared errors, and share of days on '
        'which the model beat SVI.')
P(f'The wing improvements of all XGBoost variants are large and significant (p < 0.001 for M1b and '
  f'M2) and hold on {s("wing", "xgb_svi", "win_rate_%"):.0f}% of days for M2. On the random mask, '
  f'linear interpolation, XGBoost (M1b, M2) and the GP all beat SVI significantly. On the expiry mask '
  f'the bootstrap intervals for linear, M1b, M2 and the GP exclude zero but Diebold–Mariano p-values '
  f'are above 0.05, because a few '
  f'days with large errors dominate squared-error losses; the median-day evidence (win rates of '
  f'73–88%) still favours the alternatives.')

H('4.3 Does it hold up in a crisis?', 2)
rw = tab('regime_wing', index_col=0).loc[order]
rr = tab('regime_random', index_col=0).loc[order]
T(['Model'] + [f'{REGIME[c]}' for c in rw.columns],
  [[LABEL[m]] + [f2(rw.loc[m, c]) for c in rw.columns] for m in order if m != 'mean'],
  widths=[7, 2.4, 2.4, 2.4, 2.4], highlight=best_cells(rw.drop(index='mean'), rw.columns))
caption('Table 6. Wing-mask RMSE by market regime (green = best in each regime).')
F('res_rolling.png', 'Figure 4. Error through time (21-day rolling mean). Grey band: COVID crash.')
P(f'All models are worst in the COVID crash. An XGBoost variant is the best wing model in every '
  f'regime: XGBoost on SVI in the crash ({f2(rw.loc["xgb_svi", "covid_crash"])} vs SVI '
  f'{f2(rw.loc["svi", "covid_crash"])}), recovery and 2021, and direct XGBoost in 2022 '
  f'({f2(rw.loc["xgb", "bear_2022"])} vs SVI {f2(rw.loc["svi", "bear_2022"])}). The main result '
  f'therefore survives stress. But the learned models lose the most in the crash: '
  f'direct XGBoost becomes worse than SVI ({f2(rw.loc["xgb", "covid_crash"])}), and on the random mask '
  f'XGBoost on SVI ({f2(rr.loc["xgb_svi", "covid_crash"])}) falls behind SVI '
  f'({f2(rr.loc["svi", "covid_crash"])}). In March 2020 these models had been trained on only two '
  f'calm months, so they had never seen a crash. Models fitted on the day itself (linear, GP) adapt '
  f'immediately. A practical desk would combine both, or retrain learned models more often in '
  f'turbulent periods.')
F('res_crisis_day.png', 'Figure 5. 16 March 2020, wing mask: true hidden IVs (crosses) and model '
  'predictions for three expiries.')

H('4.4 Where are the errors, and are the surfaces arbitrage-free?', 2)
F('res_regional.png', 'Figure 6. Wing-mask RMSE by moneyness z = k/√T and maturity. Errors are largest in '
  'the far wings, above all deep out-of-the-money calls (z > 0.4) at short and medium maturities; '
  'XGBoost on SVI cuts them by about two-thirds.')
T(['Model', 'RMSE (random)', 'Butterfly violations', 'Calendar violations'],
  [[LABEL[m], f2(arb.loc[m, 'rmse']), f'{100 * arb.loc[m, "butterfly_rate"]:.1f}%',
    f'{100 * arb.loc[m, "calendar_rate"]:.1f}%'] for m in order if m != 'mean'],
  widths=[7, 3, 3.2, 3.2])
caption('Table 7. Share of grid points violating static no-arbitrage conditions (random mask, '
        'average over days).')
F('res_arbitrage.png', 'Figure 7. Accuracy against no-arbitrage violations.', 13)
P(f'There is a clear trade-off. SSVI is arbitrage-free by construction (0% violations) but least '
  f'accurate. SVI rarely violates the butterfly condition ({100 * arb.loc["svi", "butterfly_rate"]:.1f}%) '
  f'but often the calendar condition ({100 * arb.loc["svi", "calendar_rate"]:.0f}%), because each '
  f'expiry is fitted separately and the slices can cross. The tree models are the most accurate in '
  f'the wings but produce the bumpiest surfaces ({100 * arb.loc["xgb_svi", "butterfly_rate"]:.0f}–'
  f'{100 * arb.loc["xgb", "butterfly_rate"]:.0f}% butterfly violations), since tree predictions are '
  f'step-shaped. The GP sits in between. A surface used for pricing would need a final smoothing or '
  f'arbitrage-repair step after an ML model.')
if HAS_NN:
    P(f'The neural net\'s penalty is designed for exactly this: it reached '
      f'{100 * arb.loc["nn", "butterfly_rate"]:.1f}% butterfly and '
      f'{100 * arb.loc["nn", "calendar_rate"]:.1f}% calendar violations, against '
      f'{100 * arb.loc["nn_nopen", "butterfly_rate"]:.1f}% and '
      f'{100 * arb.loc["nn_nopen", "calendar_rate"]:.1f}% without the penalty.')

H('4.5 Hedging', 2)
hd = tab('hedging')
hd.columns = ['model'] + list(hd.columns[1:])
hw = hd[hd['mask'] == 'wing'].set_index('model')
T(['Model', 'Hedge error, model IV ($)', 'Hedge error, market IV ($)', 'Unhedged ($)'],
  [[LABEL[m], f'{hw.loc[m, "rmse_model"]:.3f}', f'{hw.loc[m, "rmse_true_iv"]:.3f}',
    f'{hw.loc[m, "rmse_unhedged"]:.3f}'] for m in hw.index],
  widths=[7, 3.4, 3.4, 2.6])
caption('Table 8. One-day delta-hedging error for hidden wing options (RMS, dollars per option).')
P('Hedging with any model\'s IV removes about 40% of the one-day price risk of the wing options, and '
  'every model comes within about 3% of hedging with the true market IV. Delta is not very '
  'sensitive to small IV errors, so the accuracy differences above matter for pricing far '
  'strikes and for risk measures, not for one-day delta hedging.')

if os.path.exists(os.path.join(FIG, 'res_sparsity.png')):
    H('4.6 Sparsity', 2)
    F('res_sparsity.png', 'Figure 8. Error as more quotes are hidden (random mask, every 10th test day).', 13)
    P('The ranking changes as data become scarce. With 10–30% hidden, linear interpolation is '
      'best; from 50% hidden the Gaussian process is best and degrades least (0.43 at 10% hidden to '
      '0.68 at 90%). SVI (0.67 to 1.75, 2.6 times worse) and linear interpolation (3.5 times worse) '
      'degrade fastest, while the rigid SSVI barely changes but stays inaccurate. For thin markets, '
      'a flexible but regularised model such as the GP is preferable to SVI.')

H('4.7 Robustness: an independent run', 2)
P('A team member independently re-ran the Gaussian process and the neural network on a harsher '
  'version of the test: 758 days, slightly different filters (2.45 million quotes), 30% of quotes or '
  '30% of expiries hidden, and the outer 15% of strikes hidden on each side. The absolute numbers '
  'are not comparable with Table 4, but the pattern is the same:')
T(['Mask (independent run)', 'GP', 'NN + penalty', 'NN, no penalty', 'SSVI (NN prior)'],
  [['Random 30%', '0.30', '0.59', '0.59', '1.26'],
   ['30% of expiries', '0.89', '0.77', '0.77', '1.41'],
   ['Wings 15% each side', '2.36', '1.66', '1.66', '2.80']],
  widths=[5, 2.4, 2.8, 2.8, 3])
caption('Table 9. Mean daily RMSE in the independent run (volatility points).')
P('In both runs the GP is best for scattered gaps, the neural network is better at the wings and at '
  'missing expiries, and the no-arbitrage penalty changes accuracy by less than 0.01 points. The '
  'neural network beat its own SSVI starting point on more than 97% of days in every test.')

# ------------------------------------------------------------------ 5 discussion
H('5. Answer to the research question')
P('Yes, in the part of the surface that matters most for risk. Machine learning that learns from '
  'past surfaces reconstructs the far strikes with about two-thirds lower error than SVI, the gain is '
  'statistically significant, and it survives the 2022 bear market and, in reduced form, the COVID '
  'crash. Inside the observed range the honest answer is no: simple interpolation of neighbouring '
  'quotes is already better than SVI, and ML adds little. The extra complexity of ML is therefore '
  'justified for extrapolation, not for interpolation.')
P('Two qualifications follow. First, learned models need enough history: they were weakest when '
  'the market moved into a regime they had never seen. Second, accuracy and arbitrage-freeness pull '
  'in opposite directions; flexible models need an arbitrage-aware loss or a repair step before '
  'their surfaces can be used for pricing.')
H('Limitations', 2)
B(' Implied volatilities are computed by the data provider. For American SPY options, put and call '
  'IVs differ by up to 0.5 points in late 2022, a remaining early-exercise effect.', 'Data:')
B(' One underlying (SPY) over three unusually turbulent years; results may differ for calmer '
  'periods, single stocks or European index options (SPX), on which most literature is based.',
  'Scope:')
B(' XGBoost and neural-network settings were chosen by hand, not tuned on a validation set. Each '
  'day is tested with one random mask.', 'Tuning:')
B(' Learned models in March 2020 had only January–February 2020 as history.', 'History:')
H('Future work', 2)
B('Nested walk-forward tuning of XGBoost and a sweep of the neural network\'s penalty weight.')
B('One neural network trained on all past surfaces (as in the 2026 SPY study), on a GPU.')
B('An arbitrage-repair step after XGBoost, and testing on SPX options.')

# ------------------------------------------------------------------ 6 reproducibility
H('6. Code and reproducibility')
P('All code is in the project repository (README.md describes the steps). From the raw Kaggle CSV:')
restart()
N(' notebooks/01_Cleaning → 02_Filters → 03_Features: cleaning, filters, forward price and features.',
  'Data:')
N(' python -m src.run_all runs every model on every test day and mask and saves results/<model>/ '
  '(resumable; about 5 hours on a 10-core laptop).', 'Models:')
N(' python report/build_figures.py then python report/make_report.py rebuild every table, figure '
  'and this report.', 'Report:')
P('Key modules: src/masking.py (masks), src/evaluate.py (walk-forward loop), src/metrics.py, '
  'src/arbitrage.py, src/stats.py (Diebold–Mariano, bootstrap), src/hedging.py, src/ml_features.py, '
  'and one file per model in src/models/.')

# ------------------------------------------------------------------ 7 contribution
H('7. Statement of contribution')
P('[To be completed by the team before submission: one line per member describing their work.]',
  italic=True)
T(['Team member', 'Contribution'],
  [['Aryan', '[fill in]'], ['Shachi', '[fill in]'], ['Shreekrushna', '[fill in]']],
  widths=[4, 12])

# ------------------------------------------------------------------ references
H('References')
refs = [
    'Ackerer, D., Tagasovska, N. & Vatter, T. (2020). Deep smoothing of the implied volatility surface. NeurIPS 33.',
    'Andreasen, J. & Huge, B. (2011). Volatility interpolation. Risk, March 2011.',
    'Audrino, F. & Colangelo, D. (2010). Semi-parametric forecasts of the implied volatility surface using regression trees. Statistics and Computing, 20, 421–434.',
    'Carr, P. & Wu, L. (2010). Stock options and credit default swaps: a joint framework for valuation and estimation. Journal of Financial Econometrics, 8(4), 409–449.',
    'Chataigner, M., Cousin, A., Crépey, S., Dixon, M. & Gueye, D. (2021). Beyond surrogate modeling: learning the local volatility via shape constraints. SIAM Journal on Financial Mathematics, 12.',
    'De Spiegeleer, J., Madan, D., Reyners, S. & Schoutens, W. (2018). Machine learning for quantitative finance: fast derivative pricing, hedging and fitting. Quantitative Finance, 18(10).',
    'Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253–263.',
    'Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. Quantitative Finance, 14(1), 59–71.',
    'Hagan, P., Kumar, D., Lesniewski, A. & Woodward, D. (2002). Managing smile risk. Wilmott Magazine.',
    'Lind, P. P. & Gatheral, J. (2025). NN de-Americanization: an efficient method to facilitate calibration of American-style options. Quantitative Finance, 25(1).',
    'Zheng, Y., Yang, Y. & Chen, B. (2021). Incorporating prior financial domain knowledge into neural networks for implied volatility surface prediction. KDD 2021.',
    'No-arbitrage deep calibration for volatility smile and skewness (2023). arXiv:2310.16703.',
    'Volatility surface reconstruction using deep learning under no-arbitrage constraints (2026). arXiv:2605.24031.',
]
for r_ in refs:
    p = doc.add_paragraph(r_)
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.8)
    p.paragraph_format.space_after = Pt(3)
    for run in p.runs:
        run.font.size = Pt(9.5)

# ------------------------------------------------------------------ appendix A: literature
page_break()
H('Appendix A. Literature survey')
P('One page per paper, in the format given in the assignment guidelines.')
papers = [
    dict(title='Arbitrage-free SVI volatility surfaces', year='2014', journal='Quantitative Finance, 14(1)',
         link='https://arxiv.org/abs/1204.0646',
         obj='Find conditions under which the SVI formula produces a volatility surface that allows no riskless profit (static arbitrage).',
         data='Theoretical work, illustrated with S&P 500 (SPX) index options on one day (15 Sep 2011).',
         avail='Not applicable (theory); example data not public',
         methods=['The SVI parameterisation of one expiry\'s smile (five parameters).',
                  'Surface SVI (SSVI) and explicit parameter conditions ruling out butterfly and calendar arbitrage.'],
         results=['Simple conditions on the parameters guarantee an arbitrage-free surface.',
                  'Fitted surfaces can be checked for arbitrage directly from their parameters.']),
    dict(title='Managing smile risk', year='2002', journal='Wilmott Magazine',
         link='https://iphone.next-finance.net/IMG/pdf/pdf_SABR.pdf',
         obj='Build a model whose smile moves the way real market smiles move when the underlying price changes.',
         data='Interest-rate option smiles used for empirical illustration.',
         avail='Private (formula public)',
         methods=['SABR stochastic volatility model, with volatility random and correlated with the asset price.',
                  'Singular-perturbation approximation giving the implied volatility in closed form.'],
         results=['SABR predicts smile movements in the direction observed in markets, unlike local volatility models.',
                  'The closed-form approximation is accurate enough for practical use.']),
    dict(title='Volatility interpolation', year='2011', journal='Risk Magazine',
         link='https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1694972',
         obj='Build a full volatility surface from a handful of quotes while guaranteeing no arbitrage.',
         data='Market option quotes, including deliberately noisy inputs.',
         avail='Private (method public)',
         methods=['A one-step implicit finite-difference scheme stepping forward in maturity.',
                  'Arbitrage-freeness built into the construction rather than checked afterwards.'],
         results=['The surface is arbitrage-free by construction even with messy inputs.',
                  'Sets the standard that classical methods can guarantee, against which ML must be measured.']),
    dict(title='Semi-parametric forecasts of the implied volatility surface using regression trees', year='2010',
         journal='Statistics and Computing, 20', link='https://link.springer.com/article/10.1007/s11222-009-9134-y',
         obj='Improve a parametric volatility model by learning where it goes wrong.',
         data='S&P 500 index options, 777,887 observations on 1,928 days (1996–2003).',
         avail='Private (OptionMetrics-type data)',
         methods=['Boosted regression trees fitted to the errors of a starting model.',
                  'Out-of-sample evaluation over 60-day windows around market shocks (e.g. 11 September 2001).'],
         results=['Boosting the starting model clearly improves out-of-sample accuracy.',
                  'Gains hold across strikes and maturities. (Inspired our XGBoost-on-SVI model, M2.)']),
    dict(title='Machine learning for quantitative finance: fast derivative pricing, hedging and fitting', year='2018',
         journal='Quantitative Finance, 18(10)', link='https://doi.org/10.1080/14697688.2018.1495335',
         obj='Show that a flexible statistical model can replace slow numerical pricing and fit volatility surfaces quickly.',
         data='Equity and index option data.', avail='Private',
         methods=['Gaussian process regression, which also reports its uncertainty.',
                  'Applied to pricing, hedging and surface fitting.'],
         results=['Large speed gains over numerical methods with little loss of accuracy.',
                  'Established Gaussian processes as a practical tool for volatility surfaces (our model M3).']),
    dict(title='Beyond surrogate modeling: learning the local volatility via shape constraints', year='2021',
         journal='SIAM Journal on Financial Mathematics, 12', link='https://arxiv.org/abs/2212.09957',
         obj='Compare two ways of adding no-arbitrage rules to ML models, against the industry standard.',
         data='SPX European put options on 18 May 2019: 1,720 quotes for training, 1,725 for testing.',
         avail='Private',
         methods=['Gaussian process constrained to an arbitrage-free shape.',
                  'Neural network penalised when it violates no-arbitrage conditions; SSVI benchmark.'],
         results=['The constrained GP is provably arbitrage-free; the NN only penalises violations.',
                  'Both ML approaches fit held-out quotes well; regularisation is key for backtests.']),
    dict(title='Incorporating prior financial domain knowledge into neural networks for implied volatility surface prediction',
         year='2021', journal='ACM SIGKDD Conference (KDD 2021)', link='https://arxiv.org/abs/1904.12834',
         obj='Build a neural network for volatility surfaces that obeys what finance theory already says.',
         data='S&P 500 index options, OptionMetrics, 1996–2016: about 3 million quotes.',
         avail='Private',
         methods=['A network whose activation function already resembles a volatility smile.',
                  'Loss penalties for arbitrage and asymptotic conditions; trained on day t−1, tested on day t.'],
         results=['Beats SSVI and simpler networks in IV and price MAPE.',
                  'Output stays consistent with financial theory.']),
    dict(title='Deep smoothing of the implied volatility surface', year='2020',
         journal='Advances in Neural Information Processing Systems (NeurIPS 2020)',
         link='https://papers.nips.cc/paper/2020/hash/858e47701162578e5e627cd93ab0938a-Abstract.html',
         obj='Use a neural network to correct and smooth the surface produced by a standard formula.',
         data='S&P 500 options, Jan–Apr 2018 and the Sep–Dec 2008 crisis; synthetic surfaces.',
         avail='Not public',
         methods=['A neural network multiplying a prior model (SSVI or Black–Scholes).',
                  'Soft no-arbitrage penalties; daily backtests with interpolation and extrapolation splits.'],
         results=['Handles the sparse edges of the surface better than the prior alone.',
                  'Penalties remove arbitrage at little accuracy cost. (Basis of our model M4 and our masks.)']),
    dict(title='No-arbitrage deep calibration for volatility smile and skewness', year='2023',
         journal='arXiv preprint', link='https://arxiv.org/abs/2310.16703',
         obj='Fit volatility surfaces with a neural network while discouraging arbitrage.',
         data='SABR-generated synthetic data and S&P 500 intraday options (2022–2023).',
         avail='Not public',
         methods=['Neural network trained with penalty terms derived from no-arbitrage conditions.',
                  'Comparison with cubic splines, MLP and arbitrage-free smoothing; 10 random seeds.'],
         results=['The penalty sharply reduces violations at a small accuracy cost.',
                  'Gives a practical recipe for writing such penalties into a loss.']),
    dict(title='Volatility surface reconstruction using deep learning under no-arbitrage constraints', year='2026',
         journal='arXiv preprint', link='https://arxiv.org/abs/2605.24031',
         obj='Compare modern neural architectures against SVI for rebuilding a surface from partial observations.',
         data='3,900 real SPY option surfaces 2008–2025 (time-ordered split) plus synthetic Heston surfaces.',
         avail='Not public (built from a public SPY dataset)',
         methods=['Six neural designs (MLP, CNN, U-Net, Transformer, VAEs) on masked surfaces.',
                  'Random and wing masking, sparsity sweep 10–90%, regional RMSE and arbitrage metrics.'],
         results=['Attention-based and image-style networks reconstruct best, especially when data are sparse.',
                  'Closest prior work; our study adds tree models, Gaussian processes, regimes and significance tests.']),
]
for i, pp in enumerate(papers):
    if i:
        page_break()
    H(f'A.{i + 1} {pp["title"]}', 2)
    t = doc.add_table(rows=8, cols=2); t.style = 'Table Grid'
    fields = [('Title', pp['title']), ('Year', pp['year']), ('Journal', pp['journal']),
              ('Web link', pp['link']), ('Objective', pp['obj']),
              ('Data used', pp['data']), ('Data availability', pp['avail']), ('Methods used', None)]
    for j, (k_, v) in enumerate(fields):
        c0, c1 = t.rows[j].cells
        c0.text = ''; r = c0.paragraphs[0].add_run(k_); r.bold = True
        shade(c0, 'DCE6F0')
        c1.text = ''
        if v is not None:
            c1.paragraphs[0].add_run(v)
        else:
            for n_, m_ in enumerate(pp['methods'], 1):
                para = c1.paragraphs[0] if n_ == 1 else c1.add_paragraph()
                para.add_run(f'{n_}. {m_}')
        c0.width, c1.width = Cm(4), Cm(12.5)
    row = t.add_row().cells
    row[0].text = ''; r = row[0].paragraphs[0].add_run('Main results'); r.bold = True
    shade(row[0], 'DCE6F0')
    row[1].text = ''
    for n_, m_ in enumerate(pp['results'], 1):
        para = row[1].paragraphs[0] if n_ == 1 else row[1].add_paragraph()
        para.add_run(f'{n_}. {m_}')
    row[0].width, row[1].width = Cm(4), Cm(12.5)

# ------------------------------------------------------------------ appendix B: settings
page_break()
H('Appendix B. Model settings')
T(['Model', 'Settings'],
  [['SVI', 'Raw SVI per expiry (≥ 5 quotes), SLSQP from 3 starting points; bounds and Lee wing constraints; linear interpolation of total variance in T.'],
   ['SSVI', 'Power-law φ; ρ, η, γ shared; one θ per expiry as cumulative positive increments; butterfly conditions as penalties (weight 10).'],
   ['XGBoost', '500 trees, depth 6, learning rate 0.05, subsample 0.8, column sample 0.8, min child weight 20; retrained monthly on ~875,000 masked training rows from earlier days.'],
   ['Gaussian process', 'Trend in z = k/√T (5 terms) + Matérn 5/2 kernel on (k, √T) with white noise; up to 800 visible quotes; kernel by maximum likelihood each day.'],
   ['Neural network', 'w = w_SSVI × exp(MLP(k, T)); 2 hidden layers × 32 tanh units; Adam, learning rate 0.005, 300 steps; penalty weight 1 (butterfly) and 10 (calendar) on a 30 × maturities grid.'],
   ['Evaluation', 'Masks seeded by date; walk-forward history = all days before the test month; arbitrage grid = 41 strikes × all maturities of the day.']],
  widths=[3.5, 13], font=9)

try:
    doc.save(OUT)
except PermissionError:                    # the report is open in Word: save next to it
    OUT = OUT.replace('.docx', '_updated.docx')
    doc.save(OUT)
print('saved', OUT)
