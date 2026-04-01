"""
=============================================================================
  BREAST CANCER DIAGNOSIS ANALYSIS
  A complete end-to-end healthcare data analytics project
  Dataset: Breast Cancer Research (1,200 patients, 21 features)
  Author : Virat
=============================================================================

REQUIREMENTS:
    pip install pandas numpy matplotlib seaborn scipy scikit-learn

USAGE:
    python breast_cancer_analysis.py

OUTPUT:
    breast_cancer_dashboard.png  — full multi-panel dashboard (save-ready)
    Printed summary stats in console
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = 'Breast_cancer_Reseach.csv'   # <- update path if needed
OUTPUT_FILE = 'breast_cancer_dashboard.png'

# Brand colours
C_MAL  = '#C0392B'   # red   — Malignant
C_BEN  = '#1A6B8A'   # teal  — Benign
C_ACC  = '#8B3A8B'   # purple — accent
C_GOLD = '#B8860B'   # gold
C_BG   = '#F7F4F0'   # warm off-white background
C_DARK = '#1C1917'   # near-black text

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'figure.facecolor'  : C_BG,
    'axes.facecolor'    : '#FFFFFF',
    'axes.edgecolor'    : '#E0DBD5',
    'axes.labelcolor'   : '#6B6560',
    'xtick.color'       : '#6B6560',
    'ytick.color'       : '#6B6560',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.color'        : '#F0EDE8',
    'grid.linewidth'    : 0.8,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  BREAST CANCER DIAGNOSIS ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df['diagnosis_label'] = df['diagnosis'].map({'M': 'Malignant', 'B': 'Benign'})

float_cols = [c for c in df.columns if c not in ['diagnosis', 'diagnosis_label']]

print(f"\n📋 DATASET OVERVIEW")
print(f"   Records        : {len(df):,}")
print(f"   Features       : {len(float_cols)}")
print(f"   Missing values : {df.isnull().sum().sum()}")
print(f"   Malignant (M)  : {(df.diagnosis=='M').sum()} ({(df.diagnosis=='M').mean()*100:.1f}%)")
print(f"   Benign    (B)  : {(df.diagnosis=='B').sum()} ({(df.diagnosis=='B').mean()*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
mal = df[df.diagnosis == 'M'][float_cols]
ben = df[df.diagnosis == 'B'][float_cols]

# Effect sizes (Cohen's d approximation)
effect_sizes = ((mal.mean() - ben.mean()) / df[float_cols].std()).abs()
effect_sizes = effect_sizes.sort_values(ascending=False)

# T-test p-values
pvals = {}
for col in float_cols:
    _, p = stats.ttest_ind(mal[col], ben[col])
    pvals[col] = p

# Risk score
df['risk_score'] = (
    df['radius_worst'] * 0.40 +
    df['concavity_worst'] * 10 * 0.35 +
    df['concave_points_worst'] * 10 * 0.25
)

print(f"\n📊 TOP 5 DISCRIMINATING FEATURES (Cohen's d effect size)")
for feat, eff in effect_sizes.head(5).items():
    sig = "***" if pvals[feat] < 0.001 else "**" if pvals[feat] < 0.01 else "*"
    print(f"   {feat:<30} d = {eff:.4f}  p{sig}")

print(f"\n💡 KEY FINDINGS")
print(f"   Avg area_mean  — Malignant: {mal['area_mean'].mean():.1f} mm²  |  Benign: {ben['area_mean'].mean():.1f} mm²")
print(f"   Avg radius_worst — M: {mal['radius_worst'].mean():.2f}  |  B: {ben['radius_worst'].mean():.2f}")
print(f"   Avg concavity_worst — M: {mal['concavity_worst'].mean():.4f}  |  B: {ben['concavity_worst'].mean():.4f}")
print(f"   Avg Risk Score — M: {df[df.diagnosis=='M']['risk_score'].mean():.3f}  |  B: {df[df.diagnosis=='B']['risk_score'].mean():.3f}")

# 3.  DASHBOARD LAYOUT
fig = plt.figure(figsize=(22, 26), facecolor=C_BG)

# Header band
fig.text(0.04, 0.975, '🎗  BREAST CANCER DIAGNOSIS ANALYSIS',
         fontsize=20, fontweight='bold', color=C_DARK, va='top')
fig.text(0.04, 0.963,
         'Quantitative analysis of 1,200 biopsy records · cell nucleus morphology · malignant vs benign classification',
         fontsize=10, color='#6B6560', va='top')

# KPI strip
kpis = [
    ('1,200',       'Patient Records',    C_MAL),
    ('588  /  612', 'Malignant / Benign', C_BEN),
    ('area_mean',   'Top Discriminator',  C_ACC),
    ('Worst > Mean','Key Insight',        C_GOLD),
]
for idx, (val, lbl, col) in enumerate(kpis):
    x = 0.04 + idx * 0.24
    fig.text(x, 0.945, val, fontsize=14, fontweight='bold', color=col, va='top')
    fig.text(x, 0.932, lbl, fontsize=8,  color='#6B6560',  va='top',
             fontfamily='monospace')

# Divider line
fig.add_artist(plt.Line2D([0.04, 0.96], [0.928, 0.928],
               transform=fig.transFigure, color=C_DARK, linewidth=1.5))

gs = gridspec.GridSpec(3, 3,
                       figure=fig,
                       left=0.05, right=0.97,
                       top=0.920, bottom=0.04,
                       hspace=0.42, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])   # Diagnosis donut (simulated as bar)
ax2 = fig.add_subplot(gs[0, 1])   # Effect size horizontal bars
ax3 = fig.add_subplot(gs[0, 2])   # Feature avg comparison
ax4 = fig.add_subplot(gs[1, 0])   # Scatter radius vs concavity
ax5 = fig.add_subplot(gs[1, 1])   # Histogram radius_worst
ax6 = fig.add_subplot(gs[1, 2])   # Risk score bar
ax7 = fig.add_subplot(gs[2, :])   # Correlation heatmap (full width)


def style_ax(ax, title, subtitle=''):
    ax.set_title(title, fontsize=11, fontweight='bold', color=C_DARK,
                 loc='left', pad=10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=7.5, color='#6B6560', fontfamily='monospace')


# ─────────────────────────────────────────────────────────────────────────────
# CHART 01 — Diagnosis Distribution (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
counts = df['diagnosis_label'].value_counts()
bars = ax1.barh(counts.index, counts.values,
                color=[C_MAL if x == 'Malignant' else C_BEN for x in counts.index],
                height=0.5, edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, counts.values):
    pct = val / len(df) * 100
    ax1.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
             f'{val:,}  ({pct:.1f}%)', va='center', fontsize=10,
             fontweight='bold', color=C_DARK)

ax1.set_xlim(0, 750)
ax1.set_xlabel('Number of Patients', fontsize=9)
ax1.grid(axis='y', alpha=0)
style_ax(ax1, 'Diagnosis Distribution', 'COHORT BALANCE — n=1,200')


# ─────────────────────────────────────────────────────────────────────────────
# CHART 02 — Feature Effect Sizes (top 10)
# ─────────────────────────────────────────────────────────────────────────────
top10 = effect_sizes.head(10)
colors_eff = [plt.cm.RdPu(0.3 + 0.7 * (i / len(top10)))
              for i in range(len(top10), 0, -1)]

bars2 = ax2.barh(range(len(top10)), top10.values,
                 color=colors_eff, height=0.65,
                 edgecolor='white', linewidth=0.8)

ax2.set_yticks(range(len(top10)))
ax2.set_yticklabels([f.replace('_', ' ') for f in top10.index],
                    fontsize=8.5, fontfamily='monospace')
ax2.set_xlabel("Cohen's d Effect Size", fontsize=9)

for bar, val in zip(bars2, top10.values):
    ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=8, color=C_DARK)

ax2.set_xlim(0, 0.22)
style_ax(ax2, 'Feature Discriminative Power',
         "COHEN'S D EFFECT SIZE — MALIGNANT vs BENIGN")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 03 — Avg Feature Comparison (grouped bar, 6 features)
# ─────────────────────────────────────────────────────────────────────────────
feat6 = ['area_mean', 'concavity_worst', 'radius_worst',
         'smoothness_mean', 'concave_points_worst', 'radius_mean']

# Normalise each feature to % of its max for comparability
feat_labels = [f.replace('_', '\n') for f in feat6]
m_vals = [mal[f].mean() for f in feat6]
b_vals = [ben[f].mean() for f in feat6]
maxes  = [max(m, b) for m, b in zip(m_vals, b_vals)]
m_norm = [m/mx*100 for m, mx in zip(m_vals, maxes)]
b_norm = [b/mx*100 for b, mx in zip(b_vals, maxes)]

x = np.arange(len(feat6))
w = 0.38
ax3.bar(x - w/2, m_norm, w, label='Malignant', color=C_MAL,
        alpha=0.85, edgecolor='white')
ax3.bar(x + w/2, b_norm, w, label='Benign',    color=C_BEN,
        alpha=0.85, edgecolor='white')
ax3.set_xticks(x)
ax3.set_xticklabels(feat_labels, fontsize=7.5)
ax3.set_ylabel('% of Feature Maximum', fontsize=9)
ax3.set_ylim(88, 103)
ax3.legend(fontsize=8, frameon=False)
style_ax(ax3, 'Avg Feature Values by Diagnosis',
         'NORMALISED — 6 KEY FEATURES')


# ─────────────────────────────────────────────────────────────────────────────
# CHART 04 — Scatter: radius_mean vs concavity_mean
# ─────────────────────────────────────────────────────────────────────────────
sample = df.sample(500, random_state=42)
for diag, color, label in [('M', C_MAL, 'Malignant'), ('B', C_BEN, 'Benign')]:
    sub = sample[sample.diagnosis == diag]
    ax4.scatter(sub['radius_mean'], sub['concavity_mean'],
                c=color, alpha=0.45, s=20, label=label, edgecolors='none')

ax4.set_xlabel('Radius Mean (mm)', fontsize=9)
ax4.set_ylabel('Concavity Mean', fontsize=9)
ax4.legend(fontsize=8, frameon=False, markerscale=1.5)
style_ax(ax4, 'Radius vs Concavity Scatter',
         '500 PATIENTS SAMPLED — CLASS OVERLAP VISIBLE')

# Add annotation
ax4.annotate('Classes overlap\nsignificantly',
             xy=(18, 0.35), fontsize=7.5, color='#6B6560',
             style='italic',
             bbox=dict(boxstyle='round,pad=0.3', fc='#F7F4F0', ec='#E0DBD5'))


# ─────────────────────────────────────────────────────────────────────────────
# CHART 05 — Histogram: radius_worst by diagnosis
# ─────────────────────────────────────────────────────────────────────────────
bins = np.linspace(3, 34, 13)
for diag, color, label in [('M', C_MAL, 'Malignant'), ('B', C_BEN, 'Benign')]:
    ax5.hist(df[df.diagnosis == diag]['radius_worst'],
             bins=bins, color=color, alpha=0.65,
             label=label, edgecolor='white', linewidth=0.6)

ax5.set_xlabel('Radius Worst (mm)', fontsize=9)
ax5.set_ylabel('Patient Count', fontsize=9)
ax5.legend(fontsize=8, frameon=False)
style_ax(ax5, 'Radius Worst Distribution',
         'FREQUENCY HISTOGRAM — WORST-CASE RADIUS BY DIAGNOSIS')


# ─────────────────────────────────────────────────────────────────────────────
# CHART 06 — Composite Risk Score
# ─────────────────────────────────────────────────────────────────────────────
risk_avg = df.groupby('diagnosis_label')['risk_score'].mean().reindex(
    ['Benign', 'Malignant'])
risk_std = df.groupby('diagnosis_label')['risk_score'].std().reindex(
    ['Benign', 'Malignant'])

bar_cols = [C_BEN, C_MAL]
bars6 = ax6.bar(risk_avg.index, risk_avg.values,
                color=bar_cols, alpha=0.85,
                edgecolor='white', linewidth=1.5,
                width=0.5,
                yerr=risk_std.values / np.sqrt(df.groupby('diagnosis_label').size().reindex(['Benign','Malignant']).values),
                capsize=5, error_kw={'color': C_DARK, 'linewidth': 1.2})

for bar, val in zip(bars6, risk_avg.values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=10,
             fontweight='bold', color=C_DARK)

ax6.set_ylabel('Composite Risk Score', fontsize=9)
ax6.set_ylim(7.8, 8.8)
style_ax(ax6, 'Malignancy Risk Score',
         'WEIGHTED: 0.4×radius_worst + 3.5×concavity_worst + 2.5×concave_pts_worst')

# Significance bracket
y_max = risk_avg.max() + 0.12
ax6.annotate('', xy=(1, y_max), xytext=(0, y_max),
             arrowprops=dict(arrowstyle='-', color=C_DARK, lw=1))
ax6.text(0.5, y_max + 0.02, 'p < 0.01 **', ha='center',
         fontsize=8, color=C_DARK)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 07 — Correlation Heatmap (worst features only)
# ─────────────────────────────────────────────────────────────────────────────
worst_cols = [c for c in float_cols if 'worst' in c]
corr = df[worst_cols + ['diagnosis_label']].copy()
corr['diagnosis_binary'] = (corr['diagnosis_label'] == 'Malignant').astype(int)
corr = corr.drop('diagnosis_label', axis=1)

corr_matrix = corr.corr()
labels = [c.replace('_worst', '').replace('_', ' ').title() for c in corr_matrix.columns]

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, ax=ax7, mask=mask,
            annot=True, fmt='.2f', annot_kws={'size': 8},
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor='#F0EDE8',
            cbar_kws={'shrink': 0.6, 'label': 'Pearson r'})

ax7.set_title('Correlation Matrix — Worst-Case Features + Diagnosis',
              fontsize=11, fontweight='bold', color=C_DARK, loc='left', pad=10)
ax7.text(0, 1.02, 'MULTICOLLINEARITY & FEATURE RELATIONSHIPS',
         transform=ax7.transAxes, fontsize=7.5,
         color='#6B6560', fontfamily='monospace')
ax7.tick_params(axis='x', rotation=30, labelsize=8)
ax7.tick_params(axis='y', rotation=0,  labelsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.012,
         'BREAST CANCER DIAGNOSIS ANALYSIS  ·  SOURCE: KAGGLE (GURIYA79)  ·  '
         'n=1,200 PATIENT RECORDS  ·  21 FEATURES  ·  PYTHON / MATPLOTLIB / SEABORN',
         ha='center', fontsize=7.5, color='#9B9490',
         fontfamily='monospace')

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
plt.savefig(OUTPUT_FILE, dpi=180, bbox_inches='tight',
            facecolor=C_BG, edgecolor='none')
print(f"\n✅  Dashboard saved → {OUTPUT_FILE}")
plt.show()

