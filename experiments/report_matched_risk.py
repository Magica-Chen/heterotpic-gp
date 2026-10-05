"""Report matched risk for the multivariate kriging experiments."""
from experiments.common import ROOT as PACKAGE_ROOT
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = PACKAGE_ROOT
source = ROOT / 'results/E4C_v1/summary.json'
rows = json.loads(source.read_text())['scenarios']
geometries = ('rich', 'moderate', 'poor')
geometry_labels = {'rich': 'Within-region', 'moderate': 'Adjacent', 'poor': 'Remote'}
scales = (0.08, 0.12, 0.2)
styles = (('correlation_only', 'Correlation only', '#1b6a86', 'o', -0.09), ('full', 'Full covariance', '#b24e36', 's', 0.09))
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.45), sharey=True)
used = []
for ax, scale in zip(axes, scales):
    for mode, label, color, marker, shift in styles:
        subset = [next((r for r in rows if r['scenario']['geometry'] == geometry and r['scenario']['lengthscale'] == scale and (r['scenario']['fit_mode'] == mode))) for geometry in geometries]
        values = np.array([r['relative_mean_risk_reduction_percent'] for r in subset])
        bounds = np.array([r['relative_mean_risk_reduction_percent_ci95'] for r in subset])
        ax.errorbar(np.arange(3) + shift, values, yerr=np.array([values - bounds[:, 0], bounds[:, 1] - values]), color=color, marker=marker, linestyle='none', capsize=3, markersize=5, label=label)
        used.extend((dict(scenario=r['scenario']['id'], mean=float(v), ci95=ci.tolist()) for r, v, ci in zip(subset, values, bounds)))
    ax.axhline(0, color='0.4', linewidth=0.8)
    ax.set_xticks(np.arange(3), [geometry_labels[g] for g in geometries])
    ax.set_xlim(-0.4, 2.4)
    ax.set_title('$\\ell=' + f'{scale:.2f}' + '$')
    ax.grid(axis='y', alpha=0.18)
axes[0].set_ylabel('Latent-risk reduction (%)\nrelative to matching independent fit')
axes[1].legend(loc='upper center', frameon=False, fontsize=9)
fig.tight_layout()
pdf = ROOT / 'figures/e4c_matched_risk.pdf'
png = ROOT / 'figures/e4c_matched_risk.png'
fig.savefig(pdf, metadata={'CreationDate': None, 'ModDate': None})
fig.savefig(png, dpi=180)
plt.close(fig)
