from experiments.common import ROOT as PACKAGE_ROOT
import json
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
ROOT = PACKAGE_ROOT

def geometry_figure():
    source = ROOT / 'results/E6_task_v2/geometry-correlation.json'
    cells = json.loads(source.read_text())
    geometries = ('colocated', 'infill', 'partial', 'remote')
    heatmaps = [np.zeros((4, 3)) for _ in range(3)]
    for g, geometry in enumerate(geometries):
        for j, rho in enumerate((0.0, 0.4, 0.8)):
            row = cells[f'{geometry}_{rho:g}']
            diagonal = row['diagonal']['risk']['estimate']
            joint = row['joint']['risk']['estimate']
            direct_fits = row['direct']['joint_fits_mean']['estimate']
            screen_fits = row['screen']['joint_fits_mean']['estimate']
            heatmaps[0][g, j] = 100 * (1 - joint / diagonal)
            heatmaps[1][g, j] = row['screen']['risk_reduction_percent']['estimate']
            heatmaps[2][g, j] = 100 * (1 - screen_fits / direct_fits)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.75), layout='constrained')
    titles = ['(a) Coupling: JOINT vs DIAG', '(b) Screened prediction vs SEP', '(c) Coupled fits avoided']
    for index, (axis, values) in enumerate(zip(axes, heatmaps)):
        if index < 2:
            limit = max(5, float(np.ceil(np.max(np.abs(values)) / 5) * 5))
            plot = axis.imshow(values, cmap='RdBu', aspect='auto', norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit))
        else:
            plot = axis.imshow(values, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        axis.set_title(titles[index], fontsize=8, pad=7)
        axis.set_xticks(range(3), ['0', '0.4', '0.8'])
        axis.set_xlabel('Generating correlation $\\rho$')
        axis.set_yticks(range(4), ['Isotopic', 'Infill', 'Partial coverage', 'Remote'] if index == 0 else [''] * 4)
        axis.tick_params(length=0)
        for i in range(4):
            for j in range(3):
                rgba = plot.cmap(plot.norm(values[i, j]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                shown = 0.0 if abs(values[i, j]) < 0.05 else values[i, j]
                axis.text(j, i, f'{shown:.1f}', ha='center', va='center', fontsize=8, color='white' if luminance < 0.5 else 'black')
        bar = fig.colorbar(plot, ax=axis, location='bottom', shrink=0.9, pad=0.08, aspect=20)
        bar.ax.tick_params(labelsize=7)
        bar.set_label('Risk reduction (%)' if index < 2 else 'Fit reduction (%)', fontsize=7)
    outputs = []
    for extension in ('pdf', 'png'):
        output = ROOT / f'figures/e6_task_geometry_labels.{extension}'
        kwargs = {'metadata': {'CreationDate': None, 'ModDate': None}} if extension == 'pdf' else {'dpi': 180}
        fig.savefig(output, bbox_inches='tight', **kwargs)
        outputs.append({'output': str(output.relative_to(ROOT))})
    plt.close(fig)
    return {'source': str(source.relative_to(ROOT)), 'values': [values.tolist() for values in heatmaps], 'outputs': outputs}
