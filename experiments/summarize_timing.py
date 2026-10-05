"""Paired execution-time summaries for the transfer experiments."""
from collections import defaultdict
import gzip
import json
import numpy as np
from experiments.common import ROOT
from experiments.synthetic import seed_for


def main(studies=('balanced', 'robustness', 'scales', 'q1', 'q2')):
    groups = defaultdict(list)
    for path in sorted((ROOT / 'runs/section7_alignment_v1/timing').rglob('*.json.gz')):
        with gzip.open(path, 'rt') as stream:
            row = json.load(stream)
        if row['study'] not in studies:
            continue
        for branch, values in row['branches'].items():
            groups[row['study'], branch].append((row['scenario']['id'], row['replication'],
                row['scenario']['n1'], values['screen']['runtime_seconds'], values['direct']['runtime_seconds']))
    summary = {}
    for (study, branch), values in groups.items():
        x = np.asarray([[r[3], r[4]] for r in values])
        rng = np.random.default_rng(seed_for('section7_alignment_v1', 'timing', study + '_' + branch, 0))
        boot = np.zeros((2000, 2))
        strata = defaultdict(lambda: defaultdict(list))
        for index, record in enumerate(values):
            sid, replication, n1 = record[:3]
            stratum = sid if study not in ('q1', 'q2') else ('all' if study == 'q1' else str(n1))
            strata[stratum][replication].append(index)
        for stratum in sorted(strata):
            cells = np.array([x[indices].sum(axis=0) for _, indices in sorted(strata[stratum].items())])
            boot += cells[rng.integers(len(cells), size=(2000, len(cells)))].sum(axis=1) / len(x)
        saving = dict(estimate=float(100 * (1 - x[:, 0].mean() / x[:, 1].mean())),
                      ci95=np.quantile(100 * (1 - boot[:, 0] / boot[:, 1]), [.025, .975]).tolist())
        summary[study + '_' + branch] = dict(pairs=len(x), screen_seconds=float(x[:, 0].mean()),
                                            direct_seconds=float(x[:, 1].mean()), saving_percent=saving)
    destination = ROOT / 'results/section7_alignment_v1/timing_summary.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + '\n')
