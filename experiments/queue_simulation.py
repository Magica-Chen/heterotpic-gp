"""Q2 study for the multivariate kriging experiments."""
from time import perf_counter, process_time
import numpy as np
from experiments.queues import tandem_run
from experiments.synthetic import seed_for
GEOMETRIES = ('distributed', 'lower_half', 'outside')
MODELS = ('independent', 'mixture_independent', 'separable', 'autoregressive')

def simulate(xs, phase, geometry, rep):
    all_runs = []
    seeds = []
    costs = []
    for output, x in enumerate(xs):
        wall, cpu = (perf_counter(), process_time())
        values = []
        site_seeds = []
        n, w, runs = (20000, 2000, 8) if output == 0 else (2000, 200, 4)
        for index, point in enumerate(x * np.array([0.4, 1.25]) + np.array([0.35, 0.25])):
            local = []
            local_seeds = []
            for run in range(runs):
                seed = seed_for('Q2', phase, 'target' if output == 0 else geometry, rep, f'train_output{output}_site{index}_run{run}')
                local.append(tandem_run(*point, np.random.default_rng(seed), n, w, output == 0))
                local_seeds.append(seed)
            values.append(local)
            site_seeds.append(local_seeds)
        all_runs.append(np.array(values))
        seeds.append(site_seeds)
        costs.append({'output': output, 'runs': len(x) * runs, 'customers_per_run': n, 'warmup_per_run': w, 'wall_seconds': perf_counter() - wall, 'cpu_seconds': process_time() - cpu})
    return (all_runs, seeds, costs)
