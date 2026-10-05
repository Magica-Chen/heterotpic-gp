# Multivariate kriging: code and experiments

This package contains the scientific implementation, experiment configurations,
required input data and compact results for the manuscript and supplementary
information. It runs from an extracted directory using Python and the listed
dependencies. All paths are relative to the package or the selected output directory.

Repository: [Magica-Chen/heterotpic-gp](https://github.com/Magica-Chen/heterotpic-gp).
Clone the repository to obtain the code, required input data and saved results:

```bash
git clone https://github.com/Magica-Chen/heterotpic-gp.git
cd heterotpic-gp
```

## Installation

Use Python 3.11. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The supplied environment uses Linux and CPU float64 calculations. A GPU is
optional for the separate JAX model implementation in `mvgp_heter/`.

## Recreate figures and tables from the supplied results

```bash
python make_tables.py --output paper_outputs
```

This command recomputes the deterministic geometry experiments and renders the
fitted-experiment summaries. It produces PDF and PNG figures in
`paper_outputs/figures/`, LaTeX tables in `paper_outputs/tables/` and
`paper_outputs/figures/`, and numerical CSV files in `paper_outputs/csv/`.
`TABLES.txt` lists the generated table fragments. The CSV files retain the full
precision of the saved summaries; their `statistic` column identifies the nested
quantity, scenario and method. LaTeX fragments use `booktabs` and, for long tables,
`longtable`.

## Rerun the experiments

```bash
python run_experiments.py --list
python run_experiments.py --study all --workers 4 --output experiment_outputs
python make_tables.py --runs experiment_outputs
```

The complete run generates new datasets or reads the supplied application inputs,
fits the models, estimates uncertainty and measures complete-procedure execution
times. It includes the original 80-dataset comparison and the later fitted
experiments. Allow several hours and several GB of output space for the complete
suite. Elapsed time depends on the CPU and worker count. Timing measurements use
one worker and one BLAS/OpenMP thread; recorded times vary with hardware and load.

To run one experiment, substitute its identifier:

```bash
python run_experiments.py --study II-A --workers 4 --output experiment_outputs
python run_experiments.py --study AQS --workers 4 --output experiment_outputs
```

| Identifier | Manuscript / SI | Experiment and default size | Result directory |
|---|---|---|---|
| `geometry` | Section 8.1 / D | I-A–I-C: deterministic design geometry, residual channels and oracle gains | `paper_geometry` |
| `II-A` | Section 8.2 / E | 18 matched design/scale/fitting cells; 7,200 datasets | `E4C_v1` |
| `II-B` | Section 8.2 / E | Six separable/LMC design cells; 1,800 datasets, paired fitting modes | `E5_extension_v1` |
| `II-C` | Section 8.2 / E | 12 penalty cells; 6,000 outer datasets and 72,000 refit draws | `E7_v1` |
| `III` | Section 8.3 / F | 48 screening cells; 9,600 evaluation datasets and the two-candidate comparator | `E6_task_v2` |
| `IV-A` | Section 8.4 / G | Six covariance/noise families; 7,200 datasets | `E8_v1` |
| `IV-B` | Section 8.4 / G | 16 auxiliary-allocation cells; 8,000 datasets | `E8_opportunity_v2` |
| `V` | Section 8.5 / H | Balanced, robustness, scale and queue transfer controls; 22,800 datasets | `section7_alignment_v1` |
| `TQA` | Section 8.6 / I | Nine tandem-queue cells; 1,800 evaluation datasets | `Q2_calibrated_v3` |
| `AQS` | Section 8.6 / I | 910 folds with primary and mean-instrument responses; 1,820 fits of each model set | `AQS_task_v1` |
| `AQS-LMC` | Section 8.6 / I | 650 primary-instrument, unbuffered folds | `AQS_LMC_v1` |
| `original-II-B` | SI E | Six original comparison cells, 80 datasets each | `original_simulations` |

Directories in the last column are inside `experiment_outputs/results/`.
Per-dataset predictions and fitted parameters are written to `experiment_outputs/runs/`.
Study V can be restricted to a named component:

```bash
python run_experiments.py --study V --transfer-studies scales --workers 4 --output scale_outputs
```

The AQS-LMC command computes the required primary predictions when they are absent.
The complete suite runs AQS before AQS-LMC and reuses those predictions.

### Calibration and separate stages

The supplied configurations include the independently selected screening thresholds.
Evaluation uses these settings. To regenerate calibration before evaluation:

```bash
python run_experiments.py --study III --phase calibration --workers 4 --output recalibrated
python run_experiments.py --study III --workers 4 --output recalibrated
python run_experiments.py --study TQA --phase calibration --workers 4 --output recalibrated
python run_experiments.py --study TQA --workers 4 --output recalibrated
```

Calibration has 4,800 datasets for III and 900 for TQA. It writes the selected
settings inside the output directory. The package's supplied configurations stay
available for the original evaluation.

Use `--stage fit`, `--stage analyze` or `--stage timing` to run a stage separately.
The default `--stage all` runs all applicable stages. Timing applies to III, V and
TQA. A smaller exploratory fit can be run with:

```bash
python run_experiments.py --study III --stage fit --replicates 2 --limit-scenarios 1 --workers 1 --output small_example
```

Small runs write fitted records. The paper's tables and uncertainty summaries use
the complete configurations. Completed dataset files are reused when a command is
restarted. Choose a new output directory after changing settings or seeds.

## Code and data

- `experiments/`: exact Gaussian-process fitting, noise-aware and richer covariance
  models, design geometry, screening, simulation, analysis and plotting routines.
- `mvgp_heter/`: the JAX multivariate heteroscedastic GP implementation.
- `configuration/`: sample sizes, designs, fitting bounds and tolerances, bootstrap
  settings and selected calibration rules. The identifiers also name output folders.
- `data/aqs/`: the 2024 EPA annual monitor file, selected analysis rows, eligible
  instruments, coordinates and the actual outer folds and spatial bootstrap blocks.
  Data source: [EPA AirData](https://aqs.epa.gov/aqsweb/airdata/download_files.html).
  `analysis.csv` is the modelling input; transformations of responses are estimated
  within each training fold.
- `data/queue_inputs/`: the simulator responses, simulation variances, evaluation
  targets and target-fold assignments consumed by the supplementary queue transfer
  experiments. These are input datasets for the fitted comparison.
- `data/queue_reference/`: independent calibration and evaluation reference
  simulations for TQA, including run values used in the uncertainty calculation.
- `data/original_simulations/`: compact per-dataset values for the original
  80-dataset comparisons. `original-II-B` regenerates these values.
- `reference_results/`: compact saved numerical summaries, confidence intervals
  and timings corresponding to the submitted paper.

Random seeds are generated by `experiments/synthetic.py::seed_for` from the fixed
experiment, phase, scenario and replication identifiers. This retains the original
random streams across worker counts. The scripts keep numerical fitting guards
and fallbacks that define the reported methods, including covariance factorization
and optimizer-failure handling.

The saved results contain the reported calculations. Reproductions retain the
same numerical methods, experimental settings and seeds. Floating-point libraries
can affect optimizer stopping and the last digits of fitted results.

The software license is in `LICENSE`.
