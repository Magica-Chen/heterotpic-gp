"""Original simulations for the multivariate kriging experiments."""
from __future__ import annotations
from experiments.common import ROOT as PACKAGE_ROOT
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from experiments.geometry import global_bpi, summarise_pair_geometry
from experiments.oracle import OracleParams, W_matrix, average_oracle_gain, build_obs_cov, cross_cov_vector, kernel_matrix, residual_channel_terms, sample_train_test_outputs, stack_outputs
RESULTS_DIR = PACKAGE_ROOT / 'results/original_simulations'
FIGS_DIR = PACKAGE_ROOT / 'figures'
S4_REGIME_LABELS = {'rich': 'Within-region', 'moderate': 'Adjacent', 'poor': 'Remote'}
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 13, 'axes.labelweight': 'bold', 'axes.titlesize': 13, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'lines.linewidth': 2, 'text.usetex': False})

def _design_zero_overlap(regime: str, n1: int=15, n2: int=20) -> list[np.ndarray]:
    x1 = np.linspace(0.0, 1.0, n1)[:, None]
    if regime == 'interleaved':
        x2 = (np.linspace(0.0, 1.0, n2, endpoint=False) + 0.5 / n2)[:, None]
    elif regime == 'moderate':
        x2 = np.linspace(0.5, 1.5, n2)[:, None]
        for i in range(len(x2)):
            dists = np.abs(x2[i, 0] - x1[:, 0])
            if dists.min() < 0.0001:
                x2[i, 0] += 0.005
    elif regime == 'separated':
        x2 = np.linspace(1.5, 2.5, n2)[:, None]
    else:
        raise ValueError(regime)
    return [x1, x2]

def _plot_s1_design(xs: list[np.ndarray], regime: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 1.8))
    ax.scatter(xs[0][:, 0], np.zeros(len(xs[0])), label='Output 1', s=36, color='#0f4c5c', zorder=3)
    ax.scatter(xs[1][:, 0], np.ones(len(xs[1])), label='Output 2', s=36, color='#c1666b', zorder=3)
    ax.grid(False)
    ax.set_yticks([0, 1], ['Output 1', 'Output 2'])
    xmax = max(xs[0].max(), xs[1].max())
    ax.set_xlim(-0.05, xmax + 0.05)
    ax.set_xlabel('input location', fontweight='bold')
    if regime == 'interleaved':
        ax.legend(loc='upper center', ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f's1_design_{regime}.pdf')
    plt.close(fig)

def _plot_s1_coverage(geom, regime: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.plot(geom.coverage_radii, geom.dc_pq, label='$\\mathrm{DC}_{1\\to2}(r)$', color='#0f4c5c')
    ax.plot(geom.coverage_radii, geom.dc_qp, label='$\\mathrm{DC}_{2\\to1}(r)$', color='#c1666b', ls='--')
    ax.grid(False)
    ax.set_xlabel('radius $r$')
    ax.set_ylabel('directed coverage')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, loc='center right')
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f's1_coverage_{regime}.pdf')
    plt.close(fig)

def run_s2() -> pd.DataFrame:
    params = OracleParams(kernel='matern32', variance=1.0, lengthscale=0.12, Lambda=np.array([[1.0, 0.65], [0.65, 1.0]], dtype=float), noise=np.array([0.04, 0.04], dtype=float))
    x_star = np.array([[0.5]])
    motifs = {'residual': [np.array([[0.1], [0.9]]), np.array([[0.5], [0.85]])], 'redundant': [np.array([[0.45], [0.5], [0.55]]), np.array([[0.49], [0.5], [0.51]])], 'remote': [np.array([[0.45], [0.55]]), np.array([[0.05], [0.15]])]}
    rows = []
    for ell in (0.06, 0.12, 0.2):
        params.lengthscale = ell
        for name, (x1, x2) in motifs.items():
            vals = residual_channel_terms(x_star, x1, x2, params)
            vals.update({'experiment': 'S2', 'motif': name, 'lengthscale': ell})
            rows.append(vals)
    df = pd.DataFrame(rows)
    _plot_s2_residual_channel(df)
    return df

def _plot_s2_residual_channel(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 4.0))
    ax.grid(False)
    motif_style = {'residual': {'color': '#0f4c5c', 'label': 'Residual support'}, 'redundant': {'color': '#c1666b', 'label': 'Redundant support'}, 'remote': {'color': '#8a7d5d', 'label': 'Remote support'}}
    ell_markers = {0.06: 'o', 0.12: 's', 0.2: '^'}
    for motif, sub in df.groupby('motif'):
        sub = sub.sort_values('lengthscale')
        style = motif_style[motif]
        ax.plot(sub['resid_norm'], sub['delta'], color=style['color'], alpha=0.75, linewidth=1.2, zorder=2)
        for _, row in sub.iterrows():
            ax.scatter(row['resid_norm'], row['delta'], s=90, marker=ell_markers[round(float(row['lengthscale']), 2)], facecolor=style['color'], edgecolor='white', linewidth=0.9, zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('residual-channel norm $\\|\\mathbf{k}_2^{\\mathrm{res}}(x_\\star)\\|_2^2$')
    ax.set_ylabel('oracle gain $\\Delta_1(x_\\star)$')
    motif_handles = [Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=style['color'], markeredgecolor=style['color'], markersize=8, label=style['label']) for style in motif_style.values()]
    ell_handles = [Line2D([0], [0], marker=marker, linestyle='None', markerfacecolor='#666666', markeredgecolor='#666666', markersize=7, label=f'$\\ell={ell:.2f}$') for ell, marker in ell_markers.items()]
    legend1 = ax.legend(handles=motif_handles, title='Configuration', frameon=False, loc='lower right')
    ax.add_artist(legend1)
    ax.legend(handles=ell_handles, title='Lengthscale', frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIGS_DIR / 's2_residual_channel.pdf')
    plt.close(fig)

def _design_s5(regime: str, n: int=16) -> list[np.ndarray]:
    x1 = np.linspace(0.0, 1.0, n)[:, None]
    if regime == 'isotopic':
        x2 = x1.copy()
    elif regime == 'interleaved':
        x2 = (np.linspace(0.0, 1.0, n, endpoint=False) + 0.5 / n)[:, None]
    elif regime == 'separated':
        x2 = np.linspace(1.25, 2.25, n)[:, None]
    else:
        raise ValueError(regime)
    return [x1, x2]

def _fit_rho_s5(xs: list[np.ndarray], ys: list[np.ndarray], params: OracleParams) -> float:
    y_stack = stack_outputs(ys)

    def objective(theta: np.ndarray) -> float:
        rho = float(np.tanh(theta[0]))
        Sigma = build_obs_cov(xs, OracleParams(kernel=params.kernel, variance=params.variance, lengthscale=params.lengthscale, Lambda=np.array([[1.0, rho], [rho, 1.0]], dtype=float), noise=params.noise))
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            return 1000000000000.0
        alpha = np.linalg.solve(Sigma, y_stack)
        return float(0.5 * (logdet + y_stack @ alpha + len(y_stack) * np.log(2.0 * np.pi)))
    best = None
    for start in (-1.0, 0.0, 0.8, 1.5):
        res = minimize(objective, np.array([start], dtype=float), method='L-BFGS-B')
        if best is None or res.fun < best.fun:
            best = res
    return float(np.tanh(best.x[0]))

def _exact_predict(x_star: np.ndarray, xs: list[np.ndarray], ys: list[np.ndarray], params: OracleParams, target_j: int, joint: bool=True) -> tuple[float, float]:
    x_star = np.asarray(x_star, dtype=float)
    prior_var = params.Lambda[target_j, target_j] * params.variance
    if joint:
        Sigma = build_obs_cov(xs, params)
        c = np.concatenate(cross_cov_vector(x_star, xs, params, target_j))
        alpha = np.linalg.solve(Sigma, stack_outputs(ys))
        mean = float(c @ alpha)
        var = float(prior_var - c @ np.linalg.solve(Sigma, c))
        return (mean, max(var, 1e-08))
    K = params.Lambda[target_j, target_j] * kernel_matrix(params.kernel, xs[target_j], xs[target_j], params.variance, params.lengthscale) + params.noise[target_j] * np.eye(len(xs[target_j]))
    c_j = cross_cov_vector(x_star, xs, params, target_j)[target_j]
    alpha = np.linalg.solve(K, ys[target_j])
    mean = float(c_j @ alpha)
    var = float(prior_var - c_j @ np.linalg.solve(K, c_j))
    return (mean, max(var, 1e-08))

def _build_lmc_latent_cov(xs: list[np.ndarray], kernel: str, lengthscales: list[float], loadings: np.ndarray) -> np.ndarray:
    d = len(xs)
    blocks = []
    for p in range(d):
        row = []
        for q in range(d):
            block = np.zeros((len(xs[p]), len(xs[q])), dtype=float)
            for r, lengthscale in enumerate(lengthscales):
                block += loadings[p, r] * loadings[q, r] * kernel_matrix(kernel, xs[p], xs[q], variance=1.0, lengthscale=lengthscale)
            row.append(block)
        blocks.append(row)
    return np.block(blocks)

def _sample_lmc_train_test_outputs(rng: np.random.Generator, train_xs: list[np.ndarray], test_xs: list[np.ndarray], kernel: str, lengthscales: list[float], loadings: np.ndarray, noise: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    combo = [np.vstack([train_xs[j], test_xs[j]]) for j in range(len(train_xs))]
    Sigma_latent = _build_lmc_latent_cov(combo, kernel, lengthscales, loadings)
    latent = rng.multivariate_normal(np.zeros(Sigma_latent.shape[0]), Sigma_latent + 1e-08 * np.eye(Sigma_latent.shape[0]))
    train_f, train_y, test_f, test_y = ([], [], [], [])
    offset = 0
    for j, xj in enumerate(combo):
        n_total = len(xj)
        n_train = len(train_xs[j])
        f_j = latent[offset:offset + n_total]
        y_j = f_j + rng.normal(scale=np.sqrt(noise[j]), size=n_total)
        train_f.append(f_j[:n_train])
        train_y.append(y_j[:n_train])
        test_f.append(f_j[n_train:])
        test_y.append(y_j[n_train:])
        offset += n_total
    return (train_f, train_y, test_f, test_y)

def run_s5(seed: int=909, reps: int=80) -> pd.DataFrame:
    """Revisit univariate versus multivariate kriging under controlled geometry."""
    rng = np.random.default_rng(seed)
    true_params = OracleParams(kernel='matern32', variance=1.0, lengthscale=0.16, Lambda=np.array([[1.0, 0.7], [0.7, 1.0]], dtype=float), noise=np.array([0.01, 0.01], dtype=float))
    eval_grid = np.linspace(0.0, 1.0, 120)[:, None]
    rows = []
    for regime in ('isotopic', 'interleaved', 'separated'):
        xs = _design_s5(regime)
        geom = summarise_pair_geometry(xs[0], xs[1], np.linspace(0.0, 0.3, 60))
        b = global_bpi(xs, eval_grid)
        W12 = W_matrix(xs, true_params)[0, 1]
        oracle = average_oracle_gain(eval_grid, xs, true_params, 0)
        for rep in range(reps):
            _, ys, f_test, y_test = sample_train_test_outputs(rng, xs, [eval_grid, eval_grid], true_params)
            rho_hat = _fit_rho_s5(xs, ys, true_params)
            mv_params = OracleParams(kernel=true_params.kernel, variance=true_params.variance, lengthscale=true_params.lengthscale, Lambda=np.array([[1.0, rho_hat], [rho_hat, 1.0]], dtype=float), noise=true_params.noise)
            ind_params = OracleParams(kernel=true_params.kernel, variance=true_params.variance, lengthscale=true_params.lengthscale, Lambda=np.eye(2, dtype=float), noise=true_params.noise)
            mean_mv, var_mv, mean_ind, var_ind = ([], [], [], [])
            for x_star in eval_grid:
                m_mv, v_mv = _exact_predict(x_star[None, :], xs, ys, mv_params, 0, joint=True)
                m_ind, v_ind = _exact_predict(x_star[None, :], xs, ys, ind_params, 0, joint=False)
                mean_mv.append(m_mv)
                var_mv.append(v_mv + true_params.noise[0])
                mean_ind.append(m_ind)
                var_ind.append(v_ind + true_params.noise[0])
            mean_mv = np.asarray(mean_mv)
            var_mv = np.asarray(var_mv)
            mean_ind = np.asarray(mean_ind)
            var_ind = np.asarray(var_ind)
            log_mv = -0.5 * np.log(2.0 * np.pi * var_mv) - 0.5 * (y_test[0] - mean_mv) ** 2 / var_mv
            log_ind = -0.5 * np.log(2.0 * np.pi * var_ind) - 0.5 * (y_test[0] - mean_ind) ** 2 / var_ind
            rows.append({'experiment': 'S5', 'regime': regime, 'rep': rep, 'omega': geom.omega, 'tpi_1_to_2': geom.tpi_pq, 'B_1': b[0], 'W12': W12, 'avg_oracle_gain': oracle, 'rho_hat': rho_hat, 'rho_abs_error': abs(rho_hat - true_params.Lambda[0, 1]), 'rmse_diff': float(np.sqrt(np.mean((mean_ind - f_test[0]) ** 2)) - np.sqrt(np.mean((mean_mv - f_test[0]) ** 2))), 'mlpd_diff': float(np.mean(log_mv - log_ind))})
    return pd.DataFrame(rows)

def run_s5_misspecified(seed: int=1919, reps: int=80) -> pd.DataFrame:
    """Run Experiment 5 under a nonseparable two-component LMC data generator."""
    rng = np.random.default_rng(seed)
    fit_params = OracleParams(kernel='matern32', variance=1.0, lengthscale=0.16, Lambda=np.array([[1.0, 0.7], [0.7, 1.0]], dtype=float), noise=np.array([0.01, 0.01], dtype=float))
    true_lengthscales = [0.09, 0.34]
    true_loadings = np.array([[0.78, 0.44], [0.58, 0.28]], dtype=float)
    eval_grid = np.linspace(0.0, 1.0, 120)[:, None]
    rows = []
    for regime in ('isotopic', 'interleaved', 'separated'):
        xs = _design_s5(regime)
        geom = summarise_pair_geometry(xs[0], xs[1], np.linspace(0.0, 0.3, 60))
        b = global_bpi(xs, eval_grid)
        W12 = W_matrix(xs, fit_params)[0, 1]
        working_oracle = average_oracle_gain(eval_grid, xs, fit_params, 0)
        for rep in range(reps):
            _, ys, f_test, y_test = _sample_lmc_train_test_outputs(rng, xs, [eval_grid, eval_grid], kernel=fit_params.kernel, lengthscales=true_lengthscales, loadings=true_loadings, noise=fit_params.noise)
            rho_hat = _fit_rho_s5(xs, ys, fit_params)
            mv_params = OracleParams(kernel=fit_params.kernel, variance=fit_params.variance, lengthscale=fit_params.lengthscale, Lambda=np.array([[1.0, rho_hat], [rho_hat, 1.0]], dtype=float), noise=fit_params.noise)
            ind_params = OracleParams(kernel=fit_params.kernel, variance=fit_params.variance, lengthscale=fit_params.lengthscale, Lambda=np.eye(2, dtype=float), noise=fit_params.noise)
            mean_mv, var_mv, mean_ind, var_ind = ([], [], [], [])
            for x_star in eval_grid:
                m_mv, v_mv = _exact_predict(x_star[None, :], xs, ys, mv_params, 0, joint=True)
                m_ind, v_ind = _exact_predict(x_star[None, :], xs, ys, ind_params, 0, joint=False)
                mean_mv.append(m_mv)
                var_mv.append(v_mv + fit_params.noise[0])
                mean_ind.append(m_ind)
                var_ind.append(v_ind + fit_params.noise[0])
            mean_mv = np.asarray(mean_mv)
            var_mv = np.asarray(var_mv)
            mean_ind = np.asarray(mean_ind)
            var_ind = np.asarray(var_ind)
            log_mv = -0.5 * np.log(2.0 * np.pi * var_mv) - 0.5 * (y_test[0] - mean_mv) ** 2 / var_mv
            log_ind = -0.5 * np.log(2.0 * np.pi * var_ind) - 0.5 * (y_test[0] - mean_ind) ** 2 / var_ind
            rows.append({'experiment': 'S5-LMC', 'regime': regime, 'rep': rep, 'omega': geom.omega, 'tpi_1_to_2': geom.tpi_pq, 'B_1': b[0], 'W12': W12, 'working_oracle_gain': working_oracle, 'rho_hat': rho_hat, 'rmse_diff': float(np.sqrt(np.mean((mean_ind - f_test[0]) ** 2)) - np.sqrt(np.mean((mean_mv - f_test[0]) ** 2))), 'mlpd_diff': float(np.mean(log_mv - log_ind))})
    return pd.DataFrame(rows)

def write_tables(s1: pd.DataFrame | None=None, s2: pd.DataFrame | None=None, s3: pd.DataFrame | None=None, s4: pd.DataFrame | None=None, s5: pd.DataFrame | None=None, s5_misspec: pd.DataFrame | None=None, q1: pd.DataFrame | None=None) -> None:
    if s1 is not None:
        s1_t1 = s1[s1['target'] == 1] if 'target' in s1.columns else s1
        s1_tab = s1_t1.groupby('regime')[['omega', 'tpi_forward', 'W12', 'avg_oracle_gain']].mean().reindex(['interleaved', 'moderate', 'separated']).reset_index()
        s1_lines = ['\\begin{tabular}{lrrrr}', '\\hline', 'Regime & $\\omega$ & $\\tilde{\\pi}_{1\\to2}$ & $W_{12}$ & Oracle gain\\\\', '\\hline']
        for _, row in s1_tab.iterrows():
            s1_lines.append(f"{row['regime'].title()} & {row['omega']:.2f} & {row['tpi_forward']:.2f} & {row['W12']:.1f} & {row['avg_oracle_gain']:.4f}\\\\")
        s1_lines.extend(['\\hline', '\\end{tabular}'])
        (FIGS_DIR / 'table_s1_summary.tex').write_text('\n'.join(s1_lines))
    if s2 is not None:
        s2_tab = s2.copy()
        s2_tab['lower_ratio'] = s2_tab['lower_bound'] / s2_tab['delta']
        s2_tab['upper_ratio'] = s2_tab['upper_bound'] / s2_tab['delta']
        motif_labels = {'residual': 'Residual', 'redundant': 'Redundant', 'remote': 'Remote'}
        s2_lines = ['\\begin{tabular}{lrrr}', '\\hline', 'Configuration & $\\ell$ & Lower$/\\Delta_1(x_\\star)$ & Upper$/\\Delta_1(x_\\star)$\\\\', '\\hline']
        for motif in ['residual', 'redundant', 'remote']:
            sub = s2_tab[s2_tab['motif'] == motif].sort_values('lengthscale')
            for _, row in sub.iterrows():
                s2_lines.append(f"{motif_labels[motif]} & {row['lengthscale']:.2f} & {row['lower_ratio']:.3f} & {row['upper_ratio']:.1f}\\\\")
        s2_lines.extend(['\\hline', '\\end{tabular}'])
        (FIGS_DIR / 'table_s2_bounds.tex').write_text('\n'.join(s2_lines))
    if s3 is not None:
        s3_tab = s3.groupby('target')[['B_j', 'coupling', 'avg_oracle_gain', 'mlpd_diff']].mean().reset_index()
        s3_lines = ['\\begin{tabular}{rrrrr}', '\\hline', 'Output & $B_j$ & $R_j^2$ & Avg. gain & $\\Delta$MLPD\\\\', '\\hline']
        for _, row in s3_tab.iterrows():
            s3_lines.append(f"{int(row['target'])} & {row['B_j']:.3f} & {row['coupling']:.3f} & {row['avg_oracle_gain']:.3f} & {row['mlpd_diff']:.3f}\\\\")
        s3_lines.extend(['\\hline', '\\end{tabular}'])
        (FIGS_DIR / 'table_s3_summary.tex').write_text('\n'.join(s3_lines))
    if s4 is not None:
        s4_tab = s4.groupby('regime')[['W12', 'r12_error', 'avg_oracle_gain', 'avg_plugin_penalty', 'avg_realized_penalty', 'avg_plugin_margin']].mean().reindex(['rich', 'moderate', 'poor']).reset_index()
        labels = {key: '\\revboth{' + label + '}' for key, label in S4_REGIME_LABELS.items()}
        s4_lines = ['\\begin{tabular}{lrrrrrr}', '\\hline', 'Regime & $W_{12}$ & $|\\Delta R_{12}|$ & Avg. gain & Plug-in penalty & Realised cost & Plug-in margin\\\\', '\\hline']
        for _, row in s4_tab.iterrows():
            s4_lines.append(f"{labels[row['regime']]} & {row['W12']:.2f} & {row['r12_error']:.3f} & {row['avg_oracle_gain']:.4f} & {row['avg_plugin_penalty']:.4f} & {row['avg_realized_penalty']:.4f} & {row['avg_plugin_margin']:.4f}\\\\")
        s4_lines.extend(['\\hline', '\\end{tabular}'])
        (FIGS_DIR / 'table_s4_summary.tex').write_text('\n'.join(s4_lines))
    if s5 is not None:
        s5_mean = s5.groupby('regime')[['omega', 'tpi_1_to_2', 'B_1', 'W12', 'avg_oracle_gain', 'rho_abs_error', 'rmse_diff', 'mlpd_diff']].mean()
        s5_se = s5.groupby('regime')[['rmse_diff', 'mlpd_diff']].sem()
        s5_tab = s5_mean.reindex(['isotopic', 'interleaved', 'separated']).reset_index()
        labels = {'isotopic': 'Isotopic', 'interleaved': 'Interleaved', 'separated': 'Separated'}
        s5_lines = ['\\begin{tabular}{lrrrrrrr}', '\\hline', 'Regime & $\\omega$ & $\\tilde{\\pi}_{1\\to2}$ & $B_1$ & $W_{12}$ & Oracle gain & $\\Delta$RMSE & $\\Delta$MLPD\\\\', '\\hline']
        for _, row in s5_tab.iterrows():
            regime = row['regime']
            s5_lines.append(f"{labels[regime]} & {row['omega']:.2f} & {row['tpi_1_to_2']:.2f} & {row['B_1']:.3f} & {row['W12']:.1f} & {row['avg_oracle_gain']:.4f} & {row['rmse_diff']:.3f} ({s5_se.loc[regime, 'rmse_diff']:.3f}) & {row['mlpd_diff']:.3f} ({s5_se.loc[regime, 'mlpd_diff']:.3f})\\\\")
        s5_lines.extend(['\\hline', '\\end{tabular}'])
        (FIGS_DIR / 'table_s5_kriging.tex').write_text('\n'.join(s5_lines))
    if s5_misspec is not None:
        s5m_mean = s5_misspec.groupby('regime')[['omega', 'tpi_1_to_2', 'B_1', 'W12', 'working_oracle_gain', 'rmse_diff', 'mlpd_diff']].mean()
        s5m_se = s5_misspec.groupby('regime')[['rmse_diff', 'mlpd_diff']].sem()
        s5m_tab = s5m_mean.reindex(['isotopic', 'interleaved', 'separated']).reset_index()
        labels = {'isotopic': 'Isotopic', 'interleaved': 'Interleaved', 'separated': 'Separated'}
        s5m_lines = ['\\begin{tabular}{lrrrr}', '\\hline', 'Regime & $B_1$ & $W_{12}$ & $\\Delta$RMSE & $\\Delta$MLPD\\\\', '\\hline']
        for _, row in s5m_tab.iterrows():
            regime = row['regime']
            s5m_lines.append(f"{labels[regime]} & {row['B_1']:.3f} & {row['W12']:.1f} & {row['rmse_diff']:.3f} ({s5m_se.loc[regime, 'rmse_diff']:.3f}) & {row['mlpd_diff']:.3f} ({s5m_se.loc[regime, 'mlpd_diff']:.3f})\\\\")
        s5m_lines.extend(['\\hline', '\\end{tabular}'])
        (FIGS_DIR / 'table_s5_lmc_misspec.tex').write_text('\n'.join(s5m_lines))
    if q1 is not None:
        q1_mean = q1.groupby('regime')[['omega', 'tpi_1_to_2', 'B_1', 'W12', 'rho_hat', 'rmse_diff', 'mlpd_diff']].mean()
        q1_se = q1.groupby('regime')[['rmse_diff', 'mlpd_diff']].sem()
        q1_tab = q1_mean.reindex(['isotopic', 'interleaved', 'separated']).reset_index()
        labels = {'isotopic': 'Isotopic', 'interleaved': 'Interleaved', 'separated': 'Separated'}
        q1_lines = ['\\begin{tabular}{lrrrrrr}', '\\hline', 'Regime & $\\omega$ & $\\tilde{\\pi}_{1\\to2}$ & $B_1$ & $W_{12}$ & $\\Delta$RMSE & $\\Delta$MLPD\\\\', '\\hline']
        for _, row in q1_tab.iterrows():
            regime = row['regime']
            q1_lines.append(f"{labels[regime]} & {row['omega']:.2f} & {row['tpi_1_to_2']:.2f} & {row['B_1']:.3f} & {row['W12']:.1f} & {row['rmse_diff']:.3f} ({q1_se.loc[regime, 'rmse_diff']:.3f}) & {row['mlpd_diff']:.3f} ({q1_se.loc[regime, 'mlpd_diff']:.3f})\\\\")
        q1_lines.extend(['\\hline', '\\end{tabular}'])
        (FIGS_DIR / 'table_q1_queue.tex').write_text('\n'.join(q1_lines))
