"""Pair extensions for the multivariate kriging experiments."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from experiments.exact_gp import ExactGP, fit_gp
from experiments.screening import fit_marginals, fit_pair, pair_scores, gate, site_folds

def fit_models(xs, ys, config, known=None):
    if known is None:
        marginals = fit_marginals(xs, ys, config)
        return (marginals, fit_pair(xs, ys, marginals, config))
    marginals = []
    for j, (x, y) in enumerate(zip(xs, ys)):
        model = ExactGP([x], [y], config)
        theta = model.pack([known['signal_variance'][j]], [known['noise_variance'][j]], known['lengthscale'], np.eye(1))
        marginals.append(fit_gp([x], [y], config, fixed_theta=theta))
    model = ExactGP(xs, ys, config)
    theta = model.pack(known['signal_variance'], known['noise_variance'], known['lengthscale'], np.eye(len(xs)))
    return (marginals, fit_gp(xs, ys, config, fixed_theta=theta))

def paired_traces(xs, ys, evaluation, fold_seed, config, screen, known=None, envelope_sensitivity=False):
    """Training-only fitting API: no oracle predictions or assessment outcomes."""
    marginals, joint = fit_models(xs, ys, config, known)
    scores = pair_scores(marginals, evaluation, screen)
    alternative = pair_scores(marginals, evaluation, replace(screen, rho_max=0.995)) if envelope_sensitivity else None
    membership = site_folds(xs, fold_seed, screen.folds)
    folds = []
    for k in range(screen.folds):
        masks = [m != k for m in membership]
        x = [a[m] for a, m in zip(xs, masks)]
        y = [a[m] for a, m in zip(ys, masks)]
        validation_x, validation_y = (xs[0][~masks[0]], ys[0][~masks[0]])
        marginal, combined = fit_models(x, y, config, known)
        score = pair_scores(marginal, evaluation, screen)
        a = marginal[0].predict(validation_x)[0]
        b = combined.predict(validation_x)[0] if combined.success else a
        folds.append({'fold': k, 'train_indices': [np.flatnonzero(m).tolist() for m in masks], 'validation_target_indices': np.flatnonzero(~masks[0]).tolist(), 'count': len(validation_y), 'independent_sse': float(np.sum((a - validation_y) ** 2)), 'joint_sse': float(np.sum((b - validation_y) ** 2)), 'scores': score, 'gate': gate(score, screen.opportunity_threshold, screen.information_threshold), 'marginals': [f.record() for f in marginal], 'joint': combined.record(), 'scores_rho995': pair_scores(marginal, evaluation, replace(screen, rho_max=0.995)) if envelope_sensitivity else None})
    return (marginals, joint, scores, alternative, folds)
