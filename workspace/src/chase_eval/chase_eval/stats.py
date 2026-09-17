"""Aggregate statistics for RL claims: IQM + bootstrap CIs.

The rliable protocol (Agarwal et al., NeurIPS 2021) adopted by
offline_training_recipe.md 6.4 -- implemented natively (the dependency is
30 lines) so the eval stack stays installable anywhere: interquartile mean
over the pooled per-seed scores, stratified bootstrap resampling seeds, and
percentile confidence intervals. Error bars are not politeness: the seed is
a hyperparameter of the result (guide 24).
"""
from typing import Dict, Sequence, Tuple

import numpy as np


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: mean of the middle 50 % of scores."""
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0:
        return float('nan')
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    mid = v[lo:hi]
    return float(mid.mean()) if len(mid) else float(v.mean())


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0
                 ) -> Tuple[float, float, float]:
    """(IQM, ci_lo, ci_hi) for one sample with no seed axis (e.g. the
    P-controller baseline): plain percentile bootstrap over episodes."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    point = iqm(v)
    if len(v) < 2:
        return point, float('nan'), float('nan')
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = iqm(rng.choice(v, size=len(v), replace=True))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def stratified_bootstrap_ci(per_seed_scores: Dict[int, Sequence[float]],
                            n_boot: int = 2000, alpha: float = 0.05,
                            seed: int = 0) -> Tuple[float, float, float]:
    """(IQM, ci_lo, ci_hi) resampling SEEDS (the unit of replication),
    then episodes within each resampled seed."""
    rng = np.random.default_rng(seed)
    seeds = list(per_seed_scores)
    pooled = [s for scores in per_seed_scores.values() for s in scores]
    point = iqm(pooled)
    if len(seeds) < 2:
        return point, float('nan'), float('nan')
    stats = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        sample = []
        for s in chosen:
            scores = np.asarray(per_seed_scores[s], dtype=float)
            sample.extend(rng.choice(scores, size=len(scores), replace=True))
        stats[b] = iqm(sample)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
