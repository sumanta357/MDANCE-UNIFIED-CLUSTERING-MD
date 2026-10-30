"""Phase 5 - cluster validation and optimal-model selection (pipeline-added,
built on MDANCE-native distance definitions).

Every metric records its assumptions; metrics whose inputs are invalid for a
solution are reported as missing (None) and excluded from the composite
ranking with weight renormalization (documented behavior).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

from mdance.tools import bts  # MDANCE-native


@dataclass
class Metrics:
    name: str
    family: str
    native: bool
    params: dict
    n_clusters: int
    population_max: float
    population_min: float
    n_singletons: int
    silhouette: float | None = None
    davies_bouldin: float | None = None
    calinski_harabasz: float | None = None
    compactness: float | None = None       # mean within-cluster MSD (to medoid)
    separation: float | None = None        # min MSD between cluster medoids
    diversity: float | None = None         # inter-/intra- MSD ratio
    stability: float | None = None         # bootstrap bootstrap-ARI
    time_s: float = 0.0
    mem_mb: float = 0.0
    error: str | None = None
    composite: float | None = None
    rank: int | None = None
    component_scores: dict = field(default_factory=dict)


def db_distance_space(D: np.ndarray, labels: np.ndarray,
                      medoids: dict) -> float | None:
    """Davies-Bouldin generalized to MSD space (pipeline-added).

    For cluster i: S_i = mean MSD of members to cluster medoid; and
    M_ij = MSD(medoid_i, medoid_j); DB = mean_i max_{j≠i} (S_i+S_j)/M_ij.
    Requires ≥2 clusters, no singletons and separable medoids.
    """
    ids = sorted(medoids.keys())
    if len(ids) < 2:
        return None
    sizes = {c: int((labels == c).sum()) for c in ids}
    if any(s < 2 for s in sizes.values()):
        return None
    S = {c: float(D[labels == c, medoids[c]].mean()) for c in ids}
    M = {(i, j): float(D[medoids[i], medoids[j]]) for i in ids for j in ids if i != j}
    vals = []
    for i in ids:
        best = max((S[i] + S[j]) / max(M[(i, j)], 1e-12) for j in ids if j != i)
        vals.append(best)
    return float(np.mean(vals))


def ch_distance_space(D: np.ndarray, labels: np.ndarray,
                      medoids: dict) -> float | None:
    """Calinski-Harabasz generalized to MSD space (pipeline-added).

    W = Σ_k n_k · mean-MSD(members → medoid_k) (within dispersion proxy)
    B = Σ_k n_k · MSD(medoid_k → global medoid)   (between dispersion proxy)
    CH = [B/(k-1)] / [W/(n-k)]
    """
    n = len(labels)
    ids = sorted(medoids.keys())
    k = len(ids)
    if k < 2:
        return None
    cluster_sizes = {c: int((labels == c).sum()) for c in ids}
    if any(s < 1 for s in cluster_sizes.values()):
        return None
    W = sum(cluster_sizes[c] * float(D[labels == c, medoids[c]].mean())
            for c in ids)
    global_medoid = int(np.argmin(D.mean(axis=1)))
    B = sum(cluster_sizes[c] * float(D[c_, global_medoid]) for c, c_ in medoids.items())
    if W <= 0 or n <= k:
        return None
    ch = (B / (k - 1)) / (W * (n - k) / n)
    return float(ch)


def cluster_medoids(D, labels, ids=None):
    """Per-cluster medoid indices (pipeline-added; accurate argmin of the mean 
    within-cluster distance row of each member)."""
    medoids = {}
    clusters = np.unique(labels[labels >= 0])
    for c in clusters:
        idx = np.flatnonzero(labels == c)
        if len(idx) == 0:
            continue
        mean_disp = D[idx].mean(axis=1)
        medoids[int(c)] = int(idx[np.argmin(mean_disp)])
    return medoids


def evaluate_solution(sol, D, X, N_atoms, t_override=None) -> Metrics:
    """Compute all valid validation metrics for one solution."""
    labels = sol.labels
    n = len(labels)
    m = Metrics(name=sol.name, family=sol.family, native=sol.native,
                params=sol.params, n_clusters=sol.n_clusters,
                population_max=0.0, population_min=0.0, n_singletons=0,
                time_s=sol.time_s, mem_mb=sol.mem_mb, error=sol.error)
    if labels.size == 0 or (labels < 0).all():
        m.error = m.error or "no labels"
        return m
    clusters = np.unique(labels[labels >= 0])
    sizes = np.array([(labels == c).sum() for c in clusters], dtype=float)
    fracs = sizes / n
    m.population_max, m.population_min = float(fracs.max()), float(fracs.min())
    m.n_singletons = int((sizes == 1).sum())
    if D is None:
        return m
    meds = cluster_medoids(D, labels)

    # silhouette (precomputed MSD; needs >= 2 clusters, no singleton cluster
    # without another member? sklearn requires each cluster size >= 2? It
    # requires 2 <= n_labels <= n_samples-1)
    if 2 <= len(clusters) <= n - 1 and list(clusters) and (
            sizes.min() >= 1):
        try:
            m.silhouette = float(silhouette_score(D, labels, metric="precomputed"))
        except Exception:
            m.silhouette = None
    m.davies_bouldin = db_distance_space(D, labels, meds)
    m.calinski_harabasz = ch_distance_space(D, labels, meds)

    # compactness: population-weighted mean within-cluster MSD to medoid
    if len(meds) == len(clusters):
        w = sizes / n
        m.compactness = float(sum(
            fr * float(D[labels == c, meds[c]].mean())
            for c, fr in zip(meds.keys(), w)))
        # separation: min inter-cluster medoid MSD
        if len(meds) >= 2:
            med_IDS = list(meds.values())
            sep = [float(D[a, b]) for ai, a in enumerate(med_IDS)
                   for b in med_IDS[ai + 1:]]
            m.separation = float(min(sep))
        # structural diversity: mean inter-cluster MSD / mean intra-cluster MSD
        inter, intra = _diversity_terms(D, labels)
        if intra > 0:
            m.diversity = float(inter / intra)
    if t_override is not None:
        m.time_s = float(t_override)
    return m


def _diversity_terms(D: np.ndarray, labels: np.ndarray):
    """(mean inter-cluster cross MSD, mean within-cluster MSD) helpers."""
    clusters = np.unique(labels[labels >= 0])
    inter_vals, intra_vals = [], []
    for i, c1 in enumerate(clusters):
        idx1 = np.flatnonzero(labels == c1)
        if len(idx1) > 1:
            sub = D[idx1][:, idx1]
            iu = np.triu_indices(len(idx1), k=1)
            if iu[0].size:
                intra_vals.append(sub[iu].mean())
        for c2 in clusters[i + 1:]:
            idx2 = np.flatnonzero(labels == c2)
            inter_vals.append(D[idx1][:, idx2].mean())
    if not inter_vals or not intra_vals:
        return 0.0, 0.0
    return float(np.mean(inter_vals)), float(np.mean(intra_vals))


def bootstrap_stability(sol, X, D, cfg, seed) -> float | None:
    """Bootstrap subsample stability via ARI (pipeline-added).

    Re-running the same method/params on `frac`·n resampled frames and
    comparing to the full-data partition through adjusted Rand index on the
    common subset.
    """
    b = cfg["validation"]
    if not b.get("bootstrap_enabled", True) or sol.rerun is None:
        return None
    n = X.shape[0]
    frac = float(b.get("bootstrap_frac", 0.8))
    reps = int(b.get("bootstrap_n", 5))
    rng = np.random.default_rng(seed)
    seeds_tot = []
    for _ in range(reps):
        m = max(2, int(round(frac * n)))
        subset = np.sort(rng.choice(n, size=m, replace=False))
        try:
            sub_labels = sol.rerun(subset)
        except Exception:
            continue
        full_labels = sol.labels[subset]
        if len(np.unique(full_labels)) < 2 or len(np.unique(sub_labels)) < 2:
            continue
        seeds_tot.append(float(adjusted_rand_score(full_labels, sub_labels)))
    if not seeds_tot:
        return None
    return float(np.mean(seeds_tot))


def rank_solutions(metrics: list[Metrics], cfg) -> list[Metrics]:
    """Transparent weighted model ranking (pipeline-added).

    Normalization: per metric, min-max across all solutions having the value;
    direction fixed per metric; missing metrics are excluded per-solution and
    remaining weights are renormalized (documented, no silent bias).
    """
    directions = {  # True = larger is better
        "silhouette": True, "davies_bouldin": False, "calinski_harabasz": True,
        "stability": True, "population_balance": True, "compactness": False,
        "separation": True, "diversity": True,
    }
    arrays = {key: [] for key in directions}
    for m in metrics:
        arrays["silhouette"].append(m.silhouette)
        arrays["davies_bouldin"].append(m.davies_bouldin)
        arrays["calinski_harabasz"].append(m.calinski_harabasz)
        arrays["stability"].append(m.stability)
        arrays["population_balance"].append(
            m.population_min / m.population_max if m.population_max > 0 else None)
        arrays["compactness"].append(m.compactness)
        arrays["separation"].append(m.separation)
        arrays["diversity"].append(m.diversity)
    scaled = {}
    from mdance_pipeline.utils import minmax_scale
    for key, vals in arrays.items():
        arr = np.array([np.nan if v is None else float(v) for v in vals])
        out = np.full(len(arr), np.nan)
        valid = np.isfinite(arr)
        if valid.any():
            out[valid] = minmax_scale(arr[valid], invert=not directions[key])
        scaled[key] = out
    weights = dict(cfg["validation"]["weights"])
    k_min = max(2, int(cfg["clustering"].get("k_min", 2)))
    k_max = int(cfg["clustering"].get("k_max", 8))
    for m_i, m in enumerate(metrics):
        # degenerate-partition guard (documented): a single cluster, an
        # over-resolved partition (more clusters than requested k_max) or a
        # mostly-singleton split is not a useful conformational model.
        if m.n_clusters < k_min or m.n_clusters > k_max or (
                m.n_singletons >= m.n_clusters and m.n_clusters > 2):
            m.composite = None
            m.component_scores = {"excluded": "degenerate partition"}
            continue
        total_w, total = 0.0, 0.0
        comp = {}
        for key, w in weights.items():
            v = scaled[key][m_i]
            if v is not None and np.isfinite(v):
                total += float(w) * float(v)
                total_w += float(w)
                comp[key] = round(float(v) * 100.0, 2)
        m.component_scores = comp
        m.composite = round(100.0 * total / total_w, 2) if total_w > 0 else None
    valid = [m for m in metrics if m.composite is not None]
    valid.sort(key=lambda m: -m.composite)
    for rank, m in enumerate(valid, 1):
        m.rank = rank
    return valid


def filter_params_for_metrics(cfg, sols, logger):
    """Limit expensive stability evaluation to top solutions (two-stage)."""
    top = max(1, int(cfg["validation"].get("bootstrap_top_solutions", 5)))
    return top
