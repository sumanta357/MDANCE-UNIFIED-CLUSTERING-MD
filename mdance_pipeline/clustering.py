"""Phase 4 - clustering with MDANCE-native methods + documented baselines.

MDANCE-native classes used (verified against installed MDANCE 0.4.0):
  KmeansNANI, Divine, ExtendedQuality, HELM, mdBirch
Pipeline-added baselines (used explicitly for GROMACS-style benchmarking):
  sklearn Agglomerative (precomputed MSD), DBSCAN (precomputed MSD) and a
  faithful re-implementation of the Daura/gromos greedy clustering that backs
  `gmx cluster -m gromos`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

from mdance.cluster import divine as md_divine     # MDANCE-native
from mdance.cluster import equal as md_equal       # MDANCE-native (eQual)
from mdance.cluster import helm as md_helm         # MDANCE-native (HELM)
from mdance.cluster import mdbirch as md_mdbirch   # MDANCE-native
from mdance.cluster import nani as md_nani         # MDANCE-native


def _peak():
    from mdance_pipeline.utils import peak_memory_mb
    return peak_memory_mb()


@dataclass
class Solution:
    """One complete clustering solution (pipeline-added container)."""
    name: str
    family: str
    params: dict
    labels: np.ndarray
    n_clusters: int
    time_s: float
    mem_mb: float
    native: bool
    rerun: Callable | None = None
    error: str | None = None


def labels_from_clusters(clusters: dict, n: int) -> np.ndarray:
    """dict {cluster_id: [indices]} -> dense label array (pipeline-added)."""
    labels = np.full(n, -1, dtype=int)
    for cid, members in (clusters or {}).items():
        arr = np.asarray(members, dtype=int)
        labels[arr[arr < n]] = int(cid)
    return labels


def labels_from_ids(ids_list, n: int) -> np.ndarray:
    """list of per-subcluster index arrays -> dense labels (pipeline-added)."""
    labels = np.full(n, -1, dtype=int)
    for cid, arr in enumerate(ids_list or []):
        arr = np.asarray(arr, dtype=int)
        labels[arr[arr < n]] = cid
    return labels


# --------------------------------------------------------------------------
# MDANCE-native methods
# --------------------------------------------------------------------------

def run_nani(X, N_atoms, k, init_type, percentage=10, seed=42) -> Solution:
    """NANI k-means (MDANCE-native `KmeansNANI.execute_kmeans_all`)."""
    import numpy as _np
    m0, t0 = _peak(), time.time()

    def _rerun(subset: _np.ndarray) -> _np.ndarray:
        model = md_nani.KmeansNANI(data=X[subset], n_clusters=k, metric="MSD",
                                   N_atoms=N_atoms, init_type=init_type,
                                   percentage=percentage)
        labs, _, _ = model.execute_kmeans_all()
        return _np.asarray(labs, dtype=int)

    try:
        model = md_nani.KmeansNANI(data=X, n_clusters=k, metric="MSD",
                                   N_atoms=N_atoms, init_type=init_type,
                                   percentage=percentage)
        labels, centers, n_iter = model.execute_kmeans_all()
        labels = np.asarray(labels, dtype=int)
        return Solution(
            name=f"nani/{init_type}/k{k}", family=f"nani/{init_type}",
            params={"k": k, "init_type": init_type, "percentage": percentage},
            labels=labels, n_clusters=int(len(np.unique(labels))),
            time_s=time.time() - t0, mem_mb=_peak() - m0, native=True,
            rerun=_rerun)
    except Exception as exc:
        return Solution(name=f"nani/{init_type}/k{k}", family=f"nani/{init_type}",
                        params={"k": k}, labels=np.array([], int), n_clusters=0,
                        time_s=time.time() - t0, mem_mb=0.0, native=True,
                        error=str(exc))


def run_divine(X, N_atoms, k, split="weighted_MSD", anchors="nani",
               init_type="strat_all", refine=True) -> Solution:
    """Divisive clustering (MDANCE-native `Divine.run`, metric hard-coded MSD)."""
    m0, t0 = _peak(), time.time()

    def _rerun(subset):
        model = md_divine.Divine(data=X[subset], split=split, anchors=anchors,
                                 init_type=init_type, end="k", k=k,
                                 refine=refine, N_atoms=N_atoms)
        _, labs, _, _ = model.run()
        return np.asarray(labs, dtype=int)

    try:
        model = md_divine.Divine(data=X, split=split, anchors=anchors,
                                 init_type=init_type, end="k", k=k,
                                 refine=refine, N_atoms=N_atoms)
        _, labels, _, _ = model.run()
        return Solution(
            name=f"divine/{split}/{anchors}/k{k}",
            family=f"divine/{split}/{anchors}",
            params={"k": k, "split": split, "anchors": anchors},
            labels=np.asarray(labels, dtype=int),
            n_clusters=int(len(np.unique(labels))),
            time_s=time.time() - t0, mem_mb=_peak() - m0, native=True,
            rerun=_rerun)
    except Exception as exc:
        return Solution(name=f"divine/{split}/{anchors}/k{k}",
                        family=f"divine/{split}/{anchors}", params={"k": k},
                        labels=np.array([], int), n_clusters=0,
                        time_s=time.time() - t0, mem_mb=0.0, native=True,
                        error=str(exc))


def run_equal(X, N_atoms, threshold, seed_method="greedy",
              reject_lowd=True) -> Solution:
    """eQual radius clustering (MDANCE-native `ExtendedQuality.run`)."""
    m0, t0 = _peak(), time.time()

    def _rerun(subset):
        model = md_equal.ExtendedQuality(data=X[subset], threshold=threshold,
                                         metric="MSD", N_atoms=N_atoms,
                                         seed_method=seed_method,
                                         reject_lowd=reject_lowd)
        return labels_from_clusters(model.run(), len(subset))

    try:
        model = md_equal.ExtendedQuality(data=X, threshold=threshold,
                                         metric="MSD", N_atoms=N_atoms,
                                         seed_method=seed_method,
                                         reject_lowd=reject_lowd)
        clusters = model.run()
        labels = labels_from_clusters(clusters, X.shape[0])
        return Solution(
            name=f"equal/{seed_method}/t{threshold:.3g}",
            family=f"equal/{seed_method}",
            params={"threshold": float(threshold), "seed_method": seed_method},
            labels=labels, n_clusters=int(len(np.unique(labels[labels >= 0]))),
            time_s=time.time() - t0, mem_mb=_peak() - m0, native=True,
            rerun=_rerun)
    except Exception as exc:
        return Solution(name=f"equal/{seed_method}/t{threshold:.3g}",
                        family=f"equal/{seed_method}", params={},
                        labels=np.array([], int), n_clusters=0,
                        time_s=time.time() - t0, mem_mb=0.0, native=True,
                        error=str(exc))


def run_helm(X, N_atoms, k_target, initial_labels, merge_scheme="intra",
             link=None) -> Solution:
    """HELM n-ary merge of an initial partition (MDANCE-native).

    Usage mirrors the official `scripts/helm/run_helm.py`: build the verified
    cluster_dict {N0: [[indices, (c_sum, sq_sum), N], ...]} from a coarse
    initial clustering and n-ary merge toward `k_target` clusters.
    """
    m0, t0 = _peak(), time.time()
    uniq = sorted({int(l) for l in initial_labels if l >= 0})
    n0 = len(uniq)

    def build_dict(cl_labels, cluster_ids):
        d = {len(cluster_ids): []}
        for cid, orig in enumerate(cluster_ids):
            members = np.where(cl_labels == orig)[0]
            sub = X[members]
            d[len(cluster_ids)].append([
                [cid],
                (np.sum(sub, axis=0), np.sum(sub ** 2, axis=0)),
                int(len(members)),
            ])
        return d

    def _rerun(subset):
        base = _initial_partition(X[subset], k_target0)
        uniq2 = sorted({int(l) for l in base if l >= 0})
        d = build_dict(base, uniq2)
        helm = md_helm.HELM(cluster_dict=d, metric="MSD", N_atoms=N_atoms,
                            merge_scheme=merge_scheme, n_clusters=int(k_target),
                            link=link)
        merged = helm.run()
        return _decode(merged, base, uniq2)

    def _decode(merged, base, uniq):
        if not merged:
            return np.full(len(base), -1, dtype=int)
        key = min(merged.keys())
        labels = np.full(len(base), -1, dtype=int)
        for cid, entry in enumerate(merged[key]):
            for orig_cluster_id in entry[0]:
                own_id = uniq[int(orig_cluster_id)]
                members = np.where(base == own_id)[0]
                labels[members] = cid
        return labels

    k_target0 = max(int(min(len(uniq) - 1, k_target)), 1)
    try:
        if n0 <= 1:
            raise ValueError("initial partition has 1 cluster; HELM cannot merge")
        base = initial_labels
        helm_dict = build_dict(base, uniq)
        helm = md_helm.HELM(cluster_dict=helm_dict, metric="MSD", N_atoms=N_atoms,
                            merge_scheme=merge_scheme, n_clusters=int(k_target),
                            link=link)
        merged = helm.run()
        labels = _decode(merged, base, uniq)
        return Solution(
            name=f"helm/{merge_scheme}/k{k_target}", family=f"helm/{merge_scheme}",
            params={"k": int(k_target), "merge_scheme": merge_scheme,
                    "initial_k": int(n0)},
            labels=labels, n_clusters=int(len(np.unique(labels[labels >= 0]))),
            time_s=time.time() - t0, mem_mb=_peak() - m0, native=True,
            rerun=_rerun)
    except Exception as exc:
        return Solution(name=f"helm/{merge_scheme}/k{k_target}",
                        family=f"helm/{merge_scheme}", params={},
                        labels=np.array([], int), n_clusters=0,
                        time_s=time.time() - t0, mem_mb=0.0, native=True,
                        error=str(exc))


def _initial_partition(X, k, seed=42) -> np.ndarray:
    """Lean k-means initial partition for HELM seeding (pipeline-added)."""
    n = X.shape[0]
    k_eff = int(min(max(2, k), max(2, n // 10)))
    km = KMeans(n_clusters=k_eff, n_init=3, random_state=seed).fit(X)
    return np.asarray(km.labels_, dtype=int)


def run_mdbirch(X, threshold, branching_factor=50) -> Solution:
    """Streaming BIRCH (MDANCE-native `mdBirch.fit` + get_cluster_mol_ids).

    MDANCE's module needs `set_merge()` to install its internal global
    merge function (verified upstream behavior). `threshold` here is given
    in the pipeline's MDANCE-pairwise-MSD units and converted to the
    RMSD-style tolerance mdBirch expects internally (documented mapping
    t_inner = 2*sqrt(MSD/3), which preserves the sweep ordering).
    """
    m0, t0 = _peak(), time.time()
    n_features = X.shape[1]
    rms_tol = 2.0 * np.sqrt(max(float(threshold), 1e-12) / max(n_features, 1))

    def _rerun(subset):
        md_mdbirch.set_merge('radius', features=n_features)
        model = md_mdbirch.mdBirch(threshold=rms_tol,
                                   branching_factor=branching_factor)
        model.fit(X[subset])
        return labels_from_ids(model.get_cluster_mol_ids(), len(subset))

    try:
        md_mdbirch.set_merge('radius', features=n_features)
        model = md_mdbirch.mdBirch(threshold=rms_tol,
                                   branching_factor=branching_factor)
        model.fit(X)
        labels = labels_from_ids(model.get_cluster_mol_ids(), X.shape[0])
        return Solution(
            name=f"mdbirch/t{threshold:.3g}", family="mdbirch",
            params={"threshold": float(threshold),
                    "branching_factor": branching_factor},
            labels=labels, n_clusters=int(len(np.unique(labels[labels >= 0]))),
            time_s=time.time() - t0, mem_mb=_peak() - m0, native=True,
            rerun=_rerun)
    except Exception as exc:
        return Solution(name=f"mdbirch/t{threshold:.3g}", family="mdbirch",
                        params={}, labels=np.array([], int), n_clusters=0,
                        time_s=time.time() - t0, mem_mb=0.0, native=True,
                        error=str(exc))


# --------------------------------------------------------------------------
# Pipeline-added baselines for the GROMACS-style benchmark
# --------------------------------------------------------------------------

def _agglom_predict(X, D, k, linkage, subset=None):
    if linkage == "ward":
        return np.asarray(AgglomerativeClustering(
            n_clusters=k, linkage="ward").fit_predict(
            X if subset is None else X[subset]), dtype=int)
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage,
                                    metric="precomputed")
    if subset is None:
        return np.asarray(model.fit_predict(D), dtype=int)
    return np.asarray(model.fit_predict(D[subset][:, subset]), dtype=int)


def run_agglomerative(X, D, k, linkage="average") -> Solution:
    """Agglomerative on the identical MDANCE MSD matrix (pipeline-added).

    'average'/'complete' operate directly on the precomputed distance matrix;
    'ward' is fitted in coordinate space (Ward cannot use precomputed).
    """
    m0, t0 = _peak(), time.time()

    def _rerun(subset):
        return _agglom_predict(X, D, k, linkage, subset=subset)

    try:
        labels = _agglom_predict(X, D, k, linkage)
        return Solution(
            name=f"agglom/{linkage}/k{k}", family=f"agglom/{linkage}",
            params={"k": k, "linkage": linkage}, labels=labels,
            n_clusters=int(len(np.unique(labels))),
            time_s=time.time() - t0, mem_mb=_peak() - m0, native=False,
            rerun=_rerun)
    except Exception as exc:
        return Solution(name=f"agglom/{linkage}/k{k}",
                        family=f"agglom/{linkage}", params={}, labels=np.array([], int),
                        n_clusters=0, time_s=time.time() - t0, mem_mb=0.0,
                        native=False, error=str(exc))


def run_dbscan(D, eps, min_samples=5) -> Solution:
    """DBSCAN on the identical MSD matrix (pipeline-added)."""
    m0, t0 = _peak(), time.time()

    def _rerun(subset):
        model = DBSCAN(eps=float(eps), min_samples=int(min_samples),
                       metric="precomputed")
        return np.asarray(model.fit_predict(D[subset][:, subset]), dtype=int)

    try:
        model = DBSCAN(eps=float(eps), min_samples=int(min_samples),
                       metric="precomputed")
        labels = np.asarray(model.fit_predict(D), dtype=int)
        return Solution(
            name=f"dbscan/eps{eps:.3g}", family="dbscan",
            params={"eps": float(eps), "min_samples": int(min_samples)},
            labels=labels, n_clusters=int(len(np.unique(labels[labels >= 0]))),
            time_s=time.time() - t0, mem_mb=_peak() - m0, native=False,
            rerun=_rerun)
    except Exception as exc:
        return Solution(name=f"dbscan/eps{eps:.3g}", family="dbscan",
                        params={}, labels=np.array([], int), n_clusters=0,
                        time_s=time.time() - t0, mem_mb=0.0, native=False,
                        error=str(exc))


def run_gromos(D, cutoff) -> Solution:
    """Daura/gromos greedy clustering (pipeline-added; the algorithm behind
    `gmx cluster -m gromos`): repeatedly extract the frame with the most
    neighbors within `cutoff`, mark them, repeat."""
    m0, t0 = _peak(), time.time()
    n = D.shape[0]
    neigh = [np.flatnonzero(D[i] <= cutoff) for i in range(n)]
    excluded = np.zeros(n, dtype=bool)
    labels = np.full(n, -1, dtype=int)
    cid = 0
    while True:
        best_count, best = -1, -1
        for i in range(n):
            if excluded[i]:
                continue
            cand = neigh[i][~excluded[neigh[i]]]
            if len(cand) > best_count:
                best_count, best = len(cand), i
        if best < 0:
            break
        cluster = neigh[best][~excluded[neigh[best]]]
        labels[cluster] = cid
        excluded[cluster] = True
        cid += 1
    return Solution(
        name=f"gromos/c{cutoff:.3g}", family="gromos",
        params={"cutoff": float(cutoff)}, labels=labels,
        n_clusters=int(len(np.unique(labels[labels >= 0]))),
        time_s=time.time() - t0, mem_mb=_peak() - m0, native=False,
        rerun=lambda subset: _gromos_subset(D, subset, cutoff))


def _gromos_subset(D, subset, cutoff):
    n = len(subset)
    neigh = [np.flatnonzero(D[subset][:, subset][i] <= cutoff) for i in range(n)]
    excluded = np.zeros(n, dtype=bool)
    labels = np.full(n, -1, dtype=int)
    cid = 0
    while True:
        best_count, best = -1, -1
        for i in range(n):
            if excluded[i]:
                continue
            cand = neigh[i][~excluded[neigh[i]]]
            if len(cand) > best_count:
                best_count, best = len(cand), i
        if best < 0:
            break
        cluster = neigh[best][~excluded[neigh[best]]]
        labels[cluster] = cid
        excluded[cluster] = True
        cid += 1
    return labels


# --------------------------------------------------------------------------
# Sweep orchestration (pipeline-added)
# --------------------------------------------------------------------------

def run_all(X, D, N_atoms, cfg, logger) -> list[Solution]:
    """Run every compatible method per configuration; skip with reasons."""
    cl = cfg["clustering"]
    seed = int(cfg["system"]["seed"])
    n = X.shape[0]
    k_min = int(cl.get("k_min", 2))
    k_max = min(int(cl.get("k_max", 8)), max(1, n - 1))
    k_range = list(range(max(1, k_min), k_max + 1))
    D_ok = D is not None
    sols: list[Solution] = []

    logger.info("---- MDANCE-native clustering sweep (k in %s) ----", k_range)

    # --- NANI k-means (native) ---
    for init_type in cl.get("nani_init_types", ["strat_all"]):
        for k in k_range:
            sol = run_nani(X, N_atoms, k, init_type,
                           percentage=int(cl.get("nani_percentage", 10)),
                           seed=seed)
            if sol.error:
                logger.warning("skip %s: %s", sol.name, sol.error)
            else:
                sols.append(sol)

    # --- Divine (native; metric hard-coded MSD upsteam) ---
    for split in cl.get("divine_splits", ["weighted_MSD"]):
        for anchors in cl.get("divine_anchors", ["nani"]):
            if anchors == "nani":
                init_types = ["strat_all"]
            else:
                init_types = ["comp_sim"]
            for init_type in init_types:
                for k in k_range:
                    sol = run_divine(X, N_atoms, k, split=split,
                                     anchors=anchors, init_type=init_type,
                                     refine=bool(cl.get("divine_refine", True)))
                    if sol.error:
                        logger.warning("skip %s: %s", sol.name, sol.error)
                    else:
                        sols.append(sol)

    # --- eQual (native; threshold sweep) ---
    if D_ok:
        condensed = upper_desc_vals(D)
        qs = cl.get("equal_threshold_quantiles", [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95])
        thresholds = [float(np.quantile(condensed, q)) for q in qs]
    else:
        thresholds = []
    for seed_method in cl.get("equal_seed_methods", ["greedy"]):
        for thr in thresholds:
            sol = run_equal(X, N_atoms, thr, seed_method=seed_method)
            if sol.error:
                logger.warning("skip %s: %s", sol.name, sol.error)
            else:
                sols.append(sol)

    # --- mdBirch (native; threshold sweep) ---
    if D_ok:
        qs = cl.get("mdbirch_threshold_quantiles", [0.1, 0.25, 0.4, 0.55, 0.7, 0.85])
        thresholds = [float(np.quantile(upper_desc_vals(D), q)) for q in qs]
        bf = int(cl.get("mdbirch_branching_factor", 50))
        for thr in thresholds:
            sol = run_mdbirch(X, thr, branching_factor=bf)
            if sol.error:
                logger.warning("skip %s: %s", sol.name, sol.error)
            else:
                sols.append(sol)

    # --- HELM (native; refines a coarse k-means partition) ---
    if cl.get("helm_enabled", True) and n <= int(cl.get("helm_max_frames", 2000)) and D_ok:
        init_k = max(int(cl.get("helm_initial_k", 10)), k_max + 1)
        base = _initial_partition(X, init_k, seed=seed)
        for scheme in [cl.get("helm_merge_schema", "intra")]:
            for k in k_range:
                if k > len(np.unique(base)):
                    continue
                sol = run_helm(X, N_atoms, k, base,
                               merge_scheme=scheme, link=cl.get("helm_link"))
                if sol.error:
                    logger.warning("skip %s: %s", sol.name, sol.error)
                else:
                    sols.append(sol)

    # --- pipeline-added baselines ---
    if cl.get("baselines_enabled", True) and D_ok:
        for linkage in cl.get("agglomerative_linkages", ["average"]):
            for k in k_range:
                sol = run_agglomerative(X, D, k, linkage=linkage)
                if sol.error:
                    logger.warning("skip %s: %s", sol.name, sol.error)
                else:
                    sols.append(sol)
        condensed = upper_desc_vals(D)
        min_samples = int(cl.get("dbscan_min_samples", 5))
        for q in cl.get("dbscan_eps_quantiles", [0.05, 0.15, 0.3, 0.5]):
            eps = float(np.quantile(condensed, q))
            sol = run_dbscan(D, eps, min_samples=min_samples)
            if sol.error:
                logger.warning("skip %s: %s", sol.name, sol.error)
            else:
                sols.append(sol)

    logger.info("Executed %d clustering solutions.", len(sols))
    return sols


def upper_desc_vals(D: np.ndarray) -> np.ndarray:
    """Off-diagonal (i<j) values of a symmetric distance matrix."""
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]
