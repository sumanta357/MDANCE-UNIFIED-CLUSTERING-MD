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
              reject_lowd=True, min_samples: float | int = 0.05) -> Solution:
    """eQual radius clustering (MDANCE-native `ExtendedQuality.run`).

    `min_samples` follows eQual's own semantics: an int is an absolute number
    of frames, a float in (0, 1) is a fraction of the data. A small fraction
    keeps eQual's native low-density rejection while still allowing several
    radius-defined clusters (its default of 10 frames collapses typical
    trajectories into a single cluster).
    """
    m0, t0 = _peak(), time.time()

    def _rerun(subset):
        model = md_equal.ExtendedQuality(data=X[subset], threshold=threshold,
                                         metric="MSD", N_atoms=N_atoms,
                                         seed_method=seed_method,
                                         reject_lowd=reject_lowd,
                                         min_samples=min_samples)
        return labels_from_clusters(model.run(), len(subset))

    try:
        model = md_equal.ExtendedQuality(data=X, threshold=threshold,
                                         metric="MSD", N_atoms=N_atoms,
                                         seed_method=seed_method,
                                         reject_lowd=reject_lowd,
                                         min_samples=min_samples)
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

def nn_distance_thresholds(D: np.ndarray) -> list[float]:
    """Thresholds at the local distance scale (per-frame nearest neighbour).

    Quantiles of the *pairwise* distance distribution can sit far away from
    the scale at which neighbourhoods actually exist, which makes eQual and
    DBSCAN collapse to one cluster / all-noise on trajectories with a few
    well-separated states. These values are always inside the populated
    neighbourhood range, so the sweeps cross a usable threshold region.
    """
    if D is None or D.shape[0] < 3:
        return []
    Dc = np.array(D, dtype=float, copy=True)
    np.fill_diagonal(Dc, np.inf)
    nn = Dc.min(axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return []
    return sorted({float(np.quantile(nn, q)) for q in (0.25, 0.5, 0.75, 1.0)})


def run_all(X, D, N_atoms, cfg, logger, coverage: dict | None = None
            ) -> list[Solution]:
    """Run every compatible method per configuration; skip with reasons.

    K-based methods (NANI, Divine, HELM, Agglomerative) run at the single K
    chosen by the automatic Elbow procedure (``clustering.selected_k``) when
    it is available; otherwise they sweep ``k_min..k_max``. Threshold/density
    methods (eQual, mdBirch, DBSCAN, gromos) keep their native behaviour and
    are never forced onto the selected K.
    """
    cl = cfg["clustering"]
    seed = int(cfg["system"]["seed"])
    n = X.shape[0]
    k_min = int(cl.get("k_min", 2))
    k_max = min(int(cl.get("k_max", 15)), max(1, n - 1))
    selected_k = cl.get("selected_k")
    k_map = cl.get("selected_k_map") or {}

    def _k_for(family: str) -> int:
        if family in k_map:
            return max(2, min(int(k_map[family]), max(2, n - 1)))
        if selected_k is not None:
            return max(2, min(int(selected_k), max(2, n - 1)))
        return max(k_min, 2)

    if selected_k is not None or k_map:
        k_range = [selected_k] if selected_k is not None else None
        logger.info("---- K-based methods use automatic K: "
                    "%s (%s) ----",
                    (k_map or selected_k),
                    "per method" if k_map else "shared")
    else:
        k_range = list(range(max(1, k_min), k_max + 1))
        logger.info("---- MDANCE-native clustering sweep (k in %s) ----",
                    k_range)

    D_ok = D is not None
    sols: list[Solution] = []
    not_run: dict[str, str] = {}
    helm_cap = int(cl.get("helm_max_frames", 20000))
    if not D_ok:
        for fam in ("equal", "mdbirch", "dbscan", "agglom", "helm",
                    "gromos"):
            not_run[fam] = "distance matrix unavailable"
    if not cl.get("helm_enabled", True):
        not_run["helm"] = "helm_enabled: false"
    elif D_ok and n > helm_cap:
        not_run["helm"] = (f"n_frames {n} > helm_max_frames {helm_cap} "
                           f"(raise clustering.helm_max_frames to enable)")
        logger.warning("HELM not run: %s", not_run["helm"])
    if not cl.get("baselines_enabled", True):
        for fam in ("agglom", "dbscan"):
            not_run[fam] = "baselines_enabled: false"
        logger.warning("Agglomerative/DBSCAN not run: baselines disabled")

    nani_ks = k_range if k_range is not None else [_k_for("nani")]
    divine_ks = k_range if k_range is not None else [_k_for("divine")]
    helm_ks = k_range if k_range is not None else [_k_for("helm")]
    agglom_ks = k_range if k_range is not None else [_k_for("agglom")]

    # --- NANI k-means (native) ---
    for init_type in cl.get("nani_init_types", ["strat_all"]):
        for k in nani_ks:
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
                for k in divine_ks:
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
        qs = cl.get("equal_threshold_quantiles",
                    [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15,
                     0.20, 0.30])
        thresholds = [float(np.quantile(condensed, q)) for q in qs]
        # thresholds on the nearest-neighbour scale, so the sweep always enters
        # the region where >= 2 clusters can exist
        nn = nn_distance_thresholds(D)
        thresholds += [v for f in nn for v in (f, 2.0 * f) if v > 0]
        thresholds = sorted(set(thr for thr in thresholds if thr > 0))
    else:
        thresholds = []
    equal_min_samples = cl.get("equal_min_samples", 0.05)
    for seed_method in cl.get("equal_seed_methods", ["greedy"]):
        for thr in thresholds:
            sol = run_equal(X, N_atoms, thr, seed_method=seed_method,
                            min_samples=equal_min_samples)
            if sol.error:
                logger.warning("skip %s: %s", sol.name, sol.error)
            else:
                sols.append(sol)

    # --- mdBirch (native; threshold sweep) ---
    if D_ok:
        qs = cl.get("mdbirch_threshold_quantiles",
                    [0.01, 0.05, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85])
        thresholds = [float(np.quantile(upper_desc_vals(D), q)) for q in qs]
        thresholds = sorted(set(t for t in thresholds if t > 0))
        bf = int(cl.get("mdbirch_branching_factor", 50))
        for thr in thresholds:
            sol = run_mdbirch(X, thr, branching_factor=bf)
            if sol.error:
                logger.warning("skip %s: %s", sol.name, sol.error)
            else:
                sols.append(sol)

    # --- HELM (native; refines a coarse k-means partition) ---
    if cl.get("helm_enabled", True) and n <= helm_cap and D_ok:
        init_k = max(int(cl.get("helm_initial_k", 10)),
                     (max(k_range) if k_range else max(helm_ks)) + 1,
                     selected_k or 0)
        base = _initial_partition(X, init_k, seed=seed)
        for scheme in [cl.get("helm_merge_schema", "intra")]:
            for k in helm_ks:
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
            # 'ward_on_coords' is the config label for Ward fitted on
            # coordinates (Ward cannot use a precomputed matrix)
            linkage_arg = "ward" if linkage == "ward_on_coords" else linkage
            for k in agglom_ks:
                sol = run_agglomerative(X, D, k, linkage=linkage_arg)
                if sol.error:
                    logger.warning("skip %s: %s", sol.name, sol.error)
                else:
                    sols.append(sol)
        condensed = upper_desc_vals(D)
        min_samples = int(cl.get("dbscan_min_samples", 5))
        eps_list = [float(np.quantile(condensed, q))
                    for q in cl.get("dbscan_eps_quantiles",
                                    [0.01, 0.02, 0.05, 0.15, 0.30, 0.50])]
        # nearest-neighbour scale: eps that actually captures local density
        eps_list += nn_distance_thresholds(D)
        for eps in sorted(set(e for e in eps_list if e > 0)):
            sol = run_dbscan(D, eps, min_samples=min_samples)
            if sol.error:
                logger.warning("skip %s: %s", sol.name, sol.error)
            else:
                sols.append(sol)

    logger.info("Executed %d clustering solutions.", len(sols))
    if coverage is not None:
        coverage.clear()
        by_family: dict[str, list] = {}
        for s in sols:
            by_family.setdefault(str(s.family).split("/")[0], []).append(s)
        for fam in ("nani", "divine", "helm", "equal", "mdbirch",
                    "agglom", "dbscan"):
            ss = by_family.get(fam, [])
            counts = [int(s.n_clusters) for s in ss]
            usable = [c for c in counts if c >= 2]
            reason = not_run.get(fam)
            if reason is None and not ss:
                reason = "no solution produced"
            coverage[fam] = {
                "status": ("usable" if usable
                           else ("degenerate" if ss else "not_run")),
                "reason": reason or ("" if usable else
                                     "every solution had < 2 clusters"),
                "solutions": len(ss),
                "usable_solutions": len(usable),
                "cluster_counts": sorted(set(counts)),
                "failed_solutions": sum(1 for s in ss if s.error),
            }
        coverage["gromos"] = {
            "status": "deferred",
            "reason": "evaluated in the GROMOS benchmarking sweep",
            "solutions": 0, "usable_solutions": 0, "cluster_counts": [],
            "failed_solutions": 0,
        }
        for fam, info in coverage.items():
            logger.info("  method coverage %-9s %-10s %s", fam,
                        info["status"],
                        info["reason"] or f"clusters {info['cluster_counts']}")
    return sols


def upper_desc_vals(D: np.ndarray) -> np.ndarray:
    """Off-diagonal (i<j) values of a symmetric distance matrix."""
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]
