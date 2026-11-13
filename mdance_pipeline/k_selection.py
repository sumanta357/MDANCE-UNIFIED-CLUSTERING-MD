"""Automatic K selection for the K-based clustering methods (pipeline-added).

The scan runs KmeansNANI (MDANCE-native, metric='MSD') over the *same*
prepared feature matrix used by the clustering workflow, so the elbow curve
is measured in the representation the pipeline actually clusters in.

Dispersion definition (explicit, see DISPERSION_DEFINITION):

    W(K) = sum_C (1/|C|) * sum_{i<j in C} d_MSD(i, j)

computed directly on the precomputed MDANCE MSD matrix
``d_MSD(i,j) = ||x_i - x_j||^2 / (2 * N_atoms)``. For a cluster C this equals
``sum_{i in C} ||x_i - mean(C)||^2 / (2 * N_atoms)``, i.e. within-cluster
dispersion in MDANCE MSD units. It is *not* conventional Euclidean k-means
inertia re-applied to a precomputed distance matrix (which would double-count
the distance and be dimensionally wrong).
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

DISPERSION_DEFINITION = (
    "W(K) = sum_C (1/|C|) * sum_{i<j in C} d_MSD(i,j), evaluated on the "
    "precomputed MDANCE MSD matrix with d_MSD(i,j)=||x_i-x_j||^2/(2*N_atoms); "
    "for each cluster this equals sum_i ||x_i-mean(C)||^2/(2*N_atoms). "
    "Conventional Euclidean k-means inertia was deliberately NOT used."
)

SUPPORT_METRICS = {
    "silhouette": True,          # higher is better
    "davies_bouldin": False,     # lower is better
    "calinski_harabasz": True,   # higher is better
    "stability": True,           # higher is better
}


# --------------------------------------------------------------------------
# Dispersion in the MDANCE distance representation
# --------------------------------------------------------------------------
def within_cluster_dispersion(D: np.ndarray, labels: np.ndarray) -> float:
    """Sum over clusters of the mean within-cluster MDANCE MSD.

    Uses the precomputed MDANCE MSD matrix only: for cluster C,
    ``(1/|C|) * sum_{i<j} D[i, j]`` == ``sum_i ||x_i-mean(C)||^2/(2*N_atoms)``.
    Noise points (label < 0) are ignored.
    """
    D = np.asarray(D)
    labels = np.asarray(labels, dtype=int)
    total = 0.0
    for c in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == c)
        if idx.size < 2:
            continue
        block = D[np.ix_(idx, idx)]
        total += float(block[np.triu_indices(idx.size, k=1)].sum() / idx.size)
    return total


# --------------------------------------------------------------------------
# Elbow detection
# --------------------------------------------------------------------------
def detect_elbow(ks, dispersion) -> tuple[int | None, str, bool]:
    """Locate the elbow of a decreasing curve.

    Returns (position_in_ks or None, algorithm, ambiguous). Kneedle is used
    when the optional ``kneed`` package is available; otherwise a normalized
    maximum-deviation ("geometric") detection is used. A curve lacking a
    genuine knee is reported as ambiguous instead of forcing an arbitrary K.
    """
    ks = list(ks)
    v = np.asarray(dispersion, dtype=float)
    if len(v) < 3 or not np.isfinite(v).all():
        return None, "insufficient-data", True
    if v[0] <= v[-1]:
        return None, "not-decreasing", True
    try:  # optional, exact Kneedle
        from kneed import KneeLocator
        knee = KneeLocator(ks, v, curve="convex", direction="decreasing").elbow
        if knee is not None and int(knee) in ks:
            return ks.index(int(knee)), "kneed", False
    except ImportError:
        pass
    x = np.linspace(0.0, 1.0, len(v))
    y = (v - v[-1]) / max(v[0] - v[-1], 1e-15)
    deviation = (1.0 - x) - y          # distance above the chord
    idx = int(np.argmax(deviation))
    if idx in (0, len(v) - 1) or deviation[idx] < 0.08:
        return None, "geometric", True
    return idx, "geometric", False


# --------------------------------------------------------------------------
# Per-K clustering + validation metrics
# --------------------------------------------------------------------------
def _kmeans_labels(X, k, cfg, seed):
    """KmeansNANI (MDANCE-native, MSD) on X; sklearn KMeans only as fallback."""
    cl = cfg.get("clustering", {})
    ks_cfg = cfg.get("k_selection", {})
    n_atoms = int(ks_cfg.get("n_atoms", 1)) or 1
    init_type = ks_cfg.get("nani_init_type", "k-means++")
    try:
        from mdance.cluster import nani as md_nani  # MDANCE-native
        model = md_nani.KmeansNANI(
            data=X, n_clusters=int(k), metric="MSD", N_atoms=n_atoms,
            init_type=init_type,
            percentage=int(cl.get("nani_percentage", 10)))
        labels, _, _ = model.execute_kmeans_all()
        labels = np.asarray(labels, dtype=int)
        if labels.size == X.shape[0] and len(np.unique(labels)) >= 2:
            return labels, f"KmeansNANI/{init_type}"
    except Exception:
        pass
    from sklearn.cluster import KMeans
    return (np.asarray(KMeans(n_clusters=int(k), n_init=5, random_state=seed)
                       .fit_predict(X), dtype=int), "sklearn-KMeans")


def _metrics_for(X, D, labels, cfg, seed):
    """Silhouette, distance-space DB/CH, population distribution, stability."""
    from mdance_pipeline import validation as val

    out = {"silhouette": None, "davies_bouldin": None,
           "calinski_harabasz": None, "stability": None,
           "population_distribution": {}, "n_clusters": 0,
           "n_singletons": 0}
    valid = labels >= 0
    y = labels[valid]
    clusters, counts = np.unique(y, return_counts=True)
    out["n_clusters"] = int(len(clusters))
    out["n_singletons"] = int((counts == 1).sum())
    out["population_distribution"] = {str(int(c)): int(n)
                                      for c, n in zip(clusters, counts)}
    if len(clusters) < 2:
        return out
    if len(clusters) < len(y):
        try:
            from sklearn.metrics import silhouette_score
            out["silhouette"] = float(
                silhouette_score(D[np.ix_(valid, valid)], y,
                                 metric="precomputed"))
        except Exception:
            out["silhouette"] = None
    try:
        medoids = val.cluster_medoids(D, labels)
        out["davies_bouldin"] = val.db_distance_space(D, labels, medoids)
        out["calinski_harabasz"] = val.ch_distance_space(D, labels, medoids)
    except Exception:
        pass
    # Stability: mean ARI between the full K partition and refits on 80%
    # subsamples, compared on the shared frames (bootstrap ARI).
    reps = int(cfg.get("k_selection", {}).get("stability_repeats", 4))
    rng = np.random.default_rng(seed)
    scores = []
    for i in range(max(0, reps)):
        subset = np.sort(rng.choice(len(X), max(3, int(0.8 * len(X))),
                                    replace=False))
        try:
            pred, _ = _kmeans_labels(X[subset], len(clusters), cfg,
                                     seed + i + 1)
            from sklearn.metrics import adjusted_rand_score
            scores.append(float(adjusted_rand_score(labels[subset], pred)))
        except Exception:
            continue
    if scores:
        out["stability"] = float(np.mean(scores))
    return out


# --------------------------------------------------------------------------
# Selection procedure
# --------------------------------------------------------------------------
def _rank_support(rows: list[dict]) -> None:
    """Average ordinal rank across available support metrics (1 = best)."""
    sums = {r["k"]: 0.0 for r in rows}
    counts = {r["k"]: 0 for r in rows}
    for metric, higher_is_better in SUPPORT_METRICS.items():
        present = [r for r in rows if r[metric] is not None
                   and np.isfinite(r[metric])]
        present.sort(key=lambda r: r[metric], reverse=higher_is_better)
        for rank, row in enumerate(present, 1):
            sums[row["k"]] += rank
            counts[row["k"]] += 1
    for r in rows:
        r["support_rank"] = (sums[r["k"]] / counts[r["k"]]
                             if counts[r["k"]] else float("inf"))
        r["support_metric_count"] = counts[r["k"]]


def _select(rows, elbow_pos, ambiguous) -> int:
    """Elbow first, supporting metrics as the decider/validator."""
    if elbow_pos is not None and not ambiguous:
        k_e = rows[elbow_pos]["k"]
        eligible = [r for r in rows if abs(r["k"] - k_e) <= 1]
    else:
        eligible = list(rows)
    best = min(eligible, key=lambda r: (r["support_rank"], r["dispersion"],
                                        r["k"]))
    return int(best["k"])


def run(X, D, n_atoms, cfg, out_dir, logger=None) -> dict:
    """Scan K, detect the elbow, recommend a K, and save all diagnostics.

    Returns the selection report (also written to ``out_dir`` as CSV/JSON plus
    an elbow plot). The chosen K is stored in ``cfg['clustering']['selected_k']``
    only by the pipeline orchestrator, never mutated here.
    """
    ks_cfg = cfg.get("k_selection", {})
    cl = cfg.get("clustering", {})
    low = max(2, int(ks_cfg.get("k_min", cl.get("k_min", 2))))
    high = int(ks_cfg.get("k_max", cl.get("k_max", 15)))
    high = min(high, int(ks_cfg.get("max_allowed_k", 20)), X.shape[0] - 1)
    if high < low:
        raise ValueError(
            f"Cannot scan K={low}..{high} for {X.shape[0]} frames; "
            "reduce clustering.k_min or analyze more frames.")
    seed = int(cfg.get("system", {}).get("seed", 42))
    scan_cfg = dict(cfg)
    scan_cfg["k_selection"] = {**ks_cfg, "n_atoms": int(n_atoms)}

    rows = []
    algorithm = "unknown"
    for k in range(low, high + 1):
        labels, algorithm = _kmeans_labels(X, k, scan_cfg, seed)
        row = {"k": k,
               "dispersion": within_cluster_dispersion(D, labels),
               **_metrics_for(X, D, labels, scan_cfg, seed + k)}
        rows.append(row)
        if logger:
            logger.info("  K=%2d  dispersion=%.6g  silhouette=%s",
                        k, row["dispersion"],
                        None if row["silhouette"] is None
                        else round(row["silhouette"], 4))

    elbow_pos, elbow_algo, ambiguous = detect_elbow(
        [r["k"] for r in rows], [r["dispersion"] for r in rows])
    _rank_support(rows)
    selected = _select(rows, elbow_pos, ambiguous)

    os.makedirs(out_dir, exist_ok=True)
    fields = ["k", "dispersion", "n_clusters", "n_singletons", "silhouette",
              "davies_bouldin", "calinski_harabasz", "stability",
              "population_distribution", "support_rank",
              "support_metric_count"]
    with open(os.path.join(out_dir, "k_selection.csv"), "w",
              newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({**row,
                         "population_distribution": json.dumps(
                             row["population_distribution"], sort_keys=True)})

    report = {
        "selected_k": int(selected),
        "elbow_k": (int(rows[elbow_pos]["k"])
                    if elbow_pos is not None else None),
        "elbow_algorithm": elbow_algo,
        "elbow_ambiguous": bool(ambiguous),
        "k_min": low,
        "k_max": high,
        "scan_algorithm": algorithm,
        "dispersion_definition": DISPERSION_DEFINITION,
        "selection_rule": (
            "Unambiguous elbow: the elbow K and its immediate neighbours are "
            "eligible and the best average ordinal rank over silhouette, "
            "distance-space Davies-Bouldin, distance-space Calinski-Harabasz "
            "and bootstrap-ARI stability wins (ties -> lower dispersion). "
            "Ambiguous elbow (no knee in 2..k_max): every scanned K is "
            "eligible and the supporting metrics decide; the ambiguity is "
            "reported rather than an arbitrary K being forced."),
        "metrics": rows,
    }
    with open(os.path.join(out_dir, "k_selection.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    _plot(rows, elbow_pos, selected, out_dir, cfg, logger)
    if logger:
        logger.info("Automatic K selection: K=%d (elbow=%s, ambiguous=%s, "
                    "scan %d..%d)", selected, report["elbow_k"], ambiguous,
                    low, high)
    return report


def _method_labels(X, D, k, cfg, seed, n_atoms, family):
    """Run one K-based method at K (used only for per-method elbow scans)."""
    from mdance_pipeline import clustering as cl

    if family == "nani":
        sol = cl.run_nani(X, n_atoms, int(k),
                          cfg["k_selection"].get("nani_init_type",
                                                 "k-means++"),
                          percentage=int(cfg["clustering"].get(
                              "nani_percentage", 10)), seed=seed)
    elif family == "divine":
        sol = cl.run_divine(
            X, n_atoms, int(k),
            split=(cfg["clustering"].get("divine_splits") or
                   ["weighted_MSD"])[0],
            anchors=(cfg["clustering"].get("divine_anchors") or ["nani"])[0],
            refine=bool(cfg["clustering"].get("divine_refine", True)))
    elif family == "helm":
        base = cl._initial_partition(
            X, max(int(k) + 1, int(cfg["clustering"].get(
                "helm_initial_k", 10))), seed=seed)
        sol = cl.run_helm(X, n_atoms, int(k), base,
                          merge_scheme=cfg["clustering"].get(
                              "helm_merge_schema", "intra"))
    elif family == "agglom":
        sol = cl.run_agglomerative(X, D, int(k), linkage="average")
    else:
        raise ValueError(f"unsupported family for per-method K: {family}")
    if sol.error or sol.labels.size == 0:
        raise RuntimeError(f"{family} K={k} failed: {sol.error}")
    return sol.labels, sol.name


def run_per_method(X, D, n_atoms, cfg, out_dir, logger=None) -> dict:
    """Independent elbow scan for each K-based method (optional).

    Enabled with ``k_selection.per_method: true``. Each method is clustered at
    every K in the scan range, its own dispersion curve is built from those
    partitions, and the elbow + supporting metrics are evaluated per method.
    This costs one full K sweep per method, so it is off by default.
    """
    ks_cfg = cfg.get("k_selection", {})
    cl = cfg.get("clustering", {})
    low = max(2, int(ks_cfg.get("k_min", cl.get("k_min", 2))))
    high = min(int(ks_cfg.get("k_max", cl.get("k_max", 15))),
               int(ks_cfg.get("max_allowed_k", 20)), X.shape[0] - 1)
    if high < low:
        raise ValueError(f"No valid per-method K range: {low}..{high}")
    seed = int(cfg.get("system", {}).get("seed", 42))
    os.makedirs(out_dir, exist_ok=True)
    report: dict = {"per_method": {}, "k_min": low, "k_max": high,
                    "dispersion_definition": DISPERSION_DEFINITION}
    for family in ks_cfg.get("per_method_families",
                             ["nani", "divine", "helm", "agglom"]):
        rows = []
        for k in range(low, high + 1):
            try:
                labels, name = _method_labels(X, D, k, cfg, seed, n_atoms,
                                              family)
            except Exception as exc:
                if logger:
                    logger.warning("per-method K: %s K=%d failed: %s",
                                   family, k, exc)
                continue
            rows.append({"k": k, "dispersion":
                         within_cluster_dispersion(D, labels),
                         "solution": name,
                         **_metrics_for(X, D, labels, cfg, seed + k)})
        if len(rows) < 3:
            if logger:
                logger.warning("per-method K: %s has too few valid K values",
                               family)
            continue
        elbow_pos, elbow_algo, ambiguous = detect_elbow(
            [r["k"] for r in rows], [r["dispersion"] for r in rows])
        _rank_support(rows)
        selected = _select(rows, elbow_pos, ambiguous)
        report["per_method"][family] = {
            "selected_k": int(selected),
            "elbow_k": (int(rows[elbow_pos]["k"])
                        if elbow_pos is not None else None),
            "elbow_algorithm": elbow_algo,
            "elbow_ambiguous": bool(ambiguous),
            "metrics": rows,
        }
        _plot(rows, elbow_pos, selected, out_dir, cfg, logger,
              name=f"k_selection_elbow_{family}.png",
              title=f"Elbow K selection - {family}")
        if logger:
            logger.info("  per-method K %-8s -> K=%d (elbow=%s, ambiguous=%s)",
                        family, selected, report["per_method"][family]
                        ["elbow_k"], ambiguous)
    with open(os.path.join(out_dir, "k_selection_per_method.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(out_dir, "k_selection_per_method.csv"), "w",
              newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["method", "selected_k", "elbow_k", "elbow_algorithm",
                     "elbow_ambiguous"])
        for fam, info in report["per_method"].items():
            wr.writerow([fam, info["selected_k"], info["elbow_k"],
                         info["elbow_algorithm"], info["elbow_ambiguous"]])
    return report


def _plot(rows, elbow_pos, selected, out_dir, cfg, logger, name=
          "k_selection_elbow.png", title="Automatic K selection "
          "(Elbow / Kneedle)"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if logger:
            logger.warning("matplotlib unavailable; Elbow plot skipped")
        return
    ks = [r["k"] for r in rows]
    w = [r["dispersion"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ks, w, "o-", color="#1f4e79", lw=1.8, label="dispersion")
    if elbow_pos is not None:
        ax.axvline(ks[elbow_pos], ls="--", color="#b03a2e",
                   label=f"elbow K={ks[elbow_pos]}")
    ax.axvline(selected, ls=":", color="#1e8449", lw=2,
               label=f"selected K={selected}")
    ax.set_xlabel("K (clusters)")
    ax.set_ylabel("within-cluster MDANCE-MSD dispersion")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name),
                dpi=int(cfg.get("system", {}).get("dpi", 150)))
    plt.close(fig)
