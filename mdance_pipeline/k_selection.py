"""Elbow-based K selection in the prepared MDANCE feature space."""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


def within_cluster_dispersion(X, labels, n_atoms):
    """Return W = sum_C sum_{i in C} ||x_i - mean(C)||^2/(2*N_atoms).

    This is in MDANCE MSD units. For a cluster of size n_C it is also
    (1/n_C) sum_{i<j in C} d_MSD(i,j), where
    d_MSD(i,j)=||x_i-x_j||^2/(2*N_atoms). It is therefore not conventional
    Euclidean inertia applied to a precomputed distance matrix.
    """
    total = 0.0
    for cluster in np.unique(labels[labels >= 0]):
        members = np.asarray(X)[labels == cluster]
        if len(members):
            total += float(np.square(members - members.mean(axis=0)).sum()) / (2.0 * n_atoms)
    return total


def _metrics(X, D, labels, seed, stability_repeats):
    from mdance_pipeline.validation import cluster_medoids, db_distance_space, ch_distance_space

    valid = labels >= 0
    y = labels[valid]
    clusters, counts = np.unique(y, return_counts=True)
    result = {
        "silhouette": None, "davies_bouldin": None,
        "calinski_harabasz": None, "stability": None,
        "population_distribution": {str(int(c)): int(n) for c, n in zip(clusters, counts)},
    }
    if len(clusters) < 2 or len(clusters) >= len(y):
        return result
    if D is not None:
        d = D[np.ix_(valid, valid)]
        try:
            if len(y) > 2000:
                sample = np.sort(np.random.default_rng(seed).choice(len(y), 2000, replace=False))
                result["silhouette"] = float(silhouette_score(
                    d[np.ix_(sample, sample)], y[sample], metric="precomputed"))
            else:
                result["silhouette"] = float(silhouette_score(d, y, metric="precomputed"))
        except Exception:
            pass
        try:
            medoids = cluster_medoids(D, labels)
            result["davies_bouldin"] = db_distance_space(D, labels, medoids)
            result["calinski_harabasz"] = ch_distance_space(D, labels, medoids)
        except Exception:
            pass
    # Stability is mean ARI between the full KMeans assignment and deterministic
    # refits on 80% subsamples, evaluated on those same sampled frames.
    rng = np.random.default_rng(seed)
    scores = []
    for i in range(stability_repeats):
        subset = np.sort(rng.choice(len(X), max(2, int(.8 * len(X))), replace=False))
        try:
            pred = KMeans(n_clusters=len(clusters), n_init=3,
                          random_state=seed + i + 1).fit_predict(X[subset])
            scores.append(float(adjusted_rand_score(labels[subset], pred)))
        except Exception:
            pass
    if scores:
        result["stability"] = float(np.mean(scores))
    return result


def _elbow(ks, values):
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or not np.isfinite(values).all() or values[0] <= values[-1]:
        return None, "none", True
    try:
        from kneed import KneeLocator
        knee = KneeLocator(ks, values, curve="convex", direction="decreasing").elbow
        if knee is not None:
            return list(ks).index(int(knee)), "kneed", False
    except ImportError:
        pass
    x = np.linspace(0.0, 1.0, len(values))
    y = (values - values[-1]) / max(values[0] - values[-1], 1e-15)
    deviation = (1.0 - x) - y
    idx = int(np.argmax(deviation))
    if idx in (0, len(values) - 1) or deviation[idx] < 0.08:
        return None, "geometric", True
    return idx, "geometric", False


def select_k(X, D, n_atoms, cfg, out_dir, logger=None):
    """Scan configured K interval (default 2..15, maximum 20), save diagnostics."""
    cl = cfg.get("clustering", {})
    low = max(2, int(cl.get("k_min", 2)))
    high = min(20, int(cl.get("k_max", 15)), len(X) - 1)
    if high < low:
        raise ValueError(f"No valid Elbow K range: {low}..{high} for {len(X)} frames")
    seed = int(cfg.get("system", {}).get("seed", 42))
    repeats = max(0, int(cfg.get("validation", {}).get("elbow_stability_repeats", 3)))
    rows = []
    for k in range(low, high + 1):
        labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(X)
        rows.append({"k": k, "dispersion": within_cluster_dispersion(X, labels, n_atoms),
                     **_metrics(X, D, labels, seed + k, repeats)})

    elbow_idx, method, ambiguous = _elbow([r["k"] for r in rows], [r["dispersion"] for r in rows])
    # Average ordinal ranks across available validation metrics. For an
    # unambiguous curve, only the elbow and its immediate neighbors are eligible.
    directions = {"silhouette": True, "davies_bouldin": False,
                  "calinski_harabasz": True, "stability": True}
    rank_sums = {r["k"]: 0.0 for r in rows}
    rank_counts = {r["k"]: 0 for r in rows}
    for metric, higher in directions.items():
        present = [r for r in rows if r[metric] is not None and np.isfinite(r[metric])]
        present.sort(key=lambda r: r[metric], reverse=higher)
        for rank, row in enumerate(present, 1):
            rank_sums[row["k"]] += rank
            rank_counts[row["k"]] += 1
    for r in rows:
        r["support_rank"] = rank_sums[r["k"]] / rank_counts[r["k"]] if rank_counts[r["k"]] else float("inf")
        r["support_metric_count"] = rank_counts[r["k"]]
    eligible = rows if ambiguous else [r for r in rows if abs(r["k"] - rows[elbow_idx]["k"]) <= 1]
    selected = min(eligible, key=lambda r: (r["support_rank"], r["dispersion"], r["k"]))["k"]

    os.makedirs(out_dir, exist_ok=True)
    fields = ["k", "dispersion", "silhouette", "davies_bouldin", "calinski_harabasz",
              "stability", "population_distribution", "support_rank", "support_metric_count"]
    with open(os.path.join(out_dir, "k_selection.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "population_distribution": json.dumps(row["population_distribution"], sort_keys=True)})
    report = {
        "selected_k": int(selected),
        "elbow_k": int(rows[elbow_idx]["k"]) if elbow_idx is not None else None,
        "elbow_ambiguous": bool(ambiguous), "elbow_algorithm": method,
        "k_min": low, "k_max": high,
        "dispersion_definition": "sum_C sum_{i in C} ||x_i-mean(C)||^2/(2*N_atoms); equivalently sum_C (1/|C|) sum_{i<j in C} MDANCE_MSD(i,j)",
        "metrics": rows,
    }
    with open(os.path.join(out_dir, "k_selection.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ks = [r["k"] for r in rows]
        ax.plot(ks, [r["dispersion"] for r in rows], "o-", color="#005cde")
        if elbow_idx is not None:
            ax.axvline(ks[elbow_idx], ls="--", color="#de005c", label=f"elbow K={ks[elbow_idx]}")
        ax.axvline(selected, ls=":", color="#00804a", label=f"selected K={selected}")
        ax.set(xlabel="K", ylabel="Within-cluster MDANCE-MSD dispersion", title="Elbow K selection")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "k_selection_elbow.png"), dpi=200)
        plt.close(fig)
    except ImportError:
        if logger:
            logger.warning("matplotlib unavailable; skipped Elbow plot")
    if logger:
        logger.info("Automatic K=%d; elbow=%s, ambiguous=%s", selected,
                    report["elbow_k"], ambiguous)
    return int(selected), report
