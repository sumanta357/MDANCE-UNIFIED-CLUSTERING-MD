"""Phase 10 - publication-quality plots (pipeline-added, matplotlib).

Every dimensionality-reduction plot (PCA/t-SNE) is clearly labeled as
VISUALIZATION ONLY - clustering itself runs in the MSD distance space.
"""
from __future__ import annotations

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25})


def _save(fig, out_dir, name, logger):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot written: %s", path)


def plot_cluster_population(best_labels, out_dir, logger, name="cluster_population.png"):
    uniq, counts = np.unique(best_labels[best_labels >= 0], return_counts=True)
    order = np.argsort(-counts)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(np.arange(len(uniq)), counts[order], color="#4C72B0")
    for i, c in enumerate(counts[order]):
        ax.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(uniq)))
    ax.set_xticklabels([str(uniq[i]) for i in order])
    ax.set_xlabel("cluster (ordered by population)")
    ax.set_ylabel("frames")
    ax.set_title("Cluster population distribution (best MDANCE solution)")
    _save(fig, out_dir, name, logger)


def plot_silhouette(D, labels, out_dir, logger, name="silhouette_analysis.png"):
    from sklearn.metrics import silhouette_samples
    mask = labels >= 0
    labels_m = labels[mask]
    uniq = np.unique(labels_m)
    n_samples = int(mask.sum())
    if len(uniq) < 2 or len(uniq) > n_samples - 1:
        logger.info("skip silhouette plot (%d clusters / %d samples not valid)",
                    len(uniq), n_samples)
        return
    vals = silhouette_samples(D[mask][:, mask], labels_m)
    fig, ax = plt.subplots(figsize=(6, 4))
    y0 = 0
    for c in uniq:
        v = np.sort(vals[labels_m == c])
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, alpha=0.7)
        y0 += len(v) + 3
    mean = float(vals.mean())
    ax.axvline(mean, color="crimson", ls="--", label=f"mean = {mean:.3f}")
    ax.set_xlabel("silhouette coefficient (precomputed MDANCE-MSD)")
    ax.set_ylabel("samples (clustered order)")
    ax.legend()
    ax.set_title("Silhouette analysis")
    _save(fig, out_dir, name, logger)


def plot_rmsd_distribution(rmsd, out_dir, logger, name="rmsd_distribution.png"):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(rmsd, bins=40, color="#55A868", alpha=0.8)
    ax.axvline(float(np.median(rmsd)), color="k", ls="--",
               label=f"median = {np.median(rmsd):.3g} A")
    ax.set_xlabel("RMSD to reference frame (A)")
    ax.set_ylabel("frames")
    ax.legend()
    ax.set_title("RMSD distribution (aligned analysis frames)")
    _save(fig, out_dir, name, logger)


def plot_distance_heatmap(D, out_dir, logger, names,
                          name="rmsd_heatmap.png", max_frames=300):
    if D is None:
        return
    n = D.shape[0]
    if n > max_frames:
        idx = np.linspace(0, n - 1, max_frames, dtype=int)
        logger.info("heatmap subsampled to %d of %d frames", max_frames, n)
    else:
        idx = np.arange(n)
    sub = np.sqrt(np.maximum(D[idx][:, idx], 0.0))  # RMSD for visual contrast
    im = ax_img = plt.subplots(figsize=(5.4, 4.6))
    fig, ax = im
    pos = ax.imshow(sub, cmap="magma", origin="lower")
    fig.colorbar(pos, ax=ax, label="RMSD (A)")
    ax.set_title(f"Pairwise structural distance (#{sub.shape[0]} frames)\n"
                 "visualized from the MSD clustering space")
    ax.set_xlabel("frame #")
    ax.set_ylabel("frame #")
    _save(fig, out_dir, name, logger)


def scatter_embedding(X, labels, out_dir, logger, method="PCA",
                      name="pca_clustering.png"):
    mask = labels >= 0
    X_m, lab_m = X[mask], labels[mask]
    uniq = np.unique(lab_m)
    if method == "PCA":
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(X_m)
    else:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42,
                      init="random", learning_rate="auto").fit_transform(X_m)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(uniq), 1)))
    for i, c in enumerate(uniq):
        pts = coords[lab_m == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=colors[i % len(colors)],
                   label=f"c{c} (n={int((lab_m == c).sum())})")
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.set_title(f"{method} projection of coordinate features "
                 f"(VISUALIZATION ONLY - clustering used MSD distances)")
    ax.legend(fontsize=6, markerscale=1.5, ncol=2)
    _save(fig, out_dir, name, logger)


def plot_clustering_comparison(metrics, out_dir, logger,
                              name="clustering_comparison.png"):
    """Composite quality per method-family (best per family)."""
    tops = {}
    for m in metrics:
        if m.composite is None:
            continue
        if m.family not in tops or m.composite > tops[m.family][0]:
            tops[m.family] = (m.composite, m.silhouette, m.n_clusters,
                              m.time_s)
    fams = list(tops.keys())
    comp = [tops[f][0] for f in fams]
    order = np.argsort(-np.array(comp))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(np.arange(len(order)), np.array(comp)[order], color="#C44E52")
    for i, oi in enumerate(order):
        ax.text(i, comp[oi] + 1, f"{comp[oi]:.1f}", ha="center", fontsize=8)
        ax.text(i, 2, f"k={tops[fams[oi]][2]}", ha="center", fontsize=7,
                color="white")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([fams[i] if len(fams[i]) < 18 else fams[i][:16] + "..."
                        for i in order], rotation=45, ha="right")
    ax.set_ylabel("composite validation score (0-100)")
    ax.set_ylim(0, 105)
    ax.set_title("Best solution per method family (MDANCE vs baseline)")
    _save(fig, out_dir, name, logger)


def plot_representative_ranking(rows, out_dir, logger,
                                name="representative_ranking.png"):
    top = rows[: min(12, len(rows))][::-1]
    names = [f"f{o['analysis_frame']} (c{o['cluster_id']})" for o in top]
    scores = [o["final_score"] for o in top]
    colors = ["#C44E52" if i == len(top) - 1 else "#8DA0CB"
              for i in range(len(top))]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh(np.arange(len(top)), scores, color=colors)
    for i, s in enumerate(scores):
        ax.text(s + 0.4, i, f"{s:.1f}", va="center", fontsize=8)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("composite representative score (0-100)")
    ax.set_title("Candidate representative frame ranking")
    _save(fig, out_dir, name, logger)


def plot_scoring_sensitivity(sens_rows, rows, out_dir, logger,
                             name="scoring_sensitivity.png"):
    if not sens_rows:
        return
    frames = [r["winner_analysis_frame"] for r in sens_rows]
    uniq, counts = np.unique(frames, return_counts=True)
    order = np.argsort(-counts)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(np.arange(len(uniq)), counts[order], color="#64B5CD")
    ax.set_xticks(np.arange(len(uniq)))
    ax.set_xticklabels([str(uniq[i]) for i in order], fontsize=8)
    ax.set_xlabel(f"top-frame (analysis frame id)")
    ax.set_ylabel(f"times ranked #1 / {len(sens_rows)} weight draws")
    ax.set_title("Scoring sensitivity to weight perturbations")
    _save(fig, out_dir, name, logger)


def plot_gromacs_populations(best_labels, gromos_labels, out_dir, logger,
                             name="gromacs_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, (title, lab) in zip(axes, [(f"MDANCE best ({np.max(best_labels)+1} cl)", best_labels),
                                       ("gromos baseline", gromos_labels)]):
        uniq, counts = np.unique(lab[lab >= 0], return_counts=True)
        ax.bar(np.arange(len(uniq)), counts / counts.sum(), color="#4C72B0")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("cluster")
        ax.set_ylabel("population fraction")
    fig.suptitle("Population distribution: MDANCE vs GROMOS-type clustering")
    _save(fig, out_dir, name, logger)
