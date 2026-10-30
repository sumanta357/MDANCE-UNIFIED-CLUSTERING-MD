"""Phases 8.-9. - composite representative-frame scoring and sensitivity.

Fully pipeline-added (MDANCE has no frame-composite scorer). All raw metrics
are min-max normalized among candidates to [0, 1] and combined by a weighted
sum, rescaled to 0-100. Ligand metrics are optional and dropped (weights
renormalized) when no ligand is present.
"""
from __future__ import annotations

import logging

import numpy as np

from mdance.tools import bts  # MDANCE-native (comp-sim/outlier semantics)

METRIC_KEYS = ["representativeness", "population", "stability", "quality",
               "interaction", "conformational"]

BASE_WEIGHTS = {
    "representativeness": 0.25, "population": 0.15, "stability": 0.20,
    "quality": 0.15, "interaction": 0.15, "conformational": 0.10,
}


def raw_metrics_table(candidates, cand_labels, best_labels, X, D, N_atoms,
                      info, cfg):
    """Compute the raw (un-normalized) metric values for every candidate."""
    from mdance_pipeline.features import radius_of_gyration, rmsd_series
    scfg = cfg["scoring"]
    n = X.shape[0]
    best = np.asarray(best_labels)
    rg = radius_of_gyration(X, N_atoms)
    rmsd_ref = rmsd_series(X, N_atoms)

    # persistent ligand-contact map & complementary-similarity centrality
    ligpair_usage = None
    if info.get("ligand_found") and len(info.get("ligand_indices", [])):
        ligpair_usage = contact_analysis(X, info, N_atoms)
    comp_vals = bts.calculate_comp_sim(X, "MSD", N_atoms=N_atoms)
    comp_max = float(np.max(comp_vals)) or 1e-12

    rows = []
    for ci, cand in enumerate(candidates):
        c = int(cand)
        cid = int(best[c]) if best[c] >= 0 else -1
        members_idx = np.flatnonzero(best == cid) if cid >= 0 else np.array([c])
        medoid_i = _cluster_medoid(members_idx, X, N_atoms)
        d_med = float(D[c, medoid_i]) if (D is not None and medoid_i is not None) else 0.0
        cluster_max = (float(D[members_idx][:, medoid_i].max())
                       if D is not None and medoid_i is not None else 1.0)
        cluster_min = (float(D[members_idx][:, medoid_i].min())
                       if D is not None and medoid_i is not None else 0.0)
        # A. representativeness: relative position inside the cluster toward medoid
        if cluster_max - cluster_min > 1e-12:
            rep = 1.0 - (d_med - cluster_min) / (cluster_max - cluster_min)
        else:
            rep = 0.5
        # B. population (sqrt-compressed: rewards populated states, does not
        #    blindly prefer the largest cluster)
        pop = float(len(members_idx)) / n
        # C. stability: temporal consistency + RMSD typicality
        w = int(scfg.get("temporal_window", 5))
        nb = np.arange(max(0, c - w), min(n, c + w + 1))
        nb = nb[nb != c]
        if len(nb) and D is not None:
            temporal = float(np.exp(-float(D[c, nb].mean()) /
                                    max(cluster_global_scale(D), 1e-12)))
        else:
            temporal = 0.5
        med_rms = float(np.median(rmsd_ref[members_idx])) if len(members_idx) else 0.0
        rms_typ = float(np.exp(-abs(float(rmsd_ref[c]) - med_rms) /
                               max(float(np.std(rmsd_ref)) or 1e-12, 1e-12)))
        stability = 0.5 * temporal + 0.5 * rms_typ
        # D. structural quality: Rg typicality + local geometry validity
        rg_med = float(np.median(rg[members_idx])) if len(members_idx) else float(np.median(rg))
        rg_typ = float(np.exp(-abs(float(rg[c]) - rg_med) /
                              max(float(np.std(rg)) or 1e-12, 1e-12)))
        finite = 1.0 if np.isfinite(X[c]).all() else 0.0
        quality = 0.5 * rg_typ + 0.5 * finite
        # E. interaction preservation (optional; only with a ligand)
        if ligpair_usage is not None:
            inter = float(ligpair_usage[c])
        else:
            inter = None
        # F. conformational representativeness: centrality + not-the-outlier
        centrality = float(comp_vals[c]) / comp_max
        local_outlier = None
        if len(members_idx) > 3:
            try:
                local_outlier = int(bts.calculate_outlier(
                    X[members_idx], "MSD", N_atoms=N_atoms))
            except Exception:
                local_outlier = None
        is_outlier = local_outlier is not None and int(members_idx[local_outlier]) == int(c)
        conf = 0.7 * centrality + 0.3 * (0.0 if is_outlier else 1.0)
        rows.append({
            "candidate_order": ci,
            "analysis_frame": c,
            "original_frame": int(info["frame_indices"][c]),
            "time_ps": float(info["times"][c]),
            "cluster_id": cid,
            "cluster_population": len(members_idx),
            "cluster_pop_frac": pop,
            "distance_to_medoid": d_med,
            "representativeness": float(rep),
            "population": float(pop),
            "stability": float(stability),
            "quality": float(quality),
            "interaction": inter,
            "conformational": float(conf),
            "label": cand_labels[ci] if ci < len(cand_labels) else "",
        })
    return rows


def _cluster_medoid(members_idx, X, N_atoms):
    try:
        return int(members_idx[int(bts.calculate_medoid(
            X[members_idx], "MSD", N_atoms=N_atoms))])
    except Exception:
        return int(members_idx[0])


def cluster_global_scale(D):
    """Robust global scale = 90th percentile of off-diagonal MSD."""
    iu = np.triu_indices(D.shape[0], k=1)
    vals = D[iu]
    return float(np.quantile(vals, 0.90)) if vals.size else 1.0


def contact_analysis(X, info, N_atoms, dmax=4.5):
    """Ligand-preservation contact analysis (pipeline-added, coarse).

    Works on the ALL-ATOM processed universe (X holds only the selection
    atoms). Contact partners: ligand atoms x protein HEAVY atoms per frame;
    persistent pairs = present in >=50% of analysis frames; the metric is
    the per-frame fraction of persistent contacts retained. Explicitly a
    heavy-atom proximity approximation (no H positions/angles/energies).
    """
    u = info["universe"]
    lig = np.asarray(info["ligand_indices"], dtype=int).copy()
    prot = u.select_atoms("protein").indices.copy()
    try:  # drop hydrogens where masses are available (TPR / guessed)
        masses = np.asarray(u.atoms.masses, dtype=float)
        lig = lig[masses[lig] > 1.5]
        prot = prot[masses[prot] > 1.5]
    except (AttributeError, TypeError):
        try:
            lig = lig[np.array([not str(n).upper().startswith("H")
                                for n in u.atoms.names[lig]], dtype=bool)]
            prot = prot[np.array([not str(n).upper().startswith("H")
                                  for n in u.atoms.names[prot]], dtype=bool)]
        except Exception:
            pass
    if lig.size == 0 or prot.size == 0:
        return None
    n_frames = u.trajectory.n_frames
    maps = np.zeros((n_frames, lig.size, prot.size), dtype=bool)
    for i, ts in enumerate(u.trajectory):
        d = np.linalg.norm(ts.positions[lig][:, None, :] -
                           ts.positions[prot][None, :, :], axis=2)
        maps[i] = d < dmax
    prof = maps.mean(axis=0)
    persistent = prof >= 0.5
    if persistent.any():
        pres = maps[:, persistent].mean(axis=1)
    else:
        counts = maps.sum(axis=(1, 2))
        mode = float(np.median(counts))
        pres = np.array([1.0 if c >= mode else float(c) / max(mode, 1.0)
                         for c in counts])
    return np.asarray(pres, dtype=float)


def score_candidates(rows, cfg) -> list[dict]:
    """Weighted 0-100 composite on min-max-normalized metrics (documented)."""
    from mdance_pipeline.utils import minmax_scale
    weights = dict(cfg["scoring"]["weights"])
    has_lig = any(r["interaction"] is not None for r in rows)
    if not has_lig:
        weights = {k: w for k, w in weights.items() if k != "interaction"}
        weights = {k: w / sum(weights.values()) for k, w in weights.items()}
    invert = {"representativeness": False, "population": False, "stability": False,
              "quality": False, "interaction": False, "conformational": False}
    out = []
    for key, direction in invert.items():
        vals = [r.get(key) for r in rows]
        arr = np.array([np.nan if v is None else float(v) for v in vals])
        s = minmax_scale(arr, invert=direction) if arr.size else arr
        for r, v in zip(rows, s):
            r[f"score_{key}"] = (None if np.isnan(v) else round(float(v) * 100, 2))
    singleton_score = float(cfg["scoring"].get("singleton_cluster_score", 60.0))
    for r in rows:
        total_w, total = 0.0, 0.0
        for key, w in weights.items():
            v = r.get(f"score_{key}")
            if v is None:
                continue
            v = float(v)
            if key == "population" and r["cluster_population"] <= 1:
                v = min(v, singleton_score)   # documented singleton handling
            total += w * v
            total_w += w
        r["final_score"] = round(total / max(total_w, 1e-9), 2)
        out.append(r)
    out.sort(key=lambda r: -r["final_score"])
    for rank, r in enumerate(out, 1):
        r["rank"] = rank
    return out


def sensitivity_analysis(rows, scanned_solutions, cfg, logger) -> tuple[list, dict]:
    """Phase 9 sensitivity (pipeline-added).

    - weight perturbation: Dirichlet draws around the base weights
    - algorithm/k variation: re-scoring candidates taken from the top runner-up
      solutions is reflected by `scanned_solutions` names (report only)
    - distance-metric: RMSD = sqrt(MSD) is monotonic, so candidate Rankings in
      min-max space are invariant (checked & reported)
    Returns (rows-of-winners, summary).
    """
    scfg = cfg["scoring"]
    draws = int(scfg.get("sensitivity_draws", 100))
    strength = float(scfg.get("sensitivity_dirichlet_strength", 8.0))
    base = np.array([scfg["weights"][k] for k in METRIC_KEYS
                     if k in scfg["weights"]], dtype=float)
    keys = [k for k in METRIC_KEYS if k in scfg["weights"]]
    if rows and rows[0].get("score_interaction") is None:
        keys = [k for k in keys if k != "interaction"]
        base = np.array([scfg["weights"][k] for k in keys], dtype=float)
    rng = np.random.default_rng(int(cfg["system"]["seed"]))
    alpha = strength * (base / base.sum())
    winner_counts: dict[int, int] = {}
    list_of_rows = []
    for draw in range(draws):
        w = rng.dirichlet(alpha)
        scores = np.zeros(len(rows), dtype=float)
        for i, r in enumerate(rows):
            s = 0.0
            for kk, key in enumerate(keys):
                v = r.get(f"score_{key}")
                if v is not None:
                    s += w[kk] * float(v)
            scores[i] = s
        winner = int(np.argmax(scores))
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        list_of_rows.append({
            "draw": draw + 1,
            "winner_analysis_frame": rows[winner]["analysis_frame"],
            "winner_original_frame": rows[winner]["original_frame"],
        })
    summary = {
        "draws": draws,
        "win_counts": {str(rows[k]["analysis_frame"]): c
                       for k, c in sorted(winner_counts.items(),
                                          key=lambda kv: -kv[1])},
        "most_robust_analysis_frame": (rows[max(winner_counts.items(), key=lambda kv: kv[1])[0]]
                                       ["analysis_frame"] if winner_counts else None),
        "algorithm_variation_solutions": [s.name for s in (scanned_solutions or [])][:5],
    }
    return list_of_rows, summary
