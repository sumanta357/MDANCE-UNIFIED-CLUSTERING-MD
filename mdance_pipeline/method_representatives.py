"""Per-method representative extraction + independent method recommendation.

Requirement implemented here: representative frames are extracted for EVERY
clustering method, not only for the winner. For each method family the
best-ranked solution of that family is taken (parameter variants of one family
are alternatives of the same method) and, for each of its clusters, the medoid
frame and the nearest-to-centroid frame are written as individual PDB files.

Every exported structure is an ACTUAL trajectory frame (PBC-corrected and
reference-aligned by preprocessing); the original trajectory frame index and
simulation time are preserved in the CSV/JSON metadata next to the PDBs.

The recommendation of the best method is produced independently of the
composite representative-frame score (`BEST_REPRESENTATIVE_FRAME.pdb`).
"""
from __future__ import annotations

import csv
import json
import os
import re

import numpy as np

from mdance_pipeline import representatives as reps
from mdance_pipeline.utils import frame_meta

# Filesystem-safe, human-readable folder names per family
FAMILY_DIRS = {
    "nani": "NANI",
    "divine": "DIVINE",
    "equal": "eQual",
    "helm": "HELM",
    "mdbirch": "mdBIRCH",
    "agglom": "Agglomerative",
    "dbscan": "DBSCAN",
    "gromos": "GROMOS",
}


def _family_root(family: str) -> str:
    base = family.split("/")[0]
    return FAMILY_DIRS.get(base, re.sub(r"[^A-Za-z0-9_.-]+", "_", base))


def _base(family: str) -> str:
    """Method family without parameter-variant suffix (nani/strat_all -> nani)."""
    return family.split("/")[0]


def _best_per_family(sols, ranked, selected_k: int | None = None) -> dict:
    """One representative solution per METHOD (not per parameter variant).

    Preference: the best validation-ranked solution of that method. If the
    method only produced degenerate partitions (no ranking), the closest
    usable solution with >= 2 clusters is used so every method still gets its
    representative frames; the choice is recorded in the JSON summary.
    """
    by_name = {m.name: m for m in ranked if m.composite is not None}
    best: dict[str, object] = {}
    for sol in sols:
        if sol.labels.size == 0 or sol.error or sol.n_clusters < 2:
            continue
        metric = by_name.get(sol.name)
        if metric is None:
            continue
        key = _base(sol.family)
        current = best.get(key)
        if current is None or metric.rank < current[0].rank:
            best[key] = (metric, sol)

    # fallback: methods whose partitions were all excluded from ranking
    best_fallback: dict[str, object] = {}
    for sol in sols:
        if sol.labels.size == 0 or sol.error or sol.n_clusters < 2:
            continue
        key = _base(sol.family)
        if key in best:
            continue
        score = (abs(sol.n_clusters - (selected_k or sol.n_clusters)),
                 sol.name)
        cur = best_fallback.get(key)
        if cur is None or score < cur[0]:
            best_fallback[key] = (score, sol)
    for key, (_, sol) in best_fallback.items():
        best[key] = (None, sol)
    return best


def export_all_methods(pipe) -> dict:
    """Write representatives/representative_frames/<METHOD>/ for every family.

    Returns a summary dict: chosen solution per family and the recommendation.
    """
    log = pipe.logger
    N = pipe.info["n_atoms_selected"]
    X, D = pipe.dist["X"], pipe.dist["D"]
    root = os.path.join(pipe.out_dir, "representatives", "representative_frames")
    os.makedirs(root, exist_ok=True)
    write_centroid = bool(
        pipe.cfg["representatives"].get("nearest_to_centroid", True))

    selected_k = pipe.cfg["clustering"].get("selected_k")
    chosen = _best_per_family(pipe.sols, pipe.ranked, selected_k)
    # GROMOS/Daura participates only in the GROMACS benchmark, but it is a
    # clustering method too: export its closest usable cutoff solution.
    for sol in (pipe.bench_results or {}).get("solutions", []) or []:
        if sol.labels.size == 0 or sol.error or sol.n_clusters < 2:
            continue
        key = "gromos"
        cur = chosen.get(key)
        better = (cur is None or
                  abs(sol.n_clusters - (selected_k or sol.n_clusters)) <
                  abs(cur[1].n_clusters - (selected_k or sol.n_clusters)))
        if better:
            chosen[key] = (None, sol)
    summary: dict = {"families": {}, "skipped": {},
                    "recommendation": None}
    present_families = {_base(s.family) for s in pipe.sols}

    for family, (metric, sol) in sorted(chosen.items()):
        stats = reps.cluster_centroids_stats(X, sol.labels, N)
        if not stats:
            continue
        name = _family_root(family)
        dest = os.path.join(root, name)
        os.makedirs(dest, exist_ok=True)
        rows = []
        for cid in sorted(stats):
            st = stats[cid]
            med = int(st["medoid_frame"])
            near = int(st["nearest_to_centroid"])
            med_meta = frame_meta(pipe.info, med)
            near_meta = frame_meta(pipe.info, near)
            reps.write_frame_pdb(
                pipe.info, med,
                os.path.join(dest, f"cluster_{cid + 1:03d}_medoid.pdb"),
                logger=log)
            if write_centroid:
                reps.write_frame_pdb(
                    pipe.info, near,
                    os.path.join(dest, f"cluster_{cid + 1:03d}_centroid.pdb"),
                    logger=log)
            rows.append({
                "cluster_id": cid + 1,
                "population": st["population"],
                "population_frac": round(st["population"] / len(sol.labels), 6),
                "medoid_analysis_frame": med,
                "medoid_original_frame": med_meta["original_frame"],
                "medoid_time_ps": med_meta["time_ps"],
                "nearest_centroid_analysis_frame": near,
                "nearest_centroid_original_frame": near_meta["original_frame"],
                "nearest_centroid_time_ps": near_meta["time_ps"],
                "msd_medoid_to_centroid": round(float(D[med, near]), 6),
            })
        with open(os.path.join(dest, "representatives.csv"), "w",
                  newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        summary["families"][name] = {
            "solution": sol.name,
            "family": family,
            "native_mdance": bool(sol.native),
            "n_clusters": int(len(stats)),
            "validation_rank": (int(metric.rank)
                                if metric is not None else None),
            "composite": (float(metric.composite)
                          if metric is not None else None),
            "selection_note": (
                "best validation-ranked solution of this method"
                if metric is not None else
                "all ranked partitions of this method were degenerate; "
                "closest usable solution used"),
            "directory": os.path.relpath(dest, pipe.out_dir),
            "clusters": rows,
        }
        log.info("Representatives exported for %-14s (%s): %d clusters -> %s",
                 name, sol.name, len(stats),
                 os.path.relpath(dest, pipe.out_dir))

    # Methods that produced no usable (>=2 cluster) partition are reported
    # explicitly instead of being silently absent from the tree.
    for key in sorted(present_families - set(chosen)):
        fam_sols = [s for s in pipe.sols if _base(s.family) == key]
        summary["skipped"][_family_root(key)] = {
            "reason": "no partition with >= 2 clusters",
            "solutions": [s.name for s in fam_sols],
            "cluster_counts": [int(s.n_clusters) for s in fam_sols],
        }
        log.warning("No representatives for %s: %s", _family_root(key),
                    summary["skipped"][_family_root(key)]["reason"])

    # --- independent method recommendation -------------------------------
    top = pipe.ranked[0]
    top_sol = next(s for s in pipe.sols if s.name == top.name)
    emp = "No clustering solution was ranked (all were degenerate)."
    if top_sol.labels.size:
        top_stats = reps.cluster_centroids_stats(X, top_sol.labels, N)
        if top_stats:
            cid = max(top_stats, key=lambda c: top_stats[c]["population"])
            frame = int(top_stats[cid]["medoid_frame"])
            meta = frame_meta(pipe.info, frame)
            out_pdb = os.path.join(pipe.out_dir, "representatives",
                                   "RECOMMENDED_METHOD_FRAME.pdb")
            reps.write_frame_pdb(pipe.info, frame, out_pdb, logger=log)
            summary["recommendation"] = {
                "method": top.name,
                "family": top.family,
                "native_mdance": bool(top.native),
                "validation_rank": int(top.rank),
                "validation_composite": float(top.composite),
                "n_clusters": int(top.n_clusters),
                "cluster_id": int(cid) + 1,
                "population": int(top_stats[cid]["population"]),
                "analysis_frame": frame,
                "original_frame": meta["original_frame"],
                "time_ps": meta["time_ps"],
                "structure": os.path.relpath(out_pdb, pipe.out_dir),
                "note": ("Medoid (actual trajectory frame) of the largest "
                         "cluster of the top validation-ranked method. "
                         "Independent of the composite frame score in "
                         "BEST_REPRESENTATIVE_FRAME.pdb."),
            }
            emp = ""
    with open(os.path.join(pipe.out_dir, "representatives",
                           "method_recommendation.txt"), "w") as fh:
        if summary["recommendation"]:
            r = summary["recommendation"]
            fh.write("RECOMMENDED CLUSTERING METHOD (independent of the "
                     "composite frame score)\n")
            fh.write("=" * 68 + "\n")
            for key in ("method", "family", "native_mdance", "n_clusters",
                        "validation_rank", "validation_composite",
                        "cluster_id", "population", "analysis_frame",
                        "original_frame", "time_ps", "structure", "note"):
                fh.write(f"{key:<22}: {r[key]}\n")
            fh.write("\nPer-family representative directories:\n")
            for name, fam in summary["families"].items():
                rank = (f"#{fam['validation_rank']}"
                        if fam["validation_rank"] is not None else "n/a")
                fh.write(f"  {name:<14} {fam['solution']:<34} "
                         f"k={fam['n_clusters']:<3} rank={rank:<4} "
                         f"{fam['directory']}/\n")
            if summary["skipped"]:
                fh.write("\nMethods with no usable partition (nothing to "
                         "export):\n")
                for name, info in summary["skipped"].items():
                    fh.write(f"  {name:<14} {info['reason']} "
                             f"(cluster counts: {info['cluster_counts']})\n")
        else:
            fh.write(emp + "\n")
    with open(os.path.join(pipe.out_dir, "representatives",
                           "method_representatives.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary
