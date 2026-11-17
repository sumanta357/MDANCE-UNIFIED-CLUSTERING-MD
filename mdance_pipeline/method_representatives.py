"""Per-method representative extraction + independent method recommendation.

Representative frames are compared for EVERY clustering method, not only the
winner. For each method family the best-ranked solution is taken (parameter
variants are alternatives of the same method). The winning pose of every
successful method is exported once under
`02_METHOD_COMPARISON/METHOD_REPRESENTATIVES/<METHOD>.pdb`; with
`--full-output`, every cluster's medoid and nearest-to-centroid frame are
also written under `04_ADVANCED_ANALYSIS/CLUSTER_MEDOIDS/<METHOD>/`.

Every exported structure is an ACTUAL trajectory frame (PBC-corrected and
reference-aligned by preprocessing); the original trajectory frame index and
simulation time are preserved in the CSV/JSON metadata next to the PDBs.

The recommendation of the best method is produced independently of the
composite representative-frame score (`01_FINAL_RESULTS/BEST_POSE.pdb`).
"""
from __future__ import annotations

import csv
import json
import os
import re

import numpy as np

from mdance_pipeline import output_layout, representatives as reps
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
    "gmx_native": "GROMACS_native",
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


def method_slug(name: str) -> str:
    """Filesystem-safe, readable method label (e.g. 'NANI_comp_sim_k3')."""
    fam = {"agglom": "AGGLOMERATIVE", "equal": "EQUAL",
           "mdbirch": "MDBIRCH"}.get(
               str(name).split("/")[0], str(name).split("/")[0].upper())
    rest = [p for p in str(name).split("/")[1:] if p]
    slug = "_".join([fam] + rest)
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "-"
                   for ch in slug)


def collect_method_candidates(pipe) -> tuple[dict, list[dict]]:
    """All per-method representative frames as scoring candidates.

    One solution per method (the best validation-ranked of that method, or the
    closest usable one if all its partitions were degenerate). Every cluster
    contributes its medoid and, when enabled, its nearest-to-centroid frame.
    Each candidate carries the labels of ITS OWN method partition so the
    composite score is computed inside that method's cluster structure.

    Returns (chosen, candidates).
    """
    N = pipe.info["n_atoms_selected"]
    X = pipe.dist["X"]
    selected_k = pipe.cfg["clustering"].get("selected_k")
    chosen = _best_per_family(pipe.sols, pipe.ranked, selected_k)
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
    # Native GROMACS is a benchmark tier, but its real membership can also
    # contribute a representative pose and a method-comparison row.
    from mdance_pipeline import gromacs_bench
    nsol = gromacs_bench.native_solution(
        (pipe.bench_results or {}).get("native"))
    if nsol is not None and nsol.n_clusters >= 2:
        chosen["gmx_native"] = (None, nsol)

    write_centroid = bool(
        pipe.cfg["representatives"].get("nearest_to_centroid", True))
    candidates: list[dict] = []
    for family, (metric, sol) in sorted(chosen.items()):
        stats = reps.cluster_centroids_stats(X, sol.labels, N)
        if not stats:
            continue
        for cid in sorted(stats):
            st = stats[cid]
            kinds = [("medoid", st["medoid_frame"])]
            # when the medoid IS the frame nearest the centroid, one entry is
            # enough: the same pose would otherwise be scored twice
            if write_centroid and \
                    st["nearest_to_centroid"] != st["medoid_frame"]:
                kinds.append(("nearest_centroid", st["nearest_to_centroid"]))
            for kind, frame in kinds:
                candidates.append({
                    "frame": int(frame),
                    "labels": sol.labels,
                    "cluster_id": int(cid),
                    "method": sol.name,
                    "family": _family_root(family),
                    "family_raw": family,
                    "solution": sol,
                    "k": int(sol.n_clusters),
                    "kind": kind,
                    "population": int(st["population"]),
                    "tag": (f"{_family_root(family)}:{sol.name}:"
                            f"{kind}(c{cid + 1})"),
                })
    return chosen, candidates


def export_all_methods(pipe) -> dict:
    """Summarise every method and write the recommended-method pose.

    Only one representative PDB per successful method is written to the main
    tree (``02_METHOD_COMPARISON/METHOD_REPRESENTATIVES``), chosen later from
    the scored candidates. Per-cluster medoid/centroid PDBs are optional and
    go to ``04_ADVANCED_ANALYSIS/CLUSTER_MEDOIDS`` with ``--full-output``.

    Returns a summary dict: chosen solution per family and the recommendation.
    """
    log = pipe.logger
    N = pipe.info["n_atoms_selected"]
    X, D = pipe.dist["X"], pipe.dist["D"]
    full_output = bool(pipe.cfg["system"].get("full_output", False))
    root = os.path.join(pipe.out_dir, output_layout.MEDOID_DIR)
    if full_output:
        os.makedirs(root, exist_ok=True)
    write_centroid = bool(
        pipe.cfg["representatives"].get("nearest_to_centroid", True))

    chosen, _ = collect_method_candidates(pipe)
    summary: dict = {"families": {}, "skipped": {},
                    "recommendation": None}
    present_families = {_base(s.family) for s in pipe.sols}

    for family, (metric, sol) in sorted(chosen.items()):
        stats = reps.cluster_centroids_stats(X, sol.labels, N)
        if not stats:
            continue
        name = _family_root(family)
        dest = os.path.join(root, name)
        if full_output:
            os.makedirs(dest, exist_ok=True)
        rows = []
        for cid in sorted(stats):
            st = stats[cid]
            med = int(st["medoid_frame"])
            near = int(st["nearest_to_centroid"])
            med_meta = frame_meta(pipe.info, med)
            near_meta = frame_meta(pipe.info, near)
            if full_output:
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
        if full_output:
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
            "directory": (output_layout.rel(pipe.out_dir, dest)
                          if full_output else None),
            "clusters": rows,
        }
        log.info("Representatives summarised for %-14s (%s): %d clusters",
                 name, sol.name, len(stats))

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

    # methods that never produced a solution at all (disabled / too many
    # frames / failed) are reported too, instead of silently disappearing
    coverage = getattr(pipe, "method_coverage", {}) or {}
    present = {_family_root(k) for k in chosen}
    for fam, info in coverage.items():
        label = {"agglom": "AGGLOMERATIVE", "equal": "EQUAL"}.get(
            fam.upper(), fam.upper())
        if any(label == p for p in present):
            continue
        if info.get("status") in ("not_run", "deferred"):
            summary["skipped"].setdefault(label, {
                "reason": info.get("reason") or "method not executed",
                "solutions": [], "cluster_counts": [],
            })
            log.warning("No representatives for %s: %s", label,
                        summary["skipped"][label]["reason"])

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
            out_pdb = os.path.join(pipe.out_dir, output_layout.FINAL_DIR,
                                   "BEST_METHOD.pdb")
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
                         "01_FINAL_RESULTS/BEST_POSE.pdb."),
            }
            emp = ""
    with open(os.path.join(pipe.out_dir, output_layout.INTERMEDIATE_DIR,
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
                         f"k={fam['n_clusters']:<3} rank={rank:<4}\n")
            if summary["skipped"]:
                fh.write("\nMethods with no usable partition (nothing to "
                         "export):\n")
                for name, info in summary["skipped"].items():
                    fh.write(f"  {name:<14} {info['reason']} "
                             f"(cluster counts: {info['cluster_counts']})\n")
        else:
            fh.write(emp + "\n")
    with open(os.path.join(pipe.out_dir, output_layout.INTERMEDIATE_DIR,
                           "method_representatives.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary
