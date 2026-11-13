"""Export each method's representatives and a separate method/frame recommendation."""
from __future__ import annotations

import csv
import os
import re

import numpy as np

from mdance_pipeline import representatives as reps
from mdance_pipeline.pipeline import info_idx


def export_all_methods(pipe):
    """Export top-ranked solution per method family, plus independent recommendation.

    Partitions are preserved exactly as produced; each method family is handled
    separately, and exported coordinates are actual frames from the trajectory.
    """
    root = os.path.join(pipe.out_dir, "representatives", "representative_frames")
    os.makedirs(root, exist_ok=True)
    best_by_family = {}
    for metric in pipe.ranked:
        best_by_family.setdefault(metric.family, metric)

    report_rows = []
    for family, metric in best_by_family.items():
        sol = next((s for s in pipe.sols if s.name == metric.name), None)
        if sol is None or sol.labels.size == 0:
            continue
        stats = reps.cluster_centroids_stats(pipe.dist["X"], sol.labels,
                                             pipe.info["n_atoms_selected"])
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", family).strip("_") or "method"
        dest = os.path.join(root, slug)
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, "representatives.csv"), "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["cluster_id", "population", "medoid_analysis_frame",
                             "medoid_original_frame", "medoid_time_ps",
                             "nearest_centroid_analysis_frame", "nearest_centroid_original_frame",
                             "nearest_centroid_time_ps"])
            for cid, st in sorted(stats.items()):
                med = int(st["medoid_frame"])
                near = int(st["nearest_to_centroid"])
                reps.write_frame_pdb(pipe.info, med,
                    os.path.join(dest, f"cluster_{cid + 1:03d}_medoid.pdb"), pipe.logger)
                if pipe.cfg["representatives"].get("nearest_to_centroid", True):
                    reps.write_frame_pdb(pipe.info, near,
                        os.path.join(dest, f"cluster_{cid + 1:03d}_centroid.pdb"), pipe.logger)
                writer.writerow([cid, st["population"], med,
                                 info_idx(pipe.info, med, "orig"), info_idx(pipe.info, med, "t"),
                                 near, info_idx(pipe.info, near, "orig"), info_idx(pipe.info, near, "t")])
                report_rows.append((family, metric.name, cid, med, st["population"]))

    # The validation ranking independently recommends a method. Its largest
    # cluster's medoid is an actual representative frame; the usual composite
    # frame score/report remains available separately in the main pipeline.
    best_metric = pipe.ranked[0]
    best_sol = next(s for s in pipe.sols if s.name == best_metric.name)
    best_stats = reps.cluster_centroids_stats(pipe.dist["X"], best_sol.labels,
                                               pipe.info["n_atoms_selected"])
    if not best_stats:
        return
    cluster_id = max(best_stats, key=lambda c: best_stats[c]["population"])
    frame = int(best_stats[cluster_id]["medoid_frame"])
    out = os.path.join(pipe.out_dir, "representatives", "RECOMMENDED_METHOD_FRAME.pdb")
    reps.write_frame_pdb(pipe.info, frame, out, pipe.logger)
    with open(os.path.join(pipe.out_dir, "representatives", "method_recommendation.txt"), "w") as fh:
        fh.write(f"Recommended method: {best_metric.name}\n")
        fh.write(f"Family: {best_metric.family}\n")
        fh.write(f"Validation rank: {best_metric.rank}\n")
        fh.write(f"Recommended cluster: {cluster_id}\n")
        fh.write(f"Analysis frame: {frame}\n")
        fh.write(f"Original trajectory frame: {info_idx(pipe.info, frame, 'orig')}\n")
        fh.write(f"Simulation time (ps): {info_idx(pipe.info, frame, 't')}\n")
        fh.write(f"Structure: {out}\n")
        fh.write("\nPer-family solution medoid summary:\n")
        for family, name, cid, idx, population in report_rows:
            fh.write(f"{family},{name},cluster={cid},frame={idx},population={population}\n")
