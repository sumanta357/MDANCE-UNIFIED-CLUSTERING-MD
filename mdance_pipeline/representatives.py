"""Phase 7 - representative structure extraction (pipeline-added, MDANCE
native medoid/outlier routines reused).

Important design rule (explicit): k-means/BIRCH mathematical centroids are
NOT physical frames. Only ACTUAL trajectory frames are exported.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from mdance.tools import bts  # MDANCE-native


def cluster_medoid_indices(X: np.ndarray, labels: np.ndarray,
                           N_atoms: int) -> dict[int, int]:
    """Per-cluster medoid via MDANCE-native `bts.calculate_medoid` on the
    cluster submatrix (returns actual frame indices)."""
    out = {}
    for c in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == c)
        if len(idx) == 0:
            continue
        med_local = bts.calculate_medoid(X[idx], "MSD", N_atoms=N_atoms)
        out[int(c)] = int(idx[int(med_local)])
    return out


def cluster_centroids_stats(X: np.ndarray, labels: np.ndarray,
                            N_atoms: int) -> dict[int, dict]:
    """Per cluster: mathematical centroid + nearest ACTUAL frame + medoid.

    `nearest_actual_index` is the frame minimizing MSD to the centroid; the
    centroid itself is deliberately NOT exportable (not physical).
    """
    out: dict[int, dict] = {}
    for c in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == c)
        if len(idx) == 0:
            continue
        sub = X[idx]
        centroid = sub.mean(axis=0)
        msd_to_centroid = ((sub - centroid) ** 2).sum(axis=1) / (2.0 * N_atoms)
        nearest = idx[int(np.argmin(msd_to_centroid))]
        med_local = bts.calculate_medoid(sub, "MSD", N_atoms=N_atoms)
        medoid_idx = idx[int(med_local)]
        out[int(c)] = {
            "cluster_id": int(c),
            "members": idx.tolist(),
            "population": int(len(idx)),
            "centroid_coords": centroid,
            "nearest_to_centroid": int(nearest),
            "medoid_frame": int(medoid_idx),
        }
    return out


def global_medoid_index(X: np.ndarray, N_atoms: int) -> int:
    """Global medoid of the trajectory (MDANCE-native, an actual frame)."""
    return int(bts.calculate_medoid(X, "MSD", N_atoms=N_atoms))


def build_candidates(best_stats: dict, alt_stats: list[dict],
                     X: np.ndarray, D, N_atoms: int,
                     cfg) -> tuple[list[int], list[str]]:
    """Candidate frames for final scoring: medoids + nearest-to-centroid of
    every cluster of the best solution, plus (config) the medoids of runner-up
    solutions. Deduplicated, original indices."""
    seen: set[int] = set()
    candidates: list[int] = []
    labels: list[str] = []
    for cid, st in best_stats.items():
        for idx, tag in ((st["medoid_frame"], f"medoid(c{cid})"),
                         (st["nearest_to_centroid"],
                          f"nearest-centroid(c{cid})")):
            if cfg["representatives"].get("nearest_to_centroid", True) or \
                    tag.startswith("medoid"):
                if idx not in seen:
                    seen.add(idx)
                    candidates.append(idx)
                    labels.append(tag)
    for i, st in enumerate(alt_stats or []):
        for cid, info in st.items():
            idx = info["medoid_frame"]
            if idx not in seen:
                seen.add(idx)
                candidates.append(idx)
                labels.append(f"alt{i+1}-medoid(c{cid})")
    return candidates, labels


def write_frame_pdb(info, frame_idx: int, out_path: str, logger=None) -> None:
    """Write ONE original (preprocessed) frame with ALL atoms to PDB.

    Coordinates come from the processed trajectory (PBC-fixed, aligned) of the
    exact frame `frame_idx` (analysis index); no averaging / interpolation.
    Atom names, residue numbers, ligands, ions, cofactors are preserved from
    the processed topology as far as the PDB format allows.
    """
    u = info["universe"]
    u.trajectory[frame_idx]
    u.atoms.write(out_path)
    if logger:
        logger.info("Wrote frame PDB: %s (analysis frame %d -> original frame %d)",
                    out_path, frame_idx, info["frame_indices"][frame_idx])


def write_representatives(best_stats: dict, best_labels, info, out_dir,
                          logger) -> str:
    """Export representatives/global_medoid.pdb, cluster reps, candidates."""
    reps_dir = os.path.join(out_dir, "representatives")
    os.makedirs(os.path.join(reps_dir, "cluster_representatives"), exist_ok=True)
    os.makedirs(os.path.join(reps_dir, "candidate_frames"), exist_ok=True)

    gm = global_medoid_index(info["X"], info["n_atoms_selected"])
    write_frame_pdb(info, gm, os.path.join(reps_dir, "global_medoid.pdb"), logger)
    for cid, st in best_stats.items():
        for kind, idx in (("medoid", st["medoid_frame"]),
                          ("nearest_to_centroid", st["nearest_to_centroid"])):
            path = os.path.join(reps_dir, "cluster_representatives",
                                f"cluster{cid:03d}_{kind}.pdb")
            write_frame_pdb(info, idx, path, logger)
    return reps_dir
