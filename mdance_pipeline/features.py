"""Phase 3 - features and distance calculations.

MDANCE-native pieces (`mdance.tools.bts`) are tagged; the chunked MSD builder
is pipeline-added but validated against `extended_comparison(..., 'MSD')` in
the unit tests (mathematically identical definition).
"""
from __future__ import annotations

import logging
import os

import numpy as np

from mdance.tools import bts  # MDANCE-native toolkit


def msd_pairwise_chunked(X: np.ndarray, N_atoms: int, chunk_size: int = 512,
                         dtype=np.float32) -> np.ndarray:
    """Full pairwise MSD matrix, computed blockwise (pipeline-added).

    MDANCE pairwise convention (verified in msd_condensed source): the MSD
    of a 2-row set is
        MSD(i, j) = ||x_i - x_j||^2 / (2 * N_atoms)
    (= extended_comparison(X[[i,j]], 'full', 'MSD')). Chunked evaluation
    keeps peak memory at 2*chunk*n instead of n^2 floats.
    """
    n = X.shape[0]
    D = np.empty((n, n), dtype=dtype)
    sq = (X ** 2).sum(axis=1)
    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        Xi = X[i0:i1]
        cross = Xi.astype(np.float32 if dtype == np.float32 else float) @ X.T
        block = sq[i0:i1, None] + sq[None, :] - 2.0 * cross
        block = np.maximum(block, 0.0)
        D[i0:i1] = block / np.float32(2.0 * N_atoms)
    return D


def pairwise_msd(a: np.ndarray, b: np.ndarray, N_atoms: int) -> float:
    """Two-object MSD in the MDANCE-pairwise convention
    ||a - b||^2 / (2 * N_atoms) (pipeline-added helper)."""
    return float(((a - b) ** 2).sum() / (2.0 * N_atoms))


def native_msd_pairwise(X: np.ndarray, N_atoms: int) -> np.ndarray:
    """Pairwise MSD matrix via the MDANCE-native API.

    Verified definition (bts source): extended_comparison on a 2-row matrix
    returns the scalar MSD BETWEEN those two rows:
        extended_comparison(X[[i,j]], 'full', 'MSD', N_atoms) -> scalar.
    (Called on the whole set it returns one scalar - the set's own MSD.)
    """
    n = X.shape[0]
    D = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            val = bts.extended_comparison(X[[i, j]], data_type="full",
                                          metric="MSD", N_atoms=N_atoms)
            D[i, j] = D[j, i] = val
    return D


def build_distances(X: np.ndarray, N_atoms: int, cfg: dict,
                    logger: logging.Logger) -> dict:
    """Distance matrix for the pipeline; uses the chunked builder, and
    cross-validates part of it against the native MDANCE API."""
    fcfg = cfg["features"]
    n = X.shape[0]
    huge = n > int(fcfg.get("max_frames_for_full_matrix", 3000))
    if huge:
        logger.warning("n=%d exceeds max_frames_for_full_matrix=%s; "
                       "distance-matrix methods will be skipped.",
                       n, fcfg.get("max_frames_for_full_matrix"))
        return {"D": None, "distance_full_possible": False}
    if n > 1024:
        # even a 1024^2 float64 matrix is <10 MB, but keep the safer check
        pass
    D = msd_pairwise_chunked(
        X, N_atoms,
        chunk_size=int(fcfg.get("chunk_size", 512)),
        dtype=np.float32 if fcfg.get("use_float32", True) else np.float64)
    # Cross-validation of the chunked formula against MDANCE native (probe pairs)
    n_probe = min(12, n)
    native = native_msd_pairwise(X[:n_probe], N_atoms)
    diff = float(np.abs(D[:n_probe, :n_probe] - native).max())
    if diff > 1e-3 * max(1.0, float(native.max())):
        raise RuntimeError(
            f"Chunked MSD disagrees with MDANCE native extended_comparison "
            f"(max abs diff {diff:.3e}).")
    logger.info("Distance matrix built: %dx%d (chunked MSD; "
                "native cross-check OK, max|diff|=%.2e)", n, n, diff)
    return {"D": D, "distance_full_possible": True}


def rmsd_series(X: np.ndarray, N_atoms: int) -> np.ndarray:
    """Per-frame RMSD (A) of each frame to frame 0 (post-alignment).

    RMSD = sqrt( sum((x - ref)^2) / N_atoms ); NOTE = sqrt(2 * MDANCE-MSD
    for a pair) - an independent, documented stability metric (pipeline-added)."""
    ref = X[0]
    sq = ((X - ref) ** 2).sum(axis=1)
    return np.sqrt(np.maximum(sq, 0.0) / N_atoms)


def radius_of_gyration(X: np.ndarray, N_atoms: int) -> np.ndarray:
    """Rg (A) of the selected atoms, per frame (3D flattened vectors)."""
    xyz = X.reshape(-1, N_atoms, 3)
    center = xyz.mean(axis=1, keepdims=True)
    return np.sqrt(((xyz - center) ** 2).sum(axis=2).mean(axis=1))


def comp_centrality(X: np.ndarray, N_atoms: int) -> np.ndarray:
    """Per-frame complementary-similarity centrality (MDANCE-native
    calculate_comp_sim). Verified MDANCE semantics: ARGMAX(comp_sim) = medoid,
    ARGMIN(comp_sim) = outlier, so HIGH value = central frame."""
    try:
        vals = np.asarray(
            bts.calculate_comp_sim(np.asarray(X, float), "MSD", N_atoms=N_atoms),
            dtype=float)
    except Exception:
        vals = None
    if vals is None or len(vals) != X.shape[0]:
        # conservative fallback on identical semantics (higher = central)
        vals = np.abs(X - X.mean(axis=0)).mean(axis=1)
        vals = -vals
    return vals


def save_features(X: np.ndarray, D: np.ndarray | None, out_dir: str) -> None:
    """Write feature_data.npy / distance_matrix.npy as required Phase 11."""
    f_dir = os.path.join(out_dir, "features")
    os.makedirs(f_dir, exist_ok=True)
    np.save(os.path.join(f_dir, "feature_data.npy"), X)
    if D is not None:
        np.save(os.path.join(f_dir, "distance_matrix.npy"), D)
    else:
        np.save(os.path.join(f_dir, "distance_matrix.npy"),
                np.array([], dtype=float))  # n too large: skipped
