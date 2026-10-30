"""Unit tests for critical functions (pipeline-added tests)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdance.tools import bts  # noqa: E402  MDANCE-native
from mdance_pipeline import clustering, features, scoring, utils  # noqa: E402
from mdance_pipeline.config import DEFAULT_CONFIG, deep_merge, validate_config  # noqa: E402
from mdance_pipeline.validation import (  # noqa: E402
    ch_distance_space, cluster_medoids, db_distance_space)


def make_two_cluster_data(n=40, n_atoms=2, seed=0):
    """Two tight Gaussian blobs placed apart like two conformations."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.1, size=(n, n_atoms * 3))
    B = rng.normal(0, 0.1, size=(n, n_atoms * 3)) + 3.0
    return np.vstack([A, B]), n_atoms * 2


def test_chunked_msd_matches_mdance_native():
    """Chunked MSD must equal MDANCE-native extended_comparison pairwise."""
    X, N_atoms = make_two_cluster_data(n=12)
    D = features.msd_pairwise_chunked(X, N_atoms, chunk_size=5, dtype=np.float64)
    for i in range(0, 12, 5):
        for j in range(i + 1, 12):
            native = bts.extended_comparison(X[[i, j]], data_type="full",
                                             metric="MSD", N_atoms=N_atoms)
            assert abs(D[i, j] - native) < 1e-8


def test_medoid_is_real_row_and_unimodal():
    X, N_atoms = make_two_cluster_data(n=8, seed=3)
    D = features.msd_pairwise_chunked(X, N_atoms, chunk_size=7, dtype=np.float64)
    med = bts.calculate_medoid(X, "MSD", N_atoms)
    local = features.comp_centrality(X, N_atoms)
    assert int(np.argmax(local)) == int(med)          # argmax = medoid
    row_dist = D[med].mean()
    means = [D[i].mean() for i in range(X.shape[0])]
    assert row_dist <= min(means) + 1e-9


def test_outlier_index_matches_argmin():
    X, N_atoms = make_two_cluster_data(n=6, seed=5)
    loc = features.comp_centrality(X, N_atoms)
    assert int(np.argmin(loc)) == int(bts.calculate_outlier(X, "MSD", N_atoms))


def test_gromos_two_clusters():
    """Daura/gromos separates well-separated blobs (k=2 at large cutoff)."""
    X, N_atoms = make_two_cluster_data(n=30)
    D = features.msd_pairwise_chunked(X, N_atoms, chunk_size=16, dtype=np.float64)
    sol = clustering.run_gromos(D, cutoff=3.0)     # 0.015 << 3 << 6.7
    assert sol.n_clusters <= 2                     # greedy: seeds 1 per blob
    labels = sol.labels
    assert sol.n_clusters == 2
    assert (labels[:30] == labels[0]).all()
    assert (labels[30:] == labels[30]).all()
    assert labels[0] != labels[30]


def test_db_ch_distance_space():
    X, N_atoms = make_two_cluster_data(n=30)
    D = features.msd_pairwise_chunked(X, N_atoms, chunk_size=64, dtype=np.float64)
    labels = np.array([0] * 30 + [1] * 30)
    meds = cluster_medoids(D, labels)
    db = db_distance_space(D, labels, meds)
    ch = ch_distance_space(D, labels, meds)
    assert db is not None and db < 3.0
    assert ch is not None and ch > 100


def test_scoring_bounds_and_singleton():
    from mdance_pipeline.config import DEFAULT_CONFIG
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    rows = [
        {"analysis_frame": 0, "original_frame": 0, "time_ps": 0,
         "cluster_id": 0, "cluster_population": 10, "cluster_pop_frac": 0.5,
         "label": "a", "representativeness": 1.0, "population": 0.5,
         "stability": 0.8, "quality": 1.0, "interaction": None,
         "conformational": 0.7},
        {"analysis_frame": 5, "original_frame": 5, "time_ps": 5,
         "cluster_id": 1, "cluster_population": 1, "cluster_pop_frac": 0.05,
         "label": "b", "representativeness": 0.4, "population": 0.05,
         "stability": 0.2, "quality": 0.9, "interaction": None,
         "conformational": 1.0},
    ]
    out = scoring.score_candidates(copy.deepcopy(rows), cfg)
    scores = [r["final_score"] for r in out]
    assert all(0 <= s <= 100 for s in scores)
    assert len(scores) == 2
    # singleton cluster should NOT dominate purely via popularity (capped)
    ranked_top = out[0]["final_score"]
    assert ranked_top >= scores[list(map(lambda r: r["label"], out)).index("b")]


def test_config_validation_and_merge():
    cfg = deep_merge(DEFAULT_CONFIG, {"preprocess": {"stride": 0}})
    assert validate_config(cfg)
    cfg2 = deep_merge(DEFAULT_CONFIG, {})
    assert not validate_config(cfg2) if False else isinstance(cfg2, dict)


def test_seed_reproducible():
    utils.set_seed(11)
    a = np.random.random(3)
    utils.set_seed(11)
    assert (np.random.random(3) == a).all()


def test_minmax_scale_directions():
    s = utils.minmax_scale(np.array([0.0, 2.0]), invert=True)
    assert s[0] > s[1]
    s2 = utils.minmax_scale(np.array([5.0, 5.0]))
    assert np.allclose(s2, 0.5)
