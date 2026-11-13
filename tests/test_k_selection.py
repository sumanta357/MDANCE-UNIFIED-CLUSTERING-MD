import numpy as np

from mdance_pipeline.features import msd_pairwise_chunked
from mdance_pipeline.k_selection import (detect_elbow, within_cluster_dispersion)


def test_dispersion_matches_mdance_pairwise_identity():
    """W == mean within-cluster MDANCE MSD == centered sum / (2*N_atoms)."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(8, 6))
    n_atoms = 2
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    D = msd_pairwise_chunked(X, n_atoms, dtype=np.float64)

    expected_from_distances = 0.0
    expected_from_coords = 0.0
    for c in (0, 1):
        idx = np.flatnonzero(labels == c)
        block = D[np.ix_(idx, idx)]
        expected_from_distances += block[np.triu_indices(len(idx), k=1)].sum() \
            / len(idx)
        sub = X[idx]
        expected_from_coords += ((sub - sub.mean(axis=0)) ** 2).sum() \
            / (2.0 * n_atoms)

    got = within_cluster_dispersion(D, labels)
    assert np.isclose(got, expected_from_distances)
    assert np.isclose(got, expected_from_coords)


def test_dispersion_ignores_noise_and_is_not_euclidean_inertia():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(6, 4))
    n_atoms = 2
    D = msd_pairwise_chunked(X, n_atoms, dtype=np.float64)
    labels = np.array([0, 0, 1, 1, 1, -1])
    res = within_cluster_dispersion(D, labels)
    euclid_inertia = sum(((X[labels == c] - X[labels == c].mean(axis=0)) ** 2)
                         .sum() for c in (0, 1))
    assert res > 0
    # MDANCE dispersion is per-atom (divided by 2*N_atoms), inertia is not
    assert np.isclose(res, euclid_inertia / (2.0 * n_atoms))


def test_per_method_k_runs_and_returns_map(tmp_path):
    """Per-method elbow scan produces an independent K per method."""
    import copy

    from mdance_pipeline.config import DEFAULT_CONFIG
    from mdance_pipeline.features import msd_pairwise_chunked
    from mdance_pipeline.k_selection import run_per_method

    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(0.0, 0.3, (15, 24)),
                   rng.normal(1.2, 0.3, (15, 24))])
    D = msd_pairwise_chunked(X, 8, dtype=np.float64)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["k_selection"].update({"k_min": 2, "k_max": 4, "stability_repeats": 1,
                               "per_method_families": ["nani", "agglom"]})
    rep = run_per_method(X, D, 8, cfg, str(tmp_path / "ks"))
    assert set(rep["per_method"]) == {"nani", "agglom"}
    for info in rep["per_method"].values():
        assert 2 <= info["selected_k"] <= 4
        assert info["metrics"]
    assert (tmp_path / "ks" / "k_selection_per_method.csv").is_file()
    assert (tmp_path / "ks" / "k_selection_per_method.json").is_file()


def test_elbow_detection_finds_knee_and_flags_ambiguity():
    ks = [2, 3, 4, 5, 6, 7, 8]
    knee_pos, algo, ambiguous = detect_elbow(
        ks, [100.0, 60.0, 45.0, 42.0, 40.0, 39.0, 38.0])
    assert 0 < knee_pos < len(ks) - 1
    assert algo in {"kneed", "geometric"}
    assert not ambiguous

    # a linear (no-knee) curve must be reported as ambiguous, not forced
    flat_pos, _, flat_ambiguous = detect_elbow(
        ks, [40.0, 38.5, 37.0, 35.5, 34.0, 32.5, 31.0])
    assert flat_ambiguous and flat_pos is None
