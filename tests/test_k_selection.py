import numpy as np

from mdance_pipeline.features import msd_pairwise_chunked
from mdance_pipeline.k_selection import within_cluster_dispersion


def test_within_cluster_dispersion_matches_mdance_pairwise_identity():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(8, 6))
    n_atoms = 2
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    D = msd_pairwise_chunked(X, n_atoms, dtype=np.float64)
    expected = 0.0
    for c in (0, 1):
        idx = np.flatnonzero(labels == c)
        block = D[np.ix_(idx, idx)]
        expected += block[np.triu_indices(len(idx), k=1)].sum() / len(idx)
    assert np.isclose(within_cluster_dispersion(X, labels, n_atoms), expected)
