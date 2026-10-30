"""Integration test: full pipeline on a synthetic GROMACS-like trajectory.

Builds a small protein-like topology (named so that `protein and backbone`
selections work) plus a ligand, writes GRO + XTC with MDAnalysis, and runs the
complete run_mdance.py CLI flow in a temporary directory.
"""
import copy
import os
import sys

import numpy as np
import MDAnalysis as mda
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import MDAnalysis.transformations as mda_transforms  # noqa: E402

AMINO_ACIDS = ["GLY", "ALA", "SER", "VAL", "LEU", "ILE", "THR", "ASN"]


def build_universe(n_res: int = 12, roi=None) -> mda.Universe:
    atoms_per_res = 4  # N CA C O backbone-like naming
    n_atoms = n_res * atoms_per_res + 4  # +ligand
    atom_resindex = np.repeat(np.arange(n_res + 1), [atoms_per_res] * n_res + [4])
    u = mda.Universe.empty(n_atoms, n_residues=n_res + 1,
                           atom_resindex=atom_resindex,
                           residue_segindex=np.zeros(n_res + 1, int),
                           trajectory=True)
    names = (["N", "CA", "C", "O"] * n_res) + ["C1", "C2", "C3", "C4"]
    resnames = [AMINO_ACIDS[i % len(AMINO_ACIDS)] for i in range(n_res)] + ["LIG"]
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("resid", list(range(1, n_res + 2)))
    u.add_TopologyAttr("segid", ["SYS"])  # single segment
    return u


def frame_positions(t: float, transitioning: bool, n_res: int = 12):
    """Two conformational states + slow modulation (synthetic 'protein')."""
    base = np.zeros((n_res * 4 + 4, 3))
    for i in range(n_res):
        for j in range(4):
            k = i * 4 + j
            base[k] = [i * 1.5 + j * 0.3, 8.0 + 2.0 * np.sin(t + i * 0.7),
                       8.0 + 1.5 * np.cos(0.6 * t + i)]
    if transitioning:          # distinct conformational state
        base[: n_res * 2] += np.array([1.2, -0.8, 0.5])
    noise = np.random.default_rng(int(t * 10) % 5).normal(0, 0.05, base.shape)
    base += noise
    # ligand near the protein, small breathing motion
    base[n_res * 4:] = [[12.0 + 0.2 * np.sin(0.5 * t), 10.0, 10.0 + d * 0.9]
                        for d in range(4)]
    return base


@pytest.fixture
def synth_traj(tmp_path):
    n_frames, dt = 120, 10.0  # 1.2 ns
    u = build_universe()
    n_atoms = u.atoms.n_atoms
    gro = tmp_path / "structure.gro"
    xtc = tmp_path / "trajectory.xtc"
    with mda.Writer(str(gro), n_atoms=n_atoms) as W:
        ts = u.trajectory.ts
        ts.positions = frame_positions(0.0, False)
        ts.dimensions = np.array([40.0, 40.0, 40.0, 90.0, 90.0, 90.0])
        ts.time = 0.0
        ts.dt = dt
        W.write(u)
    with mda.Writer(str(xtc), n_atoms=n_atoms, multiframe=True) as W:
        for f in range(n_frames):
            ts = u.trajectory.ts
            ts.positions = frame_positions(f * dt * 0.01,
                                           1 <= f % 9 < 4)
            ts.dimensions = np.array([40.0, 40.0, 40.0, 90, 90, 90],
                                     dtype=float)
            ts.time = f * dt
            ts.dt = dt
            W.write(u)
    return str(xtc), str(gro)


INTEGRATION_OVERRIDES = {
    "system": {"output_dir": None, "seed": 7, "plots_enabled": True},
    "preprocess": {"selection": "protein and backbone", "stride": 2,
                   "start_time": 0, "end_time": None, "skip_equil_frames": 2,
                   "wrap_molecules": True, "center_protein": True,
                   "align": True, "reference": "first"},
    "features": {"chunk_size": 64, "use_float32": True,
                 "max_frames_for_full_matrix": 3000},
    "clustering": {"k_min": 2, "k_max": 4,
                   "nani_init_types": ["strat_all", "k-means++"],
                   "divine_splits": ["weighted_MSD"],
                   "divine_anchors": ["nani"],
                   "equal_seed_methods": ["greedy"],
                   "equal_threshold_quantiles": [0.25, 0.5, 0.8],
                   "mdbirch_threshold_quantiles": [0.25, 0.5, 0.8],
                   "helm_enabled": True, "helm_initial_k": 6,
                   "baselines_enabled": True,
                   "agglomerative_linkages": ["average", "ward"],
                   "dbscan_eps_quantiles": [0.1, 0.3]},
    "benchmark": {"enabled": True},
    "validation": {"bootstrap_enabled": True, "bootstrap_n": 3,
                   "bootstrap_frac": 0.8, "bootstrap_top_solutions": 4},
    "representatives": {"alternative_solutions": 1,
                        "nearest_to_centroid": True},
    "scoring": {"weights": {"representativeness": 0.25, "population": 0.15,
                            "stability": 0.20, "quality": 0.15,
                            "interaction": 0.15, "conformational": 0.10},
                "sensitivity_draws": 30},
}


def test_full_pipeline_end_to_end(synth_traj, tmp_path, capsys):
    xtc, gro = synth_traj
    out_dir = tmp_path / "MDANCE_Analysis_int"
    cfg = copy.deepcopy(__import__("mdance_pipeline.config", fromlist=["DEFAULT_CONFIG"]).DEFAULT_CONFIG)
    from mdance_pipeline.config import deep_merge
    overrides = copy.deepcopy(INTEGRATION_OVERRIDES)
    overrides["system"]["output_dir"] = str(out_dir)
    cfg = deep_merge(cfg, overrides)
    cfg["preprocess"]["xtc"] = xtc
    cfg["preprocess"]["gro"] = gro
    cfg["preprocess"]["tpr"] = None

    from mdance_pipeline.pipeline import Pipeline
    pipe = Pipeline(cfg)
    result = pipe.run()
    assert result["final"] is not None

    required = [
        "config/config.yaml",
        "preprocessing/processed.xtc",
        "preprocessing/processed.gro",
        "preprocessing/preprocessing_report.txt",
        "features/feature_data.npy",
        "features/distance_matrix.npy",
        "clustering/optimal_clustering/labels.npy",
        "clustering/optimal_clustering/model_card.txt",
        "representatives/global_medoid.pdb",
        "representatives/BEST_REPRESENTATIVE_FRAME.pdb",
        "representatives/candidate_frame_scores.csv",
        "representatives/cluster_representatives.csv",
        "benchmarking/mdance_vs_gromacs.csv",
        "benchmarking/benchmarking_report.txt",
        "scoring/scoring_metrics.csv",
        "scoring/scoring_weights.yaml",
        "scoring/sensitivity_analysis.csv",
        "plots/cluster_population.png",
        "plots/silhouette_analysis.png",
        "plots/rmsd_heatmap.png",
        "plots/pca_clustering.png",
        "plots/clustering_comparison.png",
        "plots/representative_ranking.png",
        "reports/complete_analysis_report.txt",
        "reports/clustering_summary.csv",
        "reports/clustering_summary.json",
        "reports/FINAL_FRAME_SELECTION_REPORT.txt",
        "logs/analysis.log",
    ]
    for rel in required:
        p = os.path.join(str(out_dir), rel)
        assert os.path.isfile(p), f"missing output: {rel}"
        if not rel.endswith((".npz",)):
            assert os.path.getsize(p) > 0, f"empty output: {rel}"

    # banner printed to stdout
    captured = capsys.readouterr().out
    assert "FINAL BEST REPRESENTATIVE FRAME" in captured
    assert "BEST_REPRESENTATIVE_FRAME.pdb" in captured

    # final frame is an actual analysis frame with score within [0, 100]
    with open(os.path.join(str(out_dir), "reports",
                           "FINAL_FRAME_SELECTION_REPORT.txt")) as fh:
        report = fh.read()
    assert "Selected Frame ID" in report
    import re
    score_match = re.search(r"Final Composite Score\s*:\s*([\d.]+) / 100",
                            report)
    assert score_match and 0 <= float(score_match.group(1)) <= 100

    # PDB contains real coordinates of one frame
    u_pdb = mda.Universe(os.path.join(str(out_dir), "representatives",
                                      "BEST_REPRESENTATIVE_FRAME.pdb"))
    assert u_pdb.atoms.n_atoms > 0
    assert np.isfinite(u_pdb.atoms.positions).all()

    # candidate scores CSV is ranked and finite
    import csv as csvmod
    with open(os.path.join(str(out_dir), "representatives",
                           "candidate_frame_scores.csv")) as fh:
        rows = list(csvmod.DictReader(fh))
    assert rows, "candidate_frame_scores.csv is empty"
    scores = [float(r["final_score"]) for r in rows]
    assert all(0 <= s <= 100 for s in scores)
    assert scores == sorted(scores, reverse=True)
