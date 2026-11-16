"""Integration test: full pipeline on a synthetic GROMACS-like trajectory.

Builds a small protein-like topology (named so that `protein and backbone`
selections work) plus a ligand, writes GRO + XTC with MDAnalysis, and runs the
complete run_mdance.py CLI flow in a temporary directory.
"""
import copy
import csv as csvmod
import json
import os
import re
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
    "k_selection": {"enabled": True, "k_min": 2, "k_max": 5,
                     "stability_repeats": 2},
    "clustering": {"k_min": 2, "k_max": 5,
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
        "clustering/k_selection/k_selection.csv",
        "clustering/k_selection/k_selection.json",
        "clustering/k_selection/k_selection_elbow.png",
        "representatives/global_medoid.pdb",
        "representatives/method_recommendation.txt",
        "representatives/method_representatives.json",
        "representatives/RECOMMENDED_METHOD_FRAME.pdb",
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

    # --- automatic K selection ------------------------------------------
    import json as jsonmod
    with open(os.path.join(str(out_dir), "clustering", "k_selection",
                           "k_selection.json")) as fh:
        krep = jsonmod.load(fh)
    assert krep["k_min"] == 2
    assert 2 <= krep["selected_k"] <= krep["k_max"]
    assert krep["dispersion_definition"]
    assert krep["metrics"], "no K scan rows recorded"
    assert [r["k"] for r in krep["metrics"]] == list(
        range(krep["k_min"], krep["k_max"] + 1))
    # K-based methods all used the selected K
    with open(os.path.join(str(out_dir), "reports",
                           "clustering_summary.json")) as fh:
        sols = jsonmod.load(fh)
    k_based = [s for s in sols
               if s["name"].split("/")[0] in {"nani", "divine", "helm"}
               and not s.get("error")]
    assert k_based, "no K-based solutions were produced"
    for s in k_based:
        assert s["n_clusters"] == krep["selected_k"], (
            f"{s['name']} used k={s['n_clusters']}, expected "
            f"{krep['selected_k']}")

    # --- per-method representative frames -------------------------------
    reps_root = os.path.join(str(out_dir), "representatives",
                             "representative_frames")
    families = [d for d in os.listdir(reps_root)
                if os.path.isdir(os.path.join(reps_root, d))]
    assert len(families) >= 3, f"too few method folders: {families}"
    for fam in families:
        fam_dir = os.path.join(reps_root, fam)
        pdbs = [f for f in os.listdir(fam_dir) if f.endswith(".pdb")]
        assert pdbs, f"no PDB representatives in {fam}"
        assert os.path.isfile(os.path.join(fam_dir, "representatives.csv"))
        assert any("medoid" in p for p in pdbs), f"no medoid PDB in {fam}"
    with open(os.path.join(str(out_dir), "representatives",
                           "method_representatives.json")) as fh:
        msum = jsonmod.load(fh)
    assert msum["recommendation"] is not None
    assert msum["families"], "no per-family summary written"
    for fam in msum["families"].values():
        for cl in fam["clusters"]:
            assert cl["medoid_original_frame"] >= 0
            assert cl["medoid_time_ps"] >= 0

    # --- cross-method pose comparison ------------------------------------
    pose_csv = os.path.join(str(out_dir), "representatives",
                            "pose_comparison.csv")
    assert os.path.isfile(pose_csv)
    with open(pose_csv) as fh:
        poses = list(csvmod.DictReader(fh))
    assert poses, "pose_comparison.csv is empty"
    pose_methods = {r["method"] for r in poses}
    assert len(pose_methods) >= 3, (
        f"poses from only {len(pose_methods)} method(s): {pose_methods}")
    pose_scores = [float(r["final_score"]) for r in poses]
    assert pose_scores == sorted(pose_scores, reverse=True)
    assert all(0 <= s <= 100 for s in pose_scores)
    # every pose row names the method it came from
    assert all(r["method"] and r["family"] for r in poses)
    for rel in ("representatives/POSE_COMPARISON_REPORT.txt",
                "reports/METHOD_AND_POSE_SUMMARY.txt"):
        p = os.path.join(str(out_dir), rel)
        assert os.path.isfile(p) and os.path.getsize(p) > 0
    with open(os.path.join(str(out_dir), "representatives",
                           "method_recommendation.txt")) as fh:
        rec_txt = fh.read()
    assert "CROSS-METHOD POSE COMPARISON" in rec_txt
    assert "pose winner method" in rec_txt
    with open(os.path.join(str(out_dir), "reports",
                           "METHOD_AND_POSE_SUMMARY.txt")) as fh:
        summary_txt = fh.read()
    assert "Best method (validation)" in summary_txt
    assert "Best pose (all methods)" in summary_txt

    # --- per-method coverage audit ---------------------------------------
    cov_path = os.path.join(str(out_dir), "clustering",
                            "method_coverage.json")
    assert os.path.isfile(cov_path)
    with open(cov_path) as fh:
        cov = json.load(fh)
    assert set(cov) >= {"nani", "divine", "helm", "equal", "mdbirch",
                        "agglom", "dbscan", "gromos"}
    for fam, info in cov.items():
        assert info["status"] in ("usable", "degenerate", "not_run",
                                  "deferred")
        if info["status"] in ("not_run", "degenerate"):
            assert info["reason"], f"{fam} has no reason for being missing"
    cov_txt = os.path.join(str(out_dir), "reports", "METHOD_COVERAGE.txt")
    assert os.path.isfile(cov_txt)
    with open(cov_txt) as fh:
        cov_body = fh.read()
    for label in ("NANI", "DIVINE", "HELM", "eQual", "mdBIRCH",
                  "Agglomerative", "DBSCAN", "GROMOS"):
        assert label in cov_body, f"{label} missing from coverage report"
    # every executed method must be usable on this dataset (no silent gaps)
    performed = {f for f in cov if cov[f]["status"] == "usable"}
    assert len(performed) >= 6, f"only {sorted(performed)} produced results"

    # --- final winner file names the method that produced it -------------
    best_txt = os.path.join(str(out_dir), "reports",
                            "FINAL_FRAME_SELECTION_REPORT.txt")
    with open(best_txt) as fh:
        banner = fh.read()
    named = re.search(
        r"representatives/(BEST_REPRESENTATIVE_FRAME_(\S+?)_cluster\d+_frame\d+\.pdb)",
        banner)
    assert named, "banner does not reference a method-named winner PDB"
    assert os.path.isfile(os.path.join(str(out_dir), "representatives",
                                       named.group(1)))
    assert re.search(r"BEST_REPRESENTATIVE_FRAME_(NANI|DIVINE|HELM|EQUAL|MDBIRCH|AGGLOMERATIVE|DBSCAN|GROMOS)",
                     named.group(1)), named.group(1)
    # cluster ids are 1-based everywhere user-facing
    banner_cid = re.search(r"Selected Cluster ID\s*:\s*(\d+)", banner)
    assert banner_cid, "banner has no Selected Cluster ID"
    assert int(banner_cid.group(1)) >= 1
    assert f"_cluster{banner_cid.group(1)}_frame" in named.group(1), (
        f"banner cluster {banner_cid.group(1)} disagrees with {named.group(1)}")

    # --- benchmark CSV schema and statistical comparison -----------------
    from mdance_pipeline.gromacs_bench import BENCH_HEADER
    bench_csv = os.path.join(str(out_dir), "benchmarking",
                             "mdance_vs_gromacs.csv")
    with open(bench_csv) as fh:
        breader = csvmod.reader(fh)
        bheader = next(breader)
        brows = list(breader)
    assert bheader == BENCH_HEADER, f"unexpected benchmark header: {bheader}"
    assert brows, "benchmark CSV has no rows"
    for row in brows:
        assert len(row) == len(bheader), f"ragged benchmark row: {row}"
    assert any(r[0].startswith("REFERENCE:") for r in brows)
    bidx = {h: i for i, h in enumerate(bheader)}
    grows = [r for r in brows if r[0].startswith("GROMACS")]
    assert grows, "no GROMOS comparison rows"
    for r in grows:
        assert int(r[bidx["n_clusters_gromos"]]) >= 2
        assert -1.0 <= float(r[bidx["adj_rand_index"]]) <= 1.0
        assert 0.0 <= float(r[bidx["population_L1"]]) <= 1.0
        assert 0.0 <= float(r[bidx["top5_population_divergence"]]) <= 1.0
    with open(os.path.join(str(out_dir), "benchmarking",
                           "benchmarking_report.txt")) as fh:
        btxt = fh.read()
    assert "Statistical comparison summary" in btxt
    assert "not a significance test" in btxt

    # --- GROMOS/baseline label files are stored under the right folder ---
    gdir = os.path.join(str(out_dir), "clustering", "gromos")
    bdir = os.path.join(str(out_dir), "clustering", "baselines")
    gfiles = [f for f in os.listdir(gdir) if f.endswith(".npy")]
    bfiles = [f for f in os.listdir(bdir) if f.endswith(".npy")]
    assert any(f.startswith("gromos_") for f in gfiles), (
        f"no GROMOS partitions in clustering/gromos: {gfiles}")
    assert bfiles, "no baseline partitions in clustering/baselines"
    assert not any(f.startswith(("agglom", "dbscan")) for f in gfiles), (
        f"baseline labels leaked into clustering/gromos: {gfiles}")

    # --- pose report points at a file that actually exists ---------------
    with open(os.path.join(str(out_dir), "representatives",
                           "POSE_COMPARISON_REPORT.txt")) as fh:
        ptxt = fh.read()
    pstruct = re.search(
        r"structure\s*:\s*representatives/(BEST_REPRESENTATIVE_FRAME_\S+\.pdb)",
        ptxt)
    assert pstruct, "pose report does not name the winner structure"
    assert os.path.isfile(os.path.join(str(out_dir), "representatives",
                                       pstruct.group(1))), (
        f"pose report points at missing file: {pstruct.group(1)}")

    # banner printed to stdout
    captured = capsys.readouterr().out
    assert "FINAL BEST REPRESENTATIVE FRAME" in captured
    assert "BEST_REPRESENTATIVE_FRAME.pdb" in captured

    # final frame is an actual analysis frame with score within [0, 100]
    with open(os.path.join(str(out_dir), "reports",
                           "FINAL_FRAME_SELECTION_REPORT.txt")) as fh:
        report = fh.read()
    assert "Selected Frame ID" in report
    score_match = re.search(r"Final Composite Score\s*:\s*([\d.]+) / 100",
                            report)
    assert score_match and 0 <= float(score_match.group(1)) <= 100

    # PDB contains real coordinates of one frame
    u_pdb = mda.Universe(os.path.join(str(out_dir), "representatives",
                                      "BEST_REPRESENTATIVE_FRAME.pdb"))
    assert u_pdb.atoms.n_atoms > 0
    assert np.isfinite(u_pdb.atoms.positions).all()

    # candidate scores CSV is ranked and finite
    with open(os.path.join(str(out_dir), "representatives",
                           "candidate_frame_scores.csv")) as fh:
        rows = list(csvmod.DictReader(fh))
    assert rows, "candidate_frame_scores.csv is empty"
    scores = [float(r["final_score"]) for r in rows]
    assert all(0 <= s <= 100 for s in scores)
    assert scores == sorted(scores, reverse=True)
