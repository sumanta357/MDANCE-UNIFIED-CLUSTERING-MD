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
        "README.txt",
        "01_FINAL_RESULTS/BEST_POSE.pdb",
        "01_FINAL_RESULTS/BEST_METHOD.pdb",
        "01_FINAL_RESULTS/FINAL_SUMMARY.csv",
        "01_FINAL_RESULTS/FINAL_REPORT.txt",
        "02_METHOD_COMPARISON/METHOD_RANKING.csv",
        "02_METHOD_COMPARISON/POSE_RANKING.csv",
        "02_METHOD_COMPARISON/CLUSTER_POPULATION.csv",
        "02_METHOD_COMPARISON/GROMACS_COMPARISON.csv",
        "03_PLOTS/K_SELECTION.png",
        "03_PLOTS/CLUSTER_COMPARISON.png",
        "03_PLOTS/POSE_RANKING.png",
        "03_PLOTS/GROMACS_COMPARISON.png",
        "04_ADVANCED_ANALYSIS/SENSITIVITY_ANALYSIS/sensitivity_analysis.csv",
        "04_ADVANCED_ANALYSIS/SENSITIVITY_ANALYSIS/scoring_sensitivity.png",
        "04_ADVANCED_ANALYSIS/DIAGNOSTIC_PLOTS/cluster_population.png",
        "04_ADVANCED_ANALYSIS/DIAGNOSTIC_PLOTS/silhouette_analysis.png",
        "04_ADVANCED_ANALYSIS/DIAGNOSTIC_PLOTS/rmsd_heatmap.png",
        "04_ADVANCED_ANALYSIS/DIAGNOSTIC_PLOTS/pca_clustering.png",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/config.yaml",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/features/feature_data.npy",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/features/distance_matrix.npy",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/clustering/optimal_clustering/labels.npy",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/clustering/optimal_clustering/model_card.txt",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/k_selection/k_selection.csv",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/k_selection/k_selection.json",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/k_selection/k_selection_elbow.png",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/method_coverage.json",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/METHOD_COVERAGE.txt",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/method_recommendation.txt",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/method_representatives.json",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/candidate_frame_scores.csv",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/pose_candidates_all.csv",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/solution_metrics.csv",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/solution_summary.csv",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/solution_summary.json",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/benchmarking_report.txt",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/scoring_metrics.csv",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/scoring_weights.yaml",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/complete_analysis_report.txt",
        "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/FINAL_FRAME_SELECTION_REPORT.txt",
        "05_PREPROCESSING/PROCESSED_TRAJECTORY/processed.xtc",
        "05_PREPROCESSING/REFERENCE_STRUCTURE/reference.gro",
        "05_PREPROCESSING/preprocessing_report.txt",
        "logs/analysis.log",
    ]
    for rel in required:
        p = os.path.join(str(out_dir), rel)
        assert os.path.isfile(p), f"missing output: {rel}"
        if not rel.endswith((".npz",)):
            assert os.path.getsize(p) > 0, f"empty output: {rel}"

    # --- automatic K selection ------------------------------------------
    import json as jsonmod
    with open(os.path.join(str(out_dir), "04_ADVANCED_ANALYSIS",
                           "INTERMEDIATE_RESULTS", "k_selection",
                           "k_selection.json")) as fh:
        krep = jsonmod.load(fh)
    assert krep["k_min"] == 2
    assert 2 <= krep["selected_k"] <= krep["k_max"]
    assert krep["dispersion_definition"]
    assert krep["metrics"], "no K scan rows recorded"
    assert [r["k"] for r in krep["metrics"]] == list(
        range(krep["k_min"], krep["k_max"] + 1))
    # K-based methods all used the selected K
    with open(os.path.join(str(out_dir), "04_ADVANCED_ANALYSIS",
                           "INTERMEDIATE_RESULTS", "solution_summary.json")) as fh:
        sols = jsonmod.load(fh)
    k_based = [s for s in sols
               if s["name"].split("/")[0] in {"nani", "divine", "helm"}
               and not s.get("error")]
    assert k_based, "no K-based solutions were produced"
    for s in k_based:
        assert s["n_clusters"] == krep["selected_k"], (
            f"{s['name']} used k={s['n_clusters']}, expected "
            f"{krep['selected_k']}")

    # --- per-method representative poses --------------------------------
    reps_root = os.path.join(str(out_dir), "02_METHOD_COMPARISON",
                             "METHOD_REPRESENTATIVES")
    with open(os.path.join(str(out_dir), "04_ADVANCED_ANALYSIS",
                           "INTERMEDIATE_RESULTS",
                           "method_representatives.json")) as fh:
        msum = jsonmod.load(fh)
    assert msum["recommendation"] is not None
    assert msum["families"], "no per-family summary written"
    for fam in msum["families"].values():
        assert fam["directory"] is None  # details are opt-in (--full-output)
        for cl in fam["clusters"]:
            assert cl["medoid_original_frame"] >= 0
            assert cl["medoid_time_ps"] >= 0
    # default output has no candidate/per-cluster PDBs
    for adv in ("CANDIDATE_FRAMES", "CLUSTER_MEDOIDS"):
        d = os.path.join(str(out_dir), "04_ADVANCED_ANALYSIS", adv)
        assert os.path.isdir(d)
        assert not [f for f in os.listdir(d) if f.endswith(".pdb")], (
            f"{adv} should be empty without --full-output")

    # --- simplified pose comparison --------------------------------------
    pose_csv = os.path.join(str(out_dir), "02_METHOD_COMPARISON",
                            "POSE_RANKING.csv")
    with open(pose_csv) as fh:
        poses = list(csvmod.DictReader(fh))
    assert poses, "POSE_RANKING.csv is empty"
    assert len({r["Method"] for r in poses}) >= 3
    pose_scores = [float(r["Final_Composite_Score"]) for r in poses]
    assert pose_scores == sorted(pose_scores, reverse=True)
    assert [int(r["Rank"]) for r in poses] == list(range(1, len(poses) + 1))
    assert all(0 <= s <= 100 for s in pose_scores)
    for r in poses:
        assert r["Method"] and r["Solution"] and r["Original_Frame"]
        pdb = os.path.join(str(out_dir), r["PDB_File"])
        assert os.path.isfile(pdb), f"missing per-method PDB: {pdb}"
    reps_pdbs = [f for f in os.listdir(reps_root) if f.endswith(".pdb")]
    assert len(reps_pdbs) == len(poses), (
        f"METHOD_REPRESENTATIVES has {reps_pdbs}, expected {len(poses)}")

    with open(os.path.join(str(out_dir), "04_ADVANCED_ANALYSIS",
                           "INTERMEDIATE_RESULTS",
                           "method_recommendation.txt")) as fh:
        rec_txt = fh.read()
    assert "CROSS-METHOD POSE COMPARISON" in rec_txt
    assert "pose winner method" in rec_txt
    with open(os.path.join(str(out_dir), "04_ADVANCED_ANALYSIS",
                           "INTERMEDIATE_RESULTS",
                           "METHOD_AND_POSE_SUMMARY.txt")) as fh:
        summary_txt = fh.read()
    assert "Best method (validation)" in summary_txt
    assert "Best pose (all methods)" in summary_txt

    # --- consolidated METHOD_RANKING.csv --------------------------------
    mr_path = os.path.join(str(out_dir), "02_METHOD_COMPARISON",
                           "METHOD_RANKING.csv")
    with open(mr_path) as fh:
        mrows = list(csvmod.DictReader(fh))
    assert mrows
    for col in ("Rank", "Method", "Status", "N_Clusters",
                "Cluster_Population_Frames", "Silhouette",
                "Davies_Bouldin", "Calinski_Harabasz", "Bootstrap_ARI",
                "Overall_Clustering_Score"):
        assert col in mrows[0], f"METHOD_RANKING missing {col}"
    assert all(r["Status"] in ("OK", "FAILED", "SKIPPED") for r in mrows)
    ok_ranks = [int(r["Rank"]) for r in mrows if r["Status"] == "OK"]
    assert ok_ranks == list(range(1, len(ok_ranks) + 1))

    # --- per-method coverage audit ---------------------------------------
    cov_path = os.path.join(str(out_dir), "04_ADVANCED_ANALYSIS",
                            "INTERMEDIATE_RESULTS", "method_coverage.json")
    with open(cov_path) as fh:
        cov = json.load(fh)
    assert set(cov) >= {"nani", "divine", "helm", "equal", "mdbirch",
                        "agglom", "dbscan", "gromos"}
    for fam, info in cov.items():
        assert info["status"] in ("usable", "degenerate", "not_run",
                                  "deferred")
        if info["status"] in ("not_run", "degenerate"):
            assert info["reason"], f"{fam} has no reason for being missing"
    cov_txt = os.path.join(str(out_dir), "04_ADVANCED_ANALYSIS",
                           "INTERMEDIATE_RESULTS", "METHOD_COVERAGE.txt")
    with open(cov_txt) as fh:
        cov_body = fh.read()
    for label in ("NANI", "DIVINE", "HELM", "eQual", "mdBIRCH",
                  "Agglomerative", "DBSCAN", "GROMOS"):
        assert label in cov_body, f"{label} missing from coverage report"
    performed = {f for f in cov if cov[f]["status"] == "usable"}
    assert len(performed) >= 6, f"only {sorted(performed)} produced results"

    # --- final report answers the user's questions -----------------------
    with open(os.path.join(str(out_dir), "01_FINAL_RESULTS",
                           "FINAL_REPORT.txt")) as fh:
        final_txt = fh.read()
    for phrase in ("Total trajectory frames", "Frames actually analysed",
                   "Elbow K", "Selected K used by pipeline",
                   "Why this K was selected", "CLUSTERS PRODUCED BY EACH METHOD",
                   "BEST CLUSTERING METHOD", "BEST REPRESENTATIVE POSE",
                   "Original trajectory frame",
                   "METHODS THAT FAILED OR WERE SKIPPED",
                   "GROMACS COMPARISON STATUS", "Native gmx cluster"):
        assert phrase in final_txt, f"FINAL_REPORT missing: {phrase}"
    assert f"Original trajectory frame : {poses[0]['Original_Frame']}" in final_txt
    assert f"Elbow K (Kneedle/geometric) : {krep['elbow_k']}" in final_txt
    assert (f"Selected K used by pipeline : {krep['selected_k']}"
            in final_txt)
    with open(os.path.join(str(out_dir), "01_FINAL_RESULTS",
                           "FINAL_SUMMARY.csv")) as fh:
        fs = {r["Item"]: r["Value"] for r in csvmod.DictReader(fh)}
    assert fs["Total trajectory frames"] == "120"
    assert fs["Frames analysed"] == str(len(pipe.info["frame_indices"]))
    assert fs["Elbow K"] == str(krep["elbow_k"])
    assert fs["Selected K"] == str(krep["selected_k"])
    assert fs["Best pose original frame"] == poses[0]["Original_Frame"]

    # --- benchmark CSV schema and statistical comparison -----------------
    from mdance_pipeline.gromacs_bench import BENCH_HEADER
    bench_csv = os.path.join(str(out_dir), "02_METHOD_COMPARISON",
                             "GROMACS_COMPARISON.csv")
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
    with open(os.path.join(str(out_dir), "04_ADVANCED_ANALYSIS",
                           "INTERMEDIATE_RESULTS",
                           "benchmarking_report.txt")) as fh:
        btxt = fh.read()
    assert "Statistical comparison summary" in btxt
    assert "not a significance test" in btxt

    # --- GROMOS/baseline label files are stored under advanced -----------
    gdir = os.path.join(str(out_dir), "04_ADVANCED_ANALYSIS",
                        "INTERMEDIATE_RESULTS", "clustering", "labels",
                        "gromos")
    bdir = os.path.join(str(out_dir), "04_ADVANCED_ANALYSIS",
                        "INTERMEDIATE_RESULTS", "clustering", "labels",
                        "baselines")
    gfiles = [f for f in os.listdir(gdir) if f.endswith(".npy")]
    bfiles = [f for f in os.listdir(bdir) if f.endswith(".npy")]
    assert any(f.startswith("gromos_") for f in gfiles), (
        f"no GROMOS partitions in gromos labels: {gfiles}")
    assert bfiles, "no baseline partitions in baseline labels"
    assert not any(f.startswith(("agglom", "dbscan")) for f in gfiles), (
        f"baseline labels leaked into gromos labels: {gfiles}")

    # --- best pose really is the highest-scoring, correctly mapped frame --
    with open(os.path.join(str(out_dir), "04_ADVANCED_ANALYSIS",
                           "INTERMEDIATE_RESULTS",
                           "candidate_frame_scores.csv")) as fh:
        rows = list(csvmod.DictReader(fh))
    assert rows, "candidate_frame_scores.csv is empty"
    scores = [float(r["final_score"]) for r in rows]
    assert all(0 <= s <= 100 for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert rows[0]["original_frame"] == poses[0]["Original_Frame"]
    # preprocessing maps analysis K -> original frame correctly for this run
    for r in rows:
        assert int(r["original_frame"]) == 2 + 2 * int(r["analysis_frame"])
    # BEST_POSE.pdb coordinates equal that processed analysis frame
    gro = os.path.join(str(out_dir), "05_PREPROCESSING",
                       "REFERENCE_STRUCTURE", "reference.gro")
    xtc = os.path.join(str(out_dir), "05_PREPROCESSING",
                       "PROCESSED_TRAJECTORY", "processed.xtc")
    u_proc = mda.Universe(gro, xtc)
    u_proc.trajectory[int(rows[0]["analysis_frame"])]
    ref_pos = u_proc.atoms.positions.copy()
    u_pdb = mda.Universe(os.path.join(str(out_dir), "01_FINAL_RESULTS",
                                      "BEST_POSE.pdb"))
    assert u_pdb.atoms.n_atoms == len(ref_pos)
    assert np.allclose(u_pdb.atoms.positions, ref_pos, atol=1e-3)

    # detailed banner printed to stdout still works
    captured = capsys.readouterr().out
    assert "FINAL BEST REPRESENTATIVE FRAME" in captured
    assert "01_FINAL_RESULTS/BEST_POSE.pdb" in captured
    assert scores == sorted(scores, reverse=True)
