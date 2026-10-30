"""YAML configuration loading, validation, and CLI overrides (pipeline-added)."""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "system": {
        "output_dir": "MDANCE_Analysis",
        "seed": 42,
        "dpi": 150,
        "plots_enabled": True,
        "log_level": "INFO",
    },
    "preprocess": {
        "tpr": None,             # optional
        "gro": None,             # required (structure)
        "xtc": None,             # required (trajectory)
        "selection": "protein and backbone",
        "reference": "first",    # 'first' | int frame index | null (no re-align)
        "stride": 1,
        "start_time": 0.0,       # ps
        "end_time": None,        # ps; None = to the end
        "skip_equil_frames": 0,  # original frames removed from the START
        "wrap_molecules": True,
        "center_protein": True,
        "align": True,
        "ligand_selection": "when_ligand",  # 'auto' pattern, see detect_ligand()
    },
    "features": {
        "chunk_size": 512,           # frames per chunk for the MSD matrix
        "use_float32": True,         # halve memory for large trajectories
        "max_frames_for_full_matrix": 3000,
        # Above this n, distance-matrix-based methods are skipped (memory).
    },
    "clustering": {
        "k_min": 2,
        "k_max": 8,
        "nani_init_types": ["strat_all", "comp_sim", "k-means++"],
        "nani_percentage": 10,
        "divine_splits": ["weighted_MSD"],
        "divine_anchors": ["nani"],
        "divine_refine": True,
        "equal_seed_methods": ["greedy", "medoid"],
        "equal_threshold_quantiles": [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95],
        "equal_threshold_palettes": "auto",  # auto: quantiles of pairwise MSD
        "mdbirch_threshold_quantiles": [0.10, 0.25, 0.40, 0.55, 0.70, 0.85],
        "mdbirch_branching_factor": 50,
        "helm_enabled": True,
        "helm_initial_k": 10,        # HELM refines this initial coarse partition
        "helm_merge_schema": "intra",  # {'intra','inter','half'}
        "helm_link": None,
        "helm_max_frames": 2000,
        "baselines_enabled": True,   # sklearn agglomerative / DBSCAN / gromos
        "agglomerative_linkages": ["average", "complete", "ward_on_coords"],
        "dbscan_min_samples": 5,
        "dbscan_eps_quantiles": [0.05, 0.15, 0.30, 0.50],
    },
    "benchmark": {
        "enabled": True,
        "gromos_cutoff_quantiles": [0.05, 0.10, 0.20, 0.35, 0.50, 0.70],
        "native_gmx": "auto",        # 'auto' | True | False
        "native_gmx_fitting_selection": "Backbone",
    },
    "validation": {
        "weights": {
            "silhouette": 0.22,
            "davies_bouldin": 0.18,
            "calinski_harabasz": 0.14,
            "stability": 0.16,
            "population_balance": 0.10,
            "compactness": 0.08,
            "separation": 0.07,
            "diversity": 0.05,
        },
        "bootstrap_enabled": True,
        "bootstrap_n": 6,
        "bootstrap_frac": 0.8,
        "bootstrap_top_solutions": 5,
        "bootstrap_seed": 1234,
    },
    "representatives": {
        "alternative_solutions": 2,  # extra candidate sources (top-2 runners-up)
        "nearest_to_centroid": True,
    },
    "scoring": {
        "weights": {
            "representativeness": 0.25,
            "population": 0.15,
            "stability": 0.20,
            "quality": 0.15,
            "interaction": 0.15,     # 0.0 automatically when no ligand is present
            "conformational": 0.10,
        },
        "singleton_cluster_score": 60.0,  # score for a 1-frame cluster (no spread)
        "temporal_window": 5,
        "sensitivity_draws": 120,
        "sensitivity_dirichlet_strength": 8.0,
    },
    "plots": {
        "tsne_enabled": False,       # t-SNE is slow; UMAP would add a heavy dep
        "heatmap_max_frames": 300,   # subsample the distance heatmap if bigger
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path) as fh:
            user = yaml.safe_load(fh) or {}
        cfg = deep_merge(cfg, user)
    return cfg


def apply_cli_overrides(cfg: dict, args) -> dict:
    """Apply command-line overrides (None values are ignored)."""
    p = dict(cfg["preprocess"])
    for key, argv_key in [
        ("gro", "gro"), ("xtc", "xtc"), ("tpr", "tpr"),
        ("selection", "selection"), ("stride", "stride"),
        ("start_time", "start_time"), ("end_time", "end_time"),
        ("skip_equil_frames", "skip_equil_frames"),
    ]:
        val = getattr(args, argv_key, None)
        if val is not None:
            p[key] = val
    if args.output:
        cfg["system"]["output_dir"] = args.output
    benchmark = dict(cfg["benchmark"])
    if args.benchmark_gromacs is not None:
        benchmark["enabled"] = bool(args.benchmark_gromacs)
    if args.select_best_frame is not None:
        cfg["select_best_frame"] = bool(args.select_best_frame)
    else:
        cfg.setdefault("select_best_frame", True)
    cfg["preprocess"] = p
    cfg["benchmark"] = benchmark
    return cfg


def validate_config(cfg: dict) -> list[str]:
    """Return a list of configuration errors (empty list = OK)."""
    errors: list[str] = []
    p = cfg["preprocess"]
    if not p.get("xtc"):
        errors.append("preprocess.xtc is required (--xtc or config)")
    if not p.get("gro") and not p.get("tpr"):
        errors.append("one of preprocess.tpr / preprocess.gro is required")
    if p.get("stride", 1) < 1:
        errors.append("preprocess.stride must be >= 1")
    if p.get("skip_equil_frames", 0) < 0:
        errors.append("preprocess.skip_equil_frames must be >= 0")
    if p.get("start_time", 0) < 0:
        errors.append("preprocess.start_time must be >= 0")
    if p.get("end_time") is not None and p.get("end_time") <= p.get("start_time", 0):
        errors.append("preprocess.end_time must be greater than start_time")
    c = cfg["clustering"]
    if c.get("k_min", 2) < 1:
        errors.append("clustering.k_min must be >= 1")
    if c.get("k_max", 8) < c.get("k_min", 2):
        errors.append("clustering.k_max must be >= k_min")
    v = cfg["validation"]
    if v.get("weights") and not (0 < sum(v["weights"].values()) <= 2.0):
        errors.append("validation.weights should sum to ~1.0")
    s = cfg["scoring"]
    if s.get("weights") and not (0 < sum(s["weights"].values()) <= 2.0):
        errors.append("scoring.weights should sum to ~1.0")
    for key in ("gro", "xtc", "tpr"):
        val = p.get(key)
        if val and not os.path.isfile(val):
            errors.append(f"preprocess.{key}: file not found: {val}")
    return errors
