#!/usr/bin/env python3
"""Unified entry point for MDANCE trajectory clustering and frame selection."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdance_pipeline.config import apply_cli_overrides, load_config  # noqa: E402
from mdance_pipeline.pipeline import Pipeline  # noqa: E402
from mdance_pipeline.utils import get_logger  # noqa: E402
from mdance_pipeline.k_selection import select_k  # noqa: E402
from mdance_pipeline.method_representatives import export_all_methods  # noqa: E402
from mdance_pipeline import clustering  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_mdance.py",
        description="MDANCE trajectory clustering, benchmarking, and representative frame selection.")
    ap.add_argument("--config", default=None, help="YAML configuration file (CLI overrides win)")
    ap.add_argument("--xtc", default=None, help="trajectory (.xtc)")
    ap.add_argument("--tpr", default=None, help="GROMACS run input (.tpr, optional)")
    ap.add_argument("--gro", default=None, help="structure (.gro; required if no tpr)")
    ap.add_argument("--selection", default=None, help="MDAnalysis atom selection")
    ap.add_argument("--output", default=None, help="output directory")
    ap.add_argument("--stride", type=int, default=None, help="analyze every Nth frame")
    ap.add_argument("--start-time", type=float, default=None, help="start time (ps)")
    ap.add_argument("--end-time", type=float, default=None, help="end time (ps)")
    ap.add_argument("--skip-equil-frames", type=int, default=None, help="skip equilibration frames")
    ap.add_argument("--benchmark-gromacs", dest="benchmark_gromacs", action="store_true", default=None)
    ap.add_argument("--no-benchmark-gromacs", dest="benchmark_gromacs", action="store_false")
    ap.add_argument("--select-best-frame", dest="select_best_frame", action="store_true", default=None)
    ap.add_argument("--no-select-best-frame", dest="select_best_frame", action="store_false")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = apply_cli_overrides(load_config(args.config), args)
    logger = get_logger(os.path.join(cfg["system"]["output_dir"], "logs", "analysis.log"))
    pipe = Pipeline(cfg)
    # Intercept the normal clustering phase once features/distances have been
    # prepared, run K selection on those exact arrays, then use that K only for
    # K-based algorithms. Threshold/density algorithms remain unchanged.
    original_run_all = clustering.run_all
n
    def run_with_selected_k(X, D, n_atoms, run_cfg, run_logger):
        k, report = select_k(X, D, n_atoms, run_cfg,
                             os.path.join(run_cfg["system"]["output_dir"], "clustering", "k_selection"),
                             run_logger)
        run_cfg["clustering"]["k_min"] = k
        run_cfg["clustering"]["k_max"] = k
        run_cfg["clustering"]["selected_k"] = k
        return original_run_all(X, D, n_atoms, run_cfg, run_logger)

    clustering.run_all = run_with_selected_k
    try:
        result = pipe.run()
        export_all_methods(pipe)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1
    finally:
        clustering.run_all = original_run_all
    if result["final"] is not None:
        logger.info("Pipeline finished; final analysis frame %s (score %.2f).",
                    result["final"]["analysis_frame"], result["final"]["final_score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
