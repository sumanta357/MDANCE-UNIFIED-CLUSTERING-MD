#!/usr/bin/env python3
"""Unified entry point for the MDANCE clustering / benchmarking pipeline.

Examples:
    python run_mdance.py \
        --xtc trajectory.xtc --tpr topology.tpr --gro structure.gro \
        --selection "protein and backbone" --output MDANCE_Analysis \
        --stride 10 --start-time 0 --end-time 100 \
        --benchmark-gromacs --select-best-frame

    python run_mdance.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdance_pipeline.config import apply_cli_overrides, load_config  # noqa: E402
from mdance_pipeline.pipeline import Pipeline  # noqa: E402
from mdance_pipeline.utils import get_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_mdance.py",
        description="MDANCE-based MD trajectory clustering, GROMACS "
                    "benchmarking and best-representative-frame selection.")
    ap.add_argument("--config", default=None,
                    help="YAML configuration file (CLI overrides win)")
    ap.add_argument("--xtc", default=None, help="trajectory (.xtc)")
    ap.add_argument("--tpr", default=None, help="GROMACS run input (.tpr, optional)")
    ap.add_argument("--gro", default=None, help="structure (.gro; required if no tpr)")
    ap.add_argument("--selection", default=None,
                    help="MDAnalysis atom selection (e.g. 'protein and backbone')")
    ap.add_argument("--output", default=None, help="output directory")
    ap.add_argument("--stride", type=int, default=None,
                    help="analyze every Nth frame")
    ap.add_argument("--start-time", type=float, default=None,
                    help="start time (ps)")
    ap.add_argument("--end-time", type=float, default=None, help="end time (ps)")
    ap.add_argument("--skip-equil-frames", type=int, default=None,
                    help="skip N equilibration frames at the start")
    ap.add_argument("--benchmark-gromacs", dest="benchmark_gromacs",
                    action="store_true", default=None,
                    help="run the GROMACS gromos benchmark (default yes)")
    ap.add_argument("--no-benchmark-gromacs", dest="benchmark_gromacs",
                    action="store_false", help="skip the benchmark")
    ap.add_argument("--select-best-frame", dest="select_best_frame",
                    action="store_true", default=None,
                    help="run the final frame-selection phase (default yes)")
    ap.add_argument("--no-select-best-frame", dest="select_best_frame",
                    action="store_false", help="skip frame selection")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    logger = get_logger(os.path.join(cfg["system"]["output_dir"], "logs",
                                     "analysis.log"))
    pipe = Pipeline(cfg)
    try:
        result = pipe.run()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1
    if result["final"] is not None:
        logger.info("Pipeline finished; final frame analysis index %s "
                    "(score %.2f).", result["final"]["analysis_frame"],
                    result["final"]["final_score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
