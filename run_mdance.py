#!/usr/bin/env python3
"""Unified entry point for MDANCE trajectory clustering and frame selection.

Everything (automatic Elbow-based K selection, per-method representative
frames, GROMACS benchmarking, composite frame scoring) is orchestrated by
`mdance_pipeline.pipeline.Pipeline`; this script only parses arguments.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdance_pipeline.config import (apply_cli_overrides,  # noqa: E402
                                    load_config)
from mdance_pipeline.pipeline import Pipeline  # noqa: E402
from mdance_pipeline.utils import get_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_mdance.py",
        description=("MDANCE trajectory clustering, automatic Elbow K "
                     "selection, GROMACS benchmarking, per-method "
                     "representative frames, and best-frame selection."))
    ap.add_argument("--config", default=None,
                    help="YAML configuration file (CLI overrides win)")
    ap.add_argument("--xtc", default=None, help="trajectory (.xtc)")
    ap.add_argument("--tpr", default=None, help="GROMACS run input (.tpr)")
    ap.add_argument("--gro", default=None, help="structure (.gro)")
    ap.add_argument("--selection", default=None,
                    help="MDAnalysis atom selection")
    ap.add_argument("--output", default=None, help="output directory")
    ap.add_argument("--stride", type=int, default=None,
                    help="analyze every Nth frame")
    ap.add_argument("--start-time", type=float, default=None,
                    help="start time (ps)")
    ap.add_argument("--end-time", type=float, default=None,
                    help="end time (ps)")
    ap.add_argument("--skip-equil-frames", type=int, default=None,
                    help="skip equilibration frames")
    ap.add_argument("--k-min", type=int, default=None,
                    help="lowest K for the Elbow scan (default 2)")
    ap.add_argument("--k-max", type=int, default=None,
                    help="highest K for the Elbow scan (default 15, max 20)")
    ap.add_argument("--auto-k", dest="auto_k", action="store_true",
                    default=None, help="enable Elbow-based K selection")
    ap.add_argument("--no-auto-k", dest="auto_k", action="store_false",
                    help="disable auto-K (sweep clustering.k_min..k_max)")
    ap.add_argument("--benchmark-gromacs", dest="benchmark_gromacs",
                    action="store_true", default=None)
    ap.add_argument("--no-benchmark-gromacs", dest="benchmark_gromacs",
                    action="store_false")
    ap.add_argument("--select-best-frame", dest="select_best_frame",
                    action="store_true", default=None)
    ap.add_argument("--no-select-best-frame", dest="select_best_frame",
                    action="store_false")
    ap.add_argument("--full-output", dest="full_output",
                    action="store_true", default=None,
                    help="also export every candidate/cluster PDB under "
                         "04_ADVANCED_ANALYSIS (default: final results only)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = apply_cli_overrides(load_config(args.config), args)
    if args.k_max is not None:
        cfg["k_selection"]["k_max"] = int(args.k_max)
        cfg["clustering"]["k_max"] = int(args.k_max)
    if args.k_min is not None:
        cfg["k_selection"]["k_min"] = int(args.k_min)
        cfg["clustering"]["k_min"] = int(args.k_min)
    if args.auto_k is not None:
        cfg["k_selection"]["enabled"] = bool(args.auto_k)
    logger = get_logger(os.path.join(cfg["system"]["output_dir"], "logs",
                                     "analysis.log"))
    try:
        result = Pipeline(cfg).run()
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        logger.exception("Pipeline failed: %s", exc)
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    final = result.get("final")
    if final is not None:
        logger.info("Pipeline finished; final analysis frame %s (score %.2f).",
                    final["analysis_frame"], final["final_score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
