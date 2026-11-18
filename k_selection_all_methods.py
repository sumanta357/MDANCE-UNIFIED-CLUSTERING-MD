#!/usr/bin/env python3
"""Standalone K selection for every MDANCE clustering method.

This script does ONE job -- decide how many clusters (K) each method should
use -- without running the clustering sweep, validation ranking, representative
frame extraction, GROMACS benchmark or scoring phases of ``run_mdance.py``.

It calls the *same* functions the main pipeline uses
(``mdance_pipeline.preprocess``, ``mdance_pipeline.features`` and
``mdance_pipeline.k_selection``), so the K values agree with a full pipeline
run on identical inputs.

Methods
-------
K-based methods (each gets its own independent elbow scan by default):
    nani    KmeansNANI (MDANCE-native, metric='MSD')
    divine  DIVINE split/refine
    helm    HELM hierarchical merging
    agglom Agglomerative (average linkage)

One shared scan (``--shared``, on by default) clusters with MDANCE-native
KmeansNANI at every K in the range, which is what the pipeline uses when
``k_selection.per_method`` is false.

Methods that do NOT take K are listed in the summary as ``not_applicable``
because they are driven by a distance threshold / density instead:
    eQual (threshold), mdBIRCH (threshold), DBSCAN (eps), GROMOS/Daura (cutoff).

Usage
-----
    python k_selection_all_methods.py \
        --xtc md.xtc --tpr md.tpr \
        --selection "protein and backbone or resname LIG" \
        --stride 10 --k-min 2 --k-max 15 --out K_SELECTION_RESULTS

See ``--help`` and the generated ``K_SELECTION_RESULTS/README.txt``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mdance_pipeline import features as feat  # noqa: E402
from mdance_pipeline import k_selection  # noqa: E402
from mdance_pipeline import preprocess as pre  # noqa: E402
from mdance_pipeline.config import load_config  # noqa: E402
from mdance_pipeline.utils import get_logger, set_seed  # noqa: E402

K_METHODS = ("nani", "divine", "helm", "agglom")
METHOD_LABELS = {
    "nani": "NANI (KmeansNANI)",
    "divine": "DIVINE",
    "helm": "HELM",
    "agglom": "Agglomerative",
    "shared": "Shared scan (KmeansNANI)",
    "equal": "eQual",
    "mdbirch": "mdBIRCH",
    "dbscan": "DBSCAN",
    "gromos": "GROMOS/Daura",
}
NON_K_METHODS = ("equal", "mdbirch", "dbscan", "gromos")
MAX_K_CEILING = 20


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="k_selection_all_methods.py",
        description=("Standalone elbow/Kneedle K selection for every K-based "
                     "MDANCE clustering method, using the pipeline's exact "
                     "MDANCE-MSD dispersion."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--xtc", required=True, help="trajectory file (.xtc)")
    ap.add_argument("--tpr", default=None, help="GROMACS run input (.tpr)")
    ap.add_argument("--gro", default=None, help="structure file (.gro)")
    ap.add_argument("--selection", default="protein and backbone",
                    help="MDAnalysis atom selection used for the features")
    ap.add_argument("--stride", type=int, default=1,
                    help="analyse every Nth frame")
    ap.add_argument("--start-time", type=float, default=0.0,
                    help="start time in ps")
    ap.add_argument("--end-time", type=float, default=None,
                    help="end time in ps (default: last frame)")
    ap.add_argument("--skip-equil-frames", type=int, default=0,
                    help="drop this many frames from the start")
    ap.add_argument("--ligand-selection", default="when_ligand",
                    help="ligand selection for centring ('when_ligand' = auto)")
    ap.add_argument("--no-align", action="store_true",
                    help="skip reference alignment")
    ap.add_argument("--no-wrap", action="store_true",
                    help="skip PBC wrapping")
    ap.add_argument("--k-min", type=int, default=2,
                    help="lowest K in the scan")
    ap.add_argument("--k-max", type=int, default=15,
                    help=f"highest K in the scan (hard ceiling {MAX_K_CEILING})")
    ap.add_argument("--methods", nargs="+", default=list(K_METHODS),
                    choices=list(K_METHODS),
                    help="K-based methods to scan independently")
    ap.add_argument("--no-shared", dest="shared", action="store_false",
                    help="skip the shared KmeansNANI scan")
    ap.add_argument("--stability-repeats", type=int, default=4,
                    help="bootstrap ARI refits per K")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--out", default="K_SELECTION_RESULTS",
                    help="output directory")
    ap.add_argument("--config", default=None,
                    help="optional YAML config for preprocessing defaults")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap


def build_config(args) -> dict:
    cfg = load_config(args.config)
    pre_cfg = cfg["preprocess"]
    pre_cfg["xtc"] = args.xtc
    if args.tpr:
        pre_cfg["tpr"] = args.tpr
    if args.gro:
        pre_cfg["gro"] = args.gro
    pre_cfg["selection"] = args.selection
    pre_cfg["stride"] = args.stride
    pre_cfg["start_time"] = args.start_time
    pre_cfg["end_time"] = args.end_time
    pre_cfg["skip_equil_frames"] = args.skip_equil_frames
    pre_cfg["ligand_selection"] = args.ligand_selection
    pre_cfg["wrap_molecules"] = not args.no_wrap
    pre_cfg["align"] = not args.no_align

    ks = cfg["k_selection"]
    ks["enabled"] = True
    ks["k_min"] = max(2, int(args.k_min))
    ks["k_max"] = min(int(args.k_max), MAX_K_CEILING)
    ks["max_allowed_k"] = MAX_K_CEILING
    ks["stability_repeats"] = max(0, int(args.stability_repeats))
    ks["per_method"] = bool(args.methods)
    ks["per_method_families"] = list(args.methods)

    cfg["system"]["output_dir"] = args.out
    cfg["system"]["seed"] = int(args.seed)
    cfg["system"]["log_level"] = args.log_level
    return cfg


def check_inputs(cfg: dict) -> None:
    p = cfg["preprocess"]
    if not p.get("xtc"):
        raise ValueError("--xtc is required")
    if not p.get("tpr") and not p.get("gro"):
        raise ValueError("one of --tpr / --gro is required")
    for key in ("xtc", "tpr", "gro"):
        path = p.get(key)
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f"{key} file not found: {path}")
    if cfg["k_selection"]["k_max"] < cfg["k_selection"]["k_min"]:
        raise ValueError("--k-max must be >= --k-min")


def summary_rows(scan: dict, label: str, status: str = "ok") -> dict:
    rows = scan.get("metrics") or []
    selected = scan.get("selected_k")
    sel_row = next((r for r in rows if int(r["k"]) == int(selected)),
                   {}) if selected is not None else {}
    return {
        "method": label,
        "status": status,
        "selected_k": selected if selected is not None else "",
        "elbow_k": scan.get("elbow_k") if scan.get("elbow_k") is not None
        else "",
        "elbow_algorithm": scan.get("elbow_algorithm", ""),
        "elbow_ambiguous": scan.get("elbow_ambiguous", ""),
        "dispersion_at_selected": sel_row.get("dispersion", ""),
        "silhouette_at_selected": sel_row.get("silhouette", ""),
        "davies_bouldin_at_selected": sel_row.get("davies_bouldin", ""),
        "calinski_harabasz_at_selected": sel_row.get("calinski_harabasz", ""),
        "stability_at_selected": sel_row.get("stability", ""),
        "support_rank_at_selected": sel_row.get("support_rank", ""),
        "n_clusters_at_selected": sel_row.get("n_clusters", ""),
        "n_singletons_at_selected": sel_row.get("n_singletons", ""),
    }


def write_summary(out_dir: str, rows: list[dict]) -> None:
    fields = list(rows[0].keys())
    with open(os.path.join(out_dir, "K_SELECTION_SUMMARY.csv"), "w",
              newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(out_dir, "K_SELECTION_SUMMARY.json"), "w") as fh:
        json.dump(rows, fh, indent=2)


def write_readme(out_dir: str, cfg: dict) -> None:
    p = cfg["preprocess"]
    ks = cfg["k_selection"]
    text = f"""K SELECTION - INPUTS AND OUTPUTS
================================

INPUTS (used to build the MDANCE feature matrix)
-----------------------------------------------
  trajectory (.xtc) : {p.get('xtc')}
  topology          : tpr={p.get('tpr')}  gro={p.get('gro')}
  atom selection    : {p.get('selection')}
  stride            : every {p.get('stride')} frame(s)
  start/end time    : {p.get('start_time')} / {p.get('end_time')} ps
  skip equil. frames: {p.get('skip_equil_frames')}
  wrap / align      : {p.get('wrap_molecules')} / {p.get('align')}
  K scan range      : {ks.get('k_min')}..{ks.get('k_max')} (ceiling {ks.get('max_allowed_k')})
  seed              : {cfg['system'].get('seed')}

No precomputed input files are required beyond the trajectory and topology.

OUTPUTS
-------
K_SELECTION_SUMMARY.csv / .json
    One row per method: selected K, elbow K, elbow algorithm, ambiguity flag,
    dispersion, silhouette, Davies-Bouldin, Calinski-Harabasz, bootstrap ARI
    stability, and cluster/singleton counts at the selected K. Methods that do
    not take K are present with status "not_applicable".

shared/k_selection.csv
    Full K scan of the shared KmeansNANI clustering (one row per K).
shared/k_selection.json
    Same scan plus the selected/elbow K, dispersion definition and rule.
shared/k_selection_elbow.png
    Dispersion vs K with the elbow and selected K marked.

per_method/k_selection_per_method.csv
    Per-method selected K (one row per K-based method).
per_method/k_selection_per_method.json
    Full per-method scans, including every K row.
per_method/k_selection_elbow_<method>.png
    One elbow plot per method.

K_SELECTION_REPORT.txt
    Plain-language summary table.
logs/k_selection.log
    Full run log.

HOW TO READ THE RESULT
----------------------
* W(K) = sum over clusters of the mean within-cluster MDANCE MSD
  (d_MSD(i,j) = ||x_i-x_j||^2 / (2*N_atoms)); lower is tighter.
* The elbow is chosen with Kneedle when `kneed` is installed, otherwise with a
  normalized maximum-deviation detector. No knee -> elbow_ambiguous = True and
  the supporting metrics decide instead of forcing an arbitrary K.
* Selected K = best average ordinal rank over silhouette, distance-space
  Davies-Bouldin, distance-space Calinski-Harabasz and bootstrap-ARI
  stability, restricted to the elbow +/- 1 when the elbow is unambiguous.
* Use per-method K if the methods disagree strongly; otherwise use the shared
  scan K for NANI/DIVINE/HELM/Agglomerative.
* eQual, mdBIRCH, DBSCAN and GROMOS/Daura do not take K (threshold/density
  driven), so they are reported as not_applicable.
"""
    with open(os.path.join(out_dir, "README.txt"), "w") as fh:
        fh.write(text)


def write_report(out_dir: str, rows: list[dict]) -> str:
    lines = ["K SELECTION PER METHOD", "=" * 78,
             f"{'method':<28}{'K':>4}{'elbow':>7}{'silhouette':>12}"
             f"{'DB':>8}{'CH':>10}{'ARI':>7}  status"]
    for r in rows:
        def fmt(key, spec):
            val = r.get(key, "")
            return format(val, spec) if isinstance(val, (int, float)) else "-"
        lines.append(
            f"{r['method']:<28}"
            f"{str(r['selected_k']):>4}"
            f"{str(r['elbow_k']):>7}"
            f"{fmt('silhouette_at_selected', '.4f'):>12}"
            f"{fmt('davies_bouldin_at_selected', '.3f'):>8}"
            f"{fmt('calinski_harabasz_at_selected', '.1f'):>10}"
            f"{fmt('stability_at_selected', '.3f'):>7}  {r['status']}")
    lines += ["", "Threshold/density methods (eQual, mdBIRCH, DBSCAN, "
              "GROMOS/Daura) do not take K.", ""]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "K_SELECTION_REPORT.txt"), "w") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = build_config(args)
    out_dir = cfg["system"]["output_dir"]
    try:
        check_inputs(cfg)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(os.path.join(out_dir, "logs", "k_selection.log"),
                        level=args.log_level)
    set_seed(cfg["system"].get("seed", 42))

    logger.info("Step 1/4: preprocessing trajectory")
    try:
        info = pre.preprocess_trajectory(cfg, logger)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        logger.exception("Preprocessing failed: %s", exc)
        print(f"Preprocessing failed: {exc}", file=sys.stderr)
        return 1
    n_atoms = int(info["n_atoms_selected"])
    n_frames = len(info["frame_indices"])
    logger.info("Prepared %d analysis frames x %d selected atoms",
                n_frames, n_atoms)

    logger.info("Step 2/4: building MDANCE MSD matrix")
    dist = feat.build_distances(info["X"], n_atoms, cfg, logger)
    D = dist.get("D")
    if D is None:
        msg = ("No distance matrix: the number of analysis frames exceeds "
               "features.max_frames_for_full_matrix. Increase --stride so "
               "fewer frames are analysed, or raise that limit.")
        logger.error(msg)
        print(msg, file=sys.stderr)
        return 1

    ks_cfg = cfg["k_selection"]
    # run_per_method() computes stability refits with k_selection.n_atoms, so
    # keep it in sync with the actual system (the pipeline sets this too).
    ks_cfg["n_atoms"] = n_atoms

    rows: list[dict] = []
    logger.info("Step 3/4: K scans over K=%d..%d",
                ks_cfg["k_min"], ks_cfg["k_max"])
    try:
        if args.shared:
            shared_dir = os.path.join(out_dir, "shared")
            report = k_selection.run(info["X"], D, n_atoms, cfg, shared_dir,
                                     logger)
            rows.append(summary_rows(report,
                                     METHOD_LABELS["shared"]))
        if args.methods:
            per_dir = os.path.join(out_dir, "per_method")
            per = k_selection.run_per_method(info["X"], D, n_atoms, cfg,
                                             per_dir, logger)
            for fam, scan in per.get("per_method", {}).items():
                rows.append(summary_rows(scan, METHOD_LABELS.get(fam, fam)))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        logger.exception("K selection failed: %s", exc)
        print(f"K selection failed: {exc}", file=sys.stderr)
        return 1

    for fam in NON_K_METHODS:
        rows.append({"method": METHOD_LABELS[fam], "status": "not_applicable",
                     "selected_k": "", "elbow_k": "",
                     "elbow_algorithm": "", "elbow_ambiguous": "",
                     "dispersion_at_selected": "",
                     "silhouette_at_selected": "",
                     "davies_bouldin_at_selected": "",
                     "calinski_harabasz_at_selected": "",
                     "stability_at_selected": "",
                     "support_rank_at_selected": "",
                     "n_clusters_at_selected": "",
                     "n_singletons_at_selected": ""})

    logger.info("Step 4/4: writing summary")
    write_summary(out_dir, rows)
    write_readme(out_dir, cfg)
    text = write_report(out_dir, rows)
    print()
    print(text)
    logger.info("Done. Results in %s", os.path.abspath(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
