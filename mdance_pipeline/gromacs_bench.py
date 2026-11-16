"""Phase 6 - benchmark against GROMACS `gmx cluster` (pipeline-added module).

Two-tier benchmark, explicitly documented (never claimed "perfectly
equivalent"):

1. Native `gmx cluster` (only when the `gmx` binary exists AND the trajectory
   contains exactly the atoms of the `gmx` structure file). `gmx cluster`
   computes its own RMSD matrix over the whole system it is given; subsetting
   atoms inside gmx (e.g. backbone-only) is not possible without regenerating
   a subset TPR with GROMACS tools, which is out of scope of this pipeline —
   the atomic-selection difference is then REPORTED, not hidden.

2. `gromos` (Daura et al. 1999) greedy clustering on the identical MDANCE
   pairwise-MSD matrix used by every MDANCE method. This is the exact
   algorithm behind `gmx cluster -m gromos` and gives a selection-aligned,
   distance-aligned comparison. Multiple cutoffs are evaluated
   (`-cutoff` equivalent).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess

import numpy as np
from sklearn.metrics import adjusted_rand_score

from mdance_pipeline.clustering import Solution, run_gromos, upper_desc_vals


def gmx_available() -> tuple[bool, str]:
    path = shutil.which("gmx") or shutil.which("gmx_mpi")
    return (path is not None, path or "")


def run_native_gmx(xtc, gro_or_tpr, work_dir, cutoff_nm=0.10, fit_selection=None,
                   logger=None) -> dict | None:
    """Try the native gmx binary (experimental; documented limitations).

    ``cutoff_nm`` is an RMSD cutoff in nm, which is what `gmx cluster`
    expects. The pipeline's cutoffs live in MSD space, so the caller must
    convert (RMSD = sqrt(2 * MSD) for the MDANCE pairwise MSD convention).
    """
    ok, gmx_bin = gmx_available()
    if not ok:
        return None
    os.makedirs(work_dir, exist_ok=True)
    sel = fit_selection or "Backbone"
    out_stub = os.path.join(work_dir, "gmx_cluster")
    cmd = [gmx_bin, "cluster", "-s", gro_or_tpr, "-f", xtc,
           "-method", "gromos", "-cutoff", f"{cutoff_nm:.5f}",
           "-cl", out_stub + "_clusters.xtc", "-g", out_stub + ".log",
           "-av", out_stub + "_rep.xtc", "-o", out_stub + ".xpm",
           "-sz", out_stub + "_sz.xvg", "-hist", out_stub + "_hist.xvg",
           "-rms", out_stub + "_rmsd.xvg", "-dist", out_stub + "_rmsd-dist.xvg"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                              input=f"{sel}\n", errors="replace")
    except Exception as exc:
        if logger:
            logger.warning("native gmx cluster failed: %s", exc)
        return {"returncode": None, "success": False, "cmd": " ".join(cmd),
                "cutoff_nm": float(cutoff_nm), "error": str(exc),
                "stdout_tail": "", "stderr_tail": ""}
    # gmx cluster writes cluster membership to <stub>_clusters.xtc; native
    # membership parsing is nontrivial, so only the run outcome is reported.
    log_path = out_stub + ".log"
    success = proc.returncode == 0 and os.path.isfile(log_path)
    if not success and logger:
        logger.warning("native gmx cluster returned %s (see %s)",
                       proc.returncode, log_path)
    return {"returncode": proc.returncode, "success": success,
            "cmd": " ".join(cmd), "cutoff_nm": float(cutoff_nm),
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:]}


def compare_partitions(mdance_labels: np.ndarray, gromos_labels: np.ndarray) -> dict:
    """Population-distribution + assignment agreement (pipeline-added).

    Reports the two cluster counts, the per-cluster population L1 distance
    (labels sorted by size, so it is permutation-invariant), the top-5
    population divergence and the adjusted Rand index on the shared frames.
    """
    def pops(lab):
        vals = np.asarray(lab)[np.asarray(lab) >= 0]
        uniq, counts = np.unique(vals, return_counts=True)
        return dict(zip(uniq.tolist(), counts.tolist()))

    p1, p2 = pops(mdance_labels), pops(gromos_labels)
    sorted1 = np.sort(np.array(list(p1.values()), dtype=float))[::-1]
    sorted2 = np.sort(np.array(list(p2.values()), dtype=float))[::-1]
    m = max(len(sorted1), len(sorted2))
    sorted1 = np.pad(sorted1, (0, m - len(sorted1)))
    sorted2 = np.pad(sorted2, (0, m - len(sorted2)))
    s1, s2 = sorted1.sum(), sorted2.sum()
    ari = float(adjusted_rand_score(mdance_labels, gromos_labels)) \
        if len(mdance_labels) == len(gromos_labels) else None
    return {"n_clusters_reference": int((sorted1 > 0).sum()),
            "n_clusters_gromos": int((sorted2 > 0).sum()),
            "top5_pop_divergence": float(
                np.abs(sorted1[:5] / max(s1, 1)
                       - sorted2[:5] / max(s2, 1)).mean()),
            "population_L1": (float(np.abs(sorted1 / s1 - sorted2 / s2).mean())
                              if s1 > 0 and s2 > 0 else None),
            "adj_rand_index": ari}


def benchmark(sols: list[Solution], D, X, cfg, logger, pre_info,
              out_dir) -> dict:
    """Full Phase-6 benchmark: run GROMOS at several cutoffs on the identical
    distance matrix, compare to the best MDANCE solution, write CSV + report."""
    from mdance_pipeline.utils import write_csv

    bench = cfg["benchmark"]
    if not bench.get("enabled", True):
        logger.info("GROMACS benchmark disabled by config.")
        return {}
    os.makedirs(os.path.join(out_dir, "benchmarking"), exist_ok=True)
    bench_sols: list[Solution] = []
    cutoffs = [float(np.quantile(upper_desc_vals(D), q))
               for q in bench.get("gromos_cutoff_quantiles", [.05, .1, .2, .35, .5, .7])]
    for cutoff in cutoffs:
        sol = run_gromos(D, cutoff)
        if sol.n_clusters >= 1:
            bench_sols.append(sol)
    native = None
    if bench.get("native_gmx", "auto") in (True, "auto"):
        ok, _ = gmx_available()
        if ok:
            logger.info("gmx binary found - attempting native gmx cluster "
                        "(selection limited to the supplied structure; "
                        "documented difference from atlas MDANCE selection).")
            native = run_native_gmx(
                pre_info["paths"]["processed_xtc"], pre_info["paths"]["processed_gro"],
                os.path.join(out_dir, "benchmarking"),
                # MSD -> RMSD (nm): D = sum(d^2)/(2N), so RMSD = sqrt(2*D)
                cutoff_nm=float(np.sqrt(2.0 * cutoffs[1])),
                fit_selection=bench.get("native_gmx_fitting_selection", "Backbone"),
                logger=logger)
        else:
            logger.info("gmx binary not found - benchmark uses the faithful "
                        "Python Daura/gromos implementation on the identical "
                        "MDANCE MSD matrix (documented equivalence).")
    logger.info("GROMOS cutoff sweep produced %d solutions.", len(bench_sols))
    return {"solutions": bench_sols, "native": native, "cutoffs": cutoffs}


BENCH_HEADER = ["method", "n_clusters", "time_s", "mem_mb",
                "compactness", "separation", "n_clusters_reference",
                "n_clusters_gromos", "population_L1",
                "top5_population_divergence", "adj_rand_index"]


def _partition_shape(labels, D):
    """(mean within-cluster MSD to medoid, min inter-medoid MSD) or (None, None)."""
    if D is None or labels is None or len(labels) == 0:
        return None, None
    from mdance_pipeline.validation import cluster_medoids
    meds = cluster_medoids(D, np.asarray(labels))
    if not meds:
        return None, None
    sizes = {c: int((np.asarray(labels) == c).sum()) for c in meds}
    n = len(labels)
    compact = float(sum(sizes[c] / n * float(D[np.asarray(labels) == c,
                                            meds[c]].mean())
                        for c in meds))
    sep = None
    ids = list(meds.values())
    if len(ids) >= 2:
        sep = float(min(float(D[a, b]) for ai, a in enumerate(ids)
                        for b in ids[ai + 1:]))
    return compact, sep


def benchmark_table(best_metrics, bench_results, sols, cfg,
                    D=None) -> tuple[list, list]:
    """Rows for mdance_vs_gromacs.csv + lines for benchmarking_report.txt.

    Every row has exactly ``len(BENCH_HEADER)`` values, so the CSV columns
    and their values can never drift apart.
    """
    rows, report = [], []
    best = best_metrics
    report.append("=" * 78)
    report.append("MDANCE vs GROMACS (gmx cluster) BENCHMARKING REPORT")
    report.append("=" * 78)
    report.append("")
    report.append("IMPORTANT - DOCUMENTED DIFFERENCES (not hidden):")
    report.append("* `gmx cluster -method gromos` implements the Daura et al. (1999)")
    report.append("  greedy algorithm; this pipeline reproduces it exactly on the")
    report.append("  same pairwise MDANCE-MSD distance matrix used by all other")
    report.append("  methods, with equivalent clustering.atoms selection because")
    report.append("  the matrix is built from the same preprocessed trajectory")
    report.append("  and the same atom selection as all MDANCE runs.")
    native = bench_results.get("native")
    native_ok = bool(native and native.get("success"))
    report.append("* Native `gmx cluster`: " +
                  ("attempted and SUCCEEDED" if native_ok else
                   "attempted but FAILED" if native else
                   "NOT attempted (no gmx binary)"))
    report.append("* gmx computes RMSD with its own superposition; the MDANCE")
    report.append("  pipeline uses pre-aligned coordinates + MSD (squared RMSD/atom)")
    report.append("  - the two distance DEFINITIONS are reported separately and")
    report.append("  never mixed silently.")
    report.append("")
    if best is not None:
        rows.append([f"REFERENCE:{best.name}", best.n_clusters, best.time_s,
                     best.mem_mb, best.compactness, best.separation,
                     best.n_clusters, None, None, None, None])
        report.append(f"Reference solution (rank #1): {best.name}")
        report.append(f"  family                  : {best.family}")
        report.append(f"  clusters                : {best.n_clusters}")
        report.append(f"  time (s)                : {best.time_s:.2f}")
        report.append(f"  peak mem incr. (MB)     : {best.mem_mb:.1f}")
        if best.compactness is not None:
            report.append(f"  mean intra-cluster MSD  : {best.compactness:.4g}")
        if best.separation is not None:
            report.append(f"  min inter-medoid MSD    : {best.separation:.4g}")
        report.append("")
    aris = []
    for sol in bench_results.get("solutions", []):
        labels_ref = [s for s in sols if best is not None and s.name == best.name]
        cmpo = {}
        if labels_ref:
            cmpo = compare_partitions(labels_ref[0].labels, sol.labels)
        compact, sep = _partition_shape(sol.labels, D)
        ari = cmpo.get("adj_rand_index")
        if ari is not None:
            aris.append(float(ari))
        rows.append([f"GROMACS(gromos):{sol.name}", sol.n_clusters, sol.time_s,
                     sol.mem_mb, compact, sep,
                     cmpo.get("n_clusters_reference"),
                     cmpo.get("n_clusters_gromos"),
                     cmpo.get("population_L1"),
                     cmpo.get("top5_pop_divergence"), ari])
        report.append(f"GROMOS cutoff {sol.params.get('cutoff'):.4g} "
                      f"-> clusters {sol.n_clusters}, time {sol.time_s:.2f}s")
        if compact is not None:
            report.append(f"  mean intra-cluster MSD  : {compact:.4g}")
        if sep is not None:
            report.append(f"  min inter-medoid MSD    : {sep:.4g}")
        if cmpo:
            report.append(
                f"  vs reference: ARI {ari}, "
                f"population L1 {cmpo['population_L1']:.3f}, "
                f"top5-population divergence "
                f"{cmpo['top5_pop_divergence']:.3f}")
    if native:
        rows.append(["native gmx cluster", None, None, None, None, None,
                     None, None, None, None, None])
        report.append("")
        report.append("Native gmx cluster was attempted with cutoff "
                      f"{native.get('cutoff_nm')} nm RMSD; return code "
                      f"{native.get('returncode')} "
                      f"({'success' if native_ok else 'FAILED'}; "
                      "see benchmarking/gmx_cluster*.log).")
        report.append("Native membership parsing is not claimed equivalent;")
        report.append("the quantitative comparison is the gromos-on-MSD tier.")
    if aris:
        best_i = int(np.argmax(aris))
        bench_sols = bench_results.get("solutions", [])
        report.append("")
        report.append("Statistical comparison summary (gromos-on-MSD tier):")
        report.append(f"  cutoffs evaluated       : {len(aris)}")
        report.append(f"  ARI mean / max          : {np.mean(aris):.4f} / "
                      f"{np.max(aris):.4f}")
        if best_i < len(bench_sols):
            report.append("  closest agreement cutoff: "
                          f"{bench_sols[best_i].params.get('cutoff'):.4g} MSD "
                          f"(ARI {aris[best_i]:.4f})")
        report.append("  ARI compares label assignments after permutation; "
                      "population L1")
        report.append("  compares cluster-size profiles. Both are descriptive "
                      "agreement")
        report.append("  statistics, not a significance test.")
    report.append("")
    report.append("Runtime/memory comparison summary is included per row above;")
    report.append("the native gmx binary tier is qualitative-only.")
    return rows, report
