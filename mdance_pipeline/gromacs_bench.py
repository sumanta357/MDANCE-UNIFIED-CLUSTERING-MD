"""Phase 6 - benchmark against native GROMACS `gmx cluster` (pipeline-added).

Two tiers, kept explicitly separate:

1. Native `gmx cluster -method gromos`, executed when the `gmx` binary exists.
   The pipeline writes a structure/trajectory/index containing EXACTLY the
   MDANCE analysis selection, so both tiers cluster the same atoms. Native
   per-frame membership is read from the `-clid` XVG output (and the detailed
   `-g` log is kept). The native cutoff is converted from the MDANCE MSD units
   of the distance matrix to the RMSD cutoff in nm that gmx expects.

2. MDANCE GROMOS/Daura (greedy, Daura et al. 1999) on the identical MDANCE
   pairwise-MSD matrix used by every MDANCE method. Multiple cutoffs are
   evaluated, giving a distance-aligned comparison.

Both tiers are compared against the same best MDANCE solution with ARI and
population statistics. They are never merged into one number and a failed
native run is never reported as success.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score

from mdance_pipeline import output_layout
from mdance_pipeline.clustering import Solution, run_gromos, upper_desc_vals


def gmx_available() -> tuple[bool, str]:
    path = shutil.which("gmx") or shutil.which("gmx_mpi")
    return (path is not None, path or "")


def msd_to_rmsd_nm(msd: float) -> float:
    """MDANCE MSD (Angstrom^2 / (2N)) -> RMSD cutoff in nm.

    RMSD(Angstrom) = sqrt(2 * MSD); gmx expects nm.
    """
    return float(np.sqrt(2.0 * max(float(msd), 0.0)) / 10.0)


def native_solution(native: dict | None) -> Solution | None:
    """Wrap a successful native gmx result as a Solution (for representatives)."""
    if not native or not native.get("success") or native.get("labels") is None:
        return None
    labels = np.asarray(native["labels"], dtype=int)
    if labels.size == 0:
        return None
    return Solution(
        name=f"gmx_native/gromos_{native.get('cutoff_nm'):.4f}nm",
        family="gmx_native", params={"method": "gmx cluster -method gromos",
                                     "cutoff_nm": native.get("cutoff_nm"),
                                     "selection": native.get("selection"),
                                     "n_atoms": native.get("n_atoms")},
        labels=labels, n_clusters=int(native.get("n_clusters") or
                                      len(np.unique(labels))),
        time_s=float(native.get("time_s") or 0.0), mem_mb=0.0, native=False)


def parse_clid_xvg(path: str, n_frames: int) -> tuple[np.ndarray, int]:
    """Parse `gmx cluster -clid` into dense 0-based labels.

    The XVG holds one row per frame: time and cluster number. Cluster numbers
    are normalized to 0..k-1; the assignment (and therefore ARI) is unchanged.
    """
    ids: list[int] = []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#@":
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                ids.append(int(round(float(parts[1]))))
            except ValueError:
                continue
    if len(ids) != int(n_frames):
        raise ValueError(
            f"gmx -clid produced {len(ids)} assignments for {n_frames} frames")
    raw = np.asarray(ids, dtype=int)
    _, labels = np.unique(raw, return_inverse=True)
    return labels.astype(int), int(len(np.unique(labels)))


def _write_native_inputs(pre_info: dict, work_dir: str, logger=None) -> dict:
    """Write structure/trajectory/index for the exact MDANCE atom selection."""
    import MDAnalysis as mda

    native_dir = os.path.join(work_dir, "native_selection")
    os.makedirs(native_dir, exist_ok=True)
    u = pre_info["universe"]
    selection = pre_info["selection"]
    ag = u.select_atoms(selection)
    if ag.n_atoms == 0:
        raise ValueError(f"native gmx selection '{selection}' matched 0 atoms")
    gro = os.path.join(native_dir, "selection.gro")
    xtc = os.path.join(native_dir, "selection.xtc")
    ndx = os.path.join(native_dir, "selection.ndx")
    ag.write(gro)
    with mda.Writer(xtc, n_atoms=ag.n_atoms, multiframe=True) as writer:
        for _ in u.trajectory:
            writer.write(ag)
    with open(ndx, "w") as fh:
        fh.write("[ MDANCE_SELECTION ]\n")
        for start in range(0, ag.n_atoms, 15):
            fh.write(" ".join(str(i) for i in
                              range(start + 1, min(start + 15, ag.n_atoms) + 1)))
            fh.write("\n")
    if logger:
        logger.info("Native gmx inputs: %d atoms from selection '%s' -> %s",
                    ag.n_atoms, selection, native_dir)
    return {"gro": gro, "xtc": xtc, "ndx": ndx,
            "n_atoms": int(ag.n_atoms), "selection": selection}


def run_native_gmx(pre_info: dict, work_dir: str, cutoff_nm: float = 0.10,
                   logger=None) -> dict | None:
    """Run native `gmx cluster` on the MDANCE selection and parse membership.

    ``cutoff_nm`` is an RMSD cutoff in nm (gmx convention), converted by the
    caller from the MDANCE MSD matrix. The run uses the already aligned,
    PBC-corrected trajectory and ``-nofit`` so the RMSD definition matches the
    pre-aligned coordinates used by MDANCE as closely as GROMACS allows.
    """
    ok, gmx_bin = gmx_available()
    if not ok:
        return None
    t0 = time.time()
    try:
        inputs = _write_native_inputs(pre_info, work_dir, logger)
    except Exception as exc:
        if logger:
            logger.warning("could not prepare native gmx inputs: %s", exc)
        return {"returncode": None, "success": False, "cmd": "",
                "cutoff_nm": float(cutoff_nm), "error": str(exc),
                "labels": None, "n_clusters": 0}

    out_stub = os.path.join(work_dir, "gmx_cluster")
    clid = out_stub + "_clid.xvg"
    log_path = out_stub + ".log"
    cmd = [gmx_bin, "cluster", "-s", inputs["gro"], "-f", inputs["xtc"],
           "-n", inputs["ndx"], "-method", "gromos",
           "-cutoff", f"{cutoff_nm:.6f}", "-nofit",
           "-clid", clid, "-g", log_path, "-o", out_stub + ".xpm",
           "-sz", out_stub + "_sz.xvg", "-dist", out_stub + "_dist.xvg"]
    env = dict(os.environ)
    env.setdefault("GMX_MAXBACKUP", "-1")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=3600, input="MDANCE_SELECTION\n",
                              errors="replace", env=env)
    except Exception as exc:
        if logger:
            logger.warning("native gmx cluster failed to start: %s", exc)
        return {"returncode": None, "success": False, "cmd": " ".join(cmd),
                "cutoff_nm": float(cutoff_nm), "error": str(exc),
                "labels": None, "n_clusters": 0,
                "stdout_tail": "", "stderr_tail": ""}

    success = proc.returncode == 0 and os.path.isfile(log_path)
    labels = None
    n_clusters = 0
    error = None
    if success:
        try:
            labels, n_clusters = parse_clid_xvg(clid, len(pre_info["frame_indices"]))
        except Exception as exc:
            success = False
            error = f"membership parse failed: {exc}"
    else:
        error = f"gmx cluster returned {proc.returncode}"
    if not success and logger:
        logger.warning("native gmx cluster failed: %s (see %s)", error, log_path)
    elif logger:
        logger.info("Native gmx cluster: %d clusters (cutoff %.6f nm, %d atoms)",
                    n_clusters, cutoff_nm, inputs["n_atoms"])
    return {"returncode": proc.returncode, "success": bool(success),
            "cmd": " ".join(cmd), "cutoff_nm": float(cutoff_nm),
            "time_s": time.time() - t0,
            "labels": labels, "n_clusters": int(n_clusters),
            "error": error, "selection": inputs["selection"],
            "n_atoms": inputs["n_atoms"],
            "structure": inputs["gro"], "trajectory": inputs["xtc"],
            "clid": clid, "log": log_path,
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:]}


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
    work_dir = os.path.join(out_dir, output_layout.INTERMEDIATE_DIR,
                            "gromacs_benchmark")
    os.makedirs(work_dir, exist_ok=True)
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
            q = float(bench.get("native_gmx_cutoff_quantile", 0.10))
            cutoff_msd = float(np.quantile(upper_desc_vals(D), q))
            cutoff_nm = bench.get("native_gmx_cutoff_nm")
            if cutoff_nm is None:
                cutoff_nm = msd_to_rmsd_nm(cutoff_msd)
            logger.info("gmx binary found - running native gmx cluster on "
                        "the MDANCE selection (%s), cutoff %.6f nm "
                        "(quantile %.2f).", pre_info["selection"],
                        cutoff_nm, q)
            native = run_native_gmx(pre_info, work_dir,
                                    cutoff_nm=float(cutoff_nm), logger=logger)
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
    """Rows for GROMACS_COMPARISON.csv + benchmarking_report.txt lines.

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
    native_ari = None
    if native:
        native_labels = native.get("labels")
        if native_ok and native_labels is not None and best is not None:
            ref = next((s for s in sols if s.name == best.name), None)
            cmpo = (compare_partitions(ref.labels, native_labels)
                    if ref is not None else {})
            compact, sep = _partition_shape(native_labels, D)
            native_ari = cmpo.get("adj_rand_index")
            rows.append([f"native_gmx:gromos:{native.get('cutoff_nm'):.4f}nm",
                         int(native.get("n_clusters") or 0),
                         native.get("time_s"), 0.0, compact, sep,
                         cmpo.get("n_clusters_reference"),
                         cmpo.get("n_clusters_gromos"),
                         cmpo.get("population_L1"),
                         cmpo.get("top5_pop_divergence"), native_ari])
            report.append("")
            report.append("Native gmx cluster SUCCEEDED: cutoff "
                          f"{native.get('cutoff_nm'):.6f} nm RMSD, "
                          f"{native.get('n_clusters')} clusters on the "
                          f"MDANCE selection ({native.get('selection')}, "
                          f"{native.get('n_atoms')} atoms).")
            if cmpo:
                report.append(
                    f"  vs reference: ARI {native_ari}, population L1 "
                    f"{cmpo['population_L1']:.3f}, top5-population "
                    f"divergence {cmpo['top5_pop_divergence']:.3f}")
            report.append("  Membership comes from gmx -clid and is compared "
                          "on the same frames as MDANCE.")
            report.append("  Native gmx uses its own RMSD/fitting definition;"
                          " the separate gromos-on-MSD rows above remain "
                          "the distance-aligned tier.")
        else:
            rows.append(["native gmx cluster", None, None, None, None, None,
                         None, None, None, None, None])
            report.append("")
            report.append("Native gmx cluster was attempted with cutoff "
                          f"{native.get('cutoff_nm')} nm RMSD; return code "
                          f"{native.get('returncode')} "
                          f"({'success' if native_ok else 'FAILED'}; "
                          f"{native.get('error') or ''}).")
            report.append("A failed native run is NOT counted as a successful "
                          "benchmark; see "
                          "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/"
                          "gromacs_benchmark/gmx_cluster*.log.")
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
    if native_ari is not None:
        report.append("")
        report.append("Native gmx agreement with the reference MDANCE "
                      f"partition: ARI {native_ari:.4f} (one cutoff; "
                      "descriptive).")
    report.append("")
    report.append("Runtime/memory comparison summary is included per row above;")
    report.append("the native gmx binary tier is qualitative-only.")
    return rows, report
