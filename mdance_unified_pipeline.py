#!/usr/bin/env python3
"""
mdance_unified_pipeline.py
==========================
Interactive, unified front-end for the clustering methods in MDANCE
(https://github.com/mqcomplab/MDANCE).

MODE A - frame clustering of ONE trajectory
    nani     k-means NANI                 (scan k)
    equal    eQual radial-threshold       (scan threshold, k emerges)
    helm     HELM hierarchical merging    (NANI pre-clusters -> merged down)
    divine   DIVINE divisive hierarchy    (scan k)
    mdbirch  mdBIRCH online CF-tree       (scan RMSD-like threshold)
  Every selected method is scanned, the best setting is picked (lowest DB,
  max 2nd derivative of DB, or by you), and for EVERY method you get labels,
  populations, per-cluster best (medoid-first) frames and a summary.  A
  comparison table then helps you pick the method / frame to carry forward.

MODE B - pathway clustering of MANY trajectories
    shine    SHINE pathway clustering
    prism    PRISM pathway representation (structural medoids)

Not included: CADENCE (no usable public tutorial/API yet) and PRIME (a
post-clustering CLI: prime_sim / prime_rep).

API STATUS (be aware):
  * NANI, eQual, HELM, SHINE calls follow the published MDANCE API docs.
  * DIVINE, mdBIRCH and PRISM are called through signature-aware adapters
    (only arguments the installed class accepts are passed; results are
    converted defensively).  If your MDANCE version differs, the script prints
    the real signature and skips that method instead of crashing.

Requirements:  pip install mdance MDAnalysis matplotlib
The trajectory must already be aligned / centred.
Frame numbers written out are 0-based indices into the ORIGINAL trajectory.
"""

import glob
import importlib
import inspect
import os
import sys
import time

import numpy as np

SINGLE_METHODS = ["nani", "equal", "helm", "divine", "mdbirch"]
NANI_INITS = ["strat_all", "strat_reduced", "comp_sim", "div_select", "quota",
              "k-means++", "random", "vanilla_kmeans++"]


# --------------------------------------------------------------------------- #
# Interactive helpers
# --------------------------------------------------------------------------- #
def ask(prompt, default=None, cast=str, choices=None, must_exist=False, minimum=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "":
            if default is None:
                print("  A value is required.")
                continue
            raw = str(default)
        try:
            value = cast(raw)
        except ValueError:
            print(f"  Could not read '{raw}' as {cast.__name__}.")
            continue
        if choices is not None and value not in choices:
            print(f"  Please choose one of: {', '.join(map(str, choices))}")
            continue
        if must_exist and not os.path.exists(value):
            print(f"  File not found: {value}")
            continue
        if minimum is not None and value < minimum:
            print(f"  Value must be >= {minimum}.")
            continue
        return value


def ask_yes_no(prompt, default=True):
    return ask(f"{prompt} (y/n)", "y" if default else "n", choices=["y", "n"]).lower() == "y"


def ask_float_list(prompt, default_list):
    default = ",".join(f"{v:.4g}" for v in default_list)
    while True:
        raw = ask(prompt, default)
        try:
            vals = sorted({float(x) for x in raw.replace(" ", "").split(",") if x})
            if vals:
                return vals
        except ValueError:
            pass
        print("  Please give comma-separated numbers.")


def banner(text):
    print("\n" + "=" * 74 + f"\n{text}\n" + "=" * 74)


# --------------------------------------------------------------------------- #
# Defensive-call helpers (used for DIVINE / mdBIRCH / PRISM / SHINE)
# --------------------------------------------------------------------------- #
def filtered_kwargs(callable_, **cand):
    """Keep only keyword arguments the callable accepts; error on missing required ones."""
    sig = inspect.signature(callable_)
    params = sig.parameters
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    kept = {k: v for k, v in cand.items() if k in params}
    if has_var_kw:
        # also forward unknown extras only when explicitly flagged by caller
        pass
    missing = [p.name for p in params.values()
               if p.name not in ("self", "args", "kwargs")
               and p.default is p.empty
               and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
               and p.name not in kept]
    if missing:
        raise TypeError(f"required argument(s) {missing} not provided; "
                        f"installed signature is {callable_.__name__}{sig}")
    return kept


def _is_seq(x):
    return isinstance(x, (list, tuple, np.ndarray)) and not isinstance(x, str)


def to_labels(obj, n, target_k=None):
    """Convert many plausible MDANCE return shapes into a per-frame label array."""
    if obj is None:
        raise ValueError("no result returned")
    if isinstance(obj, np.ndarray) and obj.ndim == 1 and len(obj) == n \
            and np.issubdtype(obj.dtype, np.number):
        return obj.astype(int)
    if isinstance(obj, tuple):
        for item in obj:
            try:
                return to_labels(item, n, target_k)
            except ValueError:
                continue
        raise ValueError("no element of the returned tuple looks like labels/clusters")
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and _is_seq(vals[0]) and len(vals[0]) > 0 and _is_seq(vals[0][0]):
            # {k: [cluster, cluster, ...]} hierarchy -> pick a level
            key = target_k if target_k in obj else max(obj.keys())
            return to_labels(obj[key], n, target_k)
        return to_labels(vals, n, target_k)
    if _is_seq(obj):
        seq = list(obj)
        if seq and all(_is_seq(s) for s in seq):
            labels = np.full(n, -1, dtype=int)
            for lab, members in enumerate(seq):
                members = np.asarray(members)
                if members.dtype.kind not in "iu":
                    raise ValueError("cluster members are not integer indices")
                labels[members] = lab
            if (labels >= 0).mean() < 0.9:
                raise ValueError("returned clusters cover <90% of frames")
            return labels
    raise ValueError(f"cannot interpret result of type {type(obj).__name__}")


def first_available(obj, names):
    for name in names:
        if hasattr(obj, name):
            attr = getattr(obj, name)
            return attr() if callable(attr) else attr
    return None


# --------------------------------------------------------------------------- #
# Scoring / candidates
# --------------------------------------------------------------------------- #
def mean_cluster_msd(data, labels, metric, N_atoms):
    from mdance.tools.bts import extended_comparison
    vals = []
    for c in np.unique(labels[labels >= 0]):
        sub = data[labels == c]
        if len(sub) < 2:
            vals.append(0.0)
            continue
        try:
            vals.append(float(extended_comparison(sub, data_type="full",
                                                  metric=metric, N_atoms=N_atoms)))
        except Exception:
            vals.append(np.nan)
    return float(np.nanmean(vals)) if vals else float("nan")


def make_candidate(data, labels, param, metric, N_atoms, note=""):
    """Score one clustering. Label -1 = unassigned/noise (excluded from scores)."""
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    labels = np.asarray(labels, dtype=int)
    mask = labels >= 0
    uniq = np.unique(labels[mask])
    cand = {"param": float(param), "labels": labels, "n_clusters": len(uniq),
            "noise_pct": 100.0 * (1 - mask.mean()), "ch": None, "db": None,
            "msd": None, "note": note}
    if 2 <= len(uniq) < mask.sum():
        cand["ch"] = float(calinski_harabasz_score(data[mask], labels[mask]))
        cand["db"] = float(davies_bouldin_score(data[mask], labels[mask]))
        cand["msd"] = mean_cluster_msd(data, labels, metric, N_atoms)
    return cand


def print_candidates(cands, pname):
    print(f"  {'#':>3} {pname:>10} {'clusters':>9} {'DB':>8} {'CH':>12} {'avgMSD':>10} {'noise%':>7}")
    for i, c in enumerate(cands):
        db = f"{c['db']:.4f}" if c["db"] is not None else "   n/a"
        ch = f"{c['ch']:.1f}" if c["ch"] is not None else "n/a"
        ms = f"{c['msd']:.4f}" if c["msd"] is not None else "n/a"
        print(f"  {i:>3} {c['param']:>10.4g} {c['n_clusters']:>9} {db:>8} {ch:>12} "
              f"{ms:>10} {c['noise_pct']:>7.1f}")


def pick_candidate(cands, rule, pname, method, outdir):
    """Choose the best candidate: rule 1 lowest DB, 2 max 2nd derivative, 3 ask."""
    valid = [c for c in cands if c["db"] is not None]
    if not valid:
        return None
    valid.sort(key=lambda c: c["param"])
    db = np.array([c["db"] for c in valid])
    i_min = int(np.argmin(db))
    i_2nd = None
    cs = []
    for i in range(1, len(db) - 1):
        if db[i] <= db[i - 1] and db[i] <= db[i + 1]:
            cs.append((i, db[i + 1] + db[i - 1] - 2 * db[i]))
    if cs:
        i_2nd = max(cs, key=lambda t: t[1])[0]

    plot_db(valid, i_min, i_2nd, pname, method, outdir)
    print(f"  lowest-DB          : {pname} = {valid[i_min]['param']:.4g} "
          f"({valid[i_min]['n_clusters']} clusters)")
    if i_2nd is not None:
        print(f"  max 2nd derivative : {pname} = {valid[i_2nd]['param']:.4g} "
              f"({valid[i_2nd]['n_clusters']} clusters)")
    if i_min in (0, len(valid) - 1) and len(valid) > 2:
        print("  WARNING: lowest DB is at the edge of the scanned range - widen the range.")

    if rule == "2" and i_2nd is not None:
        return valid[i_2nd]
    if rule == "3":
        print_candidates(valid, pname)
        idx = ask("Pick row # to use", i_min, int, minimum=0)
        return valid[min(idx, len(valid) - 1)]
    return valid[i_min]


def plot_db(valid, i_min, i_2nd, pname, method, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = [c["param"] for c in valid]
    ax.plot(x, [c["db"] for c in valid], "o-", color="#005cde")
    ax.axvline(valid[i_min]["param"], ls="--", color="#de005c", label="lowest DB")
    if i_2nd is not None:
        ax.axvline(valid[i_2nd]["param"], ls="--", color="#de8200", label="max 2nd deriv")
    ax.set_xlabel(pname)
    ax.set_ylabel("Davies-Bouldin (lower = better)")
    ax.set_title(method)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{method}_DB.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #
def convert_traj(top, traj, sel):
    from mdance.inputs.preprocess import gen_traj_numpy
    return gen_traj_numpy(top, traj, sel)


def load_single(outdir):
    banner("INPUT - one trajectory")
    print("The trajectory must already be ALIGNED / CENTRED.\n")
    mode = ask("Input: 1 = topology + trajectory, 2 = preprocessed .npy", "1",
               choices=["1", "2"])
    top = traj = sel = None
    if mode == "1":
        top = ask("Topology file", must_exist=True)
        traj = ask("Trajectory file", must_exist=True)
        sel = ask("MDAnalysis atom selection used for clustering", "protein and name CA")
        data = convert_traj(top, traj, sel)
        if ask_yes_no("Save the numpy array for reuse?", True):
            p = os.path.join(outdir, "traj_numpy.npy")
            np.save(p, data)
            print(f"  saved -> {p}")
    else:
        data = np.load(ask("Path to .npy (n_frames x n_atoms*3)", must_exist=True))
    n_total = data.shape[0]
    default_atoms = data.shape[1] // 3 if data.shape[1] % 3 == 0 else 1
    N_atoms = ask("Number of atoms in the clustering selection (N_atoms)",
                  default_atoms, int, minimum=1)
    sieve = ask("Sieve (every n-th frame; 1 = all)", 1, int, minimum=1)
    frame_ids = np.arange(0, n_total, sieve)
    data = np.ascontiguousarray(data[::sieve])
    print(f"Frames: {n_total} total, {len(data)} used | features: {data.shape[1]}")
    return data, frame_ids, N_atoms, sieve, top, traj, sel


# --------------------------------------------------------------------------- #
# MODE A - per-method configuration + scanning
# --------------------------------------------------------------------------- #
def run_nani(data, k, metric, N_atoms, init_type, percentage):
    from mdance.cluster.nani import KmeansNANI
    pct = percentage
    while True:
        try:
            mod = KmeansNANI(data=data, n_clusters=k, metric=metric, N_atoms=N_atoms,
                             init_type=init_type, percentage=pct)
            labels, _centers, _n_iter = mod.execute_kmeans_all()
            return np.asarray(labels)
        except ValueError as err:
            if "initiators" in str(err) and pct < 100:
                pct = min(100, pct + 10)
                continue
            raise


def cfg_nani(ctx):
    print("\n[NANI]  k-means with N-ary Natural Initialization")
    cfg = {"init_type": ask("  init_type", "strat_all", choices=NANI_INITS),
           "percentage": ask("  seed percentage (int 1-100)", 10, int, minimum=1)}
    cfg["k_start"] = ask("  start k (>=2)", 3, int, minimum=2)
    cfg["k_end"] = ask("  end k", 15, int, minimum=cfg["k_start"] + 1)
    return cfg


def scan_nani(ctx, cfg):
    cands = []
    for k in range(cfg["k_start"], cfg["k_end"] + 1):
        try:
            labels = run_nani(ctx["data"], k, ctx["metric"], ctx["N_atoms"],
                              cfg["init_type"], cfg["percentage"])
            cands.append(make_candidate(ctx["data"], labels, k, ctx["metric"], ctx["N_atoms"]))
        except Exception as err:
            print(f"    k={k}: failed ({err})")
    return cands, "k"


def cfg_equal(ctx):
    print("\n[eQual]  radial-threshold clustering (number of clusters emerges)")
    cfg = {"seed_method": ask("  seed_method", "greedy",
                              choices=["greedy", "comp_sim", "medoid",
                                       "mini_batch_kmeans", "vanilla"]),
           "min_samples": ask("  min_samples (int = frames, float<1 = fraction)", 0.01, float,
                              minimum=0.0)}
    if cfg["min_samples"] >= 1:
        cfg["min_samples"] = int(cfg["min_samples"])
    try:
        from mdance.tools.bts import extended_comparison
        base = float(extended_comparison(ctx["data"], data_type="full",
                                         metric=ctx["metric"], N_atoms=ctx["N_atoms"]))
    except Exception:
        base = 1.0
    print(f"  (ensemble-wide {ctx['metric']} = {base:.4g}; default thresholds are only "
          f"starting guesses, in the metric's units)")
    cfg["thresholds"] = ask_float_list("  thresholds to scan (comma-separated)",
                                       base * np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    return cfg


def scan_equal(ctx, cfg):
    from mdance.cluster.equal import ExtendedQuality
    n = len(ctx["data"])
    cands = []
    for thr in cfg["thresholds"]:
        try:
            mod = ExtendedQuality(data=ctx["data"], threshold=thr, metric=ctx["metric"],
                                  N_atoms=ctx["N_atoms"], seed_method=cfg["seed_method"],
                                  n_seeds=1, check_sim=False, reject_lowd=True,
                                  min_samples=cfg["min_samples"])
            clusters = mod.run()
            labels = np.full(n, -1, dtype=int)
            for lab, key in enumerate(sorted(clusters.keys())):
                labels[np.asarray(clusters[key], dtype=int)] = lab
            cands.append(make_candidate(ctx["data"], labels, thr, ctx["metric"], ctx["N_atoms"]))
        except Exception as err:
            print(f"    threshold={thr:.4g}: failed ({err})")
    return cands, "threshold"


def cfg_helm(ctx):
    print("\n[HELM]  hierarchical merging of NANI pre-clusters")
    n = len(ctx["data"])
    cfg = {"k0": ask("  number of NANI pre-clusters to merge", max(10, min(100, n // 20)),
                     int, minimum=4),
           "scheme": ask("  merge_scheme", "inter", choices=["inter", "intra", "half"]),
           "init_type": ask("  NANI init_type for the pre-clusters", "strat_all",
                            choices=NANI_INITS),
           "percentage": ask("  NANI seed percentage", 10, int, minimum=1)}
    cfg["k_min"] = ask("  lowest cluster count to merge down to", 2, int, minimum=2)
    cfg["k_max"] = ask("  highest cluster count to score", min(20, cfg["k0"] - 1), int,
                       minimum=cfg["k_min"])
    return cfg


def scan_helm(ctx, cfg):
    from mdance.cluster.helm import HELM
    data = ctx["data"]
    pre = run_nani(data, cfg["k0"], ctx["metric"], ctx["N_atoms"],
                   cfg["init_type"], cfg["percentage"])
    ids = np.unique(pre)
    clusters0 = []
    for i, c in enumerate(ids):
        sub = data[pre == c]
        clusters0.append([[i], (sub.sum(axis=0), (sub ** 2).sum(axis=0)), int(len(sub))])
    pre_map = {int(c): i for i, c in enumerate(ids)}
    pre_idx = np.array([pre_map[int(c)] for c in pre])

    mod = HELM({len(clusters0): clusters0}, ctx["metric"], ctx["N_atoms"],
               merge_scheme=cfg["scheme"], n_clusters=cfg["k_min"])
    result = mod.run()
    cands = []
    for k in sorted(result.keys()):
        if not (cfg["k_min"] <= k <= cfg["k_max"]):
            continue
        mapping = {}
        for new_label, cl in enumerate(result[k]):
            for pid in cl[0]:
                mapping[int(pid)] = new_label
        labels = np.array([mapping.get(int(p), -1) for p in pre_idx], dtype=int)
        cands.append(make_candidate(data, labels, k, ctx["metric"], ctx["N_atoms"]))
    return cands, "k"


def cfg_divine(ctx):
    print("\n[DIVINE]  deterministic divisive hierarchy (adapter - API not fully documented)")
    cfg = {"split": ask("  split strategy", "weighted_MSD",
                        choices=["MSD", "radius", "weighted_MSD"]),
           "anchors": ask("  anchors", "nani", choices=["nani", "outlier_pair", "splinter_split"])}
    cfg["k_start"] = ask("  start k (>=2)", 3, int, minimum=2)
    cfg["k_end"] = ask("  end k", 15, int, minimum=cfg["k_start"] + 1)
    return cfg


def run_divine_once(ctx, cfg, k):
    mod = importlib.import_module("mdance.cluster.divine")
    cls = getattr(mod, "Divine")
    data = ctx["data"]
    kw = filtered_kwargs(cls, data=data, X=data, matrix=data,
                         N_atoms=ctx["N_atoms"], n_atoms=ctx["N_atoms"],
                         metric=ctx["metric"], split=cfg["split"], anchors=cfg["anchors"],
                         end="k", k=k, n_clusters=k)
    obj = cls(**kw)
    result = None
    for name in ("divisive_algorithm", "run", "fit_predict", "fit", "cluster"):
        if hasattr(obj, name):
            result = getattr(obj, name)()
            break
    for src in (result,
                first_available(obj, ["labels", "labels_", "get_labels"]),
                first_available(obj, ["clusters", "cluster_dict"])):
        try:
            return to_labels(src, len(data), k)
        except ValueError:
            continue
    raise ValueError("could not extract labels from the Divine object "
                     f"(attributes: {[a for a in dir(obj) if not a.startswith('_')][:25]})")


def scan_divine(ctx, cfg):
    cands = []
    for k in range(cfg["k_start"], cfg["k_end"] + 1):
        try:
            labels = run_divine_once(ctx, cfg, k)
            cands.append(make_candidate(ctx["data"], labels, k, ctx["metric"], ctx["N_atoms"]))
        except Exception as err:
            print(f"    k={k}: failed ({err})")
            if k == cfg["k_start"]:
                print("    -> stopping DIVINE scan; check the printed message / signature.")
                break
    return cands, "k"


def cfg_mdbirch(ctx):
    print("\n[mdBIRCH]  online CF-tree clustering (adapter - API not fully documented)")
    try:
        from mdance.tools.bts import extended_comparison
        base = float(extended_comparison(ctx["data"], data_type="full",
                                         metric="MSD", N_atoms=ctx["N_atoms"]))
    except Exception:
        base = 4.0
    rmsd = max(base, 1e-6) ** 0.5
    print(f"  (ensemble-wide RMSD-like scale ~ {rmsd:.3g}; thresholds are RMSD-like, "
          f"smaller = more clusters)")
    cfg = {"thresholds": ask_float_list("  thresholds to scan (comma-separated)",
                                        rmsd * np.array([0.2, 0.35, 0.5, 0.7, 0.9])),
           "branching": ask("  branching_factor", 50, int, minimum=2)}
    return cfg


def run_mdbirch_once(ctx, cfg, thr):
    mod = importlib.import_module("mdance.cluster.mdbirch")
    data = ctx["data"]
    D = data.shape[1]
    if hasattr(mod, "set_merge"):
        mod.set_merge("radius", features=D)
    cls = getattr(mod, "mdBirch", None) or getattr(mod, "MDBirch")
    if hasattr(cls, "set_merge") and not hasattr(mod, "set_merge"):
        cls.set_merge("radius", features=D)
    kw = filtered_kwargs(cls, threshold=thr, branching_factor=cfg["branching"])
    brc = cls(**kw)
    brc.fit(data)
    for src in (first_available(brc, ["labels_", "get_labels"]),
                first_available(brc, ["get_cluster_mol_ids", "get_cluster_ids"])):
        try:
            return to_labels(src, len(data))
        except ValueError:
            continue
    if hasattr(brc, "predict"):
        return to_labels(brc.predict(data), len(data))
    raise ValueError("could not extract labels from the mdBirch object "
                     f"(attributes: {[a for a in dir(brc) if not a.startswith('_')][:25]})")


def scan_mdbirch(ctx, cfg):
    cands = []
    for thr in cfg["thresholds"]:
        try:
            labels = run_mdbirch_once(ctx, cfg, thr)
            cands.append(make_candidate(ctx["data"], labels, thr, ctx["metric"], ctx["N_atoms"]))
        except Exception as err:
            print(f"    threshold={thr:.4g}: failed ({err})")
            if thr == cfg["thresholds"][0]:
                print("    -> stopping mdBIRCH scan; check the printed message / signature.")
                break
    return cands, "threshold"


CONFIGURE = {"nani": cfg_nani, "equal": cfg_equal, "helm": cfg_helm,
             "divine": cfg_divine, "mdbirch": cfg_mdbirch}
SCAN = {"nani": scan_nani, "equal": scan_equal, "helm": scan_helm,
        "divine": scan_divine, "mdbirch": scan_mdbirch}


# --------------------------------------------------------------------------- #
# Per-cluster analysis and outputs
# --------------------------------------------------------------------------- #
def analyse_clusters(data, labels, frame_ids, metric, N_atoms, n_struct):
    from mdance.tools.bts import calculate_comp_sim, extended_comparison
    rows = []
    n_all = len(labels)
    for c in np.unique(labels[labels >= 0]):
        idx = np.where(labels == c)[0]
        sub = data[idx]
        if len(idx) > 2:
            order = np.argsort(np.asarray(calculate_comp_sim(sub, metric, N_atoms)))[::-1]
        else:
            order = np.arange(len(idx))
        top = order[:min(n_struct, len(idx))]
        best = frame_ids[idx[top]].astype(int)
        try:
            msd = float(extended_comparison(sub, data_type="full", metric=metric,
                                            N_atoms=N_atoms)) if len(idx) > 1 else 0.0
        except Exception:
            msd = float("nan")
        rows.append({"cluster": int(c), "n_frames": int(len(idx)),
                     "fraction": len(idx) / n_all, "msd": msd,
                     "medoid_frame": int(best[0]), "best_frames": best})
    return rows


def write_method_outputs(mdir, tag, rows, labels, frame_ids, cands, pname):
    os.makedirs(mdir, exist_ok=True)
    np.savetxt(os.path.join(mdir, f"labels_{tag}.csv"),
               np.column_stack([frame_ids, labels]), fmt="%d", delimiter=",",
               header="frame,cluster (-1 = unassigned)", comments="")
    with open(os.path.join(mdir, f"populations_{tag}.csv"), "w") as fh:
        fh.write("cluster,n_frames,percent\n")
        for r in rows:
            fh.write(f"{r['cluster']},{r['n_frames']},{100 * r['fraction']:.2f}\n")
    pairs = [(f, r["cluster"]) for r in rows for f in r["best_frames"]]
    np.savetxt(os.path.join(mdir, f"best_frames_indices_{tag}.csv"),
               np.array(pairs, dtype=int), fmt="%d", delimiter=",",
               header=f"Best frames per cluster ({tag}); 0-based, sieve undone\nframe,cluster",
               comments="")
    with open(os.path.join(mdir, f"cluster_summary_{tag}.csv"), "w") as fh:
        fh.write("cluster,n_frames,percent,intra_cluster_MSD,medoid_frame_0based\n")
        for r in sorted(rows, key=lambda x: -x["n_frames"]):
            fh.write(f"{r['cluster']},{r['n_frames']},{100 * r['fraction']:.2f},"
                     f"{r['msd']:.4f},{r['medoid_frame']}\n")
    with open(os.path.join(mdir, "scan_table.csv"), "w") as fh:
        fh.write(f"{pname},n_clusters,DB,CH,avg_MSD,noise_percent\n")
        for c in cands:
            fh.write(f"{c['param']:.6g},{c['n_clusters']},{c['db']},{c['ch']},"
                     f"{c['msd']},{c['noise_pct']:.2f}\n")


def final_recommendation(rows, label, outdir):
    banner(f"RESULT SUMMARY - {label}")
    ranked = sorted(rows, key=lambda r: -r["n_frames"])
    print(f"{'rank':>4} {'cluster':>7} {'frames':>8} {'%':>7} {'MSD':>10} {'medoid frame':>13}")
    for i, r in enumerate(ranked, 1):
        print(f"{i:>4} {r['cluster']:>7} {r['n_frames']:>8} {100 * r['fraction']:>7.2f} "
              f"{r['msd']:>10.3f} {r['medoid_frame']:>13}")
    min_pct = ask("\nMinimum population (%) for a 'significant' cluster", 5.0, float, minimum=0.0)
    sig = [r for r in ranked if 100 * r["fraction"] >= min_pct]
    covered = 100 * sum(r["fraction"] for r in sig)
    dom = ranked[0]
    lines = [f"Chosen clustering                 : {label}",
             f"Dominant cluster                  : {dom['cluster']} "
             f"({100 * dom['fraction']:.1f}% of frames, intra-cluster MSD {dom['msd']:.3f})",
             f"BEST REPRESENTATIVE FRAME TO TAKE : frame {dom['medoid_frame']} "
             f"(medoid of cluster {dom['cluster']}, 0-based)"]
    if len(ranked) > 1:
        gap = 100 * (dom["fraction"] - ranked[1]["fraction"])
        if gap < 5:
            lines.append(f"NOTE: cluster {ranked[1]['cluster']} is within {gap:.1f} points of "
                         f"the dominant one - use the ensemble set below instead of one frame.")
    lines.append(f"Ensemble set (clusters >= {min_pct:g}%, covering {covered:.1f}% of frames):")
    for r in sig:
        lines.append(f"    cluster {r['cluster']:>3}  medoid frame {r['medoid_frame']:>8}  "
                     f"({100 * r['fraction']:.1f}%)")
    print("\n" + "\n".join(lines))
    with open(os.path.join(outdir, "final_recommendation.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return sig


def export_structures(top, traj, rows, significant, outdir, tag):
    banner("EXPORT STRUCTURES (optional)")
    if top is None or traj is None:
        print("Only a .npy was supplied - use the frame numbers with your own trajectory.")
        return
    if not ask_yes_no("Write medoid PDBs and best-frame DCDs with MDAnalysis?", True):
        return
    import MDAnalysis as mda
    u = mda.Universe(top, traj)
    ag = u.select_atoms(ask("Atoms to export (MDAnalysis selection)", "all"))
    sdir = os.path.join(outdir, "structures")
    os.makedirs(sdir, exist_ok=True)
    for r in rows:
        u.trajectory[r["medoid_frame"]]
        ag.write(os.path.join(sdir, f"{tag}_cluster{r['cluster']}_medoid_frame{r['medoid_frame']}.pdb"))
    with mda.Writer(os.path.join(sdir, f"{tag}_significant_medoids.pdb"),
                    ag.n_atoms, multiframe=True) as W:
        for r in significant:
            u.trajectory[r["medoid_frame"]]
            W.write(ag)
    for r in rows:
        with mda.Writer(os.path.join(sdir, f"{tag}_cluster{r['cluster']}_best_frames.dcd"),
                        ag.n_atoms) as W:
            for f in r["best_frames"]:
                u.trajectory[int(f)]
                W.write(ag)
    print(f"Structures written to {sdir}")


# --------------------------------------------------------------------------- #
# MODE A driver
# --------------------------------------------------------------------------- #
def mode_single(outdir):
    data, frame_ids, N_atoms, sieve, top, traj, sel = load_single(outdir)
    banner("SETTINGS")
    metric = ask("Similarity metric (MSD for coordinates)", "MSD")
    if metric != "MSD":
        print("  NOTE: metrics other than MSD need data normalised to [0, 1] first.")
    raw = ask(f"Methods to run (comma list from {','.join(SINGLE_METHODS)} or 'all')", "all")
    methods = SINGLE_METHODS if raw.strip().lower() == "all" else \
        [m.strip().lower() for m in raw.split(",") if m.strip().lower() in SINGLE_METHODS]
    if not methods:
        sys.exit("No valid method selected.")
    rule = ask("Pick the best setting per method by: 1 = lowest DB, "
               "2 = max 2nd derivative of DB, 3 = ask me after each scan", "1",
               choices=["1", "2", "3"])
    n_struct = ask("Best frames to extract per cluster", 10, int, minimum=1)

    ctx = {"data": data, "N_atoms": N_atoms, "metric": metric, "frame_ids": frame_ids}
    cfgs = {m: CONFIGURE[m](ctx) for m in methods}

    chosen = {}
    for m in methods:
        banner(f"RUNNING {m.upper()}")
        t0 = time.time()
        try:
            cands, pname = SCAN[m](ctx, cfgs[m])
        except Exception as err:
            print(f"  {m} FAILED: {err}")
            continue
        print(f"  scan finished in {time.time() - t0:.1f} s")
        if not cands:
            print("  no valid clustering produced - skipped.")
            continue
        print_candidates(cands, pname)
        mdir = os.path.join(outdir, m)
        os.makedirs(mdir, exist_ok=True)
        best = pick_candidate(cands, rule, pname, m, mdir)
        if best is None:
            print("  no candidate with >=2 clusters - skipped.")
            continue
        rows = analyse_clusters(data, best["labels"], frame_ids, metric, N_atoms, n_struct)
        tag = f"{m}_{pname}{best['param']:.4g}"
        write_method_outputs(mdir, tag, rows, best["labels"], frame_ids, cands, pname)
        chosen[m] = {"cand": best, "rows": rows, "tag": tag, "pname": pname, "dir": mdir}
        print(f"  -> {m}: {pname}={best['param']:.4g}, {best['n_clusters']} clusters, "
              f"DB={best['db']:.4f}; files in {mdir}")

    if not chosen:
        sys.exit("No method produced a result.")

    banner("METHOD COMPARISON")
    print(f"{'method':>8} {'setting':>14} {'clusters':>9} {'DB':>8} {'CH':>12} "
          f"{'largest%':>9} {'noise%':>7}")
    for m, d in chosen.items():
        c = d["cand"]
        big = 100 * max(r["fraction"] for r in d["rows"])
        print(f"{m:>8} {d['pname'] + '=' + format(c['param'], '.4g'):>14} {c['n_clusters']:>9} "
              f"{c['db']:>8.4f} {c['ch']:>12.1f} {big:>9.1f} {c['noise_pct']:>7.1f}")
    best_method = min(chosen, key=lambda m: chosen[m]["cand"]["db"])
    print(f"\nLowest DB overall: {best_method}.  (DB favours few, compact clusters, so also look "
          f"at 'largest%' and 'noise%'; treat the cross-method comparison as a guide only.)")
    use = ask("Method to use for the final representative-frame pick", best_method,
              choices=list(chosen.keys()))

    d = chosen[use]
    sig = final_recommendation(d["rows"], d["tag"], outdir)
    export_structures(top, traj, d["rows"], sig, outdir, d["tag"])

    with open(os.path.join(outdir, "run_parameters.txt"), "w") as fh:
        fh.write(f"mode=single\ntopology={top}\ntrajectory={traj}\natom_selection={sel}\n"
                 f"N_atoms={N_atoms}\nsieve={sieve}\nmetric={metric}\nmethods={methods}\n"
                 f"selection_rule={rule}\nbest_frames_per_cluster={n_struct}\n")
        for m in methods:
            fh.write(f"{m}_config={cfgs[m]}\n")
        for m, dd in chosen.items():
            fh.write(f"{m}_selected={dd['pname']}={dd['cand']['param']}\n")
        fh.write(f"final_method={use}\n")
    banner("DONE - results in: " + os.path.abspath(outdir))


# --------------------------------------------------------------------------- #
# MODE B - pathway clustering (SHINE / PRISM)
# --------------------------------------------------------------------------- #
def load_pathways():
    banner("INPUT - many trajectories (one per pathway)")
    kind = ask("Input: 1 = .npy files (glob), 2 = trajectory files (glob) + one topology",
               "1", choices=["1", "2"])
    pattern = ask("Glob pattern for the files (e.g. 'paths/*.npy')")
    files = sorted(glob.glob(pattern))
    if len(files) < 3:
        sys.exit(f"Found {len(files)} files for '{pattern}' - need at least 3 pathways.")
    sieve = ask("Sieve (every n-th frame per pathway)", 1, int, minimum=1)
    arrays = []
    if kind == "1":
        arrays = [np.load(f) for f in files]
    else:
        top = ask("Topology file", must_exist=True)
        sel = ask("MDAnalysis atom selection", "protein and name CA")
        arrays = [convert_traj(top, f, sel) for f in files]
    nfeat = {a.shape[1] for a in arrays}
    if len(nfeat) != 1:
        sys.exit(f"Pathways have different feature counts: {sorted(nfeat)}")
    default_atoms = arrays[0].shape[1] // 3 if arrays[0].shape[1] % 3 == 0 else 1
    N_atoms = ask("Number of atoms in the selection (N_atoms)", default_atoms, int, minimum=1)
    trajs = [(i, np.ascontiguousarray(a[::sieve])) for i, a in enumerate(arrays)]
    print(f"{len(trajs)} pathways loaded; frames per pathway: "
          f"{[len(t[1]) for t in trajs][:10]}{' ...' if len(trajs) > 10 else ''}")
    return files, trajs, N_atoms, sieve


def save_dendrogram(mod, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = mod.plot()
        fig = getattr(out, "figure", None) or plt.gcf()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close("all")
        print(f"  dendrogram -> {path}")
    except Exception as err:
        print(f"  (could not save dendrogram: {err})")


def mode_pathways(outdir):
    files, trajs, N_atoms, sieve = load_pathways()
    which = ask("Method: shine / prism / both", "both", choices=["shine", "prism", "both"])
    metric = ask("Similarity metric", "MSD")
    criterion = ask("Flat-cluster criterion", "maxclust",
                    choices=["maxclust", "distance", "inconsistent", "monocrit",
                             "maxclust_monocrit"])
    t = ask("t (number of pathway clusters for 'maxclust', else distance threshold)", 3, float,
            minimum=0.0)
    link = ask("scipy linkage", "ward")
    results = {}

    if which in ("shine", "both"):
        banner("SHINE")
        merge = ask("merge_scheme", "intra",
                    choices=["intra", "inter", "semi_sum", "max", "min", "haus"])
        sampling = ask("sampling", "diversity", choices=["diversity", "quota", "None"])
        try:
            from mdance.cluster.shine import Shine
            kw = filtered_kwargs(Shine, trajs=trajs, metric=metric, N_atoms=N_atoms, t=t,
                                 criterion=criterion, link=link, merge_scheme=merge,
                                 sampling=None if sampling == "None" else sampling)
            mod = Shine(**kw)
            out = mod.run()
            clusters = out[1] if isinstance(out, tuple) and len(out) == 2 else out
            results["shine"] = np.asarray(clusters).astype(int)
            save_dendrogram(mod, os.path.join(outdir, "shine_dendrogram.png"))
        except Exception as err:
            print(f"  SHINE failed: {err}")

    if which in ("prism", "both"):
        banner("PRISM")
        option = ask("medoid-set option: 1 global sharing, 2 per-pathway, 3 two-stage", 1, int,
                     choices=[1, 2, 3])
        k = ask("k (clusters for medoid construction)", 10, int, minimum=2)
        k_final = ask("k_final (option 3 only)", k, int, minimum=2)
        wscheme = ask("weight_scheme", "weighted_avg",
                      choices=["unnormalized", "weighted_avg", "sym_avg", "product_avg"])
        try:
            prism_mod = importlib.import_module("mdance.cluster.prism")
            cls = getattr(prism_mod, "PRISM")
            kw = filtered_kwargs(cls, trajs=trajs, metric=metric, N_atoms=N_atoms,
                                 option=option, k=k, k_final=k_final, weight_scheme=wscheme,
                                 link=link, t=t, criterion=criterion)
            mod = cls(**kw)
            out = mod.run()
            cl = out[1] if isinstance(out, tuple) and len(out) == 2 else \
                (out if out is not None else first_available(mod, ["clusters", "labels_"]))
            if cl is not None:
                results["prism"] = np.asarray(cl).astype(int)
            else:
                print("  PRISM ran, but no cluster labels were returned (see the dendrogram).")
            save_dendrogram(mod, os.path.join(outdir, "prism_dendrogram.png"))
        except Exception as err:
            print(f"  PRISM failed: {err}")

    if not results:
        sys.exit("No pathway method produced labels.")

    # per-pathway medoid frame (most central frame) for convenience
    medoids = []
    try:
        from mdance.tools.bts import calculate_medoid
        for (_i, arr) in trajs:
            medoids.append(int(calculate_medoid(arr, metric, N_atoms)) * sieve)
    except Exception:
        medoids = [-1] * len(trajs)

    path = os.path.join(outdir, "pathway_clusters.csv")
    with open(path, "w") as fh:
        fh.write("pathway_id,file,medoid_frame_0based," + ",".join(f"{m}_cluster" for m in results) + "\n")
        for i, f in enumerate(files):
            fh.write(f"{i},{f},{medoids[i]}," + ",".join(str(int(results[m][i])) for m in results) + "\n")
    print(f"\nPathway assignments -> {path}")
    for m, lab in results.items():
        print(f"  {m}: " + ", ".join(f"cluster {c}: {int((lab == c).sum())} pathways"
                                     for c in np.unique(lab)))
    with open(os.path.join(outdir, "run_parameters.txt"), "w") as fh:
        fh.write(f"mode=pathways\nmethod={which}\nmetric={metric}\ncriterion={criterion}\nt={t}\n"
                 f"link={link}\nsieve={sieve}\nN_atoms={N_atoms}\nfiles={files}\n")
    banner("DONE - results in: " + os.path.abspath(outdir))


# --------------------------------------------------------------------------- #
def main():
    banner("MDANCE - unified interactive clustering pipeline")
    outdir = ask("Output directory", "mdance_results")
    os.makedirs(outdir, exist_ok=True)
    mode = ask("Mode: A = frame clustering of one trajectory (NANI/eQual/HELM/DIVINE/mdBIRCH), "
               "B = pathway clustering of many trajectories (SHINE/PRISM)", "A",
               choices=["A", "B", "a", "b"]).upper()
    if mode == "A":
        mode_single(outdir)
    else:
        mode_pathways(outdir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
