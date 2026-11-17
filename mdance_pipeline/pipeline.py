"""Unified pipeline orchestration (Phases 2-12). Pipeline-added."""
from __future__ import annotations

import csv
import json
import os
import shutil

import numpy as np

from mdance_pipeline import (clustering, features as feat, gromacs_bench,
                             k_selection, method_representatives as method_reps,
                             output_layout as layout, preprocess as pre,
                             reports, representatives as reps, scoring,
                             validation as val, utils, visualization as viz)
from mdance_pipeline.config import validate_config


class Pipeline:
    """Runs every phase; state kept on self, data on cwd as required."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out_dir = cfg["system"]["output_dir"]
        self.logger = utils.get_logger(
            os.path.join(self.out_dir, layout.LOGS_DIR, "analysis.log"),
            level=cfg["system"].get("log_level", "INFO"))
        layout.ensure_tree(self.out_dir)
        self.full_output = bool(cfg["system"].get("full_output", False))
        self.info = None
        self.dist = None
        self.sols: list = []
        self.metrics: list = []
        self.ranked: list = []
        self.best_metrics = None
        self.best_labels: np.ndarray | None = None
        self.best_stats: dict = {}
        self.score_rows: list = []
        self.sens_summary: dict = {}
        self.bench_results: dict = {}
        self.k_report: dict | None = None
        self.method_summary: dict = {}
        self.chosen_methods: dict = {}
        self.pose_rows: list[dict] = []

    # ------------------------------------------------------------- entry
    def run(self) -> dict:
        log = self.logger
        utils.set_seed(self.cfg["system"].get("seed", 42))
        with utils.timed_step(log, "Phases 1-2: input validation & preprocessing"):
            self._phase_preprocess()
        with utils.timed_step(log, "Phase 3: features & distance matrix"):
            self._phase_features()
        if self.cfg.get("k_selection", {}).get("enabled", True):
            with utils.timed_step(log, "Phase 3b: automatic K selection (Elbow)"):
                self._phase_k_selection()
        with utils.timed_step(log, "Phase 4: MDANCE + baseline clustering sweep"):
            self.method_coverage: dict = {}
            self.sols = clustering.run_all(
                self.dist["X"], self.dist["D"],
                self.info["n_atoms_selected"], self.cfg, log,
                coverage=self.method_coverage)
        with utils.timed_step(log, "Phase 5: validation & optimal model selection"):
            self._phase_validation()
        if self.cfg["benchmark"].get("enabled", True):
            with utils.timed_step(log, "Phase 6: GROMACS benchmarking"):
                self.bench_results = gromacs_bench.benchmark(
                    self.sols, self.dist["D"], self.dist["X"], self.cfg, log,
                    self.info, self.out_dir)
                self._update_gromos_coverage()
        else:
            self.method_coverage["gromos"] = {
                "status": "not_run", "reason": "benchmark.enabled: false",
                "solutions": 0, "usable_solutions": 0, "cluster_counts": [],
                "failed_solutions": 0}
            self.method_coverage["gmx_native"] = {
                "status": "not_run", "reason": "benchmark.enabled: false",
                "solutions": 0, "usable_solutions": 0, "cluster_counts": [],
                "failed_solutions": 0}
        with utils.timed_step(log, "Phase 7: representative-frame extraction"):
            self._phase_representatives()
        with utils.timed_step(log, "Phases 8-9: composite scoring & sensitivity"):
            self._phase_scoring()
        with utils.timed_step(log, "Phase 10: visualizations"):
            self._phase_plotting()
        with utils.timed_step(log, "Phases 11-12: reports & final banner"):
            self._phase_reports()
        return {"final": self.score_rows[0] if self.score_rows else None}

    # ---------------------------------------------------- method coverage
    def _update_gromos_coverage(self) -> None:
        sols = (self.bench_results or {}).get("solutions", []) or []
        # Persist the actual GROMOS/Daura partitions next to the label files
        # of the other methods (previously this folder held baselines only).
        gdir = os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                            "clustering", "labels", "gromos")
        os.makedirs(gdir, exist_ok=True)
        for sol in sols:
            if sol.labels.size:
                np.save(os.path.join(gdir,
                                     f"{sol.name.replace('/', '_')}.npy"),
                        sol.labels.astype(int))
        counts = [int(s.n_clusters) for s in sols]
        usable = [c for c in counts if c >= 2]
        if not sols:
            status, reason = "not_run", "no GROMOS benchmark solution"
        elif usable:
            status, reason = "usable", ""
        else:
            status, reason = "degenerate", "every cutoff gave < 2 clusters"
        self.method_coverage["gromos"] = {
            "status": status, "reason": reason, "solutions": len(sols),
            "usable_solutions": len(usable),
            "cluster_counts": sorted(set(counts)),
            "failed_solutions": sum(1 for s in sols if s.error),
        }
        # Native gmx tier, kept separate from the MDANCE GROMOS tier.
        native = (self.bench_results or {}).get("native")
        ndir = os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                            "clustering", "labels", "gromacs_native")
        os.makedirs(ndir, exist_ok=True)
        nsol = gromacs_bench.native_solution(native)
        if nsol is not None:
            np.save(os.path.join(ndir, "gmx_native.npy"),
                    nsol.labels.astype(int))
            nstatus = "usable" if nsol.n_clusters >= 2 else "degenerate"
            nreason = "" if nstatus == "usable" else "< 2 clusters"
            self.method_coverage["gmx_native"] = {
                "status": nstatus, "reason": nreason, "solutions": 1,
                "usable_solutions": 1 if nstatus == "usable" else 0,
                "cluster_counts": [int(nsol.n_clusters)],
                "failed_solutions": 0,
            }
            self.native_solution = nsol
        elif native:
            self.method_coverage["gmx_native"] = {
                "status": "not_run",
                "reason": native.get("error") or "native gmx did not finish",
                "solutions": 0, "usable_solutions": 0,
                "cluster_counts": [], "failed_solutions": 1,
            }
        else:
            native_mode = self.cfg["benchmark"].get("native_gmx", "auto")
            self.method_coverage["gmx_native"] = {
                "status": "not_run",
                "reason": ("disabled by config" if native_mode is False
                           else "gmx binary not found"),
                "solutions": 0, "usable_solutions": 0,
                "cluster_counts": [], "failed_solutions": 0,
            }

    def _write_method_coverage(self) -> None:
        """Persist why every method did or did not contribute results."""
        entries = self.method_coverage or {}
        with open(os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                               "method_coverage.json"), "w") as fh:
            json.dump(entries, fh, indent=2)
        labels = {"nani": "NANI", "divine": "DIVINE", "helm": "HELM",
                  "agglom": "Agglomerative", "equal": "eQual",
                  "mdbirch": "mdBIRCH", "dbscan": "DBSCAN",
                  "gromos": "GROMOS", "gmx_native": "GROMACS native"}
        lines = ["PER-METHOD COVERAGE (why each method produced results or "
                 "not)", "=" * 78, ""]
        for fam, label in labels.items():
            info = entries.get(fam, {})
            status = info.get("status", "not_run")
            counts = info.get("cluster_counts") or []
            lines.append(f"{label:<16} {status:<11} "
                         f"solutions {info.get('solutions', 0):<3} "
                         f"usable {info.get('usable_solutions', 0):<3} "
                         f"clusters {counts}")
            if info.get("reason"):
                lines.append(f"{'':<16} reason: {info['reason']}")
            if info.get("failed_solutions"):
                lines.append(f"{'':<16} failed runs: "
                             f"{info['failed_solutions']}")
        used = [labels.get(f, f) for f, i in entries.items()
                if i.get("status") in ("usable", "degenerate")]
        missing = [labels.get(f, f) for f, i in entries.items()
                   if i.get("status") not in ("usable", "degenerate")]
        lines += ["",
                  f"Methods evaluated : {', '.join(used) or 'none'}",
                  f"Methods not run   : {', '.join(missing) or 'none'}",
                  "",
                  "Each evaluated method contributes its best representative "
                  "pose to the main comparison",
                  "(>= 2 clusters); degenerate partitions are listed in",
                  "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/"
                  "method_representatives.json under 'skipped'."]
        with open(os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                               "METHOD_COVERAGE.txt"), "w") as fh:
            fh.write("\n".join(lines))

    # ------------------------------------------------------------ phase 2
    def _phase_preprocess(self) -> dict:
        log = self.logger
        errors = validate_config(self.cfg)
        if errors:
            raise ValueError("Config errors:\n  " + "\n  ".join(errors))
        pre.validate_inputs(self.cfg, log)
        info = pre.preprocess_trajectory(self.cfg, log)
        self.info = info
        txt = pre.preprocessing_report(info, self.cfg)
        with open(os.path.join(self.out_dir, layout.PREPROCESSING_DIR,
                               "preprocessing_report.txt"), "w") as fh:
            fh.write(txt)
        import yaml
        with open(os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                               "config.yaml"), "w") as fh:
            yaml.safe_dump(self.cfg, fh, sort_keys=False)
        log.info("Preprocessing complete.")
        return info

    # ------------------------------------------------------------ phase 3
    def _phase_features(self) -> dict:
        X = self.info["X"]
        D = feat.build_distances(X, self.info["n_atoms_selected"], self.cfg,
                                 self.logger)["D"]
        feat.save_features(X, D, self.out_dir)
        rmsd = feat.rmsd_series(X, self.info["n_atoms_selected"])
        rgyr = feat.radius_of_gyration(X, self.info["n_atoms_selected"])
        self.dist = {"X": X, "D": D, "rmsd": rmsd, "rgyr": rgyr}
        return self.dist

    # ----------------------------------------------------------- phase 3b
    def _phase_k_selection(self) -> dict:
        """Elbow-based K selection on the prepared MDANCE MSD representation.

        Default: one K for all K-based methods. With
        ``k_selection.per_method: true`` each of NANI / Divine / HELM /
        Agglomerative gets its own elbow scan and therefore its own K.
        """
        ks_dir = os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                              "k_selection")
        report = k_selection.run(
            self.dist["X"], self.dist["D"],
            self.info["n_atoms_selected"], self.cfg, ks_dir, self.logger)
        self.k_report = report
        src = os.path.join(ks_dir, "k_selection_elbow.png")
        dst = os.path.join(self.out_dir, layout.PLOTS_DIR, "K_SELECTION.png")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        self.cfg["clustering"]["selected_k"] = int(report["selected_k"])
        if self.cfg["k_selection"].get("per_method", False):
            per = k_selection.run_per_method(
                self.dist["X"], self.dist["D"],
                self.info["n_atoms_selected"], self.cfg, ks_dir, self.logger)
            self.k_report_per_method = per
            self.cfg["clustering"]["selected_k_map"] = {
                fam: int(v["selected_k"])
                for fam, v in per.get("per_method", {}).items()}
            self.logger.info("Per-method K: %s",
                             self.cfg["clustering"]["selected_k_map"])
        return report

    # ------------------------------------------------------------ phase 4-5
    def _phase_validation(self) -> None:
        log = self.logger
        X, D = self.dist["X"], self.dist["D"]
        metrics = []
        for sol in self.sols:
            metrics.append(val.evaluate_solution(sol, D, X,
                                                 self.info["n_atoms_selected"]))
        ranked = val.rank_solutions(metrics, self.cfg)
        top_n = max(1, int(self.cfg["validation"].get("bootstrap_top_solutions", 5)))
        seed = int(self.cfg["validation"].get("bootstrap_seed", 123))
        for m in ranked[:top_n]:
            sol = next((s for s in self.sols if s.name == m.name), None)
            if sol is None or sol.rerun is None:
                continue
            with utils.timed_step(log, f"bootstrap stability: {m.name}"):
                m.stability = val.bootstrap_stability(sol, X, D, self.cfg, seed)
        self.metrics = metrics
        self.ranked = val.rank_solutions(metrics, self.cfg)
        if not self.ranked:
            raise RuntimeError("No valid clustering solution was produced - "
                               "check the logs for method failures.")
        self.best_metrics = self.ranked[0]
        sol = next((s for s in self.sols if s.name == self.best_metrics.name),
                   None)
        if sol is None:
            raise RuntimeError("Best solution ranking lost its source")
        self.best_labels = np.asarray(sol.labels, dtype=int)
        self._write_clustering_outputs()
        log.info("Optimal solution: %s (k=%d, composite %.2f)",
                 self.best_metrics.name, self.best_metrics.n_clusters,
                 self.best_metrics.composite)

    def _write_clustering_outputs(self) -> None:
        opt_dir = os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                               "clustering", "optimal_clustering")
        os.makedirs(opt_dir, exist_ok=True)
        np.save(os.path.join(opt_dir, "labels.npy"),
                self.best_labels.astype(int))
        m = self.best_metrics
        with open(os.path.join(opt_dir, "model_card.txt"), "w") as fh:
            fh.write(
                f"optimal method  : {m.name}\n"
                f"family          : {m.family}\n"
                f"native MDANCE   : {m.native}\n"
                f"params          : {m.params}\n"
                f"n_clusters      : {m.n_clusters}\n"
                f"composite score : {m.composite}\n\n"
                f"component scores (0-100):\n" +
                "\n".join(f"  {k} = {v}" for k, v in m.component_scores.items()))
        rows = [[mt.name, mt.native, mt.n_clusters, mt.silhouette,
                 mt.davies_bouldin, mt.calinski_harabasz, mt.compactness,
                 mt.separation, mt.diversity, mt.stability, mt.population_max,
                 mt.population_min, mt.n_singletons, mt.time_s, mt.mem_mb,
                 mt.composite, mt.rank] for mt in self.ranked]
        utils.write_csv(os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                                     "solution_metrics.csv"),
                        ["method", "native_MDANCE", "n_clusters", "silhouette",
                         "davies_bouldin", "calinski_harabasz", "compactness",
                         "separation", "diversity", "stability_ARI",
                         "pop_max_frac", "pop_min_frac", "n_singletons",
                         "time_s", "mem_mb", "composite_score", "rank"], rows)
        # native / benchmark / baseline separation subdirectories: only the
        # Daura/GROMOS solutions belong in clustering/gromacs; pipeline-added
        # baselines (Agglomerative, DBSCAN, ...) go to clustering/baselines.
        for sol in self.sols:
            if sol.labels.size == 0:
                continue
            fam = str(sol.family).split("/")[0]
            if sol.native:
                sub = "mdance"
            elif fam in ("gromos", "gromacs"):
                sub = "gromacs"
            else:
                sub = "baselines"
            path = os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                                "clustering", "labels", sub,
                                f"{sol.name.replace('/', '_')}.npy")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, sol.labels.astype(int))

    # ------------------------------------------------------------ phase 7
    def _phase_representatives(self) -> None:
        log = self.logger
        N = self.info["n_atoms_selected"]
        self.best_stats = reps.cluster_centroids_stats(
            self.dist["X"], self.best_labels, N)
        reps_dir = os.path.join(self.out_dir, layout.MEDOID_DIR)
        if self.full_output:
            gm = reps.global_medoid_index(self.dist["X"], N)
            reps.write_frame_pdb(self.info, gm,
                                 os.path.join(reps_dir, "global_medoid.pdb"),
                                 logger=log)
            for cid, st in self.best_stats.items():
                for kind, idx in (("medoid", st["medoid_frame"]),
                                  ("nearest_to_centroid",
                                   st["nearest_to_centroid"])):
                    reps.write_frame_pdb(
                        self.info, idx,
                        os.path.join(reps_dir,
                                     f"cluster{cid + 1:03d}_{kind}.pdb"),
                        logger=log)
        # Population summary of the best solution (main comparison folder)
        rows = []
        largest = max((st["population"] for st in self.best_stats.values()),
                      default=0)
        for cid in sorted(self.best_stats.keys()):
            st = self.best_stats[cid]
            med = self.dist["X"][st["medoid_frame"]]
            nearest = self.dist["X"][st["nearest_to_centroid"]]
            msd = float(((med - nearest) ** 2).sum() / (2.0 * N))
            rows.append([self.best_metrics.name, cid + 1, st["population"],
                         round(st["population"] / len(self.best_labels) * 100, 2),
                         "yes" if st["population"] == largest else "no",
                         info_idx(self.info, st["medoid_frame"], "orig"),
                         info_idx(self.info, st["nearest_to_centroid"], "orig"),
                         round(msd, 6)])
        utils.write_csv(
            os.path.join(self.out_dir, layout.METHOD_DIR,
                         "CLUSTER_POPULATION.csv"),
            ["Method", "Cluster_ID", "Population_Frames",
             "Population_Percent", "Largest_Cluster",
             "Medoid_Original_Frame", "Nearest_To_Centroid_Original_Frame",
             "MSD_Medoid_Nearest"], rows)
        # alternative solutions provide extra candidate frames (Phase 8/9)
        self.alt_stats = []
        for m in self.ranked[1:1 + int(self.cfg["representatives"].get(
                "alternative_solutions", 2))]:
            sol = next((s for s in self.sols if s.name == m.name), None)
            if sol is None or sol.labels.size == 0:
                continue
            st = reps.cluster_centroids_stats(self.dist["X"], sol.labels, N)
            self.alt_stats.append((m, st))
        # candidate_frames PDBs happen after ranking (Phase 8), so all
        # candidates here are medoids of best+alternatives
        log.info("Representatives extracted for %d clusters (%d alternates).",
                 len(self.best_stats), len(self.alt_stats))
        # Every method gets its own representative frames (independent of the
        # winning method) plus a method recommendation of its own.
        self.method_summary = {}
        if self.cfg["representatives"].get("export_all_methods", True):
            self.method_summary = method_reps.export_all_methods(self)
        # Consolidated cluster-population table for every successful method.
        fams = self.method_summary.get("families") or {}
        if fams:
            pop_rows = []
            for name, fam in fams.items():
                largest = max((c["population"] for c in fam["clusters"]),
                              default=0)
                for c in fam["clusters"]:
                    pop_rows.append([
                        name, c["cluster_id"], c["population"],
                        round(c["population_frac"] * 100, 2),
                        "yes" if c["population"] == largest else "no",
                        c["medoid_original_frame"],
                        c["nearest_centroid_original_frame"],
                        c["msd_medoid_to_centroid"]])
            utils.write_csv(
                os.path.join(self.out_dir, layout.METHOD_DIR,
                             "CLUSTER_POPULATION.csv"),
                ["Method", "Cluster_ID", "Population_Frames",
                 "Population_Percent", "Largest_Cluster",
                 "Medoid_Original_Frame",
                 "Nearest_To_Centroid_Original_Frame",
                 "MSD_Medoid_Nearest"], pop_rows)

    # ------------------------------------------------------------ phase 8-9
    def _phase_scoring(self) -> None:
        log = self.logger
        X, D = self.dist["X"], self.dist["D"]
        N = self.info["n_atoms_selected"]
        # Cross-method pose comparison: every method's representative frames
        # (medoid + nearest-to-centroid of every cluster) compete in ONE ranked
        # table, each scored inside its own method's partition.
        self.pose_candidates: list[dict] = []
        if self.cfg["scoring"].get("cross_method_poses", True):
            self.chosen_methods, self.pose_candidates = \
                method_reps.collect_method_candidates(self)
        if not self.pose_candidates:
            # fallback: winner + runner-up solutions only (legacy behaviour)
            cands, tags = [], []
            for cid in sorted(self.best_stats):
                for idx, tag in ((self.best_stats[cid]["medoid_frame"],
                                  f"medoid(c{cid})"),
                                 (self.best_stats[cid]["nearest_to_centroid"],
                                  f"nearest-centroid(c{cid})")):
                    cands.append(idx)
                    tags.append(tag)
            self.pose_candidates = [
                {"frame": int(c), "labels": self.best_labels,
                 "cluster_id": int(self.best_labels[c]),
                 "method": self.best_metrics.name,
                 "family": self.best_metrics.family.split("/")[0],
                 "k": int(self.best_metrics.n_clusters), "kind": "medoid",
                 "tag": tags[i]} for i, c in enumerate(cands)]
        if not self.pose_candidates:
            raise RuntimeError("No candidate representative frames found.")
        log.info("Cross-method pose comparison: %d candidates from %d methods.",
                 len(self.pose_candidates),
                 len({c["method"] for c in self.pose_candidates}))
        rows = scoring.cross_method_metrics(
            self.pose_candidates, X, D, N, self.info, self.cfg)
        for r in rows:
            r["algorithm"] = r.get("source_method") or self.best_metrics.name
            r["metric"] = "MSD (MDANCE)"
            r["optimal_k"] = r.get("source_k")
        rows = scoring.score_candidates(rows, self.cfg)
        self.score_rows = rows
        self._write_cross_method_table(rows)

        sens_rows, self.sens_summary = scoring.sensitivity_analysis(
            rows, self.ranked[1:4], self.cfg, log)
        self.sens_rows = sens_rows

        keys = ["rank", "analysis_frame", "original_frame", "time_ps",
                "cluster_id", "cluster_population", "cluster_pop_frac",
                "distance_to_medoid", "algorithm", "metric", "optimal_k",
                "source_family", "source_kind", "label",
                "representativeness", "population", "stability",
                "quality", "interaction", "conformational",
                "score_representativeness", "score_population",
                "score_stability", "score_quality", "score_interaction",
                "score_conformational", "final_score"]
        with open(os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                               "candidate_frame_scores.csv"), "w",
                  newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            # cluster ids are 0-based internally; all user-facing files are
            # 1-based, matching cluster_NNN_* PDB names and the banner.
            wr.writerows([{**r, "cluster_id": r["cluster_id"] + 1}
                          for r in rows])

        # Every candidate PDB is optional (default: final results only).
        if self.full_output:
            for r in rows:
                idx = r["analysis_frame"]
                fam = (r.get("source_family") or "pose").replace("/", "_")
                reps.write_frame_pdb(
                    self.info, idx,
                    os.path.join(self.out_dir, layout.CANDIDATE_DIR,
                                 f"score{r['final_score']:.2f}_{fam}"
                                 f"_cluster{r['cluster_id'] + 1}"
                                 f"_frame{r['original_frame']}.pdb"),
                    logger=log)
        self._annotate_recommendation(rows)

        import yaml
        eff = dict(self.cfg["scoring"]["weights"])
        if rows and rows[0].get("score_interaction") is None:
            eff.pop("interaction", None)
            tot = sum(eff.values())
            eff = {k: round(v / tot, 6) for k, v in eff.items()}
        with open(os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                               "scoring_weights.yaml"), "w") as fh:
            yaml.safe_dump({
                "effective_weights": eff,
                "raw_weights": self.cfg["scoring"]["weights"],
                "normalization": ("per-metric min-max normalization among "
                                  "candidates -> [0,100]; weighted sum with "
                                  "renormalized weights (missing metrics are "
                                  "excluded, never silently zeroed)"),
            }, fh, sort_keys=False)
        with open(os.path.join(self.out_dir, layout.SENSITIVITY_DIR,
                               "sensitivity_analysis.csv"), "w",
                  newline="") as fh:
            if sens_rows:
                wr = csv.DictWriter(fh, fieldnames=list(sens_rows[0]))
                wr.writeheader()
                wr.writerows(sens_rows)
        utils.write_csv(
            os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                         "scoring_metrics.csv"),
            ["rank", "method", "family", "kind", "candidate_label",
             "analysis_frame", "original_frame", "cluster_id",
             "final_score"] +
            [f"score_{k}" for k in scoring.METRIC_KEYS],
                         [[(r["cluster_id"] + 1 if k == "cluster_id" else r.get(k, ""))
               for k in (["rank", "source_method",
                                      "source_family", "source_kind", "label",
                                      "analysis_frame", "original_frame",
                                      "cluster_id", "final_score"] +
                                     [f"score_{k}" for k in scoring.METRIC_KEYS])]
             for r in rows])

    # --------------------------------------------------- cross-method poses
    def _write_cross_method_table(self, rows: list[dict]) -> None:
        """Ranked cross-method pose table + one best pose per method."""
        log = self.logger
        reps_dir = os.path.join(self.out_dir, layout.INTERMEDIATE_DIR)
        header = ["rank", "method", "family", "kind", "method_k",
                  "cluster_id", "cluster_population", "analysis_frame",
                  "original_frame", "time_ps"] + \
                 [f"score_{k}" for k in scoring.METRIC_KEYS] + ["final_score"]
        table = []
        for r in rows:
            table.append([r["rank"], r.get("source_method", ""),
                          r.get("source_family", ""), r.get("source_kind", ""),
                          r.get("source_k"), r["cluster_id"] + 1,
                          r["cluster_population"], r["analysis_frame"],
                          r["original_frame"], round(r["time_ps"], 3)] +
                         [r.get(f"score_{k}") for k in scoring.METRIC_KEYS] +
                         [r["final_score"]])
        utils.write_csv(os.path.join(reps_dir, "pose_candidates_all.csv"),
                        header, table)

        best_per_family: dict[str, dict] = {}
        for r in rows:
            fam = r.get("source_family") or "?"
            if fam not in best_per_family or \
                    r["final_score"] > best_per_family[fam]["final_score"]:
                best_per_family[fam] = r

        # One best representative PDB per successfully completed method.
        ordered = sorted(best_per_family.items(),
                         key=lambda kv: -kv[1]["final_score"])
        pose_rows = []
        for rank, (fam, r) in enumerate(ordered, 1):
            fname = "".join(ch if (ch.isalnum() or ch in "-_.") else "_"
                            for ch in fam) + ".pdb"
            pdb_path = os.path.join(self.out_dir, layout.METHOD_REPS_DIR,
                                    fname)
            reps.write_frame_pdb(self.info, r["analysis_frame"], pdb_path,
                                 logger=log)
            r["method_rank"] = rank
            r["structure_file"] = layout.rel(self.out_dir, pdb_path)
            pose_rows.append([
                rank, fam, r.get("source_method", ""),
                r["cluster_id"] + 1, r["original_frame"],
                ("nearest-to-centroid" if r.get("source_kind") ==
                 "nearest_centroid" else r.get("source_kind", "medoid")),
                r.get("score_representativeness"), r.get("score_population"),
                r.get("score_stability"), r.get("score_quality"),
                r.get("score_interaction"), r.get("score_conformational"),
                r["final_score"], r["structure_file"]])
        utils.write_csv(
            os.path.join(self.out_dir, layout.METHOD_DIR, "POSE_RANKING.csv"),
            ["Rank", "Method", "Solution", "Cluster_ID", "Original_Frame",
             "Representative_Type", "Representativeness_Score",
             "Population_Score", "Stability_Score", "Quality_Score",
             "Interaction_Score", "Conformational_Score",
             "Final_Composite_Score", "PDB_File"], pose_rows)
        self.pose_rows = pose_rows
        top = rows[0]
        lines = [
            "CROSS-METHOD REPRESENTATIVE POSE COMPARISON",
            "=" * 70, "",
            "Every method contributes the medoid (and nearest-to-centroid) of "
            "every cluster; each pose is scored inside its OWN method "
            "partition.",
            f"Candidates compared      : {len(rows)}",
            f"Methods compared         : {len(best_per_family)}",
            "",
            "Best pose per method (by composite pose score):",
        ]
        for fam, r in sorted(best_per_family.items(),
                             key=lambda kv: -kv[1]["final_score"]):
            lines.append(
                f"  {fam:<14} {r.get('source_method', ''):<34}"
                f" cluster {r['cluster_id'] + 1:<3}"
                f" frame {r['analysis_frame']:<5}"
                f" (orig {r['original_frame']:<5}) score {r['final_score']:6.2f}")
        lines += ["", "OVERALL BEST POSE:",
                  f"  method          : {top.get('source_method', '')}",
                  f"  family          : {top.get('source_family', '')}",
                  f"  pose kind       : {top.get('source_kind', '')}",
                  f"  cluster         : {top['cluster_id'] + 1}"
                  f" (population {top['cluster_population']})",
                  f"  analysis frame  : {top['analysis_frame']}",
                  f"  original frame  : {top['original_frame']}",
                  f"  time            : {top['time_ps'] / 1000:.3f} ns",
                  f"  composite score : {top['final_score']:.2f} / 100",
                  f"  per-method file : {top.get('structure_file', '')}",
                  "  overall pose    : 01_FINAL_RESULTS/BEST_POSE.pdb"]
        with open(os.path.join(reps_dir, "POSE_COMPARISON_REPORT.txt"),
                  "w") as fh:
            fh.write("\n".join(lines))
        self.logger.info("Best cross-method pose: %s cluster %d, score %.2f",
                         top.get("source_method", "?"),
                         top["cluster_id"] + 1, top["final_score"])

    def _annotate_recommendation(self, rows: list[dict]) -> None:
        """State recommended method vs pose winner, and flag disagreement."""
        path = os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                            "method_recommendation.txt")
        top = rows[0]
        rec_name = self.ranked[0].name if self.ranked else "?"
        best_per_family: dict[str, dict] = {}
        for r in rows:
            fam = r.get("source_family") or "?"
            if fam not in best_per_family or \
                    r["final_score"] > best_per_family[fam]["final_score"]:
                best_per_family[fam] = r
        agree = top.get("source_method") == rec_name
        lines = ["",
                 "CROSS-METHOD POSE COMPARISON (independent layer)",
                 "=" * 68,
                 f"poses compared          : {len(rows)}",
                 f"pose winner method      : {top.get('source_method', '')}",
                 f"pose winner family      : {top.get('source_family', '')}",
                 f"pose winner cluster     : {top['cluster_id'] + 1}"
                 f" (population {top['cluster_population']})",
                 f"pose winner frame       : analysis {top['analysis_frame']}"
                 f" / original {top['original_frame']}"
                 f" / {top['time_ps'] / 1000:.3f} ns",
                 f"pose winner score       : {top['final_score']:.2f} / 100",
                 f"recommended method      : {rec_name}"
                 f" (validation rank #{self.ranked[0].rank if self.ranked else '?'})",
                 f"recommendation agrees   : {'YES' if agree else 'NO'}",
                 "",
                 "Best pose score per method:"]
        for fam, r in sorted(best_per_family.items(),
                             key=lambda kv: -kv[1]["final_score"]):
            lines.append(f"  {fam:<14} {r['final_score']:6.2f}"
                         f"  ({r.get('source_method', '')})")
        with open(path, "a") as fh:
            fh.write("\n".join(lines) + "\n")
        with open(os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                               "METHOD_AND_POSE_SUMMARY.txt"), "w") as fh:
            fh.write("\n".join([
                "BEST CLUSTERING METHOD AND ITS SELECTED POSE",
                "=" * 70, "",
                "Two independent answers are reported:",
                "  1. best METHOD by cluster validation (ranking in "
                "02_METHOD_COMPARISON/METHOD_RANKING.csv)",
                "  2. best POSE by composite frame score across ALL method "
                "representatives",
                "",
                f"Best method (validation) : {self.ranked[0].name}",
                f"  cluster validation score: {self.ranked[0].composite}",
                f"  representative structure: "
                "01_FINAL_RESULTS/BEST_METHOD.pdb",
                "",
                f"Best pose (all methods)  : {top.get('source_method', '')}",
                f"  frame                 : analysis "
                f"{top['analysis_frame']} / original "
                f"{top['original_frame']} / "
                f"{top['time_ps'] / 1000:.3f} ns",
                f"  cluster               : {top['cluster_id'] + 1}"
                f" (population {top['cluster_population']})",
                f"  composite pose score  : {top['final_score']:.2f} / 100",
                f"  structure             : "
                f"{top.get('structure_file', '')}",
                f"  same method as above  : {'YES' if agree else 'NO'}",
                "", "Details:",
                "  04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/"
                "pose_candidates_all.csv",
                "  04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/"
                "POSE_COMPARISON_REPORT.txt",
                "  04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/"
                "FINAL_FRAME_SELECTION_REPORT.txt",
            ]))

    # ------------------------------------------------------------ phase 10
    def _phase_plotting(self) -> None:
        log = self.logger
        cfg = self.cfg
        if not cfg["system"].get("plots_enabled", True):
            log.info("plots disabled by config")
            return
        p_dir = os.path.join(self.out_dir, layout.PLOTS_DIR)
        d_dir = os.path.join(self.out_dir, layout.DIAGNOSTIC_PLOTS_DIR)
        s_dir = os.path.join(self.out_dir, layout.SENSITIVITY_DIR)
        labels = self.best_labels
        # --- four primary plots -------------------------------------------
        viz.plot_clustering_comparison(self.metrics, p_dir, log,
                                       name="CLUSTER_COMPARISON.png")
        viz.plot_representative_ranking(self.score_rows, p_dir, log,
                                        name="POSE_RANKING.png")
        if self.bench_results.get("solutions"):
            bs = max(self.bench_results["solutions"],
                     key=lambda s: s.n_clusters if s.n_clusters else -1)
            native = gromacs_bench.native_solution(
                self.bench_results.get("native"))
            viz.plot_gromacs_populations(
                labels, bs.labels, p_dir, log,
                name="GROMACS_COMPARISON.png",
                native_labels=(native.labels if native is not None else None))
        # --- diagnostic plots (advanced, retained not deleted) ------------
        viz.plot_cluster_population(labels, d_dir, log)
        if self.dist["D"] is not None:
            viz.plot_silhouette(self.dist["D"], labels, d_dir, log)
            viz.plot_distance_heatmap(
                self.dist["D"], d_dir, log, None,
                max_frames=cfg["plots"].get("heatmap_max_frames", 300))
        viz.plot_rmsd_distribution(self.dist["rmsd"], d_dir, log)
        viz.scatter_embedding(self.dist["X"], labels, d_dir, log, "PCA")
        if cfg["plots"].get("tsne_enabled"):
            try:
                viz.scatter_embedding(self.dist["X"], labels, d_dir, log,
                                      "t-SNE", "tsne_clustering.png")
            except Exception as exc:
                log.warning("t-SNE plot skipped: %s", exc)
        if getattr(self, "sens_rows", None):
            viz.plot_scoring_sensitivity(self.sens_rows, self.score_rows,
                                         s_dir, log)

    # ------------------------------------------------------------ phase 11-12
    def _write_best_structures(self, best_row: dict) -> None:
        """BEST_POSE.pdb (highest scored) and BEST_METHOD.pdb (recommended)."""
        log = self.logger
        best_path = os.path.join(self.out_dir, layout.FINAL_DIR, "BEST_POSE.pdb")
        reps.write_frame_pdb(self.info, best_row["analysis_frame"], best_path,
                             logger=log)
        best_row["structure_file"] = layout.rel(self.out_dir, best_path)

        rec = (self.method_summary or {}).get("recommendation")
        method_path = os.path.join(self.out_dir, layout.FINAL_DIR,
                                   "BEST_METHOD.pdb")
        if rec and os.path.isfile(method_path):
            return
        # export_all_methods disabled: still provide the recommended-method
        # pose from the top validation-ranked solution's largest cluster.
        top = self.ranked[0]
        sol = next((s for s in self.sols if s.name == top.name), None)
        if sol is None or sol.labels.size == 0:
            return
        stats = reps.cluster_centroids_stats(
            self.dist["X"], sol.labels, self.info["n_atoms_selected"])
        if not stats:
            return
        cid = max(stats, key=lambda c: stats[c]["population"])
        reps.write_frame_pdb(self.info, stats[cid]["medoid_frame"],
                             method_path, logger=log)

    @staticmethod
    def _family_status(info: dict) -> str:
        """Map coverage to the three user-facing states."""
        status = info.get("status", "not_run")
        reason = (info.get("reason") or "").lower()
        if status == "degenerate":
            return "FAILED"
        if status == "deferred":
            return "SKIPPED"
        if status == "not_run":
            skipped = any(token in reason for token in (
                "disabled", "not enabled", "not requested", "deferred",
                "enabled: false", "binary not found"))
            return "SKIPPED" if skipped else "FAILED"
        return "SKIPPED"

    def _method_ranking_rows(self) -> list[dict]:
        """One consolidated row per method family, in ranking order."""
        labels = {"nani": "NANI", "divine": "DIVINE", "helm": "HELM",
                  "agglom": "Agglomerative", "equal": "eQual",
                  "mdbirch": "mdBIRCH", "dbscan": "DBSCAN",
                  "gromos": "GROMOS", "gmx_native": "GROMACS native"}
        chosen = self.chosen_methods
        if not chosen:
            chosen = method_reps._best_per_family(
                self.sols, self.ranked,
                self.cfg["clustering"].get("selected_k"))
        coverage = self.method_coverage or {}
        families = list(labels)
        for fam in list(coverage) + list(chosen):
            if fam not in families:
                families.append(fam)
        rows = []
        for fam in families:
            info = coverage.get(fam, {})
            pair = chosen.get(fam)
            row = {"method": labels.get(fam, fam.upper()),
                   "status": "", "solution": "", "n_clusters": "",
                   "population_frames": "", "population_percent": "",
                   "silhouette": "", "davies_bouldin": "",
                   "calinski_harabasz": "", "bootstrap_ari": "",
                   "overall_score": "", "rank": "", "notes": ""}
            if pair is not None:
                metric, sol = pair
                counts = np.unique(sol.labels[sol.labels >= 0],
                                   return_counts=True)[1]
                largest = int(counts.max()) if counts.size else 0
                row.update({
                    "status": "OK", "solution": sol.name,
                    "n_clusters": int(sol.n_clusters),
                    "population_frames": largest,
                    "population_percent": round(
                        largest / max(len(sol.labels), 1) * 100, 2),
                    "silhouette": (metric.silhouette
                                   if metric is not None else ""),
                    "davies_bouldin": (metric.davies_bouldin
                                       if metric is not None else ""),
                    "calinski_harabasz": (metric.calinski_harabasz
                                          if metric is not None else ""),
                    "bootstrap_ari": (metric.stability
                                      if metric is not None else ""),
                    "overall_score": (metric.composite
                                      if metric is not None else ""),
                    "rank": "",
                    "notes": (f"best validation-ranked variant (validation "
                              f"rank #{metric.rank})" if metric is not None
                              else ("native gmx cluster (benchmark tier; not "
                                    "in the MDANCE validation ranking)"
                                    if fam == "gmx_native" else
                                    "closest usable variant (not "
                                    "validation-ranked)")),
                })
            else:
                row["status"] = self._family_status(info)
                row["notes"] = info.get("reason") or "no usable solution"
            rows.append(row)
        ok = [r for r in rows if r["status"] == "OK"]
        ok.sort(key=lambda r: (-float(r["overall_score"])
                               if r["overall_score"] != "" else float("inf"),
                               r["method"]))
        for i, r in enumerate(ok, 1):
            r["rank"] = i
        rest = [r for r in rows if r["status"] != "OK"]
        rest.sort(key=lambda r: (r["method"], r["status"]))
        return ok + rest

    def _gromacs_status(self) -> tuple[str, str, str]:
        """(native, MDANCE GROMOS, comparison) status strings."""
        if not self.cfg["benchmark"].get("enabled", True):
            return ("not attempted (benchmark disabled)",
                    "not run (benchmark disabled)", "not available")
        native = (self.bench_results or {}).get("native")
        sols = (self.bench_results or {}).get("solutions", []) or []
        if native is None:
            native_txt = ("not attempted (disabled by config)"
                          if self.cfg["benchmark"].get("native_gmx") is False
                          else "not attempted (no gmx binary found)")
        elif native.get("success"):
            native_txt = (f"SUCCESS: {native.get('n_clusters')} clusters, "
                          f"cutoff {native.get('cutoff_nm'):.6f} nm, "
                          f"{native.get('n_atoms')} selected atoms")
        else:
            native_txt = (f"FAILED ({native.get('error') or 'gmx error'}); "
                          "NOT counted as a successful benchmark")
        native_ok = bool(native and native.get("success"))
        if sols:
            gromos_txt = (f"{len(sols)} cutoff partition(s) produced by the "
                          "MDANCE implementation")
        else:
            gromos_txt = "no usable partition produced"
        if native_ok and sols:
            cmp_txt = ("quantitative for BOTH native gmx and MDANCE GROMOS "
                       "in 02_METHOD_COMPARISON/GROMACS_COMPARISON.csv")
        elif native_ok:
            cmp_txt = ("native gmx compared quantitatively; MDANCE GROMOS "
                       "tier unavailable")
        elif sols:
            cmp_txt = ("MDANCE GROMOS compared quantitatively; native gmx "
                       "did not produce a partition")
        else:
            cmp_txt = "not available (no GROMACS partition from either tier)"
        return native_txt, gromos_txt, cmp_txt

    def _final_report_lines(self, best_row: dict, method_rows: list[dict],
                            gromacs: tuple[str, str, str]) -> list[str]:
        info = self.info
        kr = self.k_report or {}
        rec = (self.method_summary or {}).get("recommendation") or {}
        total = info.get("n_frames_total", len(info["frame_indices"]))
        analysed = len(info["frame_indices"])
        elbow = kr.get("elbow_k")
        selected = kr.get("selected_k",
                          self.cfg["clustering"].get("selected_k", "n/a"))
        metrics = {int(r["k"]): r for r in kr.get("metrics", [])}
        selrow = metrics.get(int(selected)) if selected != "n/a" else None
        elbowrow = metrics.get(int(elbow)) if elbow is not None else None

        lines = [
            "MDANCE UNIFIED CLUSTERING - FINAL REPORT",
            "=" * 70, "",
            "WHAT WAS ANALYSED", "-" * 70,
            f"Trajectory file          : {self.cfg['preprocess'].get('xtc')}",
            f"Atom selection           : {info['selection']}",
            f"Total trajectory frames  : {total}",
            f"Frames actually analysed : {analysed}",
            f"Frame filtering          : stride "
            f"{self.cfg['preprocess'].get('stride', 1)}, skipped equilibration "
            f"{self.cfg['preprocess'].get('skip_equil_frames', 0)} frame(s)",
            f"Analysed time window     : {info['times'][0]/1000:.3f} - "
            f"{info['times'][-1]/1000:.3f} ns",
            f"Ligand detected          : "
            f"{'yes' if info['ligand_found'] else 'no'}",
            "",
            "HOW MANY CLUSTERS (K)", "-" * 70,
            f"Elbow K (Kneedle/geometric) : {elbow if elbow is not None else 'none detected'}",
            f"Selected K used by pipeline : {selected}",
        ]
        if selrow:
            lines.append(
                f"Selected K quality         : silhouette "
                f"{utils.fmt_float(selrow.get('silhouette'), 4)}, "
                f"Davies-Bouldin "
                f"{utils.fmt_float(selrow.get('davies_bouldin'), 4)}, "
                f"Calinski-Harabasz "
                f"{utils.fmt_float(selrow.get('calinski_harabasz'), 4)}, "
                f"bootstrap ARI "
                f"{utils.fmt_float(selrow.get('stability'), 4)}")
        if elbow is None:
            why = ("The elbow scan found no clear knee, so the pipeline did "
                   "not force an arbitrary K. It compared every scanned K "
                   "using silhouette, Davies-Bouldin, Calinski-Harabasz and "
                   "bootstrap ARI, then chose K=" + str(selected) + ".")
        elif int(elbow) == int(selected):
            why = ("The detected elbow K and the selected K agree. The elbow "
                   "was confirmed by the supporting validation metrics "
                   "(silhouette, Davies-Bouldin, Calinski-Harabasz and "
                   "bootstrap ARI).")
        else:
            why = (
                f"The elbow was detected at K={elbow}, but the pipeline "
                f"does not assume the elbow must be used. It compares K="
                f"{elbow} with its immediate neighbours (K-1 and K+1) and "
                f"picks the best average rank across silhouette, "
                f"Davies-Bouldin, Calinski-Harabasz and bootstrap ARI. For "
                f"this trajectory K={selected} ranked best, so selected K="
                f"{selected} differs from elbow K={elbow}.")
            if elbowrow and selrow:
                why += (f" At K={elbow} the silhouette was "
                        f"{utils.fmt_float(elbowrow.get('silhouette'), 4)} "
                        f"and at K={selected} it was "
                        f"{utils.fmt_float(selrow.get('silhouette'), 4)}; "
                        "the combined rank, not the elbow alone, decided.")
        lines += [f"Why this K was selected    : {why}", "",
                  "CLUSTERS PRODUCED BY EACH METHOD", "-" * 70,
                  f"{'Method':<18}{'Status':<10}{'Clusters':<10}Notes"]
        for r in method_rows:
            lines.append(
                f"{r['method']:<18}{r['status']:<10}"
                f"{str(r['n_clusters'] or '-'):<10}{r['notes']}")
        lines += [
            "",
            "BEST CLUSTERING METHOD (cluster validation)", "-" * 70,
            f"Method                    : {self.ranked[0].name}",
            f"Overall clustering score   : {self.ranked[0].composite} / 100",
            f"Clusters                   : {self.ranked[0].n_clusters}",
            f"Recommended representative : 01_FINAL_RESULTS/BEST_METHOD.pdb",
        ]
        if rec:
            lines.append(f"  (cluster {rec['cluster_id']}, original frame "
                         f"{rec['original_frame']}, "
                         f"{rec['time_ps']/1000:.3f} ns)")
        lines += [
            "",
            "BEST REPRESENTATIVE POSE (highest composite frame score)",
            "-" * 70,
            f"Winning method / family   : "
            f"{best_row.get('source_method', '')} / "
            f"{best_row.get('source_family', '')}",
            f"Representative type       : "
            f"{best_row.get('source_kind', 'medoid')}",
            f"Best cluster ID           : {best_row['cluster_id'] + 1}",
            f"Cluster population        : {best_row['cluster_population']} "
            f"frames ({best_row['cluster_pop_frac']*100:.2f}%)",
            f"Original trajectory frame : {best_row['original_frame']}",
            f"Analysis frame            : {best_row['analysis_frame']}",
            f"Simulation time           : {best_row['time_ps']/1000:.3f} ns",
            f"Final composite score     : {best_row['final_score']:.2f} / 100",
            "Best PDB file             : 01_FINAL_RESULTS/BEST_POSE.pdb",
            "",
            "METHODS THAT FAILED OR WERE SKIPPED", "-" * 70,
        ]
        bad = [r for r in method_rows if r["status"] != "OK"]
        if bad:
            for r in bad:
                lines.append(f"{r['method']:<18}{r['status']:<10}"
                             f"{r['notes']}")
        else:
            lines.append("None - every enabled method produced a usable partition.")
        native_txt, gromos_txt, cmp_txt = gromacs
        lines += [
            "",
            "GROMACS COMPARISON STATUS", "-" * 70,
            f"Native gmx cluster        : {native_txt}",
            f"MDANCE GROMOS implementation: {gromos_txt}",
            f"MDANCE vs GROMACS         : {cmp_txt}",
            "Note: a failed native gmx run is reported as FAILED and is never "
            "treated as a successful benchmark.",
            "",
            "WHERE TO LOOK NEXT", "-" * 70,
            "1. 01_FINAL_RESULTS/FINAL_REPORT.txt  - this file",
            "2. 01_FINAL_RESULTS/BEST_POSE.pdb    - use this structure",
            "3. 01_FINAL_RESULTS/BEST_METHOD.pdb  - recommended-method pose",
            "4. 02_METHOD_COMPARISON/METHOD_RANKING.csv - why a method won",
            "5. 02_METHOD_COMPARISON/POSE_RANKING.csv   - all method winners",
            "6. 03_PLOTS/                         - the four key figures",
            "7. 04_ADVANCED_ANALYSIS/             - diagnostics and optional "
            "detail",
            "",
        ]
        return lines

    def _phase_reports(self) -> None:
        log = self.logger
        self._write_method_coverage()
        best_row = self.score_rows[0]
        self._write_best_structures(best_row)
        method_rows = self._method_ranking_rows()
        self.method_rows = method_rows
        utils.write_csv(
            os.path.join(self.out_dir, layout.METHOD_DIR,
                         "METHOD_RANKING.csv"),
            ["Rank", "Method", "Status", "Solution", "N_Clusters",
             "Cluster_Population_Frames", "Cluster_Population_Percent",
             "Silhouette", "Davies_Bouldin", "Calinski_Harabasz",
             "Bootstrap_ARI", "Overall_Clustering_Score", "Notes"],
            [[r["rank"], r["method"], r["status"], r["solution"],
              r["n_clusters"], r["population_frames"],
              r["population_percent"], r["silhouette"],
              r["davies_bouldin"], r["calinski_harabasz"],
              r["bootstrap_ari"], r["overall_score"], r["notes"]]
             for r in method_rows])

        gromacs = self._gromacs_status()
        self.gromacs_status = gromacs
        report_lines = self._final_report_lines(best_row, method_rows, gromacs)
        with open(os.path.join(self.out_dir, layout.FINAL_DIR,
                               "FINAL_REPORT.txt"), "w") as fh:
            fh.write("\n".join(report_lines) + "\n")
        summary = [
            ("Total trajectory frames", self.info.get(
                "n_frames_total", len(self.info["frame_indices"]))),
            ("Frames analysed", len(self.info["frame_indices"])),
            ("Elbow K", (self.k_report or {}).get("elbow_k", "not detected")),
            ("Selected K", (self.k_report or {}).get(
                "selected_k", self.cfg["clustering"].get("selected_k"))),
            ("Best clustering method", self.ranked[0].name),
            ("Best clustering score", self.ranked[0].composite),
            ("Best pose method", best_row.get("source_method", "")),
            ("Best pose family", best_row.get("source_family", "")),
            ("Best cluster ID", best_row["cluster_id"] + 1),
            ("Best pose original frame", best_row["original_frame"]),
            ("Best pose time (ns)", round(best_row["time_ps"] / 1000, 3)),
            ("Best pose score (0-100)", best_row["final_score"]),
            ("Best PDB file", "01_FINAL_RESULTS/BEST_POSE.pdb"),
            ("Recommended method PDB", "01_FINAL_RESULTS/BEST_METHOD.pdb"),
            ("Failed/skipped methods", "; ".join(
                f"{r['method']}={r['status']}" for r in method_rows
                if r["status"] != "OK") or "none"),
            ("GROMACS native status", gromacs[0]),
            ("GROMACS comparison", gromacs[2]),
        ]
        utils.write_csv(os.path.join(self.out_dir, layout.FINAL_DIR,
                                     "FINAL_SUMMARY.csv"),
                        ["Item", "Value"], summary)

        # Full solution-level summary stays available for reproducibility.
        rows = [[s.name, s.native, s.n_clusters, s.time_s, s.mem_mb,
                 (s.error or "OK")[:80]] for s in self.sols]
        utils.write_csv(os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                                     "solution_summary.csv"),
                        ["method", "native_MDANCE", "n_clusters", "time_s",
                         "mem_mb", "status"], rows)
        with open(os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                               "solution_summary.json"), "w") as fh:
            json.dump([dict(
                name=s.name, native=s.native, n_clusters=s.n_clusters,
                time_s=round(s.time_s, 3), mem_mb=round(s.mem_mb, 2),
                params=s.params, error=(s.error or None))
                for s in self.sols], fh, indent=2, default=str)

        # Write the human-readable layout guide once per run.
        with open(os.path.join(self.out_dir, "README.txt"), "w") as fh:
            fh.write(layout_readme())
        explanations = [
            ("Representativeness", "relative position of the candidate inside "
             "its cluster toward the cluster medoid (1 = medoid itself)"),
            ("Population", "cluster population fraction (min-max normalised "
             "across candidates); populated conformational states are "
             "rewarded, and a one-frame cluster is capped so it cannot win "
             "on population alone"),
            ("Stability", "0.5 * temporal consistency (decay of the mean MSD "
             "to the +/-5 surrounding frames) + 0.5 * typicality of the "
             "frame's RMSD-to-reference inside its cluster"),
            ("Quality", "0.5 * radius-of-gyration typicality + 0.5 * "
             "coordinate finiteness (NaN/inf -> 0)"),
            ("Interaction", "fraction of persistent (>=50% frame occupancy) "
             "heavy-atom protein-ligand contacts retained (documented "
             "approximation; omitted without a ligand)"),
            ("Conformational", "0.7 * MDANCE complementary-similarity "
             "centrality (argmax=medoid semantics) + 0.3 * non-outlier flag"),
        ]
        banner = reports.final_frame_report(best_row, self.score_rows[1:],
                                            explanations, None, self.cfg)
        with open(os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                               "FINAL_FRAME_SELECTION_REPORT.txt"), "w") as fh:
            fh.write(banner)
        print(banner)
        log.info("Final frame: analysis %d / original %d, score %.2f",
                 best_row["analysis_frame"], best_row["original_frame"],
                 best_row["final_score"])

        bench_table, bench_report = gromacs_bench.benchmark_table(
            self.ranked[0], self.bench_results, self.sols, self.cfg,
            D=self.dist["D"])
        utils.write_csv(os.path.join(self.out_dir, layout.METHOD_DIR,
                                     "GROMACS_COMPARISON.csv"),
                        gromacs_bench.BENCH_HEADER, bench_table)
        with open(os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                               "benchmarking_report.txt"), "w") as fh:
            fh.write("\n".join(bench_report))
        self._complete_report()

    def _complete_report(self) -> None:
        cfg = self.cfg
        info = self.info
        best_row = self.score_rows[0]
        lines = [
            "MDANCE UNIFIED PIPELINE - COMPLETE ANALYSIS REPORT",
            "=" * 70, "",
            f"Trajectory            : {cfg['preprocess'].get('xtc')}",
            f"Atom selection        : {info['selection']}",
            f"Analysis frames       : {len(info['frame_indices'])}",
            f"Time window           : {info['times'][0]:.1f} - "
            f"{info['times'][-1]:.1f} ps",
            f"Ligand present        : {info['ligand_found']}", "",
            "MDANCE coverage (verified API: docs/verified_mdance_api.md)",
            "  native methods used : KmeansNANI (per-initializer), Divine,",
            "                        ExtendedQuality/eQual, HELM (merge), mdBirch",
            "  excluded with reason: Shine & PRISM (multi-trajectory pathway",
            "                        clusterers), extended-similarity metrics",
            "                        (fingerprint input required; coordinates",
            "                        use MSD), Prime rep_frames (different data",
            "                        workflow - documented)",
            "", "Automatic K selection (Elbow, on the MDANCE MSD matrix):",
        ]
        kr = self.k_report or {}
        if kr:
            lines += [
                f"  scan range         : K = {kr['k_min']}..{kr['k_max']}"
                f"  ({kr['scan_algorithm']})",
                f"  elbow              : {kr['elbow_k']}"
                f"  (algorithm={kr['elbow_algorithm']},"
                f" ambiguous={kr['elbow_ambiguous']})",
                f"  selected K         : {kr['selected_k']}",
                "  dispersion         : " + kr["dispersion_definition"],
                "  used by            : NANI, Divine, HELM, Agglomerative",
                "  not applied to     : eQual, mdBIRCH, DBSCAN, GROMOS"
                " (native thresholds)",
                "  results            : 03_PLOTS/K_SELECTION.png; details in "
                "04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/k_selection/",
            ]
        else:
            lines.append("  disabled (k_selection.enabled = false)")
        lines += ["", "Method execution statistics (all solutions run):"]
        for s in self.sols:
            lines.append(f"  {s.name:<44} {'native' if s.native else 'added':<7}"
                         f" k={s.n_clusters:>3} {s.time_s:7.2f}s "
                         f"{s.mem_mb:7.1f}MB "
                         + (str(s.error)[:46] if s.error else "ok"))
        lines += ["", "Validation ranking (top 10, weights documented in config):"]
        for m in self.ranked[:10]:
            lines.append(f"  #{m.rank:>2} {m.name:<44} composite={m.composite}"
                         f" silhouette={m.silhouette} db={m.davies_bouldin}"
                         f" stab={m.stability}")
        rec = (self.method_summary or {}).get("recommendation")
        lines += ["", "RECOMMENDED CLUSTERING METHOD (independent of the frame "
                  "score):"]
        if rec:
            lines += [
                f"  method            : {rec['method']} (rank #{rec['validation_rank']})",
                f"  clusters          : {rec['n_clusters']}",
                f"  representative    : cluster {rec['cluster_id']}"
                f" medoid, analysis frame {rec['analysis_frame']}"
                f" (original {rec['original_frame']}, {rec['time_ps']} ps)",
                f"  structure         : {rec['structure']}",
                "  details           : 04_ADVANCED_ANALYSIS/"
                "INTERMEDIATE_RESULTS/method_recommendation.txt",
            ]
        else:
            lines.append("  none")
        fams = (self.method_summary or {}).get("families") or {}
        if fams:
            lines += ["", "Per-method representative pose "
                      "(02_METHOD_COMPARISON/METHOD_REPRESENTATIVES/):"]
            for name, fam in fams.items():
                lines.append(f"  {name:<14} {fam['solution']:<34}"
                             f" k={fam['n_clusters']:<3}"
                             f" rank=#{fam['validation_rank']}")
        lines += ["",
                  "FINAL SELECTION (rationale: 04_ADVANCED_ANALYSIS/"
                  "INTERMEDIATE_RESULTS/FINAL_FRAME_SELECTION_REPORT.txt)",
                  f"  analysis frame  : {best_row['analysis_frame']}",
                  f"  original frame  : {best_row['original_frame']}",
                  f"  time (ns)       : {best_row['time_ps']/1000:.3f}",
                  f"  cluster         : {best_row['cluster_id'] + 1}",
                  f"  final score     : {best_row['final_score']}",
                  "  sensitivity     : " + json.dumps(
                      self.sens_summary.get("win_counts", {})),
                  ]
        with open(os.path.join(self.out_dir, layout.INTERMEDIATE_DIR,
                               "complete_analysis_report.txt"), "w") as fh:
            fh.write("\n".join(lines))


def layout_readme() -> str:
    """Short, human-readable guide written at the output root."""
    return "\n".join([
        "MDANCE UNIFIED CLUSTERING - OUTPUT GUIDE",
        "=" * 60,
        "",
        "START HERE: 01_FINAL_RESULTS/FINAL_REPORT.txt",
        "",
        "01_FINAL_RESULTS/       the answer (report, summary, best pose)",
        "    BEST_POSE.pdb       highest-scoring representative pose",
        "    BEST_METHOD.pdb     pose from the best-ranked clustering method",
        "    FINAL_SUMMARY.csv   one-line key numbers",
        "    FINAL_REPORT.txt    plain-language explanation",
        "",
        "02_METHOD_COMPARISON/   method and pose comparison",
        "    METHOD_RANKING.csv  one row per method, incl. FAILED/SKIPPED",
        "    POSE_RANKING.csv    best pose per successful method",
        "    CLUSTER_POPULATION.csv",
        "    GROMACS_COMPARISON.csv",
        "    METHOD_REPRESENTATIVES/<METHOD>.pdb",
        "",
        "03_PLOTS/               four primary figures only",
        "    K_SELECTION.png, CLUSTER_COMPARISON.png, POSE_RANKING.png,",
        "    GROMACS_COMPARISON.png",
        "",
        "04_ADVANCED_ANALYSIS/   diagnostics and optional detail",
        "    CANDIDATE_FRAMES/   every scored candidate (--full-output only)",
        "    CLUSTER_MEDOIDS/    per-cluster medoids (--full-output only)",
        "    SENSITIVITY_ANALYSIS/",
        "    INTERMEDIATE_RESULTS/  K scan, labels, raw metrics, reports",
        "    DIAGNOSTIC_PLOTS/   non-primary plots (kept, not deleted)",
        "",
        "05_PREPROCESSING/       processed trajectory + reference structure",
        "    PROCESSED_TRAJECTORY/processed.xtc, processed_pbc.xtc",
        "    REFERENCE_STRUCTURE/reference.gro",
        "    preprocessing_report.txt",
        "",
        "Use --full-output to add every candidate/cluster PDB under",
        "04_ADVANCED_ANALYSIS; the default tree contains final results only.",
        "",
        "WHERE OLD FILES WENT",
        "    best pose / recommended pose -> 01_FINAL_RESULTS/BEST_*.pdb",
        "    final banner/report          -> 01_FINAL_RESULTS/FINAL_REPORT.txt",
        "    clustering_summary.csv       -> 02_METHOD_COMPARISON/"
        "METHOD_RANKING.csv",
        "    pose_comparison.csv          -> 02_METHOD_COMPARISON/"
        "POSE_RANKING.csv (summary) + 04_ADVANCED_ANALYSIS/"
        "INTERMEDIATE_RESULTS/pose_candidates_all.csv (all rows)",
        "    mdance_vs_gromacs.csv        -> 02_METHOD_COMPARISON/"
        "GROMACS_COMPARISON.csv",
        "    k_selection_elbow.png        -> 03_PLOTS/K_SELECTION.png",
        "    other plots                  -> 04_ADVANCED_ANALYSIS/"
        "DIAGNOSTIC_PLOTS/",
        "    candidate/per-cluster PDBs   -> 04_ADVANCED_ANALYSIS/ "
        "(--full-output)",
        "",
    ])


def info_idx(info, analysis_idx: int, which: str):
    if which == "orig":
        return int(info["frame_indices"][analysis_idx])
    return round(float(info["times"][analysis_idx]), 3)
