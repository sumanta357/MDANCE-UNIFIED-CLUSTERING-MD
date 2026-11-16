"""Unified pipeline orchestration (Phases 2-12). Pipeline-added."""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from mdance_pipeline import (clustering, features as feat, gromacs_bench,
                             k_selection, method_representatives as method_reps,
                             preprocess as pre, reports,
                             representatives as reps, scoring,
                             validation as val, utils,
                             visualization as viz)
from mdance_pipeline.config import validate_config


class Pipeline:
    """Runs every phase; state kept on self, data on cwd as required."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out_dir = cfg["system"]["output_dir"]
        self.logger = utils.get_logger(
            os.path.join(self.out_dir, "logs", "analysis.log"),
            level=cfg["system"].get("log_level", "INFO"))
        utils.ensure_dirs(self.out_dir, [
            "config", "preprocessing", "features", "clustering",
            "clustering/mdance", "clustering/gromacs",
            "clustering/baselines",
            "clustering/optimal_clustering",
            "representatives",
            "representatives/representative_frames",
            "representatives/cluster_representatives",
            "representatives/candidate_frames", "benchmarking", "scoring",
            "plots", "reports", "logs"])
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

    # ------------------------------------------------------------- entry
    def run(self) -> dict:
        log = self.logger
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
        gdir = os.path.join(self.out_dir, "clustering", "gromos")
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

    def _write_method_coverage(self) -> None:
        """Persist why every method did or did not contribute results."""
        entries = self.method_coverage or {}
        with open(os.path.join(self.out_dir, "clustering",
                               "method_coverage.json"), "w") as fh:
            json.dump(entries, fh, indent=2)
        labels = {"nani": "NANI", "divine": "DIVINE", "helm": "HELM",
                  "agglom": "Agglomerative", "equal": "eQual",
                  "mdbirch": "mdBIRCH", "dbscan": "DBSCAN",
                  "gromos": "GROMOS"}
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
                  "Each evaluated method exports representative frames for "
                  "every cluster",
                  "it produced (>= 2 clusters); degenerate partitions are "
                  "listed in",
                  "representatives/method_representatives.json under ",
                  "'skipped'."]
        with open(os.path.join(self.out_dir, "reports",
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
        with open(os.path.join(self.out_dir, "preprocessing",
                               "preprocessing_report.txt"), "w") as fh:
            fh.write(txt)
        import yaml
        with open(os.path.join(self.out_dir, "config", "config.yaml"), "w") as fh:
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
        ks_dir = os.path.join(self.out_dir, "clustering", "k_selection")
        report = k_selection.run(
            self.dist["X"], self.dist["D"],
            self.info["n_atoms_selected"], self.cfg, ks_dir, self.logger)
        self.k_report = report
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
        np.save(os.path.join(self.out_dir, "clustering",
                             "optimal_clustering", "labels.npy"),
                self.best_labels.astype(int))
        m = self.best_metrics
        with open(os.path.join(self.out_dir, "clustering",
                               "optimal_clustering", "model_card.txt"),
                  "w") as fh:
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
        utils.write_csv(os.path.join(self.out_dir, "clustering",
                                     "clustering_summary.csv"),
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
            path = os.path.join(self.out_dir, "clustering", sub,
                                f"{sol.name.replace('/', '_')}.npy")
            np.save(path, sol.labels.astype(int))

    # ------------------------------------------------------------ phase 7
    def _phase_representatives(self) -> None:
        log = self.logger
        N = self.info["n_atoms_selected"]
        self.best_stats = reps.cluster_centroids_stats(
            self.dist["X"], self.best_labels, N)
        reps_dir = os.path.join(self.out_dir, "representatives")
        gm = reps.global_medoid_index(self.dist["X"], N)
        reps.write_frame_pdb(self.info, gm,
                             os.path.join(reps_dir, "global_medoid.pdb"),
                             logger=log)
        for cid, st in self.best_stats.items():
            for kind, idx in (("medoid", st["medoid_frame"]),
                              ("nearest_to_centroid", st["nearest_to_centroid"])):
                reps.write_frame_pdb(
                    self.info, idx,
                    os.path.join(reps_dir, "cluster_representatives",
                                 f"cluster{cid + 1:03d}_{kind}.pdb"), logger=log)
        # cluster_representatives.csv
        rows = []
        for cid in sorted(self.best_stats.keys()):
            st = self.best_stats[cid]
            med = self.dist["X"][st["medoid_frame"]]
            nearest = self.dist["X"][st["nearest_to_centroid"]]
            msd = float(((med - nearest) ** 2).sum() / (2.0 * N))
            rows.append([cid + 1, st["population"],
                         st["population"] / len(self.best_labels),
                         info_idx(self.info, st["medoid_frame"], "orig"),
                         info_idx(self.info, st["medoid_frame"], "t"),
                         info_idx(self.info, st["nearest_to_centroid"], "orig"),
                         info_idx(self.info, st["nearest_to_centroid"], "t"),
                         msd])
        utils.write_csv(
            os.path.join(reps_dir, "cluster_representatives.csv"),
            ["cluster_id", "population", "population_frac",
             "medoid_original_frame", "medoid_time_ps",
             "nearest_to_centroid_original_frame", "nearest_to_centroid_time_ps",
             "msd_between_medoid_and_nearest"], rows)
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
            _, self.pose_candidates = method_reps.collect_method_candidates(
                self)
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
        with open(os.path.join(self.out_dir, "representatives",
                               "candidate_frame_scores.csv"), "w",
                  newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            # cluster ids are 0-based internally; all user-facing files are
            # 1-based, matching cluster_NNN_* PDB names and the banner.
            wr.writerows([{**r, "cluster_id": r["cluster_id"] + 1}
                          for r in rows])

        # candidate_frame PDBs for every candidate
        reps_dir = os.path.join(self.out_dir, "representatives")
        for r in rows:
            idx = r["analysis_frame"]
            fam = (r.get("source_family") or "pose").replace("/", "_")
            reps.write_frame_pdb(
                self.info, idx,
                os.path.join(reps_dir, "candidate_frames",
                             f"score{r['final_score']:.2f}_{fam}"
                             f"_cluster{r['cluster_id'] + 1}"
                             f"_frame{r['original_frame']}.pdb"), logger=log)
        self._annotate_recommendation(rows)

        import yaml
        eff = dict(self.cfg["scoring"]["weights"])
        if rows and rows[0].get("score_interaction") is None:
            eff.pop("interaction", None)
            tot = sum(eff.values())
            eff = {k: round(v / tot, 6) for k, v in eff.items()}
        with open(os.path.join(self.out_dir, "scoring",
                               "scoring_weights.yaml"), "w") as fh:
            yaml.safe_dump({
                "effective_weights": eff,
                "raw_weights": self.cfg["scoring"]["weights"],
                "normalization": ("per-metric min-max normalization among "
                                  "candidates -> [0,100]; weighted sum with "
                                  "renormalized weights (missing metrics are "
                                  "excluded, never silently zeroed)"),
            }, fh, sort_keys=False)
        with open(os.path.join(self.out_dir, "scoring",
                               "sensitivity_analysis.csv"), "w",
                  newline="") as fh:
            if sens_rows:
                wr = csv.DictWriter(fh, fieldnames=list(sens_rows[0]))
                wr.writeheader()
                wr.writerows(sens_rows)
        utils.write_csv(
            os.path.join(self.out_dir, "scoring", "scoring_metrics.csv"),
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
        """Ranked cross-method pose table + per-method best pose."""
        reps_dir = os.path.join(self.out_dir, "representatives")
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
        utils.write_csv(os.path.join(reps_dir, "pose_comparison.csv"), header,
                        table)

        best_per_family: dict[str, dict] = {}
        for r in rows:
            fam = r.get("source_family") or "?"
            if fam not in best_per_family or \
                    r["final_score"] > best_per_family[fam]["final_score"]:
                best_per_family[fam] = r
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
                  "  structure       : representatives/"
                  f"BEST_REPRESENTATIVE_FRAME_"
                  f"{method_reps.method_slug(top.get('source_method', ''))}"
                  f"_cluster{top['cluster_id'] + 1}"
                  f"_frame{top['original_frame']}.pdb",
                  "                    (stable alias: "
                  "BEST_REPRESENTATIVE_FRAME.pdb)"]
        with open(os.path.join(reps_dir, "POSE_COMPARISON_REPORT.txt"),
                  "w") as fh:
            fh.write("\n".join(lines))
        self.logger.info("Best cross-method pose: %s cluster %d, score %.2f",
                         top.get("source_method", "?"),
                         top["cluster_id"] + 1, top["final_score"])

    def _annotate_recommendation(self, rows: list[dict]) -> None:
        """State recommended method vs pose winner, and flag disagreement."""
        path = os.path.join(self.out_dir, "representatives",
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
        with open(os.path.join(self.out_dir, "reports",
                               "METHOD_AND_POSE_SUMMARY.txt"), "w") as fh:
            fh.write("\n".join([
                "BEST CLUSTERING METHOD AND ITS SELECTED POSE",
                "=" * 70, "",
                "Two independent answers are reported:",
                "  1. best METHOD by cluster validation (ranking in "
                "clustering/clustering_summary.csv)",
                "  2. best POSE by composite frame score across ALL method "
                "representatives",
                "",
                f"Best method (validation) : {self.ranked[0].name}",
                f"  cluster validation score: {self.ranked[0].composite}",
                f"  representative structure: "
                "representatives/RECOMMENDED_METHOD_FRAME.pdb",
                "",
                f"Best pose (all methods)  : {top.get('source_method', '')}",
                f"  frame                 : analysis "
                f"{top['analysis_frame']} / original "
                f"{top['original_frame']} / "
                f"{top['time_ps'] / 1000:.3f} ns",
                f"  cluster               : {top['cluster_id'] + 1}"
                f" (population {top['cluster_population']})",
                f"  composite pose score  : {top['final_score']:.2f} / 100",
                f"  structure             : representatives/"
                f"BEST_REPRESENTATIVE_FRAME_{method_reps.method_slug(top.get('source_method', ''))}"
                f"_cluster{top['cluster_id'] + 1}"
                f"_frame{top['original_frame']}.pdb",
                f"  same method as above  : {'YES' if agree else 'NO'}",
                "", "Details:",
                "  representatives/pose_comparison.csv",
                "  representatives/POSE_COMPARISON_REPORT.txt",
                "  reports/FINAL_FRAME_SELECTION_REPORT.txt",
            ]))

    # ------------------------------------------------------------ phase 10
    def _phase_plotting(self) -> None:
        log = self.logger
        cfg = self.cfg
        if not cfg["system"].get("plots_enabled", True):
            log.info("plots disabled by config")
            return
        p_dir = os.path.join(self.out_dir, "plots")
        labels = self.best_labels
        viz.plot_cluster_population(labels, p_dir, log)
        if self.dist["D"] is not None:
            viz.plot_silhouette(self.dist["D"], labels, p_dir, log)
            viz.plot_distance_heatmap(
                self.dist["D"], p_dir, log, None,
                max_frames=cfg["plots"].get("heatmap_max_frames", 300))
        viz.plot_rmsd_distribution(self.dist["rmsd"], p_dir, log)
        viz.scatter_embedding(self.dist["X"], labels, p_dir, log, "PCA")
        if cfg["plots"].get("tsne_enabled"):
            try:
                viz.scatter_embedding(self.dist["X"], labels, p_dir, log,
                                      "t-SNE", "tsne_clustering.png")
            except Exception as exc:
                log.warning("t-SNE plot skipped: %s", exc)
        viz.plot_clustering_comparison(self.metrics, p_dir, log)
        viz.plot_representative_ranking(self.score_rows, p_dir, log)
        if getattr(self, "sens_rows", None):
            viz.plot_scoring_sensitivity(self.sens_rows, self.score_rows,
                                         p_dir, log)
        if self.bench_results.get("solutions"):
            bs = max(self.bench_results["solutions"],
                     key=lambda s: s.n_clusters if s.n_clusters else -1)
            viz.plot_gromacs_populations(labels, bs.labels, p_dir, log)

    # ------------------------------------------------------------ phase 11-12
    def _phase_reports(self) -> None:
        log = self.logger
        self._write_method_coverage()
        best_row = self.score_rows[0]
        reps_dir = os.path.join(self.out_dir, "representatives")
        # method-named, self-describing winner file + stable alias
        best_slug = method_reps.method_slug(
            best_row.get("source_method") or self.best_metrics.name)
        best_named = (f"BEST_REPRESENTATIVE_FRAME_{best_slug}"
                      f"_cluster{best_row['cluster_id'] + 1}"
                      f"_frame{best_row['original_frame']}.pdb")
        best_row["structure_file"] = os.path.join("representatives",
                                                  best_named)
        reps.write_frame_pdb(
            self.info, best_row["analysis_frame"],
            os.path.join(reps_dir, best_named), logger=log)
        reps.write_frame_pdb(
            self.info, best_row["analysis_frame"],
            os.path.join(reps_dir, "BEST_REPRESENTATIVE_FRAME.pdb"),
            logger=None)
        rows = [[s.name, s.native, s.n_clusters, s.time_s, s.mem_mb,
                 (s.error or "OK")[:80]] for s in self.sols]
        utils.write_csv(os.path.join(self.out_dir, "reports",
                                     "clustering_summary.csv"),
                        ["method", "native_MDANCE", "n_clusters", "time_s",
                         "mem_mb", "status"], rows)
        with open(os.path.join(self.out_dir, "reports",
                               "clustering_summary.json"), "w") as fh:
            json.dump([{k: v for k, v in dict(
                name=s.name, native=s.native, n_clusters=s.n_clusters,
                time_s=round(s.time_s, 3), mem_mb=round(s.mem_mb, 2),
                params=s.params, error=(s.error or None)).items()}
                for s in self.sols], fh, indent=2, default=str)
        explanations = [
            ("Representativeness", "relative position of the candidate inside "
             "its cluster toward the cluster medoid (1 = medoid itself)"),
            ("Population", "sqrt-compressed cluster population fraction; "
             "populated conformational states are rewarded while the largest "
             "cluster is not automatically preferred"),
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
        with open(os.path.join(self.out_dir, "reports",
                               "FINAL_FRAME_SELECTION_REPORT.txt"), "w") as fh:
            fh.write(banner)
        print(banner)
        log.info("Final frame: analysis %d / original %d, score %.2f",
                 best_row["analysis_frame"], best_row["original_frame"],
                 best_row["final_score"])

        bench_table, bench_report = gromacs_bench.benchmark_table(
            self.ranked[0], self.bench_results, self.sols, self.cfg,
            D=self.dist["D"])
        utils.write_csv(os.path.join(self.out_dir, "benchmarking",
                                     "mdance_vs_gromacs.csv"),
                        gromacs_bench.BENCH_HEADER, bench_table)
        with open(os.path.join(self.out_dir, "benchmarking",
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
                "  results            : clustering/k_selection/"
                "(k_selection.csv, .json, k_selection_elbow.png)",
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
                "  details           : representatives/method_recommendation.txt",
            ]
        else:
            lines.append("  none")
        fams = (self.method_summary or {}).get("families") or {}
        if fams:
            lines += ["", "Per-method representative frames "
                      "(representatives/representative_frames/):"]
            for name, fam in fams.items():
                lines.append(f"  {name:<14} {fam['solution']:<34}"
                             f" k={fam['n_clusters']:<3}"
                             f" rank=#{fam['validation_rank']}  "
                             f"{fam['directory']}/")
        lines += ["",
                  "FINAL SELECTION (rationale: reports/FINAL_FRAME_SELECTION_REPORT.txt)",
                  f"  analysis frame  : {best_row['analysis_frame']}",
                  f"  original frame  : {best_row['original_frame']}",
                  f"  time (ns)       : {best_row['time_ps']/1000:.3f}",
                  f"  cluster         : {best_row['cluster_id'] + 1}",
                  f"  final score     : {best_row['final_score']}",
                  "  sensitivity     : " + json.dumps(
                      self.sens_summary.get("win_counts", {})),
                  ]
        with open(os.path.join(self.out_dir, "reports",
                               "complete_analysis_report.txt"), "w") as fh:
            fh.write("\n".join(lines))


def info_idx(info, analysis_idx: int, which: str):
    if which == "orig":
        return int(info["frame_indices"][analysis_idx])
    return round(float(info["times"][analysis_idx]), 3)
