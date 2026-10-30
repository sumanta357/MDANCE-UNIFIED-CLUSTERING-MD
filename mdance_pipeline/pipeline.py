"""Unified pipeline orchestration (Phases 2-12). Pipeline-added."""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from mdance_pipeline import (clustering, features as feat, gromacs_bench,
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
            "clustering/optimal_clustering", "representatives",
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

    # ------------------------------------------------------------- entry
    def run(self) -> dict:
        log = self.logger
        with utils.timed_step(log, "Phases 1-2: input validation & preprocessing"):
            self._phase_preprocess()
        with utils.timed_step(log, "Phase 3: features & distance matrix"):
            self._phase_features()
        with utils.timed_step(log, "Phase 4: MDANCE + baseline clustering sweep"):
            self.sols = clustering.run_all(
                self.dist["X"], self.dist["D"],
                self.info["n_atoms_selected"], self.cfg, log)
        with utils.timed_step(log, "Phase 5: validation & optimal model selection"):
            self._phase_validation()
        if self.cfg["benchmark"].get("enabled", True):
            with utils.timed_step(log, "Phase 6: GROMACS benchmarking"):
                self.bench_results = gromacs_bench.benchmark(
                    self.sols, self.dist["D"], self.dist["X"], self.cfg, log,
                    self.info, self.out_dir)
        with utils.timed_step(log, "Phase 7: representative-frame extraction"):
            self._phase_representatives()
        with utils.timed_step(log, "Phases 8-9: composite scoring & sensitivity"):
            self._phase_scoring()
        with utils.timed_step(log, "Phase 10: visualizations"):
            self._phase_plotting()
        with utils.timed_step(log, "Phases 11-12: reports & final banner"):
            self._phase_reports()
        return {"final": self.score_rows[0] if self.score_rows else None}

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
        # native/added separation subdirectories (requested structure)
        for sol in self.sols:
            if sol.labels.size == 0:
                continue
            sub = "mdance" if sol.native else "gromacs"
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
                                 f"cluster{cid:03d}_{kind}.pdb"), logger=log)
        # cluster_representatives.csv
        rows = []
        for cid in sorted(self.best_stats.keys()):
            st = self.best_stats[cid]
            med = self.dist["X"][st["medoid_frame"]]
            nearest = self.dist["X"][st["nearest_to_centroid"]]
            msd = float(((med - nearest) ** 2).sum() / (2.0 * N))
            rows.append([cid, st["population"],
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

    # ------------------------------------------------------------ phase 8-9
    def _phase_scoring(self) -> None:
        log = self.logger
        X, D = self.dist["X"], self.dist["D"]
        N = self.info["n_atoms_selected"]
        candidates, cand_labels = [], []
        seen: set[int] = set()
        st0 = self.best_stats
        for cid in sorted(st0.keys()):
            for idx, tag in ((st0[cid]["medoid_frame"], f"medoid(c{cid})"),
                             (st0[cid]["nearest_to_centroid"],
                              f"nearest-centroid(c{cid})")):
                if idx not in seen:
                    seen.add(idx)
                    candidates.append(idx)
                    cand_labels.append(tag)
        for alt_i, (m, st) in enumerate(self.alt_stats, 1):
            for cid in sorted(st.keys()):
                idx = st[cid]["medoid_frame"]
                if idx not in seen:
                    seen.add(idx)
                    candidates.append(idx)
                    cand_labels.append(f"alt{alt_i}[{m.name[:18]}]-medoid(c{cid})")
        if not candidates:
            raise RuntimeError("No candidate representative frames found.")
        log.info("Scoring %d candidate frames.", len(candidates))
        rows = scoring.raw_metrics_table(
            candidates, cand_labels, self.best_labels, X, D, N, self.info,
            self.cfg)
        best_sol = next(
            (s for s in self.sols if s.name == self.best_metrics.name), None)
        best_name = best_sol.name if best_sol else self.best_metrics.name
        best_k = int(len(np.unique(self.best_labels[self.best_labels >= 0])))
        for r in rows:
            r["algorithm"] = best_name
            r["metric"] = "MSD (MDANCE)"
            r["optimal_k"] = best_k
        rows = scoring.score_candidates(rows, self.cfg)
        self.score_rows = rows

        sens_rows, self.sens_summary = scoring.sensitivity_analysis(
            rows, self.ranked[1:4], self.cfg, log)
        self.sens_rows = sens_rows

        keys = ["rank", "analysis_frame", "original_frame", "time_ps",
                "cluster_id", "cluster_population", "cluster_pop_frac",
                "distance_to_medoid", "algorithm", "metric", "optimal_k",
                "label", "representativeness", "population", "stability",
                "quality", "interaction", "conformational",
                "score_representativeness", "score_population",
                "score_stability", "score_quality", "score_interaction",
                "score_conformational", "final_score"]
        with open(os.path.join(self.out_dir, "representatives",
                               "candidate_frame_scores.csv"), "w",
                  newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)

        # candidate_frame PDBs for every candidate
        reps_dir = os.path.join(self.out_dir, "representatives")
        for r in rows:
            idx = r["analysis_frame"]
            reps.write_frame_pdb(
                self.info, idx,
                os.path.join(reps_dir, "candidate_frames",
                             f"score{r['final_score']:.2f}_cluster{r['cluster_id']}"
                             f"_frame{r['original_frame']}.pdb"), logger=log)

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
            ["rank", "candidate_label", "analysis_frame", "original_frame",
             "cluster_id", "final_score"] +
            [f"score_{k}" for k in scoring.METRIC_KEYS],
            [[r.get(k, "") for k in (["rank", "label", "analysis_frame",
                                      "original_frame", "cluster_id",
                                      "final_score"] +
                                     [f"score_{k}" for k in scoring.METRIC_KEYS])]
             for r in rows])

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
        best_row = self.score_rows[0]
        reps_dir = os.path.join(self.out_dir, "representatives")
        reps.write_frame_pdb(
            self.info, best_row["analysis_frame"],
            os.path.join(reps_dir, "BEST_REPRESENTATIVE_FRAME.pdb"), logger=log)
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
            self.ranked[0], self.bench_results, self.sols, self.cfg)
        utils.write_csv(os.path.join(self.out_dir, "benchmarking",
                                     "mdance_vs_gromacs.csv"),
                        ["method", "n_clusters", "time_s", "mem_mb",
                         "compactness", "separation", "n_clusters_gromacs",
                         "adj_rand_index"], bench_table)
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
            "", "Method execution statistics (all solutions run):",
        ]
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
        lines += ["",
                  "FINAL SELECTION (rationale: reports/FINAL_FRAME_SELECTION_REPORT.txt)",
                  f"  analysis frame  : {best_row['analysis_frame']}",
                  f"  original frame  : {best_row['original_frame']}",
                  f"  time (ns)       : {best_row['time_ps']/1000:.3f}",
                  f"  cluster         : {best_row['cluster_id']}",
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
