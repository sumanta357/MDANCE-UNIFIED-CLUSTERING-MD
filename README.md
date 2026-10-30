# MDANCE-UNIFIED-CLUSTERING-MD

A complete, unified, reproducible Python pipeline for **MDANCE-based MD
trajectory clustering**, **benchmarking against GROMACS `gmx cluster`**, and
**automated selection of ONE best representative frame** from the trajectory
for downstream structural / protein-ligand interaction analysis.

The scientific goal is *the final frame*, not the clustering: the pipeline
ends by scoring every candidate representative frame (0–100), ranking them,
reporting why the winner was chosen, and exporting
`representatives/BEST_REPRESENTATIVE_FRAME.pdb` (an **actual trajectory
frame**, never an averaged centroid).

* Verified MDANCE API inventory (0.4.0): `docs/verified_mdance_api.md`
* Native MDANCE methods used: `KmeansNANI` (8 initializers), `Divine`,
  `ExtendedQuality` (eQual), `HELM`, `mdBirch` — plus `mdance.tools.bts`
  medoid / outlier / complementary-similarity routines.
* `Shine` / `PRISM` are multi-trajectory pathway clusterers and are not run on
  single-frame populations (documented limitation, not hidden).
* Pipeline-added (clearly labeled in the code): preprocessing, chunked MSD
  matrix, validation metrics, gromos/Daura benchmark, composite scoring,
  sensitivity analysis, plots, reports.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate    # optional but advised
# CPU-only torch first (mdance needs torch; avoid GB of CUDA wheels):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install mdance==0.4.0
pip install -r requirements.txt   # numpy scipy scikit-learn MDAnalysis matplotlib pyyaml pytest
```

Conda equivalent: `environment.yml`.

## Usage

```bash
python run_mdance.py \
    --xtc trajectory.xtc --tpr topology.tpr --gro structure.gro \
    --selection "protein and backbone" --output MDANCE_Analysis \
    --stride 10 --start-time 0 --end-time 100 \
    --benchmark-gromacs --select-best-frame
```

Or with a YAML configuration:

```bash
python run_mdance.py --config configs/config.yaml
```

Example configurations in `configs/`: backbone (`config.yaml`), C-alpha
(`config_calpha.yaml`), protein-ligand (`config_protein_ligand.yaml`),
binding-site (`config_binding_site.yaml`). Partial overrides are merged over
the built-in defaults.

If no `--tpr` is provided, the pipeline uses the `.gro` as topology (a GRO
lacks bonds; neighbor searches that need bonds degrade gracefully — documented
in the reports).

## Output structure

```
MDANCE_Analysis/
├── config/config.yaml                  # resolved configuration
├── preprocessing/{processed.xtc, processed.gro, preprocessing_report.txt}
├── features/{feature_data.npy, distance_matrix.npy}
├── clustering/{mdance/, gromacs/, optimal_clustering/, clustering_summary.csv}
├── representatives/{global_medoid.pdb, cluster_representatives/,
│                    candidate_frames/, BEST_REPRESENTATIVE_FRAME.pdb,
│                    candidate_frame_scores.csv, cluster_representatives.csv}
├── benchmarking/{mdance_vs_gromacs.csv, benchmarking_report.txt}
├── scoring/{scoring_metrics.csv, scoring_weights.yaml, sensitivity_analysis.csv}
├── plots/  *.png
├── reports/{complete_analysis_report.txt, clustering_summary.{csv,json},
│            FINAL_FRAME_SELECTION_REPORT.txt}
├── logs/analysis.log
requirements.txt, environment.yml, README.md
```

## What the pipeline does (phases)

1. **Verify MDANCE** — every MDANCE call is validated against source files
   (`docs/verified_mdance_api.md`).
2. **Preprocess** — load XTC/TPR/GRO, validate, PBC (wrap→unwrap→center),
   reference-frame alignment, time-window/stride/equilibration filters.
   **All atoms kept** (ligand/ions/RNA/cofactors preserved).
3. **Features/distances** — MDANCE convention `(frames, 3·N_atoms)`;
   chunked MSD matrix, cross-validated against the MDANCE-native definition.
4. **Cluster** — every compatible MDANCE method over configurable k /
   initializer / threshold grids, timed and memory-tracked, reproducible seeds.
5. **Validate & select** — silhouette, distance-space Davies–Bouldin and
   Calinski–Harabasz, compactness/separation/diversity, population balance,
   bootstrap-ARI stability, multi-criteria weighted ranking (documented
   normalization and weight renormalization for missing metrics).
6. **Benchmark vs GROMACS** — Daura/gromos algorithm on the *identical*
   MDANCE-MSD matrix (the algorithm behind `gmx cluster -m gromos`) at several
   cutoffs; native `gmx cluster` attempted when a `gmx` binary exists;
   differences explicitly documented rather than claimed equivalent.
7. **Representatives** — per-cluster medoid (actual frame, MDANCE-native),
   nearest-actual-frame-to-centroid, global medoid; centroids are never
   exported (not physical frames).
8. **Best frame** — candidates (medoids + nearest-to-centroids + runner-up
   medoids) scored 0–100 from six min-max-normalized metric groups with
   YAML-configurable weights; ligand metrics optional (weights renormalize).
9. **Sensitivity** — Dirichlet weight perturbations; winner frequency; most
   robust frame reported.
10. **Visualization** — population, silhouette, RMSD, distance heatmap, PCA,
    method comparison, candidate ranking, sensitivity (+optional t-SNE; PCA /
    t-SNE are labeled visualization-only, clustering uses MSD space).
11. **Reports** — the exact FINAL-BEST-REPRESENTATIVE-FRAME banner, plus
    complete/benchmarking/preprocessing reports, CSV/JSON summaries, logs.

## Tests

```bash
pytest tests/ -x -q
```

* `tests/test_unit.py` — chunked-vs-native MSD equivalence, medoid/outlier
  semantics, gromos clustering, DB/CH distance-space metrics, scoring
  normalization, config validation.
* `tests/test_integration.py` — builds a synthetic GROMACS-like trajectory
  (GRO + XTC via MDAnalysis writers, protein-like names so selections work)
  and executes the **full** `run_mdance.py` pipeline end-to-end, asserting the
  required output tree, banner and `BEST_REPRESENTATIVE_FRAME.pdb` against
  real trajectory inputs.

## Scientific assumptions (explicit)

* Distances: MDANCE-MSD = squared atom-normalized Euclidean distance of the
  aligned selection (RMSD² per atom). All clustering on one space.
* Silhouette/silhouette-like metrics need ≥2 clusters; DB/CH are computed as
  documented distance-space generalizations (not the scikit-learn feature-space
  variants); missing metrics are excluded with weight renormalization.
* `KmeansNANI` uses scikit-learn `KMeans` internally (Euclidean refinement of
  NANI seeds) — this is the MDANCE-verified design; returned `centers` are
  **means, not frames** and are never exported.
* Seeded python/numpy RNG at startup; NANI initializers consume the
  global numpy RNG (seeded at startup).
* `Divine` fixes its internal metric to `'MSD'` (verified upstream behavior).
* The best frame's "highest score" ≠ biologically optimal; scoring weights
  and every normalization are exposed and configurable.
