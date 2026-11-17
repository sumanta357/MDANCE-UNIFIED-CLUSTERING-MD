# MDANCE-UNIFIED-CLUSTERING-MD

A complete, unified, reproducible Python pipeline for **MDANCE-based MD
trajectory clustering**, **benchmarking against GROMACS `gmx cluster`**, and
**automated selection of ONE best representative frame** from the trajectory
for downstream structural / protein-ligand interaction analysis.

Two behaviours are central:

* **Automatic K selection** — an Elbow/**Kneedle** scan over K = 2..15
(extendable to 20), measured in the MDANCE MSD representation, chooses the K
used by **NANI, DIVINE, HELM and Agglomerative** (one shared K by default;
`k_selection.per_method: true` gives each of them its own scan and K). eQual,
mdBIRCH, DBSCAN and GROMOS/Daura keep their native threshold/density
behaviour and are never forced onto that K. Results:
`03_PLOTS/K_SELECTION.png` (details in
`04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/k_selection/`).
* **One best representative pose per method** — each successful method's
best-scoring representative (actual trajectory frame, original frame index
and simulation time) is written to
`02_METHOD_COMPARISON/METHOD_REPRESENTATIVES/<METHOD>.pdb`, with one row per
method in `POSE_RANKING.csv`.
* **Explicit method coverage** — `METHOD_COVERAGE.txt` +
`method_coverage.json` under `04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/`
list, for every method, whether it produced usable partitions and why not when
it did not (disabled, too many frames, threshold sweep gave <2 clusters,
failed runs). Threshold sweeps (eQual,
mdBIRCH, DBSCAN) also test thresholds at the nearest-neighbour distance scale,
so they no longer collapse to one cluster / all-noise, and the HELM frame cap
is 20 000 (configurable).
* **Cross-method pose comparison** — all those poses compete in one ranked
table, each scored inside its own method's partition, so
`01_FINAL_RESULTS/BEST_POSE.pdb` can come from **any** method.
`METHOD_AND_POSE_SUMMARY.txt` reports both the best method by cluster
validation and the best pose, and whether they agree. See
`docs/automatic_k_and_representatives.md`.

The scientific goal is *the final frame*, not the clustering: the pipeline
ends by scoring every candidate representative frame (0–100), ranking them,
reporting why the winner was chosen, and exporting
`01_FINAL_RESULTS/BEST_POSE.pdb` (an **actual trajectory frame**, never an
averaged centroid).

* **Plain-language guide to the workflow and every clustering method:**
  `docs/METHODS_EXPLAINED.md`
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

Add `--full-output` to also export every scored candidate frame and every
per-cluster medoid/centroid PDB under `04_ADVANCED_ANALYSIS/`. The default run
writes only the final results, one pose per successful method and the four
primary plots.

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
├── README.txt                     # where to look first
├── 01_FINAL_RESULTS/
│   ├── BEST_POSE.pdb              # highest-scoring representative (use this)
│   ├── BEST_METHOD.pdb            # recommended-method representative
│   ├── FINAL_SUMMARY.csv          # key numbers on one screen
│   └── FINAL_REPORT.txt           # plain-language report
├── 02_METHOD_COMPARISON/
│   ├── METHOD_RANKING.csv         # one row per method, incl. FAILED/SKIPPED
│   ├── POSE_RANKING.csv           # best pose per successful method
│   ├── CLUSTER_POPULATION.csv
│   ├── GROMACS_COMPARISON.csv
│   └── METHOD_REPRESENTATIVES/<METHOD>.pdb
├── 03_PLOTS/
│   ├── K_SELECTION.png
│   ├── CLUSTER_COMPARISON.png
│   ├── POSE_RANKING.png
│   └── GROMACS_COMPARISON.png
├── 04_ADVANCED_ANALYSIS/
│   ├── CANDIDATE_FRAMES/          # every candidate (--full-output)
│   ├── CLUSTER_MEDOIDS/           # per-cluster medoids (--full-output)
│   ├── SENSITIVITY_ANALYSIS/
│   ├── DIAGNOSTIC_PLOTS/
│   └── INTERMEDIATE_RESULTS/      # K scan, labels, raw metrics, details
├── 05_PREPROCESSING/
│   ├── PROCESSED_TRAJECTORY/{processed.xtc, processed_pbc.xtc}
│   ├── REFERENCE_STRUCTURE/reference.gro
│   └── preprocessing_report.txt
└── logs/analysis.log
```

By default only final results, one pose per successful method and the four
primary plots are exported. Add `--full-output` to also write every scored
candidate frame and per-cluster medoid/centroid PDB under
`04_ADVANCED_ANALYSIS/`. Diagnostics are moved there rather than deleted.

## What the pipeline does (phases)

1. **Verify MDANCE** — every MDANCE call is validated against source files
   (`docs/verified_mdance_api.md`).
2. **Preprocess** — load XTC/TPR/GRO, validate, PBC (wrap→unwrap→center),
   reference-frame alignment, time-window/stride/equilibration filters.
   **All atoms kept** (ligand/ions/RNA/cofactors preserved).
3. **Features/distances** — MDANCE convention `(frames, 3·N_atoms)`;
   chunked MSD matrix, cross-validated against the MDANCE-native definition.
3b. **Automatic K selection** — KmeansNANI scan over K = 2..15 (configurable,
   max 20); dispersion `W(K) = sum_C (1/|C|) sum_{i<j in C} d_MSD(i,j)` on the
   precomputed MDANCE MSD matrix (= `sum_i ||x_i-mean(C)||^2/(2*N_atoms)`),
   **not** Euclidean k-means inertia; Kneedle/geometric elbow detection with
   explicit ambiguity reporting; supporting silhouette, distance-space
   Davies–Bouldin/Calinski–Harabasz, population distribution and bootstrap-ARI
   stability; CSV/JSON/PNG saved.
4. **Cluster** — every compatible MDANCE method. NANI, DIVINE, HELM and
   Agglomerative run at the selected K; eQual (threshold sweep), mdBIRCH
   (threshold sweep), DBSCAN (eps sweep) and GROMOS (cutoff sweep) keep their
   native K-free behaviour. Timed, memory-tracked, reproducible seeds;
   methods that disagree keep their own partitions.
5. **Validate & select** — silhouette, distance-space Davies–Bouldin and
   Calinski–Harabasz, compactness/separation/diversity, population balance,
   bootstrap-ARI stability, multi-criteria weighted ranking (documented
   normalization and weight renormalization for missing metrics).
6. **Benchmark vs GROMACS** — Daura/gromos algorithm on the *identical*
   MDANCE-MSD matrix (the algorithm behind `gmx cluster -m gromos`) at several
   cutoffs; native `gmx cluster -method gromos` runs automatically when the
   `gmx` binary exists. Native input files contain exactly the MDANCE atom
   selection, the MSD cutoff is converted to an RMSD cutoff in nm, and native
   per-frame membership is parsed from `-clid` for a real quantitative
   comparison. `02_METHOD_COMPARISON/GROMACS_COMPARISON.csv` reports cluster
   counts, runtime/memory, compactness/separation and partition-agreement
   statistics (ARI, population L1, top-5 population divergence) for native
   GROMACS and the MDANCE GROMOS tier separately. Native success/failure is
   reported explicitly; a failed run is never treated as a successful
   benchmark.
7. **Representatives** — **per method**: the best representative pose of every
   successful method is scored and exported as
   `02_METHOD_COMPARISON/METHOD_REPRESENTATIVES/<METHOD>.pdb` with original
   frame index and time in `POSE_RANKING.csv`; the recommended-method pose is
   `01_FINAL_RESULTS/BEST_METHOD.pdb`. Per-cluster medoid/centroid PDBs are
   opt-in under `04_ADVANCED_ANALYSIS/CLUSTER_MEDOIDS/` (`--full-output`).
   Centroids are never exported (not physical frames). The recommended method
   and its representative frame are reported separately from the composite
   frame score.
8. **Best frame (cross-method)** — the medoid of every cluster of every
   method (plus the nearest-to-centroid frame when different) is scored 0–100
   from six min-max-normalized metric groups with YAML-configurable weights,
   each pose evaluated inside its own method's partition; ligand metrics are
   optional (weights renormalize). The winner can come from any method and is
   exported as `01_FINAL_RESULTS/BEST_POSE.pdb`; the best pose per method is
   `02_METHOD_COMPARISON/POSE_RANKING.csv` + `METHOD_REPRESENTATIVES/`. Detail
   is retained in `04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/`.
9. **Sensitivity** — Dirichlet weight perturbations; winner frequency; most
   robust frame reported.
10. **Visualization** — population, silhouette, RMSD, distance heatmap, PCA,
    method comparison, candidate ranking, sensitivity (+optional t-SNE; PCA /
    t-SNE are labeled visualization-only, clustering uses MSD space).
11. **Reports** — `01_FINAL_RESULTS/FINAL_REPORT.txt` (plain language) and
    `FINAL_SUMMARY.csv`, plus the detailed banner and CSV/JSON diagnostics in
    `04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/`, logs and preprocessing
    reports.

## Tests

```bash
pytest tests/ -x -q
```

* `tests/test_unit.py` — chunked-vs-native MSD equivalence, medoid/outlier
  semantics, gromos clustering, DB/CH distance-space metrics, scoring
  normalization, config validation.
* `tests/test_k_selection.py` — dispersion identity (distance vs centered
  coordinates) and elbow detection incl. the ambiguous/no-knee case.
* `tests/test_integration.py` — builds a synthetic GROMACS-like trajectory
  (GRO + XTC via MDAnalysis writers, protein-like names so selections work)
  and executes the **full** pipeline end-to-end, asserting the required output
  tree, the K-selection results, that K-based methods used the selected K, the
  per-method `representative_frames/` exports with frame/time metadata, the
  independent method recommendation, the banner and
  `BEST_REPRESENTATIVE_FRAME.pdb` against real trajectory inputs.

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
