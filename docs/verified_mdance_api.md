# Technical Report — Verified MDANCE APIs (Phase 1)

Date: 2026-10-30
Reference: official repository https://github.com/mqcomplab/MDANCE, branch `main`
(this mirrors PyPI release 0.4.0), plus the **installed local copy of `MDANCE 0.4.0`**
(`/usr/local/lib/python3.10/dist-packages/mdance`), which was used as the
authoritative source for every signature quoted below.

Everything listed under **VERIFIED** was confirmed by reading the installed
source files of MDANCE 0.4.0. Everything under **NOTE / NOT AVAILABLE** is
either not implemented in 0.4.0 or is pipeline-supplemental code, clearly
labeled later in the code as `pipeline-added`.

---

## 1. Package layout (installed 0.4.0)

```
mdance/
  __init__.py, version
  cluster/   nani.py  helm.py  divine.py  equal.py  mdbirch.py  shine.py  prism.py
  data/      (bundled NumPy/CSV/DCD sample data)
  inputs/    preprocess.py  (trajectory → NumPy conversion, Normalizer)
  outputs/   postprocess.py (helpers for writing coordinate files)
  prime/     rep_frames.py  sim_calc.py    (Prime-style similarity workflows)
  tools/     bts.py  esim.py  isim.py     (MSD + extended similarity toolkit)
  cli/       prime_rep.py  prime_sim.py
```

Note: scripts such as `scripts/helm/run_helm.py` document the officially
intended usage of HELM (merging an initially coarse partition).

## 2. Verified trajectory → data conversion (`mdance.inputs.preprocess`)

```python
mdance.inputs.preprocess.gen_traj_numpy(prmtopFileName, trajFileName, atomSel,
                                        verbose=True)
```
* Uses MDAnalysis; `atomSel` is an MDAnalysis selection string.
* Returns `np.ndarray` of shape `(n_frames, 3 * n_selected_atoms)` — **one row
  per frame, flattened coordinates of the selected atoms**. The number of
  selected atoms is what all MDANCE functions call `N_atoms` (not the number
  of features; `features = 3 * N_atoms`).
* No PBC/alignment is performed inside `gen_traj_numpy` — the docs and the
  bundled examples state that trajectories must be pre-aligned; this is why the
  pipeline performs PBC/centering/alignment with MDAnalysis *itself* (Phase 2)
  and can then use either `gen_traj_numpy` or its own equivalent extraction.
* `Normalizer`, `read_cpptraj`, `normalize_file` exist but are documented in
  MDANCE as intended for **non-MD** data; they are not used for trajectories.

## 3. Verified distance/similarity toolkit (`mdance.tools.bts`)

All VERIFIED from installed `bts.py`:

| Function | Signature | Purpose |
|---|---|---|
| `mean_sq_dev` | `(matrix, N_atoms)` | *O(N)* mean square deviation (MSD) of an *n*-ary object; for a single row it is exactly the MDANCE-native pairwise MSD |
| `msd_condensed` | `(c_sum, sq_sum, N, N_atoms)` | MSD of a group from its column sums / squared column sums / count |
| `extended_comparison` | `(matrix, data_type='full', metric='MSD', N=None, N_atoms=1, **kwargs)` | *n*-ary comparison; `data_type='full'`/`'condensed'`; `metric='MSD'` for coordinates |
| `calculate_comp_sim` | `(matrix, metric, N_atoms=1)` | complementary-similarity values per row (centrality ranking) |
| `calculate_medoid` | `(matrix, metric, N_atoms=1)` | index of the medoid (an **actual row** of the input) |
| `calculate_outlier` | `(matrix, metric, N_atoms=1)` | index of the outlier |
| `trim_outliers` | `(matrix, n_trimmed, metric, N_atoms, criterion='comp_sim')` | remove *n* most outlying rows |
| `diversity_selection` | `(matrix, percentage, metric, N_atoms=1, method='strat', start='medoid')` | stratified/diversity sampling used by NANI |
| `quota_sampling` | `(data, metric, percentage=10, n_bins=10, hard_cap=True, N_atoms=1, comp_sim=None)` | quota-based sampling |
| `rep_sample` | `(data, metric='MSD', N_atoms=1, n_bins=10, n_samples=100, hard_cap=True)` | representative sampling |
| `refine_dis_matrix` | `(matrix)` | convert a full distance matrix to condensed form |

Key conventions verified inside `bts`:

* `MSD(i, j) = ||x_i − x_j||² / N_atoms` (mean square deviation **per atom**,
  in Ångström² — the square of the atom-weighted RMSD between the flattened
  selections). RMSD = `sqrt(MSD)`. MDANCE's `extended_comparison` with
  `metric='MSD'` and `data_type='full'` returns the full
  `(n_frames, n_frames)` matrix of these values; `'condensed'` uses
  `refine_dis_matrix` (lower triangular).
* Extended-similarity indices (e.g. EIC indices) require binned/fingerprint
  data and are **not defined for Cartesian coordinates**; the pipeline
  therefore uses `metric='MSD'` for all structural distances. This is a *verified
  compatibility requirement*, not a limitation we invented.

## 4. Verified clustering classes (`mdance.cluster`)

### 4.1 `KmeansNANI` (`mdance/cluster/nani.py`)

```python
KmeansNANI(data, n_clusters, metric, N_atoms, init_type='strat_all', **kwargs)
```
* `init_type` ∈ {`'comp_sim'`, `'div_select'`, `'k-means++'`, `'random'`,
  `'vanilla_kmeans++'`, `'strat_all'`, `'strat_reduced'`, `'quota'`} (checked
  in `_check_init_type`).
* `percentage` (default 10) is accepted via `kwargs` for the percentage-based
  initializers.
* Methods: `initiate_kmeans()`, `kmeans_clustering(initiators)` ->
  `labels, centers, n_iter`, `execute_kmeans_all()` -> `labels, centers,
  n_iter`. The final refinement is scikit-learn `KMeans` on the coordinates
  (Euclidean), seeded by the NANI initializers; `n_init=1` is the documented
  usage (return value `centers` is a **coordinate mean — not a physical frame**).
* `compute_scores(data, labels)` returns `(CH, DB)` from scikit-learn in
  feature space.
* Every frame IS assigned (`labels` for all rows).

### 4.2 `Divine` (`mdance/cluster/divine.py`)

```python
Divine(data, split='weighted_MSD', anchors='nani', init_type='strat_all',
       end='k', k=None, refine=True, N_atoms=1, threshold=0, percentage=10)
```
* `run()` -> `(clusters, labels, scores, msds)`; `clusters` is a list of
  index lists; validated `split` ∈ {`MSD`, `radius`, `weighted_MSD`}, `anchors`
  ∈ {`nani`, `outlier_pair`, `splinter_split`}, `end` ∈ {`k`, `points`}.
* Divisive top-down hierarchical clustering; internal k-means refinement when
  `refine=True`.
* Note: `metric` is hard-coded to `'MSD'` inside `Divine`.

### 4.3 `ExtendedQuality` (the eQual algorithm, `mdance/cluster/equal.py`)

```python
ExtendedQuality(data, threshold, metric, N_atoms, seed_method='greedy',
                n_seeds=1, check_sim=False, reject_lowd=True, **kwargs)
```
* `run()` -> `renumbered_clusters` — a `dict {cluster_id: [frame indices]}`.
* `seed_method` ∈ {`'comp_sim'`, `'greedy'`, `'medoid'`,
  `'mini_batch_kmeans'`, `'vanilla'`}.
* Radius/threshold-based *non*-k-dependent clustering; the number of clusters
  is an emergent property of `threshold` (an MSD value). It also exposes
  `calculate_best_frames(clusters, n_structures=10, sorted_by='frame')` and
  `labels(clusters, sieve)`.

### 4.4 `HELM` (`mdance/cluster/helm.py`)

```python
HELM(cluster_dict, metric, N_atoms, merge_scheme='inter', n_clusters=None,
     eps=None, trim_start=False, ...);  HELM(...)()  # callable alias of run()
```
* Merges an **initial partition** (per the official `scripts/helm/run_helm.py`)
  whose `cluster_dict` format is verified as
  `{N0: [[indices, (c_sum, sq_sum), N], ...]}` with
  `c_sumik = np.sum(cluster_rows, axis=0)`,
  `sq_sumik = np.sum(cluster_rows**2, axis=0)`.
* `merge_scheme` ∈ {`'intra'`, `'inter'`, `'half'`}; termination by exactly one
  of `n_clusters` / `eps`; optional SciPy `link` path (`e.g. 'ward'`).
* Returns the full ladder of merged partitions `{Nk: [...]}`.
* The official example seeds HELM from *another clustering result* (its bundled
  `labels_60`), which the pipeline follows: HELM refines an initial NANI
  k-means partition. Running HELM over ~1000 singletons is (O(n³)) cluster
  merges and is not supported.

### 4.5 `mdBirch` (`mdance/cluster/mdbirch.py`)

```python
mdBirch(*, threshold=1, branching_factor=50)
```
* `fit(X)` (single pass, online BIRCH with RMSD-calibrated merges;
  merge criterion `radius` only), `get_centroids()`,
  `get_cluster_mol_ids()` (list of per-subcluster index lists; per-cluster id
  assignment is reconstructed by the pipeline).
* `threshold` is an MSD tolerance.

### 4.6 `Shine` / `PRISM` (trajectory/pathway clusterers)

```python
Shine(trajs, metric, N_atoms, t, criterion, link='ward', ...)
PRISM(trajs, metric, t, criterion, link='ward', ...)
```
* Verified to operate on **lists of trajectories** (pathway clustering across
  multiple per-trajectory MDANCE matrices), not on flat single-frame
  populations. A single-trajectory input cannot define meaningful intra-set
  dissimilarities, so the pipeline does not use them for frame clustering;
  their role and exclusion rationale are documented in reports.

## 5. Verified representative-frame capabilities (MDANCE-native)

* `bts.calculate_medoid(...)` — medoid = **actual trajectory frame**.
* `bts.calculate_outlier(...)` — outlier detection.
* `bts.calculate_comp_sim(...)` — per-frame centrality ordering.
* `ExtendedQuality.calculate_best_frames(...)` — best frames per cluster.
* K-means centers of `KmeansNANI` and `mdBirch.get_centroids()` are
  **mathematical centroids** — they are NOT necessarily physically valid
  structures. The pipeline therefore ranks *actual frames* only (Phase 7/8).
* `mdance.prime.rep_frames` implements a Prime-workflow representative-frame
  chooser that expects Prime similarity folders/summary files (different data
  layout than this pipeline) and is documented as not directly reusable here.

## 6. Verified GPU / parallelization status

* Installed `bts.py`/`nani.py`/`equal.py`/`helm.py`/`divine.py` use NumPy and
  `torch` only for the optional `align_traj`/shape-GMM alignment helpers.
  **No GPU clustering backend and no `n_jobs`/`parallel` parameters exist** in
  the clustering classes of 0.4.0 (verified). The pipeline's own
  parallelism/batching is limited where crucial (documented below).

## 7. Verified dependencies / versions

`pyproject.toml` of MDANCE 0.4.0:
Python ≥ 3.10; numpy, scikit-learn ≥ 0.24, MDAnalysis ≥ 2.0, scipy ≥ 1.6,
matplotlib ≥ 3.3, cycler, torch (CPU wheel sufficient), shapeGMMTorch.
Consequence for `requirements.txt` / `environment.yml`.

## 8. APIs that do NOT exist in 0.4.0 (do not use)

`K_Means`, `md_cluster`, `mdance.cluster.kmeans`, `mdance.tools.proto`,
`mean_sq_dist`/`gen_sq_dist`/`sum_sq_dist` (the function is
`mean_sq_dev`), `NURV`, `sim_fitness`, `sbac_ccf`, a `DBSCAN` class,
`cadence.py`, `gen_traj_numpy`-style *(data IS the verified converter)*, an
HDF5 "cluster dict" class, and `color_index` / `mini_scale_unit` conventions —
all reported absent from the verified installed source.

---

## 9. What the pipeline adds beyond MDANCE (explicitly labeled `pipeline-added`)

1. Preprocessing (PBC correction, centering, alignment, time windows,
   integrity checks) — uses MDAnalysis as MDANCE recommends but computes it
   itself; then hands MDANCE a clean `(frames, 3*N_atoms)` matrix in the exact
   verified convention.
2. A chunked, mathematically identical MSD matrix builder (valuable for large
   trajectories) — validated in the unit tests against
   `bts.extended_comparison(... 'MSD')` to prove equivalence.
3. Baseline clustering for benchmarking breadth (clever relations to `gmx
   cluster`): SciPy/sklearn hierarchical (average/complete linkage on the same
   MDANCE distance matrix), DBSCAN, and a faithful implementation of the
   Daura/gromos greedy exhaustive-neighbor algorithm used by `gmx cluster`.
4. Cluster-quality metrics (silhouette/DB/CH in distance space), bootstrap
   stability, GROMACS benchmarking tables, multi-criteria model selection,
   composite frame scoring (0–100), sensitivity analysis, plots, and reports.

Every MDANCE function call in the code is comment-tagged `# MDANCE-native`,
and every supplemental function with `# pipeline-added`.
