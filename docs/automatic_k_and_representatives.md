# Automatic K selection and per-method representative frames

## 1. Automatic K selection (Elbow / Kneedle)

`k_selection.enabled: true` (default) runs **before** the clustering sweep in
`clustering/k_selection/`:

* `k_selection.csv` — one row per K: dispersion, n_clusters, n_singletons,
  silhouette, distance-space Davies–Bouldin, distance-space
  Calinski–Harabasz, bootstrap-ARI stability, population distribution,
  support rank.
* `k_selection.json` — the same data plus the selected/elbow K, the elbow
  algorithm and whether the elbow was ambiguous, the dispersion definition
  and the selection rule.
* `k_selection_elbow.png` — K vs within-cluster dispersion with the elbow and
  the selected K marked.

Scan range: **K = 2..15 by default**, configurable with
`k_selection.k_max` and capped at `k_selection.max_allowed_k: 20` (also
limited by the number of analysis frames). The old fixed default of 8 is gone.

### Dispersion definition (mathematically consistent with MDANCE)

```
W(K) = sum_C ( 1/|C| * sum_{i<j in C} d_MSD(i, j) )
```

evaluated on the **precomputed MDANCE MSD matrix** with
`d_MSD(i,j) = ||x_i - x_j||^2 / (2*N_atoms)`. For each cluster this equals
`sum_i ||x_i - mean(C)||^2 / (2*N_atoms)` in MDANCE MSD units, so it is the
within-cluster dispersion of the representation MDANCE actually clusters in.

Conventional Euclidean k-means inertia is **not** applied to the precomputed
distance matrix (that would square distances twice and be dimensionally
wrong). The K scan clusters with MDANCE-native `KmeansNANI` (`metric='MSD'`)
on the same feature matrix; scikit-learn KMeans is used only as a fallback if
the native call fails, and the algorithm used is recorded in the JSON.

### Elbow detection and recommendation

* Kneedle (`kneed.KneeLocator`, convex/decreasing) when available; otherwise a
  normalized maximum-deviation detector.
* If a clear elbow exists, the elbow K and its immediate neighbours are
  eligible and the best average ordinal rank over silhouette, Davies–Bouldin,
  Calinski–Harabasz and bootstrap-ARI stability wins (ties → lower
  dispersion).
* If **no** clear knee exists, the curve is reported as
  `elbow_ambiguous: true`, every scanned K stays eligible, and the supporting
  metrics decide. No arbitrary K is forced.
* Population distribution and stability are always recorded, so the choice is
  auditable.

### How the selected K is applied

| Method | Uses selected K? |
|---|---|
| NANI (KmeansNANI) | **yes** |
| DIVINE | **yes** |
| HELM | **yes** |
| Agglomerative | **yes** |
| eQual | no — native threshold/seed behavior |
| mdBIRCH | no — native threshold behavior |
| DBSCAN | no — native eps/density behavior |
| GROMOS/Daura | no — native distance cutoff |

Disable with `k_selection.enabled: false` (or `--no-auto-k`); K-based methods
then sweep `clustering.k_min..clustering.k_max` as before. Partitions are never
reconciled: methods that disagree keep their own labels and populations.

## 2. Representative frames for every method

`representatives.export_all_methods: true` (default) writes, for each method
that produced at least one partition with ≥ 2 clusters:

```
representatives/representative_frames/
├── NANI/            cluster_001_medoid.pdb, cluster_001_centroid.pdb, representatives.csv
├── DIVINE/          ...
├── HELM/            ...
├── Agglomerative/   ...
├── eQual/           ...   (when the threshold sweep yields ≥ 2 clusters)
├── mdBIRCH/         ...
├── DBSCAN/          ...
└── GROMOS/          ...   (from the GROMACS benchmark sweep)
```

Each `representatives.csv` records, per cluster: population, population
fraction, medoid analysis frame, **original trajectory frame**, simulation
time (ps), nearest-to-centroid frame/original frame/time, and the MSD between
them. Every PDB is a real trajectory frame (PBC-corrected, reference-aligned
by preprocessing), never an averaged centroid.

The chosen solution per method is the best validation-ranked one of that
method; `method_representatives.json` records the choice and, for methods with
only degenerate partitions (e.g. eQual returning a single cluster), the reason
nothing was exported.

## 3. Independent method recommendation

`representatives/method_recommendation.txt` and
`representatives/RECOMMENDED_METHOD_FRAME.pdb` give the top
validation-ranked method and the medoid of its largest cluster. This is
**separate** from the composite 0–100 frame score, which still produces
`representatives/BEST_REPRESENTATIVE_FRAME.pdb` and the final banner.
