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

### Per-method K (optional)

`k_selection.per_method: true` runs an **independent elbow scan for each** of
NANI, Divine, HELM and Agglomerative (`k_selection.per_method_families`): each
method is clustered at every K in the scan range, its own dispersion curve is
built from its own partitions, and the same elbow/supporting-metric rule picks
its K. Results go to `k_selection_per_method.csv` / `.json` and
`k_selection_elbow_<family>.png`; the per-method K values are logged and used
in the sweep. Methods whose scan fails (e.g. a method that refuses very small
K on few frames) fall back to the shared selected K, with a logged warning.
This costs one full K sweep per method, so it is **off by default**.

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

## 3. Cross-method pose comparison (the final pose)

`scoring.cross_method_poses: true` (default) makes **every method's
representative poses compete in one ranked table**. For each method the chosen
solution's clusters contribute their medoid and (when it differs) the frame
nearest the centroid, giving up to `2 * n_clusters` poses per method:

* `representatives/pose_comparison.csv` — ranked rows with `method`, `family`,
  `kind`, `method_k`, cluster, population, analysis/original frame, time, the
  six component scores and `final_score`;
* `representatives/POSE_COMPARISON_REPORT.txt` — best pose per method plus the
  overall winner;
* `representatives/candidate_frames/*.pdb` — one PDB per compared pose, named
  `score<...>_<METHOD>_cluster<N>_frame<orig>.pdb`;
* `representatives/BEST_REPRESENTATIVE_FRAME.pdb` and the final banner — the
  winning pose, **whatever method produced it**.

Each pose is scored inside **its own** method's partition: population,
representativeness (distance to that cluster's medoid), stability and quality
typicality all refer to the method that generated the pose. Only the six
component metrics are min-max normalized across the whole cross-method
candidate set.

## 4. Independent method recommendation

`representatives/method_recommendation.txt` and
`representatives/RECOMMENDED_METHOD_FRAME.pdb` give the top
validation-ranked method and the medoid of its largest cluster, and the same
file records whether the cross-method pose winner came from that method.
`reports/METHOD_AND_POSE_SUMMARY.txt` states both answers together:

1. **best method** by cluster validation (ranking in
   `clustering/clustering_summary.csv`), with its representative structure;
2. **best pose** by the composite frame score across all methods, with its
   frame indices, time and score, plus `same method as above: YES/NO`.

The composite frame score remains separate from the cluster-validation
ranking; both are reported explicitly instead of being conflated.
