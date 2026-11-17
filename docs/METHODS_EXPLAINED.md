# MDANCE Unified Pipeline — Methods Explained

This guide explains, in plain language, what the pipeline does, what every
clustering method is doing, and how to read the results. It is written for
someone who knows MD trajectories but does not want to reverse-engineer the
source code.

* For the verified MDANCE API details, see `docs/verified_mdance_api.md`.
* For the automatic-K and representative-file mechanics, see
  `docs/automatic_k_and_representatives.md`.
* For installation and command-line usage, see `README.md`.

---

## 1. The 60-second overview

You give the pipeline a trajectory, a topology, and an atom selection (for
example `"protein and backbone"`). It then:

1. **Cleans the trajectory** — PBC wrapping/unwrapping, centering, alignment
   to a reference frame, time window and stride filtering.
2. **Measures structural similarity** — builds one distance matrix where the
   distance between any two frames is MDANCE's **MSD** (mean squared
   deviation).
3. **Picks a number of clusters (K)** automatically with an elbow scan.
4. **Runs eight clustering method families** (with their parameter sweeps)
   on that same MDANCE distance representation.
5. **Scores and ranks** every clustering solution with cluster-quality
   metrics.
6. **Benchmarks** its GROMOS/Daura implementation against the GROMACS
   `gmx cluster -method gromos` algorithm.
7. **Exports representative frames** for every cluster of every method.
8. **Chooses one final representative frame** by a 0–100 composite score
   across all methods.
9. **Writes CSV/JSON/TXT reports and plots** that explain every choice.

The scientific target is the **final frame** — usually for docking, MD
analysis, or figure preparation — not the clustering itself.

---

## 2. Basic ideas you need

| Term | Meaning in this pipeline |
|---|---|
| **Frame** | One snapshot of the trajectory. Only real frames are ever exported, never averaged coordinates. |
| **Analysis frame** | A frame that survived `stride`, time-window, and equilibration filtering. It has an internal index (0, 1, 2, …) and an **original frame index** in the input XTC. |
| **MSD** | MDANCE's mean squared deviation, `sum((x_i - x_j)^2) / (2 * N_atoms)`. It is a squared distance, not RMSD. `RMSD = sqrt(2 * MSD)`. |
| **Distance matrix** | An `n_frames × n_frames` table of MSD values. Matrix-based methods use this exact table; methods that need raw coordinates compute the same MSD definition on the same feature matrix, which keeps comparisons fair. |
| **Cluster** | A group of frames that are structurally similar to each other. |
| **Medoid** | The real frame in a cluster that is most central (lowest average MSD to the other members). |
| **Centroid** | The mathematical average of a cluster. It is usually **not** a physically valid structure, so it is never exported. |
| **Representative frame** | A real frame chosen to stand for a cluster: the medoid, or the frame nearest the centroid. |
| **K** | The requested number of clusters. Not every method uses K. |
| **Noise label (-1)** | Frames some methods refuse to put in any cluster (DBSCAN, eQual). They are not exported as a cluster. |
| **Native** | A method implemented inside the MDANCE package itself. |
| **Pipeline-added** | A method implemented by this pipeline for comparison, clearly labeled in code and reports. |

---

## 3. The workflow at a glance

| Phase | What happens | Main outputs |
|---|---|---|
| 1–2 Preprocess | Load, validate, PBC, center, align, filter frames | `05_PREPROCESSING/` |
| 3 Features | Build the MDANCE MSD matrix | `04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/features/` |
| 3b K selection | Elbow scan over K = 2..15 | `03_PLOTS/K_SELECTION.png` + `04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/k_selection/` |
| 4 Clustering | Run all eight methods | `02_METHOD_COMPARISON/METHOD_RANKING.csv` |
| 5 Validation | Score and rank every solution | `04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/solution_metrics.csv` |
| 6 Benchmark | GROMOS/Daura sweep vs the best solution | `02_METHOD_COMPARISON/GROMACS_COMPARISON.csv` |
| 7 Representatives | Best representative pose per method | `02_METHOD_COMPARISON/METHOD_REPRESENTATIVES/` |
| 8–9 Pose scoring | Score all representative poses 0–100 | `02_METHOD_COMPARISON/POSE_RANKING.csv` |
| 10 Plots | Four primary plots + diagnostics | `03_PLOTS/` + `04_ADVANCED_ANALYSIS/DIAGNOSTIC_PLOTS/` |
| 11–12 Reports | Final plain-language report | `01_FINAL_RESULTS/` |

---

## 4. Automatic K selection (the elbow)

### What it does

Clustering methods that need K must be told how many groups to make. This
pipeline estimates K from the data instead of using a fixed value:

1. For every K from 2 to 15 (configurable, hard cap 20), cluster the frames
   with MDANCE's NANI k-means.
2. Measure how tightly packed the clusters are:

   ```
   W(K) = sum over clusters C of ( 1/|C| * sum of MSD(i,j) for i<j in C )
   ```

   Lower W means tighter clusters. This is computed directly from the MDANCE
   MSD matrix, so it matches the representation every other method uses.
3. Find the "elbow" — the K after which adding more clusters stops buying much
   improvement. The pipeline uses the Kneedle algorithm when available and a
   geometric fallback otherwise.
4. If a clear elbow exists, only `K-1`, `K`, and `K+1` around the elbow are
   candidates. If the curve has no clear knee, the pipeline **says so**
   (`elbow_ambiguous: true`) and lets the supporting metrics decide instead of
   inventing a K.
5. Supporting metrics break ties: silhouette, distance-space Davies–Bouldin,
   distance-space Calinski–Harabasz, and bootstrap-ARI stability.

The scan range, the elbow, the ambiguity flag, every K's metrics, and the
selection rule are all written to
`04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/k_selection/k_selection.json`
and `.csv`, with a plot in `k_selection_elbow.png`.

### Which methods use K?

| Method | Uses the automatic K? |
|---|---|
| NANI | **Yes** |
| DIVINE | **Yes** |
| HELM | **Yes** |
| Agglomerative | **Yes** |
| eQual | No — threshold based |
| mdBIRCH | No — threshold based |
| DBSCAN | No — density based |
| GROMOS/Daura | No — cutoff based |

Threshold and density methods keep their native behaviour because forcing K on
them would change what makes them useful. With `--no-auto-k`, the K-based
methods sweep `clustering.k_min..k_max` instead.

Optional: `k_selection.per_method: true` runs a separate elbow scan for NANI,
DIVINE, HELM, and Agglomerative, so each can choose its own K. This is slower
(one full K sweep per method) and is off by default.

---

## 5. The eight clustering methods

All methods use the *same* preprocessed frames and the *same* MDANCE MSD
definition — either as the precomputed distance matrix or as the metric on the
same feature matrix. They differ only in how they group frames.

### 5.1 NANI (KmeansNANI)

**Plain idea:** k-means, but the starting centres are chosen smartly from the
trajectory instead of at random.

**Quick facts**

| | |
|---|---|
| Type | Partitional, centroid based |
| Uses K | Yes |
| Native MDANCE | Yes (`mdance.cluster.nani.KmeansNANI`) |
| Default settings | initializers `strat_all`, `comp_sim`, `k-means++`; `percentage: 10` |
| Output names | `nani/strat_all/k3`, `nani/comp_sim/k3`, … |

**How it works.** NANI picks a small, representative sample of frames as
starting points ("initiators"). The `init_type` controls how that sample is
chosen:

* `strat_all` — stratified sampling across the conformational space.
* `comp_sim` — starts from frames with high complementary similarity
  (central frames).
* `k-means++` — the classic distance-weighted seeding.
* `quota` — fills bins of conformational space with a quota per bin.

The pipeline runs every configured initializer; NANI itself finishes with a
refinement step. Different initializers can produce different partitions; all
are kept and ranked separately, which is why you may see several NANI rows for
one K.

**Strengths:** fast, robust, good default choice for MD data. Complementary
similarity starts tend to pick genuinely populated states.

**Watch out:** like all k-means methods it assumes roughly round clusters and
will split a large cluster rather than call it noise.

---

### 5.2 DIVINE (divisive clustering)

**Plain idea:** start with everything in one group and repeatedly split
clusters until you reach K, then optionally clean up the boundaries with
k-means refinement.

**Quick facts**

| | |
|---|---|
| Type | Divisive, hierarchical |
| Uses K | Yes |
| Native MDANCE | Yes (`mdance.cluster.divine.Divine`) |
| Default settings | `split: weighted_MSD`, `anchors: nani`, `refine: true` |
| Output names | `divine/weighted_MSD/nani/k3`, … |

**How it works.** DIVINE performs top-down divisive clustering: it chooses a
cluster and an anchor frame inside it, splits that cluster around the anchor,
and repeats until K clusters exist. With `refine: true` the final boundaries
are adjusted by an internal k-means refinement.

* `split` controls how the split direction is scored (`MSD`, `radius`,
  `weighted_MSD`).
* `anchors` controls how the split anchor is chosen (`nani` starts from NANI
  seeds, `outlier_pair`, `splinter_split`).

**Strengths:** good when one conformational state is much broader than the
others, because it spends splits where the data is badly represented.

**Watch out:** divisions are greedy and not reversible; early splits shape the
final result. Compare with NANI before trusting a subtle state.

---

### 5.3 HELM (hierarchical merging)

**Plain idea:** make a fine-grained partition first, then merge neighbouring
clusters until only K remain — like coarsening a map.

**Quick facts**

| | |
|---|---|
| Type | Agglomerative merging |
| Uses K | Yes |
| Native MDANCE | Yes (`mdance.cluster.helm.HELM`) |
| Default settings | initial partition from k-means (`helm_initial_k: 10`), `merge_scheme: intra` |
| Frame limit | `helm_max_frames: 20000` (configurable) |
| Output names | `helm/intra/k3`, … |

**How it works.** HELM starts from a coarse partition (k-means with at least
`helm_initial_k` clusters) and merges the pair of clusters that are most
similar, using n-ary cluster summaries rather than pair-by-pair distances. The
`merge_scheme` controls whether merging looks inside clusters (`intra`),
between them (`inter`), or a mix (`half`).

**Strengths:** produces compact, well-separated groups and is less sensitive to
where k-means initially placed centres.

**Watch out:** it is the most expensive native method. Above
`helm_max_frames` it is skipped and
`04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/METHOD_COVERAGE.txt` says so.
It also cannot merge a starting partition that already has only one cluster.

---

### 5.4 Agglomerative (average / complete / Ward)

**Plain idea:** start with every frame as its own group and repeatedly join the
two closest groups until K remain.

**Quick facts**

| | |
|---|---|
| Type | Hierarchical agglomerative |
| Uses K | Yes |
| Native MDANCE | **No** — pipeline-added baseline |
| Defaults | linkages `average`, `complete`, `ward_on_coords` |
| Output names | `agglom/average/k3`, `agglom/complete/k3`, `agglom/ward/k3` |

**How it works.**

* `average` joins the pair with the smallest average distance between all
  members. It runs directly on the precomputed MSD matrix.
* `complete` joins the pair whose *worst* pair of members is closest. Also on
  the MSD matrix.
* `ward` minimises the increase in within-cluster variance. Ward cannot work
  from a precomputed matrix, so it is fitted on the coordinate vectors; the
  config label is `ward_on_coords` and the result is reported as `ward`.

**Strengths:** deterministic, no random seeds, and a useful sanity check
against the MDANCE-native methods.

**Watch out:** single-linkage-style chaining is avoided by using average or
complete, but all hierarchical methods commit to early merges. Ward uses a
different distance geometry from the other methods, which is stated in the
reports.

---

### 5.5 eQual (ExtendedQuality)

**Plain idea:** put a ball of radius *t* around a seed frame; every frame
inside the ball joins that cluster; repeat with a new seed until no frames are
left. Frames in sparse regions can be rejected as low-density.

**Quick facts**

| | |
|---|---|
| Type | Radius / density based |
| Uses K | No |
| Native MDANCE | Yes (`mdance.cluster.equal.ExtendedQuality`) |
| Defaults | seed methods `greedy`, `medoid`; `equal_min_samples: 0.05` |
| Output names | `equal/greedy/t0.545`, `equal/medoid/t1.2`, … |

**How it works.** The threshold *t* is an MSD radius. Seeds are chosen by
`seed_method`:

* `greedy` — repeatedly take the frame with the most unassigned neighbours.
* `medoid` — seed from central frames first.
* `comp_sim` — seed from high complementary-similarity frames.

`min_samples` is eQual's native low-density rejection: a value below 1 is a
fraction of frames, an integer is an absolute count. Frames that do not pass
the density test are left unassigned (-1).

The pipeline sweeps many thresholds: quantiles of the pairwise MSD distribution
**plus** thresholds on the nearest-neighbour distance scale. The latter matter,
because pairwise quantiles can be far outside the range where neighbourhoods
actually exist, which would otherwise collapse eQual to one cluster or all
noise.

**Strengths:** does not assume round clusters, and can leave genuinely
transitional frames unassigned.

**Watch out:** the result depends strongly on *t*. Read the threshold in the
solution name and compare several thresholds in
`02_METHOD_COMPARISON/METHOD_RANKING.csv` and the solution-level
`04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/solution_metrics.csv`; do not treat
one eQual row as definitive.

---

### 5.6 mdBIRCH (streaming BIRCH)

**Plain idea:** summarise the data as a tree of small "micro-clusters", then
group the summaries. It is designed to handle data in a stream with limited
memory.

**Quick facts**

| | |
|---|---|
| Type | Hierarchical, threshold based |
| Uses K | No |
| Native MDANCE | Yes (`mdance.cluster.mdbirch.mdBirch`) |
| Defaults | threshold quantiles `0.10 … 0.85` of pairwise MSD, `branching_factor: 50` |
| Output names | `mdbirch/t0.805`, … |

**How it works.** New frames are added to the closest existing micro-cluster
if they fall within `threshold`; otherwise a new micro-cluster is created.
Micro-clusters are then merged into final clusters. The pipeline converts its
MSD threshold into the RMSD-style tolerance mdBIRCH expects internally
(`t_inner = 2 * sqrt(MSD / n_features)`) and sweeps several thresholds.

**Strengths:** memory-efficient and good at separating dense states.

**Watch out:** it is easy to over-resolve. On a small trajectory you may see
20–60 clusters. Partitions with more clusters than `clustering.k_max` are
excluded from the validation ranking as degenerate, but their representative
frames can still be exported so the method is not silently dropped. Check
`METHOD_COVERAGE.txt`.

---

### 5.7 DBSCAN

**Plain idea:** a frame is "core" if it has at least `min_samples` neighbours
within radius `eps`. Core frames and their neighbours form clusters; everything
else is noise.

**Quick facts**

| | |
|---|---|
| Type | Density based |
| Uses K | No |
| Native MDANCE | **No** — pipeline-added baseline |
| Defaults | `dbscan_min_samples: 5`, `eps` quantiles `0.05, 0.15, 0.30, 0.50` + nearest-neighbour scale |
| Output names | `dbscan/eps1.54`, … |

**How it works.** Runs scikit-learn DBSCAN with `metric="precomputed"` on the
MDANCE MSD matrix. The `eps` sweep includes distance quantiles and
nearest-neighbour-scale values so that a usable density region is actually
tested.

**Strengths:** finds arbitrarily shaped clusters and explicitly labels noise,
which is useful for detecting transition frames.

**Watch out:** with `min_samples: 5` on a small number of analysis frames,
DBSCAN can label everything as noise or produce one cluster. Those runs are
reported as degenerate. Lower `min_samples` if you have few frames.

---

### 5.8 GROMOS / Daura

**Plain idea:** the algorithm behind `gmx cluster -method gromos`. Repeatedly
find the frame with the most neighbours inside a cutoff, make that a cluster,
remove its members, and repeat.

**Quick facts**

| | |
|---|---|
| Type | Greedy neighbour clustering |
| Uses K | No |
| Native MDANCE | **No** — faithful pipeline implementation |
| Defaults | cutoff sweep at quantiles `0.05, 0.10, 0.20, 0.35, 0.50, 0.70` of pairwise MSD |
| Output names | `gromos/c1.2`, `gromos/c3.82`, … |

**How it works.** For a cutoff *c*, count for every remaining frame how many
other remaining frames lie within *c*. The frame with the largest count becomes
the centre of a cluster containing all its neighbours; those frames are
removed; repeat. It is deterministic and uses no random seed.

GROMOS runs in the **benchmark phase**, not the ordinary clustering sweep, and
its partitions are saved under
`04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/clustering/labels/gromos/`. They
also participate in
the cross-method pose comparison and get their own representative frames.

**Strengths:** directly comparable to a standard GROMACS workflow, and a good
independent check that the MDANCE partitions are sensible.

**Watch out:** a too-small cutoff produces one cluster per frame group; a
too-large cutoff merges everything. That is why the cutoff is swept and
reported, not hidden.

---

### 5.9 Side-by-side summary

| Method | Family | Native? | Uses K? | Main knob | Typical speed | Good for |
|---|---|---|---|---|---|---|
| NANI | k-means | Yes | Yes | `init_type`, `percentage` | Fast | General default |
| DIVINE | Divisive | Yes | Yes | `split`, `anchors` | Medium | Uneven cluster widths |
| HELM | Merging | Yes | Yes | `helm_initial_k`, `merge_scheme` | Slow | Compact, separated states |
| Agglomerative | Hierarchical | No | Yes | `linkage` | Medium | Deterministic cross-check |
| eQual | Radius | Yes | No | `threshold`, `seed_method` | Fast | Arbitrary shapes, noise |
| mdBIRCH | Threshold tree | Yes | No | `threshold` | Fast | Dense states, memory limits |
| DBSCAN | Density | No | No | `eps`, `min_samples` | Fast | Noise / transition frames |
| GROMOS | Greedy neighbours | No | No | `cutoff` | Fast | GROMACS comparison |

---

## 6. How clustering solutions are ranked (validation)

Every solution — every method, every parameter setting — is scored with the
same metrics, all computed from the same MSD matrix.

| Metric | Plain meaning | Better when | Weight |
|---|---|---|---|
| Silhouette | How much closer a frame is to its own cluster than to the nearest other cluster | Higher | 0.22 |
| Davies–Bouldin | Within-cluster spread divided by between-cluster distance, generalised to MSD space | Lower | 0.18 |
| Calinski–Harabasz | Between-cluster separation divided by within-cluster spread | Higher | 0.14 |
| Bootstrap stability | Re-run the method on 80% subsamples and measure ARI agreement | Higher | 0.16 |
| Population balance | Smallest cluster fraction divided by largest | Higher | 0.10 |
| Compactness | Population-weighted mean MSD from members to their medoid | Lower | 0.08 |
| Separation | Smallest MSD between any two cluster medoids | Higher | 0.07 |
| Diversity | Mean between-cluster MSD divided by mean within-cluster MSD | Higher | 0.05 |

Details that matter:

* Each metric is min-max normalised across the solutions that have it. Missing
  metrics (for example DB with singletons) are excluded **for that solution**
  and the remaining weights are renormalised — they are never silently set to
  zero.
* Stability is only computed for the top
  `validation.bootstrap_top_solutions` solutions (default 5), because it
  requires re-running the method.
* **Degenerate partitions are excluded** from the ranking: fewer than
  `clustering.k_min` clusters, more than `clustering.k_max` clusters, or a
  partition where every cluster is a singleton. Their reason is recorded in
  the output.
* The winner here is the **recommended method**, not necessarily the source of
  the final frame. See section 8.

Results: `02_METHOD_COMPARISON/METHOD_RANKING.csv` (one row per method, with
FAILED/SKIPPED states) and
`04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/solution_metrics.csv` (every
parameter variant with all metrics and ranks).

---

## 7. Benchmarking against GROMACS

The benchmark answers two questions: *does the pipeline reproduce GROMACS-style
clustering?* and *how similar are the partitions numerically?*

### Tier 1 — GROMOS/Daura on the identical matrix

The pipeline runs its own faithful Daura implementation at several cutoffs on
the **same** MDANCE MSD matrix used by every other method. This removes every
difference except the clustering algorithm itself. For each cutoff it reports:

* cluster count and runtime/memory,
* mean intra-cluster MSD and minimum inter-medoid MSD,
* **ARI** (adjusted Rand index) against the validation-ranked best solution,
* **population L1** — difference in the cluster-size profile (0 = identical
  sizes; 1 = completely different),
* **top-5 population divergence** — average difference among the five largest
  cluster fractions.

How to read ARI: 1.0 means the two partitions are identical up to relabelling;
about 0 means agreement no better than chance; negative means worse than
chance. ARI is **descriptive**, not a significance test — the report says this
explicitly. There are no p-values.

### Tier 2 — native `gmx cluster`

If a `gmx` binary is available, the pipeline runs
`gmx cluster -method gromos` automatically on inputs built from exactly the
MDANCE analysis atom selection (structure + trajectory + index group). Its
cutoff is converted from the MDANCE MSD units of the distance matrix to an
RMSD cutoff in nm (`RMSD_nm = sqrt(2 * MSD_A^2) / 10`), because `gmx` expects
nm. Per-frame native membership is parsed from the `-clid` XVG, so the native
partition receives the same ARI / population-L1 / top-5 comparison against
the best MDANCE solution as the GROMOS tier. The detailed `-g` log and all
gmx output files are kept in
`04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/gromacs_benchmark/`.

Native GROMACS uses its own RMSD/superposition definition and is reported as
a separate row/tier, never mixed into the MDANCE GROMOS row. A failed native
run is reported as FAILED with its error and is never counted as a successful
benchmark. Configure the native cutoff with
`benchmark.native_gmx_cutoff_quantile` or override it with
`benchmark.native_gmx_cutoff_nm`.

Files: `02_METHOD_COMPARISON/GROMACS_COMPARISON.csv` and
`04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/benchmarking_report.txt`.

---

## 8. Representative frames and the final 0–100 score

### Medoid vs nearest-to-centroid

For each cluster the pipeline exports two real frames when they differ:

* **medoid** — the most central actual frame;
* **nearest to centroid** — the real frame closest to the mathematical average.

Both are actual trajectory frames with PBC correction and alignment applied.
A centroid itself is never written as a structure.

Every method that produced at least one partition with two or more clusters
contributes its single highest-scoring representative to
`02_METHOD_COMPARISON/METHOD_REPRESENTATIVES/<METHOD>.pdb` (one row per method
in `POSE_RANKING.csv`). With `--full-output`, the per-cluster
`cluster_NNN_medoid.pdb` / `cluster_NNN_centroid.pdb` files and a
`representatives.csv` are also written under
`04_ADVANCED_ANALYSIS/CLUSTER_MEDOIDS/<METHOD>/`.

### The six pose metrics

Every candidate pose (medoid or nearest-centroid of every cluster of every
method) is scored from 0 to 100:

| Component | Weight | What it rewards |
|---|---|---|
| Representativeness | 0.25 | Being close to the centre of its own cluster (the medoid itself scores highest) |
| Population | 0.15 | Belonging to a well-populated conformational state |
| Stability | 0.20 | Being temporally consistent (similar to nearby frames) and having a typical RMSD for its cluster |
| Quality | 0.15 | Having a typical radius of gyration and finite coordinates |
| Interaction | 0.15 | Retaining persistent protein–ligand contacts (only when a ligand exists; the weight is dropped and the rest renormalised otherwise) |
| Conformational | 0.10 | High MDANCE complementary-similarity centrality and not being a local outlier |

Each component is min-max normalised across all candidates, then combined with
the configured weights. A cluster with one frame gets a capped population
score (60) so a singleton cannot win on popularity alone.

### Two different answers

The pipeline deliberately reports **two independent results**:

1. **Recommended method** — the best cluster-quality ranking from section 6.
   Its representative frame is `01_FINAL_RESULTS/BEST_METHOD.pdb`.
2. **Best pose** — the highest composite frame score across *all* methods.
   Its file is `01_FINAL_RESULTS/BEST_POSE.pdb`.

`04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/METHOD_AND_POSE_SUMMARY.txt` states
both, and whether the same method produced them. A disagreement is informative: it means the method with the
best overall cluster structure did not contain the single most representative
frame.

Because each pose is scored inside **its own** method's partition, methods
that produce identical partitions can produce identical scores and tie. In
that case the frame is the same; use `METHOD_AND_POSE_SUMMARY.txt` and the
partition agreement, not the tie order, to choose a method.

### Sensitivity

Weights are perturbed 120 times (Dirichlet draws around the configured
weights). `04_ADVANCED_ANALYSIS/SENSITIVITY_ANALYSIS/sensitivity_analysis.csv`
records which frame wins in each
draw. If one frame wins most draws, the choice is robust to weight assumptions;
if the wins are spread out, the ranking is genuinely uncertain.

---

## 9. Output files: what to read first

```
MDANCE_Analysis/
├── README.txt                    where to look first
├── 01_FINAL_RESULTS/
│   ├── FINAL_REPORT.txt          ← start here (plain language)
│   ├── FINAL_SUMMARY.csv         key numbers
│   ├── BEST_POSE.pdb             highest-scoring pose
│   └── BEST_METHOD.pdb           recommended-method pose
├── 02_METHOD_COMPARISON/
│   ├── METHOD_RANKING.csv        one row per method, incl. FAILED/SKIPPED
│   ├── POSE_RANKING.csv          best pose per successful method
│   ├── CLUSTER_POPULATION.csv
│   ├── GROMACS_COMPARISON.csv
│   └── METHOD_REPRESENTATIVES/<METHOD>.pdb
├── 03_PLOTS/
│   ├── K_SELECTION.png
│   ├── CLUSTER_COMPARISON.png
│   ├── POSE_RANKING.png
│   └── GROMACS_COMPARISON.png
├── 04_ADVANCED_ANALYSIS/
│   ├── CANDIDATE_FRAMES/        all candidates (--full-output)
│   ├── CLUSTER_MEDOIDS/         per-cluster PDBs (--full-output)
│   ├── SENSITIVITY_ANALYSIS/
│   ├── DIAGNOSTIC_PLOTS/
│   └── INTERMEDIATE_RESULTS/
│       ├── k_selection/         elbow CSV/JSON + original plot
│       ├── features/            feature matrix + distance matrix
│       ├── clustering/          label arrays + optimal model card
│       ├── solution_metrics.csv / solution_summary.csv / .json
│       ├── candidate_frame_scores.csv / pose_candidates_all.csv
│       ├── METHOD_COVERAGE.txt / method_coverage.json
│       ├── method_recommendation.txt / method_representatives.json
│       ├── scoring_weights.yaml / scoring_metrics.csv
│       ├── benchmarking_report.txt
│       ├── FINAL_FRAME_SELECTION_REPORT.txt
│       └── complete_analysis_report.txt
├── 05_PREPROCESSING/
│   ├── PROCESSED_TRAJECTORY/{processed.xtc, processed_pbc.xtc}
│   ├── REFERENCE_STRUCTURE/reference.gro
│   └── preprocessing_report.txt
└── logs/analysis.log
```

`--full-output` adds every scored candidate PDB and per-cluster medoid PDB;
the default tree holds final results and one pose per successful method only.

**Recommended reading order**

1. `01_FINAL_RESULTS/FINAL_REPORT.txt` — the answer and why, in plain language.
2. `01_FINAL_RESULTS/FINAL_SUMMARY.csv` — the key numbers on one screen.
3. `02_METHOD_COMPARISON/METHOD_RANKING.csv` — how methods ranked and which
   failed or were skipped.
4. `02_METHOD_COMPARISON/POSE_RANKING.csv` — the best pose per method.
5. `03_PLOTS/K_SELECTION.png` — elbow K vs selected K.
6. `04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/benchmarking_report.txt` — the
   GROMOS comparison detail.
7. `04_ADVANCED_ANALYSIS/SENSITIVITY_ANALYSIS/sensitivity_analysis.csv` — how
   stable the winner is.

**Cluster numbers are 1-based in all reports, CSVs, and filenames.**

---

## 10. Configuration: the settings that matter

| Setting | Default | When to change it |
|---|---|---|
| `preprocess.selection` | `protein and backbone` | Use `name CA` for speed, or pocket residues for binding-site work |
| `preprocess.stride` | 1 | Increase to 5–10 for long trajectories; keep enough frames for clustering |
| `preprocess.skip_equil_frames` | 0 | Drop the equilibration part of the run |
| `k_selection.k_max` | 15 | Raise to 20 for very heterogeneous trajectories |
| `k_selection.per_method` | false | true if methods should choose independent K values |
| `clustering.helm_max_frames` | 20000 | Raise for long trajectories, at the cost of HELM runtime |
| `clustering.dbscan_min_samples` | 5 | Lower (2–3) for short trajectories with few analysis frames |
| `benchmark.enabled` | true | Keep on for the GROMACS comparison |
| `scoring.weights` | see section 8 | Raise `interaction` for protein–ligand work, `stability` for dynamics-focused work |
| `scoring.cross_method_poses` | true | false to score only the winner and runner-up solutions |

Ready-made configs:

* `configs/config.yaml` — backbone, general purpose.
* `configs/config_calpha.yaml` — Cα only, faster.
* `configs/config_protein_ligand.yaml` — ligand interaction scoring.
* `configs/config_binding_site.yaml` — pocket-focused selection.

---

## 11. Troubleshooting

**A method is missing or says `not_run`.**
Open `04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/METHOD_COVERAGE.txt` or
`method_coverage.json`. The
reason is recorded: disabled in config, HELM over its frame cap, distance
matrix unavailable (too many frames), or the method produced no solution.

**A method says `degenerate`.**
Its best partition had fewer than two clusters, or every cluster was a
singleton. Threshold methods (eQual, DBSCAN, mdBIRCH, GROMOS) need a usable
threshold range; the sweep already includes nearest-neighbour-scale values.
For short trajectories, lowering `dbscan_min_samples` or using more frames
(often via a smaller stride) helps.

**Very few analysis frames.**
`--stride 10` on 251 frames gives 26 analysis frames. With so few frames, K is
limited, DBSCAN with `min_samples: 5` can label everything as noise, and many
threshold sweeps collapse to one cluster. Use a smaller stride if you need
finer states; clustering 20–30 frames is only appropriate for a coarse view.

**mdBIRCH produced dozens of clusters.**
That is over-resolution. Those partitions are excluded from ranking as
degenerate when they exceed `clustering.k_max`, but their representatives are
still exported. Reduce its threshold quantiles if you want coarser partitions.

**HELM was skipped.**
Your analysis-frame count exceeded `clustering.helm_max_frames` (default
20000). The reason is logged and reported, not hidden.

**No ligand was detected.**
The interaction component is dropped and the other weights are renormalised;
`04_ADVANCED_ANALYSIS/INTERMEDIATE_RESULTS/scoring_weights.yaml` shows the
effective weights. Set
`preprocess.ligand_selection` explicitly if auto-detection missed your ligand.

**Memory use is high.**
The full distance matrix grows as frames². `features.max_frames_for_full_matrix`
(default 3000) skips matrix-based methods above that size; `use_float32` and
`chunk_size` control memory during construction. For very long trajectories,
analyze a representative subset with stride.

**`gmx` is not installed.**
The benchmark still runs the GROMOS/Daura tier on the identical MSD matrix and
records that native `gmx` was not attempted.

---

## 12. Scientific honesty and limits

* **Every exported structure is a real frame.** No averaged centroid is ever
  exported as a physical structure.
* **Methods are not reconciled.** If NANI and DBSCAN disagree, both partitions
  are kept and reported. The pipeline does not force a consensus.
* **The pose score is a heuristic, not a biological truth.** It combines
  documented geometric and temporal metrics with configurable weights. The
  sensitivity analysis shows how much the winner depends on those weights.
* **Benchmark statistics are descriptive.** ARI, population L1, and top-5
  divergence measure agreement between partitions. They are not hypothesis
  tests and carry no p-values.
* **Native and pipeline-added components are labeled.** NANI, DIVINE, HELM,
  eQual, and mdBIRCH are MDANCE-native; Agglomerative, DBSCAN, GROMOS/Daura,
  validation metrics, scoring, benchmarking, and reports are pipeline-added.
* **HELM and Shine/PRISM differ in scope.** Shine and PRISM are
  multi-trajectory pathway clusterers and are not run on single-trajectory
  frames. This limitation is stated rather than hidden.

---

## 13. Glossary

| Term | Definition |
|---|---|
| **ARI** | Adjusted Rand Index: agreement between two partitions, corrected for chance. 1 = identical, 0 = random. |
| **Bootstrap stability** | Mean ARI between a full-data partition and refits on random 80% subsets. |
| **Cutoff** | MSD radius used by GROMOS/Daura to decide whether two frames are neighbours. |
| **Dispersion W(K)** | Total within-cluster MDANCE MSD used by the elbow scan. Lower = tighter clusters. |
| **Elbow / knee** | The K where extra clusters stop producing large dispersion improvements. |
| **eps** | DBSCAN neighbourhood radius in MSD units. |
| **Medoid** | The actual frame with the lowest average MSD to its cluster members. |
| **MSD** | Mean squared deviation, `sum(dx²)/(2N)`. `RMSD = sqrt(2·MSD)`. |
| **Silhouette** | Per-frame measure of how much better it fits its own cluster than the nearest other cluster. |
| **Singleton** | A cluster containing exactly one frame. |
| **Threshold** | Generic radius/tolerance knob for eQual, mdBIRCH, and GROMOS. |
