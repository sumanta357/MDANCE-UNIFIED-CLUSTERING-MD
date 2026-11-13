# Automatic K-selection and per-method representatives

The command-line workflow selects K on the preprocessed coordinate feature matrix used by MDANCE. The default scan is K=2..15; set `clustering.k_max: 20` in YAML to extend it. Values are capped at 20 and at the available frame count. NANI, DIVINE, HELM, and Agglomerative use the recommended K. eQual, mdBIRCH, DBSCAN, and Daura/GROMOS retain their threshold/density-based behavior.

Dispersion is defined as

`W = sum_C sum_{i in C} ||x_i - mean(C)||^2 / (2*N_atoms)`.

For cluster size n_C, this equals `(1/n_C) sum_{i<j in C} d_MSD(i,j)` with MDANCE pair distance `d_MSD(i,j)=||x_i-x_j||^2/(2*N_atoms)`. Thus no conventional K-means inertia is inferred from a precomputed distance matrix. Supporting silhouette, distance-space Davies–Bouldin and Calinski–Harabasz, bootstrap stability, and population distributions are saved with K-selection results. If the curve has no clear knee, the JSON records the ambiguity and the supporting metrics choose the recommendation.

Outputs: `clustering/k_selection/k_selection.csv`, `k_selection.json`, and `k_selection_elbow.png`. Each method family's highest-ranked solution gets separate medoid and nearest-centroid actual-frame PDBs in `representatives/representative_frames/<family>/`, plus frame/time metadata. The independent method recommendation is written to `representatives/method_recommendation.txt` and `RECOMMENDED_METHOD_FRAME.pdb`.

The full `Pipeline` API remains usable programmatically; to enable the same K-selection and all-method exports outside `run_mdance.py`, call `select_k` on its prepared X/D arrays and `export_all_methods(pipe)` after the run.
