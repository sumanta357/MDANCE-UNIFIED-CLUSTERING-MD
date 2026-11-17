"""Canonical, simplified output directory layout (pipeline-added).

The pipeline computes exactly the same numbers as before; this module only
decides where each artefact is written. The main tree contains the results a
user actually needs, while detailed diagnostics live under
``04_ADVANCED_ANALYSIS`` and are never mixed into the primary report files.
"""
from __future__ import annotations

import os

FINAL_DIR = "01_FINAL_RESULTS"
METHOD_DIR = "02_METHOD_COMPARISON"
METHOD_REPS_DIR = os.path.join(METHOD_DIR, "METHOD_REPRESENTATIVES")
PLOTS_DIR = "03_PLOTS"
ADVANCED_DIR = "04_ADVANCED_ANALYSIS"
CANDIDATE_DIR = os.path.join(ADVANCED_DIR, "CANDIDATE_FRAMES")
MEDOID_DIR = os.path.join(ADVANCED_DIR, "CLUSTER_MEDOIDS")
SENSITIVITY_DIR = os.path.join(ADVANCED_DIR, "SENSITIVITY_ANALYSIS")
INTERMEDIATE_DIR = os.path.join(ADVANCED_DIR, "INTERMEDIATE_RESULTS")
DIAGNOSTIC_PLOTS_DIR = os.path.join(ADVANCED_DIR, "DIAGNOSTIC_PLOTS")
PREPROCESSING_DIR = "05_PREPROCESSING"
PROCESSED_DIR = os.path.join(PREPROCESSING_DIR, "PROCESSED_TRAJECTORY")
REFERENCE_DIR = os.path.join(PREPROCESSING_DIR, "REFERENCE_STRUCTURE")
LOGS_DIR = "logs"

#: Directories created at startup (empty detail folders are intentional: they
#: show where optional ``--full-output`` artefacts will appear).
TREE = [
    FINAL_DIR, METHOD_DIR, METHOD_REPS_DIR, PLOTS_DIR,
    ADVANCED_DIR, CANDIDATE_DIR, MEDOID_DIR, SENSITIVITY_DIR,
    INTERMEDIATE_DIR, DIAGNOSTIC_PLOTS_DIR,
    PREPROCESSING_DIR, PROCESSED_DIR, REFERENCE_DIR, LOGS_DIR,
]


def ensure_tree(out_dir: str) -> None:
    """Create the simplified output tree under ``out_dir``."""
    for sub in TREE:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def p(out_dir: str, *parts: str) -> str:
    """Join a path below the output root."""
    return os.path.join(out_dir, *parts)


def rel(out_dir: str, path: str) -> str:
    """Path relative to the output root, always using '/' separators."""
    return os.path.relpath(path, out_dir).replace(os.sep, "/")
