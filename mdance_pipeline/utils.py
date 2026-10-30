"""Shared utilities: seeding, logging, timing, memory tracking, output helpers.

Mostly pipeline-added; the MDATE-native pieces are tagged at call sites.
"""
# pipeline-added module (MDANCE has no such orchestration utilities)
from __future__ import annotations

import csv
import json
import logging
import os
import random
import resource
import sys
import time
from contextlib import contextmanager

import numpy as np


def set_seed(seed: int | None) -> None:
    """Seed python, numpy and random for reproducibility.

    Note: scikit-learn KMeans accepts random_state explicitly; MDANCE nani
    initializers consume the global numpy RNG internally, so the global seed
    is required for reproducible NANI runs.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def get_logger(logfile: str | None = None, name: str = "mdance_pipeline",
               level: int | str = logging.INFO) -> logging.Logger:
    """Create (or reuse) a logger that logs to stdout and optionally a file."""
    logger = logging.getLogger(name)
    if getattr(logger, "_configured", False):
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
        fh = logging.FileHandler(logfile, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    logger._configured = True  # type: ignore[attr-defined]
    return logger


def peak_memory_mb() -> float:
    """Peak RSS of the process in MB (pipeline-added, resource-based)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def current_memory_mb() -> float:
    """Current RSS of the process in MB."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return 0.0


@contextmanager
def timed_step(logger: logging.Logger, label: str):
    """Context manager recording wall time and peak memory for one step."""
    t0 = time.time()
    m0 = peak_memory_mb()
    logger.info("== %s", label)
    try:
        yield
    except Exception as exc:
        logger.error("%s FAILED after %.1fs: %s", label, time.time() - t0, exc)
        raise
    else:
        logger.info("== %s finished in %.1fs (peak RSS %.0f MB)",
                    label, time.time() - t0, peak_memory_mb())
    finally:
        pass


def ensure_dirs(base: str, subdirs: list[str]) -> None:
    for d in [""] + subdirs:
        os.makedirs(os.path.join(base, d), exist_ok=True)


def write_csv(path: str, header: list[str], rows: list) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [v if not isinstance(v, float) or not np.isfinite(v) else
                 round(v, 6) for v in row])


def write_json(path: str, obj) -> None:
    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(x) for x in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return _clean(o.tolist())
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    with open(path, "w") as fh:
        json.dump(_clean(obj), fh, indent=2, default=str)


def fmt_float(x, digits: int = 2) -> str:
    """Format a float for human-readable reports (never 'nan')."""
    if x is None:
        return "N/A"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "N/A"
    return f"{xf:.{digits}f}"


def minmax_scale(values: np.ndarray, invert: bool = False) -> np.ndarray:
    """Min-max scale to [0, 1]; constant arrays → 0.5 everywhere.

    invert=True flips direction so that *smaller* raw values → larger scores.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0 or not np.isfinite(v).any():
        return np.full(v.shape, 0.5)
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < 1e-12:
        out = np.full(v.shape, 0.5)
    else:
        out = (v - vmin) / (vmax - vmin)
    if invert:
        out = 1.0 - out
    return out
