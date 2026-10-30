"""Phase 2 - trajectory preprocessing (PBC, centering, alignment, filters).

Uses MDAnalysis (the library MDANCE itself uses in `gen_traj_numpy`).
The heavy lifting is MDAnalysis-native; the orchestration is pipeline-added.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import MDAnalysis as mda
import MDAnalysis.transformations as transforms
from MDAnalysis.analysis import align as mda_align
from MDAnalysis.analysis import rms as mda_rms

PROTEIN_WATER_ION_RESNAMES = {
    "SOL", "WAT", "HOH", "TIP3", "TIP4", "ALK", "SPC", "SOD", "NA", "NA+",
    "CL", "CL-", "K", "K+", "MG", "ZN", "CA", "ION", "IONS",
}


def validate_inputs(cfg: dict, logger: logging.Logger) -> dict:
    """Input validation & compatibility checks (pipeline-added)."""
    p = cfg["preprocess"]
    report_lines: list[str] = []
    info: dict = {}

    for key in ("tpr", "gro", "xtc"):
        val = p.get(key)
        if val:
            if not os.path.isfile(val):
                raise FileNotFoundError(f"{key} file not found: {val}")
            size = os.path.getsize(val)
            if size == 0:
                raise ValueError(f"{key} file is empty: {val}")
            info[f"{key}_size_mb"] = size / 1e6
            logger.info("Input %-4s : %s (%.1f MB)", key, val, size / 1e6)
            report_lines.append(f"{key}: {val} ({size} bytes)")

    top = p.get("tpr") or p.get("gro")
    logger.info("Building topology from: %s (TPR preferred when available)",
                top or "<missing>")
    u = mda.Universe(top)
    info["n_atoms"] = u.atoms.n_atoms
    gromacs_tpr = p.get("tpr") is not None
    report_lines.append(f"n_atoms (topology): {u.atoms.n_atoms}")
    report_lines.append("topology source: " + ("TPR" if gromacs_tpr else "GRO"))

    ut = mda.Universe(top, p["xtc"], format="XTC")
    n_frames = ut.trajectory.n_frames
    info["n_frames"] = n_frames
    info["dt_ps"] = float(ut.trajectory.dt)
    info["total_time_ns"] = float(ut.trajectory.totaltime) / 1000.0
    logger.info("Trajectory: %d frames, dt = %.1f ps (total %.3f ns)",
                n_frames, info["dt_ps"], info["total_time_ns"])
    if n_frames == 0:
        raise ValueError("Trajectory contains no frames (corrupted XTC).")
    report_lines.append(f"n_frames: {n_frames}; dt: {info['dt_ps']:.3f} ps")

    sel = u.select_atoms(p["selection"])
    info["n_selection"] = sel.n_atoms
    if sel.n_atoms == 0:
        raise ValueError(
            f"Atom selection '{p['selection']}' matches 0 atoms. "
            "Check the selection string / topology.")
    logger.info("Selection '%s' -> %d atoms", p["selection"], sel.n_atoms)
    report_lines.append(f"selection '{p['selection']}' -> {sel.n_atoms} atoms")

    # Coordinate integrity check (non-finite coordinate detection).
    non_finite = 0
    ts = ut.trajectory[0]
    if not np.isfinite(ts.positions).all():
        non_finite += 1
        logger.warning("Frame 0 contains non-finite coordinates!")
    report_lines.append(f"non-finite coordinates in frame 0: {non_finite}")
    info["non_finite_frame0"] = bool(non_finite)

    info["report_lines"] = report_lines
    return info


def detect_ligand(u: mda.Universe, cfg_sel: str | None) -> tuple[bool, str | None]:
    """Auto-detect a bound ligand/cofactor (heuristic, pipeline-added).

    Heuristic: all residues that are not protein-like, not water/ions, and
    smaller than 60 residues. Returns (found, selection string). The user can
    override with `preprocess.ligand_selection`.
    """
    if cfg_sel and cfg_sel != "when_ligand":
        ag = u.select_atoms(cfg_sel)
        return (ag.n_atoms > 0, cfg_sel if ag.n_atoms > 0 else None)
    not_protein = u.atoms - u.select_atoms("protein")
    residue_grps = not_protein.residues
    keep_mask = []
    for res in residue_grps:
        if res.resname.strip().upper() in PROTEIN_WATER_ION_RESNAMES:
            keep_mask.append(False)
        elif len(res.atoms) > 100:
            keep_mask.append(False)   # likely a nucleic acid / big fragment
        else:
            keep_mask.append(True)
    keep = np.array(keep_mask, dtype=bool)
    # guard against oceans of water: ligand-like residues must be a small
    # fraction of ALL residues in the universe (not of the non-protein subset)
    if keep.sum() == 0 or keep.sum() > len(u.residues) * 0.5:
        return False, None
    names = sorted({r.resname for r in residue_grps[keep]})
    return True, " or ".join([f"resname {n}" for n in names])


def classify_frames(cfg: dict, u: mda.Universe) -> tuple[np.ndarray, np.ndarray]:
    """Frame interval / time-window / equilibration / stride selection.

    Returns (original_frame_indices, times_ps). Original indices and times are
    retained throughout the pipeline (requirement of the requested workflow).
    """
    p = cfg["preprocess"]
    idxs: list[int] = []
    times: list[float] = []
    start, end = p.get("start_time", 0.0), p.get("end_time")
    skip = int(p.get("skip_equil_frames", 0))
    for ts in u.trajectory:
        if ts.frame < skip:
            continue
        t = float(ts.time)
        if t is None:
            t = ts.frame * float(u.trajectory.dt)
        if t < start:
            continue
        if end is not None and t > end:
            continue
        idxs.append(ts.frame)
        times.append(t)
    stride = max(int(p.get("stride", 1)), 1)
    idxs = idxs[::stride]
    times = (np.asarray(times)[::stride]).tolist()
    if not idxs:
        raise ValueError(
            "No frames survive the start_time/end_time/stride/equilibration "
            "filters. Relax preprocess.* settings.")
    return np.asarray(idxs, dtype=int), np.asarray(times, dtype=float)


def preprocess_trajectory(cfg: dict, logger: logging.Logger) -> dict:
    """Full Phase-2 preprocessing (pipeline-added; MDAnalysis-native ops).

    Steps:
      1. load universe (TPR preferred, GRO fallback)
      2. validate selection
      3. classify frames (window/stride/equil)
      4. wrap → unwrap → center (PBC handling, box aware)
      5. write all-atom processed trajectory (ALL atoms: ligands & ions kept)
      6. rigid-body align to the reference frame (analysis-selected atoms)
      7. export processed.gro (reference frame) + features & metadata

    Returns dict with feature matrix, frame bookkeeping, ligand info.
    """
    p = cfg["preprocess"]
    out_dir = cfg["system"]["output_dir"]
    pre_dir = os.path.join(out_dir, "preprocessing")
    os.makedirs(pre_dir, exist_ok=True)

    top = p.get("tpr") or p.get("gro")
    u = mda.Universe(top, p["xtc"], format="XTC")
    u_top = mda.Universe(top)
    sel_atoms = u.select_atoms(p["selection"])
    logger.info("Loading trajectory (%d frames) ...", u.trajectory.n_frames)

    keep_idx, keep_time = classify_frames(cfg, u)
    logger.info("Analysis frames: %d (original indices kept: %d ... %d)",
                len(keep_idx), keep_idx[0], keep_idx[-1])

    # ----- PBC / center -----------------------------------------------------
    # GRO topologies carry no bonds: fall back to per-atom handling there.
    has_bonds = False
    try:
        has_bonds = u.bonds is not None and len(u.bonds) > 0
    except Exception:
        has_bonds = False
    compound = "fragments" if has_bonds else "atoms"
    protein_ag = u.select_atoms("protein")
    ag_center = protein_ag if protein_ag.n_atoms else u.atoms
    transform_list = []
    if p.get("wrap_molecules", True):
        transform_list.append(transforms.wrap(u.atoms, compound=compound))
    if p.get("wrap_molecules", True) and compound == "fragments":
        transform_list.append(transforms.unwrap(u.atoms, compound="fragments"))
    if p.get("center_protein", True):
        transform_list.append(
            transforms.center_in_box(ag_center, center="geometry"))
    if transform_list:
        u.trajectory.add_transformations(*transform_list)

    proc_xtc_prefix = os.path.join(pre_dir, "processed_pbc.xtc")
    with mda.Writer(proc_xtc_prefix, n_atoms=u.atoms.n_atoms, multiframe=True) as W:
        for ts in u.trajectory:
            if ts.frame in set(keep_idx.tolist()):
                W.write(u)
    logger.info("Wrote PBC-corrected all-atom trajectory: %s", proc_xtc_prefix)

    # ----- alignment to reference frame -------------------------------------
    final_xtc = proc_xtc_prefix
    ref_frame_in_processed = 0
    if p.get("reference") is not None and p.get("align", True):
        u_m = mda.Universe(top, proc_xtc_prefix, format="XTC")
        u_r = mda.Universe(top, proc_xtc_prefix, format="XTC")
        r = p.get("reference")
        if isinstance(r, int):
            ref_frame_in_processed = min(max(0, r), u_r.trajectory.n_frames - 1)
        u_r.trajectory[ref_frame_in_processed]
        ref_ag = u_r.select_atoms(p["selection"])
        if u_m.trajectory.n_frames > 1:
            mda_align.AlignTraj(u_m, u_r, select=p["selection"],
                                filename=os.path.join(pre_dir, "processed.xtc"),
                                in_memory=False).run()
            final_xtc = os.path.join(pre_dir, "processed.xtc")
            logger.info("Aligned trajectory written: %s (ref frame %d)",
                        final_xtc, ref_frame_in_processed)
        else:
            final_xtc = proc_xtc_prefix
            logger.info("Only one analysis frame - skipping alignment.")

    # ----- processed.gro (reference frame, all atoms) ------------------------
    u_g = mda.Universe(top, final_xtc, format="XTC")
    u_g.trajectory[ref_frame_in_processed]
    gro_path = os.path.join(pre_dir, "processed.gro")
    u_g.atoms.write(gro_path)
    logger.info("Wrote reference-frame structure: %s", gro_path)

    # ----- feature extraction (MDANCE `gen_traj_numpy` convention) -----------
    feature_rows: list[np.ndarray] = []
    frame_indices: list[int] = []
    times: list[float] = []
    n_frames_seen = u_g.trajectory.n_frames
    for i, ts in enumerate(u_g.trajectory):
        # MDANCE gen_traj_numpy convention: one row per frame, flattened as
        # x1,y1,z1, x2,y2,z2, ... of the SELECTED atoms
        feature_rows.append(
            ts.positions[sel_atoms.indices].astype(np.float64).reshape(-1))
        times.append(float(ts.time))
        frame_indices.append(int(ts.frame))
    X = np.asarray(feature_rows, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite tool positions survived preprocessing; "
                         "the trajectory is corrupted.")
    u_final = mda.Universe(top, final_xtc, format="XTC")  # kept for PDB writing
    sel_final = u_final.select_atoms(p["selection"])

    # RMSF of the selection over the analysis frames (stability metrics)
    rmsf_vals = np.zeros(sel_final.n_atoms, dtype=float)
    try:
        rmsf_an = mda_rms.RMSF(sel_final)
        rmsf_an.run(start=0, stop=n_frames_seen)
        rmsf_vals = np.asarray(rmsf_an.results.rmsf, dtype=float)
    except Exception as exc:  # pragma: no cover
        logger.warning("RMSF computation failed: %s", exc)

    ligand_found, ligand_sel = detect_ligand(u_final, p.get("ligand_selection"))
    ligand_indices = u_final.select_atoms(ligand_sel).indices \
        if ligand_found and ligand_sel else np.array([], dtype=int)
    logger.info("Ligand/cofactor detected: %s%s",
                ligand_found, f" (selection: {ligand_sel})" if ligand_found else "")

    info = {
        "paths": {"processed_xtc": final_xtc, "processed_gro": gro_path,
                  "pbc_xtc": proc_xtc_prefix},
        "X": X,
        "frame_indices": np.asarray(frame_indices, dtype=int),
        "times": np.asarray(times, dtype=float),
        "n_atoms_selected": int(sel_atoms.n_atoms),
        "selection": p["selection"],
        "rmsf": rmsf_vals,
        "ligand_found": bool(ligand_found),
        "ligand_selection": ligand_sel,
        "ligand_indices": ligand_indices,
        "universe": u_final,
        "selection_indices": sel_final.indices,
        "ref_frame_in_processed": int(ref_frame_in_processed),
        "dt_ps": float(u_g.trajectory.dt),
    }
    return info


def preprocessing_report(info: dict, cfg: dict) -> str:
    p = cfg["preprocess"]
    lines: list[str] = [
        "MDANCE PIPELINE - PREPROCESSING REPORT", "=" * 60,
        f"Trajectory              : {p.get('xtc')}",
        f"Topology                : {p.get('tpr') or p.get('gro')}"
        f" (source: {'TPR' if p.get('tpr') else 'GRO'})",
        f"Atom selection          : {p['selection']}",
        f"Selected atoms          : {info['n_atoms_selected']} "
        f"({info['X'].shape[1]} features per frame)",
        f"Analysis frames         : {len(info['frame_indices'])}",
        f"Original frame indices  : kept and reported for every frame",
        f"First/last original idx : {info['frame_indices'][0]} / "
        f"{info['frame_indices'][-1]}",
        f"First/last time (ns)    : {info['times'][0]/1000:.3f} / "
        f"{info['times'][-1]/1000:.3f}",
        f"Time step               : {info['dt_ps']:.3f} ps",
        f"Equilibration skipped   : {p.get('skip_equil_frames', 0)} frames",
        f"Time window             : {p.get('start_time')} - {p.get('end_time')} ps",
        f"Stride                  : {p.get('stride', 1)}",
        f"Wrap molecules          : {p.get('wrap_molecules', True)}",
        f"Center protein          : {p.get('center_protein', True)}",
        f"Alignment               : {p.get('align', True)} "
        f"(reference frame {info['ref_frame_in_processed']}, "
        f"selection '{p['selection']}')",
        f"Processed XTC (all atm) : {info['paths']['processed_xtc']}",
        f"Processed GRO           : {info['paths']['processed_gro']}",
        f"Ligand detected         : {info['ligand_found']}"
        + (f" (selection: {info['ligand_selection']})" if info['ligand_found'] else ""),
        "",
        "Notes:",
        "- PBC handling: wrap fragments -> unwrap fragments -> center protein",
        "  in the unit cell (MDAnalysis transformations).",
        "- Alignment: rigid-body, least-squares fit of the selected atoms of",
        "  every frame onto the reference frame (MDAnalysis AlignTraj).",
        "- ALL atoms (protein, ligand, ions, water) are kept in the output",
        "  trajectory; only the *feature/selection* space is restricted.",
        "- A mathematical k-means centroid is NOT a physical frame; the",
        "  pipeline only exports actual frames (see representative modules).",
    ]
    return "\n".join(lines)
