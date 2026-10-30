"""Phases 11-12 - reports and final selection banner (pipeline-added)."""
from __future__ import annotations


def final_frame_report(best_row, top_rows, metric_explanations,
                       selected_cluster_info, cfg) -> str:
    """The exact banner required by the project brief (Phase 12)."""
    def line(t=""):
        return t

    wsum = cfg["scoring"]["weights"]
    def w(k, default=0.0): return float(wsum.get(k, default))
    tot_w = sum(w(k) for k in ("representativeness", "population", "stability",
                               "quality", "conformational"))
    scale = 1.0
    if w("interaction"):
        if best_row.get("score_interaction") is None:
            tot_w = sum(w(k) for k in wsum if k != "interaction")

    def contrib(k):
        v = best_row.get(f"score_{k}")
        if v is None:
            return "N/A (no ligand)"
        return f"{float(v):.2f}"

    lines = [
        "=" * 52,
        "FINAL BEST REPRESENTATIVE FRAME",
        "=" * 52,
        f"Selected Frame ID       : {best_row['analysis_frame']}",
        f"Original Frame Index    : {best_row['original_frame']}",
        f"Simulation Time         : {best_row['time_ps']/1000:.2f} ns",
        f"Selected Algorithm      : {best_row.get('algorithm', '')}",
        f"Distance Metric         : {best_row.get('metric', 'MSD')}",
        f"Optimal Number of Clusters: {best_row.get('optimal_k', '')}",
        f"Selected Cluster ID     : {best_row['cluster_id']}",
        f"Cluster Population      : {best_row['cluster_pop_frac']*100:.2f} %",
        "",
        f"Final Composite Score   : {best_row['final_score']:.2f} / 100",
        "",
        f"Representativeness Score: {contrib('representativeness')}",
        f"Cluster Population Score: {contrib('population')}",
        f"Structural Stability Score: {contrib('stability')}",
        f"Structural Quality Score: {contrib('quality')}",
        f"Interaction Preservation Score: {contrib('interaction')}",
        f"Conformational Representativeness Score: {contrib('conformational')}",
        "",
        "Output Structure:",
        "representatives/BEST_REPRESENTATIVE_FRAME.pdb",
        "=" * 52,
        "",
        "NEXT 5 HIGHEST-RANKED CANDIDATE FRAMES",
        "-" * 52,
    ]
    for r in top_rows[:5]:
        lines.append(
            f"#{r['rank']:<2} analysis frame {r['analysis_frame']:>6}"
            f" (original {r['original_frame']:>6})  "
            f"cluster {r['cluster_id']:>3} (pop {r['cluster_population']:>5}"
            f" = {r['cluster_pop_frac']*100:5.2f}%)   score {r['final_score']:>6.2f}")
    lines += ["", "METRIC DEFINITIONS (all explicit):"]
    for name, expl in metric_explanations:
        lines.append(f"* {name}: {expl}")
    lines += [
        "",
        "SCIENTIFIC NOTES:",
        "- The selected frame is an ACTUAL trajectory frame; coordinates are the",
        "  PBC-corrected, reference-aligned coordinates of that single frame -",
        "  never averaged or interpolated.",
        "- A k-means/BIRCH centroid is never exported because it is generally not",
        "  a physically valid structure; every candidate is an observed frame.",
        "- A high score means 'most representative & structurally typical', not",
        "  proof of biological optimality; interaction metrics are coarse heavy-",
        "  atom contact approximations where ligand information exists.",
    ]
    return "\n".join(lines)
