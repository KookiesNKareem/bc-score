"""Generate publication figures 1-3 (and invoke 4-5 + Extended Data) in Nature Methods / Nature Biotechnology style."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "paper" / "figures"
NCS_JSON = PROJECT / "results" / "loto_benchmark_results.json"
OOF_CSV = PROJECT / "results" / "candidate_oof_predictions.csv"

INK = "#24313B"
C_NAVY = "#254B70"
C_TEAL = "#007F82"
C_BLUE = "#6786A1"
C_ORANGE = "#C87820"
C_RED = "#B04A50"
C_GREEN = "#4A7C59"
C_GREY = "#7A848D"
C_PURPLE = "#7654AA"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.6,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.8,
        "axes.labelweight": "normal",
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.6,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": "#6B737B",
        "axes.linewidth": 0.65,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save_both(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_p = FIG_DIR / f"{stem}.pdf"
    png_p = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_p, bbox_inches="tight")
    fig.savefig(png_p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_p.name} and {png_p.name}")


def load_oof_table() -> list[dict]:
    with OOF_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate_fig1(rep: dict) -> None:
    import runpy
    with plt.rc_context():
        runpy.run_path(str(Path(__file__).with_name("plot_fig1_pipeline.py")), run_name="__main__")


def generate_fig2_benchmark(rep: dict) -> None:
    """Figure 2: 3-panel Nature Methods benchmark (a: LOTO AP bars + CIs; b: Pooled Precision-Recall curves; c: 11 Multi-Hit targets)."""
    bench = rep["unified_benchmark_and_sensitivity"]["methods"]
    oof_rows = load_oof_table()
    y = np.array([int(r["label"]) for r in oof_rows])
    t = np.array([r["target_id"] for r in oof_rows])
    uniq_t = sorted(np.unique(t))

    rng = np.random.default_rng(2026)
    method_order = [
        ("spatial_ppi_v2", "SpatialPPI v2 (1.8 s CPU)"),
        ("rosetta_dg", "PyRosetta ΔG (4.5 s CPU)"),
        ("bcebm_full_equal", "BC-Score (Ours, 0.043 s)"),
        ("af2_ig_ipae", "AF2-Init-Guess iPAE"),
        ("colab_af2m_ipsae", "AF2-Multimer ipSAE"),
        ("boltz1_ipsae", "Boltz-1 ipSAE (35 s)"),
        ("af3_ipsae", "AlphaFold3 ipSAE (180 s)"),
        ("boltz1_plus_bcebm", "Boltz-1 + BC-Score (Ours)"),
        ("af3_plus_bcebm", "AF3 + BC-Score (Ours)"),
        ("boltz1_af3_plus_bcebm", "Ensemble + BC-Score (Ours)"),
    ]

    ci_gap, ci_mtap = {}, {}
    t_idx = {u: np.where(t == u)[0] for u in uniq_t}
    for m_key, _ in method_order:
        sc = np.array([float(r[m_key]) for r in oof_rows])
        b_g, b_m = [], []
        for _ in range(300):
            s_all, t_aps = [], []
            for u in uniq_t:
                iu = t_idx[u]
                pos_u, neg_u = iu[y[iu] == 1], iu[y[iu] == 0]
                bp = rng.choice(pos_u, size=len(pos_u), replace=True)
                bn = rng.choice(neg_u, size=len(neg_u), replace=True)
                bu = np.concatenate([bp, bn])
                s_all.append(bu)
                t_aps.append(average_precision_score(y[bu], sc[bu]))
            ba = np.concatenate(s_all)
            b_g.append(average_precision_score(y[ba], sc[ba]))
            b_m.append(float(np.mean(t_aps)))
        ci_gap[m_key] = (np.quantile(b_g, 0.025), np.quantile(b_g, 0.975))
        ci_mtap[m_key] = (np.quantile(b_m, 0.025), np.quantile(b_m, 0.975))

    fig = plt.figure(figsize=(11.2, 4.15), facecolor="white")
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.18, 0.88, 1.12], wspace=0.34)

    # ------------------------------------------------------------------
    # Panel a: LOTO Ranking Precision (Input-PDB, GPU Co-Folders, Hybrids)
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0])
    y_pos = np.arange(len(method_order))[::-1]
    h = 0.36

    # Subtle category background bands
    ax_a.axhspan(6.5, 9.55, color="#F3F7F9", zorder=0)
    ax_a.axhspan(-0.55, 2.5, color="#F2F8F8", zorder=0)

    for idx_m, (m_key, _) in enumerate(method_order):
        yp = y_pos[idx_m]
        g_val = bench[m_key]["global_ap"]
        m_val = bench[m_key]["mean_target_ap"]
        g_lo, g_hi = ci_gap[m_key]
        m_lo, m_hi = ci_mtap[m_key]
        is_ours = "bcebm" in m_key

        ax_a.barh(yp + h / 2, g_val, height=h, color=C_NAVY, alpha=0.94 if is_ours else 0.78,
                  label="Pooled Global AP" if idx_m == 0 else None, zorder=3)
        ax_a.errorbar(g_val, yp + h / 2, xerr=[[max(0, g_val - g_lo)], [max(0, g_hi - g_val)]],
                      fmt="none", ecolor=INK, elinewidth=0.85, capsize=1.8, zorder=4)
        ax_a.text(g_hi + 0.010, yp + h / 2, f"{g_val:.3f}", va="center", fontsize=6.6,
                  fontweight="bold" if is_ours else "normal", color=C_NAVY)

        ax_a.barh(yp - h / 2, m_val, height=h, color=C_TEAL, alpha=0.92 if is_ours else 0.75,
                  label="Mean Target AP (13)" if idx_m == 0 else None, zorder=3)
        ax_a.errorbar(m_val, yp - h / 2, xerr=[[max(0, m_val - m_lo)], [max(0, m_hi - m_val)]],
                      fmt="none", ecolor=INK, elinewidth=0.85, capsize=1.8, zorder=4)
        ax_a.text(m_hi + 0.010, yp - h / 2, f"{m_val:.3f}", va="center", fontsize=6.6,
                  fontweight="bold" if is_ours else "normal", color=C_TEAL)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([m[1] for m in method_order], fontsize=7.0)
    for tick_lbl, (m_key, _) in zip(ax_a.get_yticklabels(), method_order):
        if "bcebm" in m_key:
            tick_lbl.set_fontweight("bold")
    ax_a.set_xlim(0, 0.94)
    ax_a.set_ylim(-0.55, len(method_order) - 0.45)
    ax_a.axhline(6.5, color="#CBD5E0", ls="--", lw=0.8)
    ax_a.axhline(2.5, color="#CBD5E0", ls="--", lw=0.8)
    ax_a.text(0.925, 9.20, "Input PDB only", ha="right", va="center", fontsize=6.1, color=C_GREY, style="italic", weight="bold")
    ax_a.text(0.925, 6.15, "GPU co-folders", ha="right", va="center", fontsize=6.1, color=C_GREY, style="italic", weight="bold")
    ax_a.text(0.925, 2.22, "Co-folder +\nBC-Score", ha="right", va="center", fontsize=6.1, color=C_TEAL, style="italic", weight="bold")
    ax_a.set_axisbelow(True)
    ax_a.grid(axis="x", color="#E4E8EB", lw=0.5)
    ax_a.set_xlabel("Out-of-fold Average Precision (AP)")
    ax_a.set_title("Overall LOTO ranking precision (N = 3,532)", loc="left", pad=8)
    ax_a.text(-0.28, 1.04, "a", transform=ax_a.transAxes, fontsize=11, fontweight="bold", color=INK)
    ax_a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), frameon=True, facecolor="white", framealpha=0.94, fontsize=6.3)

    # ------------------------------------------------------------------
    # Panel b: Pooled Out-of-Fold Precision-Recall Curves (N = 3,532)
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1])
    pr_specs = [
        ("boltz1_af3_plus_bcebm", "Ensemble + BC-Score (0.508)", C_NAVY, "-", 1.9),
        ("af3_plus_bcebm", "AF3 + BC-Score (0.468)", "#1B8A6B", "-", 1.8),
        ("bcebm_full_equal", "BC-Score alone (0.383)", C_TEAL, "-", 2.0),
        ("af3_ipsae", "AlphaFold3 ipSAE (0.361)", C_ORANGE, "--", 1.5),
        ("boltz1_ipsae", "Boltz-1 ipSAE (0.336)", C_PURPLE, "-.", 1.4),
        ("rosetta_dg", "PyRosetta ΔG (0.135)", C_GREY, ":", 1.3),
    ]
    for col_k, lbl_p, col_p, ls_p, lw_p in pr_specs:
        sc = np.array([float(r[col_k]) for r in oof_rows])
        prec, rec, _ = precision_recall_curve(y, sc)
        # Subsample mildly for smooth rendering
        idx_s = np.linspace(0, len(prec) - 1, min(400, len(prec))).astype(int)
        ax_b.plot(rec[idx_s], prec[idx_s], color=col_p, ls=ls_p, lw=lw_p, label=lbl_p)

    prev_all = float(np.mean(y))
    ax_b.axhline(prev_all, color="#B0BAC2", ls="--", lw=0.9, label=f"Random ({prev_all*100:.1f}%)")
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(0.0, 1.02)
    ax_b.set_xlabel("Recall (fraction of 391 binders)")
    ax_b.set_ylabel("Precision (positive predictive value)")
    ax_b.set_title("Pooled precision–recall curves", loc="left", pad=8)
    ax_b.text(-0.18, 1.04, "b", transform=ax_b.transAxes, fontsize=11, fontweight="bold", color=INK)
    ax_b.set_axisbelow(True)
    ax_b.grid(True, color="#E4E8EB", lw=0.5)
    ax_b.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, fontsize=6.1)

    # ------------------------------------------------------------------
    # Panel c: Per-Target AP Across All 11 Multi-Hit Campaigns
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[2])
    rep_targets = ["EGFR", "VirB8", "FGFR2", "InsulinR", "Pdl1", "IL7Ra", "SARS_CoV2_RBD", "TrkA", "IL2Ra", "Mdm2", "sntx"]
    sorted_targets = sorted(rep_targets, key=lambda u: bench["bcebm_full_equal"]["by_target"][u])
    prev_map = rep["dataset_filtering"]["per_target_summary"]
    yt = np.arange(len(sorted_targets))

    for i_u, u in enumerate(sorted_targets):
        p_val = prev_map[u]["prevalence"]
        r_val = bench["rosetta_dg"]["by_target"][u]
        e_val = bench["bcebm_full_equal"]["by_target"][u]
        a_val = bench["af3_ipsae"]["by_target"][u]
        h_val = bench["af3_plus_bcebm"]["by_target"][u]
        lo_v = min(p_val, r_val, e_val, a_val, h_val)
        hi_v = max(p_val, r_val, e_val, a_val, h_val)
        ax_c.plot([lo_v, hi_v], [i_u, i_u], color="#DCE1E5", lw=1.7, zorder=1)

        # Dodge vertically when BC-Score and AF3+BC-Score are within 0.025 (e.g. sntx at 0.99-1.00)
        dy_eh = 0.14 if abs(e_val - h_val) < 0.03 else 0.0

        ax_c.scatter(p_val, i_u, color=INK, s=42, marker="|", linewidths=1.4, zorder=2,
                     label="Random (prevalence)" if i_u == 0 else None)
        ax_c.scatter(r_val, i_u, color=C_GREY, s=30, marker="o", zorder=3,
                     label="PyRosetta ΔG" if i_u == 0 else None)
        ax_c.scatter(a_val, i_u, color=C_ORANGE, s=34, marker="D", zorder=4,
                     label="AlphaFold3 ipSAE" if i_u == 0 else None)
        ax_c.scatter(e_val, i_u - dy_eh, color=C_NAVY, s=40, marker="o", edgecolor="white", lw=0.4, zorder=5,
                     label="BC-Score (Ours)" if i_u == 0 else None)
        ax_c.scatter(h_val, i_u + dy_eh, color=C_TEAL, s=42, marker="s", edgecolor="white", lw=0.4, zorder=6,
                     label="AF3 + BC-Score (Ours)" if i_u == 0 else None)

    short_t = {"SARS_CoV2_RBD": "SARS-CoV-2"}
    t_labels = [
        f"{short_t.get(u, u)} ({prev_map[u]['n_positives']}/{prev_map[u]['n_total']})"
        for u in sorted_targets
    ]
    ax_c.set_yticks(yt)
    ax_c.set_yticklabels(t_labels, fontsize=6.8)
    ax_c.set_xlim(0.0, 1.05)
    ax_c.set_ylim(-2.2, 10.6)
    ax_c.set_xlabel("Within-target Average Precision (AP)")
    ax_c.set_title("Per-target AP (11 multi-hit targets)", loc="left", pad=8)
    ax_c.text(-0.26, 1.04, "c", transform=ax_c.transAxes, fontsize=11, fontweight="bold", color=INK)
    ax_c.set_axisbelow(True)
    ax_c.grid(axis="x", color="#E4E8EB", lw=0.5)
    ax_c.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=6.1)

    save_both(fig, "fig2_loto_benchmark")


def generate_fig3_heatmap(rep: dict) -> None:
    """Figure 3: Mechanistic Dissection (a: 4-Stage Sidechain Source Ablation; b: 5-Module + Combined BC-Score Heatmap with Marginal Mean AP)."""
    oof_rows = load_oof_table()
    y = np.array([int(r["label"]) for r in oof_rows])
    t = np.array([r["target_id"] for r in oof_rows])
    ab = rep["matched_full_pipeline_attribution_ablation"]

    fig = plt.figure(figsize=(11.0, 3.95), facecolor="white")

    # ------------------------------------------------------------------
    # Panel a: Matched 4-Way Sidechain Attribution Ablation
    # ------------------------------------------------------------------
    ax_a = fig.add_axes([0.13, 0.18, 0.27, 0.71])
    ab_specs = [
        ("c1_monomer_sequence_only", "Monomer & sequence\nonly (C1 alone)"),
        ("full_pipeline_rigid_template_sidechains", "Canonical rigid-template\nsidechains"),
        ("full_pipeline_supplied_sidechains", "Supplied complex-packed\nsidechains (PDB)"),
        ("full_pipeline_2point_gnn_reconstructed_full_equal", "Partner-blind reconstructed\nsidechains (BC-Score)"),
    ]
    y_pos = np.arange(len(ab_specs))[::-1]
    h = 0.34

    for idx_s, (a_key, _) in enumerate(ab_specs):
        yp = y_pos[idx_s]
        g_val = ab[a_key]["global_ap"]
        m11_val = ab[a_key]["mean_target_ap_11"]
        is_ours = "reconstructed" in a_key
        c_g = C_NAVY if is_ours else "#7A8B99"
        c_m = C_TEAL if is_ours else "#9CB3C5"

        ax_a.barh(yp + h / 2, g_val, height=h, color=c_g, alpha=0.94,
                  label="Pooled Global AP" if idx_s == 3 else None)
        ax_a.barh(yp - h / 2, m11_val, height=h, color=c_m, alpha=0.90,
                  label="Multi-Hit Mean AP (11)" if idx_s == 3 else None)
        fw = "bold" if is_ours else "normal"
        ax_a.text(g_val + 0.010, yp + h / 2, f"{g_val:.3f}", va="center", fontsize=6.9, fontweight=fw, color=INK)
        ax_a.text(m11_val + 0.010, yp - h / 2, f"{m11_val:.3f}", va="center", fontsize=6.9, fontweight=fw, color=INK)

    ax_a.axhline(0.5, color="#CBD5E0", ls="--", lw=0.85)
    # Clean bracket comparing Supplied PDB (y=1) vs Partner-blind Reconstructed BC-Score (y=0)
    bx = 0.605
    ax_a.plot([0.585, bx, bx, 0.585], [0.83, 0.83, -0.17, -0.17], color=C_TEAL, lw=1.1)
    ax_a.text(
        bx + 0.012, 0.33,
        "+0.102 Pooled AP\n+0.121 Target AP\n(p < 0.001 vs. PDB)",
        fontsize=6.2, fontweight="bold", color=C_TEAL, va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#EDF7F7", edgecolor=C_TEAL, lw=0.6)
    )

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([s[1] for s in ab_specs], fontsize=7.1)
    for tick_lbl, (a_key, _) in zip(ax_a.get_yticklabels(), ab_specs):
        if "reconstructed" in a_key:
            tick_lbl.set_fontweight("bold")
    ax_a.set_xlim(0.0, 0.89)
    ax_a.set_ylim(-0.55, len(ab_specs) - 0.35)
    ax_a.set_axisbelow(True)
    ax_a.grid(axis="x", color="#E4E8EB", lw=0.5)
    ax_a.set_xlabel("Out-of-fold Average Precision (AP)")
    ax_a.set_title("Matched full-pipeline sidechain ablation", loc="left", pad=8)
    ax_a.text(-0.42, 1.04, "a", transform=ax_a.transAxes, fontsize=11, fontweight="bold", color=INK)
    ax_a.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, fontsize=6.4)

    # ------------------------------------------------------------------
    # Panel b: 6x11 Heatmap (C1-C5 + Combined BC-Score) + Marginal Mean AP Bar
    # ------------------------------------------------------------------
    ax_b = fig.add_axes([0.54, 0.20, 0.34, 0.69])
    ax_m = fig.add_axes([0.89, 0.20, 0.085, 0.69])

    target_order = [
        "sntx", "Mdm2", "IL2Ra", "TrkA", "SARS_CoV2_RBD", "IL7Ra", "Pdl1", "InsulinR", "FGFR2", "VirB8", "EGFR",
    ]
    comp_keys = ["S1_fold", "S2_strain", "S3_intf", "S4_alchem", "S5_funnel", "bcebm_full_equal"]
    comp_labels = [
        "C1: Monomer foldability",
        "C2: Sidechain & hotspots",
        "C3: Polar & electrostatics",
        "C4: Alchemical specificity",
        "C5: Relaxation & funnel",
        "Combined BC-Score",
    ]

    mat = np.zeros((len(comp_keys), len(target_order)), dtype=np.float64)
    for j, u in enumerate(target_order):
        m = t == u
        for i, ck in enumerate(comp_keys):
            sc = np.array([float(r[ck]) for r in oof_rows])[m]
            mat[i, j] = float(average_precision_score(y[m], sc))

    im = ax_b.imshow(mat, cmap="YlGnBu", vmin=0.05, vmax=0.95, aspect="auto")

    # Crisp white gridlines between heatmap cells and bold divider above Combined BC-Score
    ax_b.set_xticks(np.arange(-0.5, len(target_order), 1), minor=True)
    ax_b.set_yticks(np.arange(-0.5, len(comp_labels), 1), minor=True)
    ax_b.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax_b.tick_params(which="minor", bottom=False, left=False)
    ax_b.axhline(4.5, color=INK, lw=1.6)

    short_names = {"SARS_CoV2_RBD": "SARS-2"}
    prev_map = rep["dataset_filtering"]["per_target_summary"]
    col_labels = [f"{short_names.get(u, u)}\n({prev_map[u]['prevalence']:.2f})" for u in target_order]
    ax_b.set_xticks(np.arange(len(target_order)))
    ax_b.set_xticklabels(col_labels, fontsize=6.3, rotation=32, ha="right")
    ax_b.set_yticks(np.arange(len(comp_labels)))
    ax_b.set_yticklabels(comp_labels, fontsize=7.2)
    ax_b.get_yticklabels()[-1].set_fontweight("bold")

    for j in range(mat.shape[1]):
        best_c = int(np.argmax(mat[:5, j]))
        for i in range(mat.shape[0]):
            val = mat[i, j]
            txt_col = "white" if val >= 0.48 else INK
            fw = "bold" if (i == best_c or i == 5) else "normal"
            ax_b.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_col, fontsize=6.7, fontweight=fw)

    ax_b.set_title("Module-level precision across 11 multi-hit campaigns", loc="left", pad=8)
    ax_b.text(-0.34, 1.04, "b", transform=ax_b.transAxes, fontsize=11, fontweight="bold", color=INK)

    # Marginal horizontal bar plot of 11-target Mean AP for each row
    row_means = np.mean(mat, axis=1)
    y_rows = np.arange(len(comp_labels))
    bar_cols = ["#6786A1"] * 5 + [C_TEAL]
    ax_m.barh(y_rows, row_means, height=0.62, color=bar_cols, alpha=0.92)
    ax_m.axhline(4.5, color=INK, lw=1.4)
    for yi, rm in enumerate(row_means):
        ax_m.text(rm + 0.02, yi, f"{rm:.2f}", va="center", fontsize=6.6,
                  fontweight="bold" if yi == 5 else "normal", color=C_TEAL if yi == 5 else INK)
    ax_m.set_ylim(len(comp_labels) - 0.5, -0.5)
    ax_m.set_xlim(0, 0.72)
    ax_m.set_yticks([])
    ax_m.set_xticks([0.0, 0.25, 0.50])
    ax_m.set_xticklabels(["0", "0.25", "0.50"], fontsize=6.4)
    ax_m.set_xlabel("Mean AP", fontsize=7.2)
    ax_m.set_title("11-Target\nMean AP", loc="center", fontsize=7.4, pad=4)
    ax_m.grid(axis="x", color="#E4E8EB", lw=0.5)
    ax_m.set_axisbelow(True)

    save_both(fig, "fig3_sidechain_ablation_and_modules")


def main() -> None:
    rep = json.loads(NCS_JSON.read_text())
    generate_fig1(rep)
    generate_fig2_benchmark(rep)
    generate_fig3_heatmap(rep)
    print("Generated upgraded Figures 1-3.")


if __name__ == "__main__":
    main()
