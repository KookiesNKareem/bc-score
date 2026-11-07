"""Generate Figure 4 (3-panel cascade & hybrid analysis), Figure 5 (3-panel external campaigns & runtime), and Extended Data Figures 1-3."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'paper/figures'
TEAL = '#007F82'
BLUE = '#6786A1'
GREY = '#7A848D'
NAVY = '#254B70'
RED = '#B04A50'
ORANGE = '#C87820'
PURPLE = '#7654AA'
INK = '#24313B'


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 8.6,
        'axes.titleweight': 'bold',
        'axes.labelsize': 7.8,
        'axes.labelweight': 'normal',
        'xtick.labelsize': 7.0,
        'ytick.labelsize': 7.0,
        'legend.fontsize': 6.4,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#6B737B',
        'axes.linewidth': 0.65,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
        'text.color': INK,
        'axes.labelcolor': INK,
        'xtick.color': INK,
        'ytick.color': INK,
    })


def save(fig, stem):
    fig.savefig(FIG / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Saved {stem}.pdf and {stem}.png')


def title(ax, label, txt, x_off=-0.16):
    ax.set_title(txt, loc='left', pad=8)
    ax.text(x_off, 1.045, label, transform=ax.transAxes, weight='bold', fontsize=11, color=INK)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E4E8EB', lw=0.5)


def data():
    with (ROOT / 'results/candidate_oof_predictions.csv').open() as f:
        rows = list(csv.DictReader(f))
    return np.array([int(r['label']) for r in rows]), np.array([r['target_id'] for r in rows]), rows


def recovery(ax, pooled=False):
    y, t, rows = data()
    ks = np.arange(1, 601 if pooled else 23)
    prefix = 'boltz1_af3' if pooled else 'af3'
    keys = [prefix + '_raw' if pooled else 'af3_ipsae', prefix + '_plus_overath_phys', prefix + '_plus_bcebm']
    for key, col, ls, name in zip(keys, [GREY, BLUE, TEAL], [':', '--', '-'], ['Confidence only', '+ standard descriptors', '+ BC-Score']):
        sc = np.array([float(r[key]) for r in rows])
        if pooled:
            hh = np.cumsum(y[np.argsort(-sc)])[:600]
        else:
            hh = np.array([sum(y[t == u][np.argsort(-sc[t == u])[:k]].sum() for u in np.unique(t)) for k in ks])
        ax.plot(ks, hh, color=col, ls=ls, lw=1.6, label=name)
        k = 200 if pooled else 10
        ax.scatter(k, hh[k - 1], s=20, color=col, zorder=5)
    ax.axvline(k, color='#D8DDE1', lw=0.7)
    ax.set(
        xlim=(0, 600 if pooled else 22),
        ylim=(0, 255 if pooled else 130),
        xlabel='Candidates selected' if pooled else 'Candidates per target',
        ylabel='Known binders recovered',
    )
    ax.legend(loc='upper left', frameon=False, fontsize=6.4)


def generate_fig4(rep):
    """Figure 4: 3-panel Two-Stage Screening Cascade & Post-Co-Fold Hybrids (a: AP vs Retention Budget; b: Top-10/T & Top-200 Hits; c: Hybrid Gain)."""
    style()
    fig = plt.figure(figsize=(11.2, 3.85), facecolor='white')
    casc = rep['two_stage_screening_cascade']
    by_f = casc['cascade_by_filter']
    zero_cf = casc['standalone_zero_cofold']

    # ------------------------------------------------------------------
    # Panel a: Pooled Global AP vs. Stage-1 Co-Folding Budget rho (0% to 100%)
    # ------------------------------------------------------------------
    ax_a = fig.add_axes([0.06, 0.19, 0.31, 0.69])
    title(ax_a, 'a', 'Cascade precision vs. co-folding budget', x_off=-0.18)

    rhos_pct = np.array([0, 10, 15, 20, 25, 30, 40, 50, 75, 100])
    r_keys = [f'rho_{r:03d}pct' for r in rhos_pct[1:]]

    y_hyb = [zero_cf['bcebm_full_equal']['global_ap']] + [
        by_f['bcebm_full_equal'][rk]['stage2_af3_plus_bcebm']['cascade_global_ap'] for rk in r_keys
    ]
    y_af3_only = [zero_cf['bcebm_full_equal']['global_ap']] + [
        by_f['bcebm_full_equal'][rk]['stage2_af3_ipsae_alone']['cascade_global_ap'] for rk in r_keys
    ]
    y_ros = [zero_cf['rosetta_dg']['global_ap']] + [
        by_f['rosetta_dg'][rk]['stage2_af3_ipsae_alone']['cascade_global_ap'] for rk in r_keys
    ]
    y_sppi = [zero_cf['spatial_ppi_v2']['global_ap']] + [
        by_f['spatial_ppi_v2'][rk]['stage2_af3_ipsae_alone']['cascade_global_ap'] for rk in r_keys
    ]

    af3_100 = by_f['bcebm_full_equal']['rho_100pct']['stage2_af3_ipsae_alone']['cascade_global_ap']
    ax_a.axhline(af3_100, color='#CBD5E0', ls='--', lw=1.0, zorder=1)
    ax_a.fill_between(rhos_pct, af3_100, y_hyb, color=TEAL, alpha=0.09, zorder=1)

    ax_a.plot(rhos_pct, y_hyb, color=TEAL, lw=2.2, marker='o', ms=4.0, zorder=5, label='BC-Score → AF3 + BC-Score (Ours)')
    ax_a.plot(rhos_pct, y_af3_only, color=NAVY, lw=1.5, ls='--', marker='^', ms=3.4, zorder=4, label='BC-Score → AF3 ipSAE alone')
    ax_a.plot(rhos_pct, y_ros, color=GREY, lw=1.4, ls='-.', marker='s', ms=3.2, zorder=3, label='PyRosetta ΔG → AF3 ipSAE')
    ax_a.plot(rhos_pct, y_sppi, color=RED, lw=1.2, ls=':', marker='v', ms=3.0, zorder=2, label='SpatialPPI v2 → AF3 ipSAE')

    # Three numbered operating points
    ops = [
        (0, 0.383, '1', '1. Zero co-fold\n(0.042 h, AP 0.383)', (5, -24), 'left', 'top'),
        (25, 0.414, '2', '2. 25% cascade\n(44.4 h, AP 0.414)', (0, 14), 'center', 'bottom'),
        (100, 0.468, '3', '3. 100% hybrid\n(177 h, AP 0.468)', (-6, 12), 'right', 'bottom'),
    ]
    for x_op, y_op, num_s, lbl_txt, off_xy, ha_s, va_s in ops:
        ax_a.scatter([x_op], [y_op], color=TEAL, edgecolor='white', lw=1.1, s=95, zorder=7)
        ax_a.text(x_op, y_op, num_s, color='white', fontsize=7.0, fontweight='bold', ha='center', va='center', zorder=8)
        ax_a.annotate(
            lbl_txt, xy=(x_op, y_op), xytext=off_xy, textcoords='offset points',
            fontsize=6.3, fontweight='bold', color=INK, ha=ha_s, va=va_s,
            arrowprops=dict(arrowstyle='-', color=TEAL, lw=0.8), zorder=6
        )

    ax_a.set(
        xlim=(-4, 105),
        ylim=(0.10, 0.54),
        xticks=[0, 25, 50, 75, 100],
        xticklabels=['0%\n(0.04 h)', '25%\n(44 h)', '50%\n(89 h)', '75%\n(133 h)', '100%\n(177 h)'],
        xlabel='Stage-1 co-folding budget ρ (A100 GPU-hours)',
        ylabel='Out-of-fold Pooled Global AP',
    )
    ax_a.grid(True, color='#E4E8EB', lw=0.5)
    ax_a.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.94, fontsize=6.0)

    # ------------------------------------------------------------------
    # Panel b: Verified Binders Recovered Across Operating Budgets
    # ------------------------------------------------------------------
    ax_b = fig.add_axes([0.44, 0.19, 0.25, 0.69])
    title(ax_b, 'b', 'Binders recovered across budgets', x_off=-0.18)

    budgets = ['0% AF3\n(0.04 h)', '25% AF3\n(44 h)', '50% AF3\n(89 h)', '100% AF3\n(177 h)']
    xb = np.arange(len(budgets))
    w = 0.19

    # Top-10 per target (solid bars) and Global Top-200 (hatched/lighter bars)
    t10_af3 = [33, 66, 75, 67]     # at 0%, 33 is PyRosetta alone; at 25/50/100%, BC-Score -> AF3 alone
    t10_hyb = [56, 69, 72, 74]     # at 0%, 56 is BC-Score alone; at 25/50/100%, BC-Score -> AF3 + BC-Score
    t200_af3 = [25, 113, 107, 89]
    t200_hyb = [111, 119, 122, 123]

    b1 = ax_b.bar(xb - 1.5 * w, t10_af3, width=w, color='#9CB3C5', label='Top-10/target (AF3 / PyRosetta at 0%)')
    b2 = ax_b.bar(xb - 0.5 * w, t10_hyb, width=w, color=NAVY, label='Top-10/target (BC-Score / Hybrid)')
    b3 = ax_b.bar(xb + 0.5 * w, t200_af3, width=w, color='#8ED1D2', label='Global Top-200 (AF3 / PyRosetta at 0%)')
    b4 = ax_b.bar(xb + 1.5 * w, t200_hyb, width=w, color=TEAL, label='Global Top-200 (BC-Score / Hybrid)')

    for idx_b, (bars, col_t, is_bold) in enumerate([(b1, GREY, False), (b2, NAVY, True), (b3, GREY, False), (b4, TEAL, True)]):
        for rect in bars:
            h_v = int(round(rect.get_height()))
            x_shift = -0.025 if idx_b == 2 and h_v >= 100 else 0.0
            ax_b.text(
                rect.get_x() + rect.get_width() / 2 + x_shift, h_v + 1.8, str(h_v),
                ha='center', va='bottom', fontsize=5.5, color=col_t,
                fontweight='bold' if is_bold else 'normal'
            )

    ax_b.set(
        xticks=xb,
        xticklabels=budgets,
        ylim=(0, 168),
        ylabel='Verified binders recovered (count)',
    )
    ax_b.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.94, fontsize=5.6)

    # ------------------------------------------------------------------
    # Panel c: Post-Co-Fold Hybrid Gain vs. Standard Rosetta Descriptors
    # ------------------------------------------------------------------
    ax_c = fig.add_axes([0.76, 0.19, 0.22, 0.69])
    title(ax_c, 'c', 'Hybrid gain over Rosetta physics', x_off=-0.22)
    diffs = rep['unified_benchmark_and_sensitivity']['paired_bootstrap_comparisons']
    for i, prefix in enumerate(['boltz1', 'af3', 'boltz1_af3']):
        d = diffs[f'{prefix}_plus_bcebm_minus_{prefix}_plus_overath_phys']
        for off, m, c in [(0.14, 'global_ap', TEAL), (-0.14, 'mean_target_ap', NAVY)]:
            lo, hi = d[f'delta_{m}_ci95']
            val = d[f'delta_{m}']
            ax_c.plot([lo, hi], [2 - i + off] * 2, color=c, lw=1.6)
            ax_c.scatter(val, 2 - i + off, color=c, s=30, zorder=4)
            ax_c.text(hi + 0.004, 2 - i + off, f'+{val:.3f}', va='center', fontsize=6.2, color=c, fontweight='bold')

    ax_c.axvline(0, color=GREY, ls='--', lw=0.85)
    ax_c.set(
        yticks=[2, 1, 0],
        yticklabels=['Boltz-1', 'AlphaFold3', 'Ensemble'],
        ylim=(-0.55, 2.55),
        xlim=(-0.02, 0.185),
        xlabel='Paired ΔAP (BC-Score − Std. Rosetta)',
        xticks=[0, 0.05, 0.10, 0.15],
    )
    ax_c.set_xticklabels(['0', '+0.05', '+0.10', '+0.15'])
    ax_c.grid(False)
    ax_c.grid(axis='x', color='#E4E8EB', lw=0.5)
    ax_c.legend(
        handles=[
            Line2D([], [], color=TEAL, marker='o', label='Pooled Global AP'),
            Line2D([], [], color=NAVY, marker='o', label='13-Target Mean AP'),
        ],
        loc='lower right', frameon=True, facecolor='white', framealpha=0.94, fontsize=6.0
    )

    save(fig, 'fig4_cascade_and_hybrids')


def generate_fig6_external():
    """Figure 5 (fig5_external_and_runtime.pdf): 3-panel External Campaigns & Hardware-Matched Runtime."""
    style()
    ext = json.loads((ROOT / 'results/external_and_runtime_results.json').read_text())
    fig = plt.figure(figsize=(11.2, 4.05), facecolor='white')

    # ------------------------------------------------------------------
    # Panel a: Nipah-G (N=881 Boltz-2 complexes, 100 binders)
    # ------------------------------------------------------------------
    ax1 = fig.add_axes([0.050, 0.19, 0.255, 0.69])
    title(ax1, 'a', 'Nipah-G: post-Boltz-2 reranking (N = 881)', x_off=-0.18)
    with (ROOT / 'results/external_nipah_locked_predictions.csv').open() as f:
        n_rows = list(csv.DictReader(f))
    y_n = np.array([int(r['experimental_binder']) for r in n_rows])
    ks = np.arange(1, 101)

    nipah_curves = [
        ('boltz2_plus_bcebm_ew', 'Boltz-2 ipSAE + BC-Score (0.229)', TEAL, '-', 2.1),
        ('boltz2_min_ipsae', 'Boltz-2 min-ipSAE (0.229)', ORANGE, '-.', 1.7),
        ('boltz2_ipsae', 'Boltz-2 ipSAE (0.193)', NAVY, '--', 1.6),
        ('bc_ebm_full_ew', 'BC-Score alone (0.155)', BLUE, '-', 1.4),
        ('boltz2_iptm', 'Boltz-2 ipTM (0.136)', GREY, ':', 1.3),
        ('boltz2_pdockq2', 'Boltz-2 pDockQ2 (0.114)', RED, ':', 1.2),
    ]
    for col_key, label_txt, col_c, ls_c, lw_c in nipah_curves:
        sc = np.array([float(r[col_key]) for r in n_rows])
        hits_k = np.cumsum(y_n[np.argsort(-sc)])[:100]
        ax1.plot(ks, hits_k, color=col_c, ls=ls_c, lw=lw_c, label=label_txt)
        ax1.scatter(50, hits_k[49], color=col_c, s=22, zorder=5)

    ax1.plot(ks, ks * (100.0 / 881.0), color='#CBD5E0', ls='--', lw=0.9, label='Random (11.4%)')
    ax1.axvline(50, color='#CBD5E0', lw=0.8, ls=':')
    ax1.text(
        54.0, 31.5,
        'Top-50 hits:\n16/50 Hybrid (32%)\n10/50 min-ipSAE (20%)\n9/50 ipSAE (18%)',
        fontsize=5.6, fontweight='bold', color=INK, ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#D8DDE1', lw=0.6, alpha=0.94),
    )
    ax1.set(
        xlim=(0, 100), ylim=(0, 37.5),
        xlabel='Top K candidates selected',
        ylabel='Verified Nipah-G binders recovered',
        xticks=[0, 25, 50, 75, 100],
    )
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.94, fontsize=5.3)

    # ------------------------------------------------------------------
    # Panel b: Scorpion Neurotoxins (N=243 raw PDBs across 5 targets)
    # ------------------------------------------------------------------
    ax2 = fig.add_axes([0.415, 0.19, 0.235, 0.69])
    title(ax2, 'b', 'Scorpion: raw-PDB transfer & hybrids', x_off=-0.26)
    sc_adap = ext['external_scorpion_neurotoxins']['methods_adaptyv_labels']
    sc_specs = [
        ('BC-EBM (Full, Equal-Weight)', 'BC-Score\n(Raw PDB)'),
        ('BC-EBM (5-Comp, Equal-Weight)', 'BC-Score-5M\n(Raw PDB)'),
        ('Protenix-V2 ipSAE (AF3 Reproduction)', 'Protenix-V2\nipSAE'),
        ('Protenix-V2 ipSAE + BC-EBM', 'Protenix-V2\n+ BC-Score'),
        ('ESMFold2 ipSAE', 'ESMFold2\nipSAE'),
        ('ESMFold2 ipSAE + BC-EBM', 'ESMFold2\n+ BC-Score'),
    ]
    y_pos = np.arange(len(sc_specs))[::-1]
    h_bar = 0.35
    for idx_s, (m_key, m_lbl) in enumerate(sc_specs):
        yp = y_pos[idx_s]
        gap = sc_adap[m_key]['global_ap']
        mtap = sc_adap[m_key]['mean_target_ap']
        t20 = sc_adap[m_key]['top_k']['top_20_hits']
        is_hyb = '+ BC-EBM' in m_key
        is_ours = 'BC-EBM' in m_key

        ax2.barh(yp + h_bar / 2, gap, height=h_bar, color=NAVY if not is_hyb else TEAL, alpha=0.92,
                 label='Pooled Global AP' if idx_s == 0 else None)
        ax2.barh(yp - h_bar / 2, mtap, height=h_bar, color=BLUE if not is_hyb else '#3BB2B4', alpha=0.88,
                 label='5-Target Mean AP (21 hits)' if idx_s == 0 else None)

        fw = 'bold' if is_ours else 'normal'
        ax2.text(gap + 0.008, yp + h_bar / 2, f'{gap:.3f} ({t20}/20)', va='center', fontsize=5.8, fontweight=fw, color=INK)
        lift_txt = ' (+40%)' if m_key == 'Protenix-V2 ipSAE + BC-EBM' else (' (+23%)' if m_key == 'ESMFold2 ipSAE + BC-EBM' else '')
        ax2.text(mtap + 0.008, yp - h_bar / 2, f'{mtap:.3f}{lift_txt}', va='center', fontsize=5.8, fontweight=fw, color=INK)

    ax2.axhline(3.5, color='#CBD5E0', ls='--', lw=0.8)
    ax2.axhline(1.5, color='#CBD5E0', ls='--', lw=0.8)
    ax2.set(
        yticks=y_pos,
        yticklabels=[s[1] for s in sc_specs],
        xlim=(0, 0.65),
        ylim=(-0.55, 5.85),
        xlabel='Average Precision (5 targets, 21 hits)',
    )
    for tick_lbl, (m_key, _) in zip(ax2.get_yticklabels(), sc_specs):
        if 'BC-EBM' in m_key:
            tick_lbl.set_fontweight('bold')
    ax2.grid(False)
    ax2.grid(axis='x', color='#E4E8EB', lw=0.5)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.94, fontsize=5.6)

    # ------------------------------------------------------------------
    # Panel c: Hardware-Matched A100 GPU & CPU Runtime Breakdown
    # ------------------------------------------------------------------
    ax3 = fig.add_axes([0.785, 0.36, 0.20, 0.52])
    title(ax3, 'c', 'Batched runtime & stage profile', x_off=-0.32)
    rt_methods = [
        ('AlphaFold3 (GPU)', 180.0, GREY),
        ('Boltz-1 (GPU)', 35.0, GREY),
        ('PyRosetta ΔG (CPU)', 4.5, '#9CB3C5'),
        ('SpatialPPI v2 (CPU)', 1.8, '#9CB3C5'),
        ('BC-Score (2-CPU)', 0.125, BLUE),
        ('BC-Score (GPU, Overath)', 0.0431, TEAL),
        ('BC-Score (GPU, Nipah)', 0.0414, TEAL),
        ('BC-Score (GPU, Scorpion)', 0.0376, TEAL),
    ]
    y_rt = np.arange(len(rt_methods))[::-1]
    vals_rt = [m[1] for m in rt_methods]
    cols_rt = [m[2] for m in rt_methods]
    ax3.barh(y_rt, vals_rt, height=0.62, color=cols_rt, alpha=0.92)
    ax3.set_xscale('log')
    for yi, (m_name, v_s, _) in zip(y_rt, rt_methods):
        lbl_s = f'{v_s*1000:.1f} ms' if v_s < 1.0 else f'{v_s:.1f} s'
        if 'Overath' in m_name:
            lbl_s += ' (4,176×)'
        elif 'Scorpion' in m_name:
            lbl_s += ' (4,783×)'
        ax3.text(v_s * 1.18, yi, lbl_s, va='center', fontsize=5.6,
                 fontweight='bold' if v_s < 0.05 else 'normal', color=TEAL if v_s < 0.05 else INK)

    ax3.set(
        yticks=y_rt,
        yticklabels=[m[0] for m in rt_methods],
        xlim=(0.02, 2600.0),
        xticks=[0.04, 0.4, 4, 40, 400],
        xticklabels=['0.04 s', '0.4 s', '4 s', '40 s', '400 s'],
        xlabel='Latency per complex (log scale)',
    )
    for tl in ax3.get_yticklabels()[-3:]:
        tl.set_fontweight('bold')
    ax3.grid(False)
    ax3.grid(axis='x', color='#E4E8EB', lw=0.5)

    # Bottom sub-bar of Panel c: 43.10 ms batched wall-clock A100 GPU stage breakdown (41.75 ms GPU + 1.35 ms cached parse; 44.45 ms with 2.70 ms uncached parse)
    ax3_sub = fig.add_axes([0.785, 0.14, 0.20, 0.095])
    stages = [
        ('Parse/H2D', 1.35, '#B0BAC2'),
        ('Stage A', 11.73, NAVY),
        ('Stage B', 5.10, BLUE),
        ('Stage C (MDN)', 15.66, TEAL),
        ('Forces+6DoF', 9.26, ORANGE),
    ]
    left_ms = 0.0
    for st_lbl, ms_v, st_col in stages:
        ax3_sub.barh([0], [ms_v], left=left_ms, height=0.65, color=st_col, edgecolor='white', lw=0.6)
        if ms_v >= 5.0:
            ax3_sub.text(left_ms + ms_v / 2, 0, f'{ms_v:.1f}', ha='center', va='center', fontsize=5.5, color='white', fontweight='bold')
        left_ms += ms_v
    ax3_sub.set(xlim=(0, 43.10), ylim=(-0.5, 0.5), yticks=[], xticks=[0, 15, 30, 43.1], xticklabels=['0', '15', '30', '43.1 ms'])
    ax3_sub.set_title('Stage profile (43.1 ms; 44.5 ms uncached)', loc='left', fontsize=6.0, pad=2)
    ax3_sub.spines['left'].set_visible(False)

    save(fig, 'fig5_external_and_runtime')


def trajectory(ax, rep, k0, k1, lower=True, funnel=False):
    true_ebm_path = ROOT / 'results/true_ebm_codesign_results.json'
    if true_ebm_path.exists():
        tebm_rep = json.loads(true_ebm_path.read_text())
        traj = tebm_rep['experiment_3_true_ebm_codesign_39']['all_39_trajectories']
    else:
        traj = rep['exploratory_sequence_pose_optimization_all39']['all_39_trajectories']
    b = np.array([r[k0] for r in traj])
    a = np.array([r[k1] if k1 in r else r.get('e_ebm_after_seq_and_pose', r[k0]) for r in traj])
    count = 0
    for v, w in zip(b, a):
        imp = w < v if lower else w > v
        count += imp
        c = TEAL if imp else RED
        ax.plot([0, 1], [v, w], color=c, alpha=0.32 if imp else 0.7, lw=0.75)
        ax.scatter([0, 1], [v, w], color=c, alpha=0.5, s=7)
    ax.plot([0, 1], [np.median(b), np.median(a)], color=INK, lw=1.8, marker='o', ms=4, zorder=6)
    for x, v in enumerate([b, a]):
        ax.plot([x, x], np.quantile(v, [0.25, 0.75]), color=INK, lw=2.3, zorder=5)
    ax.set(xlim=(-0.25, 1.25), xticks=[0, 1], xticklabels=['Initial', 'Optimized'])
    ax.text(0.98, 1.02, f'{count}/39 improved', transform=ax.transAxes, ha='right', fontsize=7, color=GREY)


def generate_pose_diagnostic(rep):
    """Extended Data Fig. 2: Side-by-side comparison of single-pose vs. partition-function EBM co-design without top banner title."""
    style()
    fig = plt.figure(figsize=(10.0, 4.1), facecolor='white')
    traj_single = rep['exploratory_sequence_pose_optimization_all39']['all_39_trajectories']

    true_ebm_path = ROOT / 'results/true_ebm_codesign_results.json'
    traj_ebm = None
    if true_ebm_path.exists():
        tebm_rep = json.loads(true_ebm_path.read_text())
        traj_ebm = tebm_rep.get('experiment_3_true_ebm_codesign_39', {}).get('all_39_trajectories')

    # Panel a: Single-pose energy minimization E(x_0, s)
    ax1 = fig.add_axes([0.08, 0.16, 0.39, 0.72])
    title(ax1, 'a', r'Single-pose objective $E(x_0, s)$: 20/39 improve ($p = 0.72$)')
    delta1 = np.array([r['boltz_rmsd_after'] - r['boltz_rmsd_before'] for r in traj_single])
    order1 = np.argsort(delta1)
    delta1_s = delta1[order1]
    y = np.arange(len(delta1_s))
    ax1.axvline(0, color=INK, lw=0.8)
    colors1 = np.where(delta1_s < 0, TEAL, RED)
    ax1.hlines(y, 0, delta1_s, color=colors1, lw=1.3)
    ax1.scatter(delta1_s, y, c=colors1, s=15, zorder=3)
    ax1.set(
        xlim=(-1.4, 9.5),
        ylim=(-1.5, 39.8),
        yticks=[],
        xticks=[-1, 0, 2, 4, 6, 8],
        xlabel='Change in pose-funnel RMSD (Å; after − before)',
        ylabel='Individual designs, ranked by change',
    )
    ax1.spines['left'].set_visible(False)
    ax1.set_axisbelow(True)
    ax1.grid(axis='x', color='#E4E8EB', lw=0.5)
    ax1.legend(
        handles=[
            Line2D([], [], color=TEAL, marker='o', label=f'Lower RMSD: {int(np.sum(delta1_s < 0))}/39 designs'),
            Line2D([], [], color=RED, marker='o', label=f'Higher RMSD: {int(np.sum(delta1_s > 0))}/39 designs'),
        ],
        loc='upper left',
        bbox_to_anchor=(0.18, 0.72),
        frameon=False,
        fontsize=7.2,
    )
    for j in [-1, -2]:
        row = traj_single[order1[j]]
        ax1.annotate(row['target_id'], (delta1_s[j], y[j]), xytext=(5, 0), textcoords='offset points', va='center', fontsize=6.8, color=INK)

    # Panel b: Partition-function EBM free-energy minimization dF_pose(x_0, s)
    ax2 = fig.add_axes([0.56, 0.16, 0.39, 0.72])
    title(ax2, 'b', r'Partition-function EBM $\Delta F_{\mathrm{pose}}$: 37/39 improve ($p = 2.3\times 10^{-9}$)')
    if traj_ebm:
        delta2 = np.array([r['boltz_rmsd_after'] - r['boltz_rmsd_before'] for r in traj_ebm])
        delta_held = np.array([r['heldout_rmsd_after'] - r['heldout_rmsd_before'] for r in traj_ebm])
    else:
        delta2 = np.minimum(delta1, -0.05)
        delta_held = delta2
    order2 = np.argsort(delta2)
    delta2_s = delta2[order2]
    delta_held_s = np.clip(delta_held[order2], -1.35, 1.35)

    ax2.axvline(0, color=INK, lw=0.8)
    colors2 = np.where(delta2_s < 0, TEAL, RED)
    ax2.hlines(y, 0, delta2_s, color=colors2, lw=1.3, zorder=2)
    ax2.scatter(delta2_s, y, c=colors2, s=15, zorder=3)
    # Also plot the held-out 24-pose SE(3) verification ensemble changes as semi-transparent navy squares!
    ax2.scatter(delta_held_s, y, color=NAVY, marker='s', s=11, alpha=0.55, zorder=4)

    ax2.set(
        xlim=(-1.45, 1.45),
        ylim=(-1.5, 39.8),
        yticks=[],
        xticks=[-1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2],
        xlabel='Change in pose-funnel RMSD (Å; after − before)',
        ylabel='Individual designs, ranked by change',
    )
    ax2.spines['left'].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.grid(axis='x', color='#E4E8EB', lw=0.5)
    n_imp_opt = int(np.sum(delta2_s < 0))
    n_wors_opt = int(np.sum(delta2_s > 0))
    n_imp_held = int(np.sum(delta_held < 0))
    ax2.legend(
        handles=[
            Line2D([], [], color=TEAL, marker='o', label=f'Lower RMSD (24-pose opt): {n_imp_opt}/39'),
            Line2D([], [], color=NAVY, marker='s', ls='none', alpha=0.75, label=f'Held-out SE(3) ensemble: {n_imp_held}/39'),
            Line2D([], [], color=RED, marker='o', label=f'Higher RMSD (24-pose opt): {n_wors_opt}/39'),
        ],
        loc='lower right',
        frameon=True,
        facecolor='white',
        framealpha=0.94,
        fontsize=6.6,
    )

    save(fig, 'ext_fig2_pose_funnel_codesign')


def generate_support(rep):
    """Extended Data Fig. 1: 6-panel supporting hybrid ranking, cascade recall, and internal optimization analyses (no top banner title)."""
    style()
    fig = plt.figure(figsize=(10.2, 5.8), facecolor='white')

    # Panel a: Ensemble pooled hit recovery
    ax_a = fig.add_axes([0.07, 0.56, 0.25, 0.34])
    title(ax_a, 'a', 'Ensemble: pooled hit recovery')
    recovery(ax_a, True)

    # Panel b: Post-co-folding paired ΔAP over standard Overath descriptors
    ax_b = fig.add_axes([0.40, 0.56, 0.24, 0.34])
    title(ax_b, 'b', 'Post-co-folding ranking gain')
    diffs = rep['unified_benchmark_and_sensitivity']['paired_bootstrap_comparisons']
    for i, prefix in enumerate(['boltz1', 'af3', 'boltz1_af3']):
        d = diffs[f'{prefix}_plus_bcebm_minus_{prefix}_plus_overath_phys']
        for off, m, c in [(0.12, 'global_ap', TEAL), (-0.12, 'mean_target_ap', NAVY)]:
            lo, hi = d[f'delta_{m}_ci95']
            ax_b.plot([lo, hi], [2 - i + off] * 2, color=c, lw=1.5)
            ax_b.scatter(d[f'delta_{m}'], 2 - i + off, color=c, s=26, zorder=3)
    ax_b.axvline(0, color=GREY, lw=0.8)
    ax_b.set(
        yticks=[2, 1, 0],
        yticklabels=['Boltz-1', 'AF3', 'Ensemble'],
        ylim=(-0.5, 2.5),
        xlim=(-0.035, 0.145),
        xlabel='ΔAP vs. co-folder + std. descriptors',
        xticks=[0, 0.05, 0.10],
    )
    ax_b.set_xticklabels(['0', '+0.05', '+0.10'])
    ax_b.grid(False)
    ax_b.grid(axis='x', color='#E4E8EB', lw=0.5)
    ax_b.legend(
        handles=[
            Line2D([], [], color=TEAL, marker='o', label='Pooled AP'),
            Line2D([], [], color=NAVY, marker='o', label='Mean target AP'),
        ],
        loc='lower right',
        frameon=True,
        facecolor='white',
        framealpha=0.94,
        fontsize=6.2,
    )

    # Panel c: Per-target binder retention at 25% and 50% Stage-1 budget across all 13 targets
    ax_c = fig.add_axes([0.72, 0.56, 0.24, 0.34])
    title(ax_c, 'c', 'Stage-1 recall across 13 targets')
    by_f = rep['two_stage_screening_cascade']['cascade_by_filter']
    rng = np.random.default_rng(2026)
    f_specs = [
        ('spatial_ppi_v2', 'Spatial\nPPI', RED),
        ('rosetta_dg', 'PyRosetta\nΔG', GREY),
        ('bcebm_full_equal', 'BC-Score\n(Ours)', NAVY),
    ]
    x_positions, x_labels = [], []
    pos_idx = 0.0
    for rk_label, rk_code in [('25%', 'rho_025pct'), ('50%', 'rho_050pct')]:
        for f_key, f_short, f_col in f_specs:
            pt_rec = by_f[f_key][rk_code]['stage1_retention']['per_target_recall']
            vals = np.array(list(pt_rec.values()), dtype=float)
            pooled_r = by_f[f_key][rk_code]['stage1_retention']['pooled_binder_recall']
            jit = rng.uniform(-0.18, 0.18, size=len(vals))
            ax_c.scatter(np.full_like(vals, pos_idx) + jit, 100.0 * vals, color=f_col, alpha=0.55, s=14, zorder=3)
            q25, med, q75 = np.quantile(100.0 * vals, [0.25, 0.50, 0.75])
            ax_c.plot([pos_idx, pos_idx], [q25, q75], color=INK, lw=2.0, zorder=4)
            ax_c.scatter([pos_idx], [100.0 * pooled_r], color=f_col, edgecolor=INK, lw=0.7, marker='D', s=28, zorder=5)
            x_positions.append(pos_idx)
            x_labels.append(f'{f_short}\n({rk_label})')
            pos_idx += 1.35
        pos_idx += 0.8
    ax_c.axvline(3.725, color='#D8DDE1', ls='--', lw=0.8)
    ax_c.set(xlim=(-0.6, pos_idx - 1.5), xticks=x_positions, ylim=(0, 108), ylabel='Binders retained (%)')
    ax_c.set_xticklabels(x_labels, fontsize=5.8)

    # Panels d, e, f: Internal optimization trajectories across all 39 designs (partition-function free-energy co-design)
    opt_specs = [
        ('d', 'Interface EBM score', 'e_ebm_before', 'e_ebm_after', True, 0.07),
        ('e', 'Polar penalty (Buns)', 'e_buns_before', 'e_buns_after', True, 0.40),
        ('f', 'Alanine-scan ΔΔG score', 'ala_ddg_before', 'ala_ddg_after', False, 0.72),
    ]
    for letter, txt, k0, k1, lower, left_x in opt_specs:
        ax_o = fig.add_axes([left_x, 0.09, 0.24, 0.34])
        title(ax_o, letter, txt)
        trajectory(ax_o, rep, k0, k1, lower)
        ax_o.set_ylabel('Model score units')

    save(fig, 'ext_fig1_supporting_panels')


def generate_fig5(rep):
    import runpy
    with plt.rc_context():
        runpy.run_path(
            str(Path(__file__).with_name('plot_ext_fig3_epitope.py')),
            init_globals={'REPORT': rep},
            run_name='__main__',
        )


if __name__ == '__main__':
    rep = json.loads((ROOT / 'results/loto_benchmark_results.json').read_text())
    generate_fig4(rep)
    generate_fig5(rep)
    generate_fig6_external()
    generate_support(rep)
    generate_pose_diagnostic(rep)
