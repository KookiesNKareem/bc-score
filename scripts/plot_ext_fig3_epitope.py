"""Generate Extended Data Figure 3 (ext_fig3_same_epitope_pairs.pdf): Pairwise same-epitope binder discrimination without top banner title."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/figures'
INK = '#24313B'
NAVY = '#254B70'
TEAL = '#007F82'
GREY = '#7A848D'
ORANGE = '#C87820'
PURPLE = '#7654AA'
BLUE = '#6786A1'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 8.8,
    'axes.titleweight': 'bold',
    'axes.labelsize': 8,
    'axes.labelweight': 'normal',
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
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

fig = plt.figure(figsize=(6.4, 3.35), facecolor='white')

rep = globals().get('REPORT')
if rep is None:
    rep = json.loads((ROOT / 'results/loto_benchmark_results.json').read_text())

ep = rep['overlapping_epitope_pairs_by_threshold']
xs = np.array([50, 60, 70, 80, 90])
ks = [f'{x}pct' for x in xs]

ax = fig.add_axes([0.13, 0.25, 0.56, 0.64])
ax.set(
    xlim=(48, 91.5),
    ylim=(46, 76.5),
    xticks=xs,
    yticks=[50, 55, 60, 65, 70, 75],
    ylabel='Verified binder ranked higher (%)',
    xlabel='Minimum target epitope Jaccard contact overlap (%)',
)
ax.set_title('Same-epitope pairwise discrimination across overlap regimes', loc='left', pad=8)
ax.xaxis.labelpad = 6
ax.yaxis.labelpad = 6
ax.tick_params(width=0.6, length=3)
ax.set_axisbelow(True)
ax.grid(axis='y', color='#E4E8EB', lw=0.5)
ax.axhline(50, color='#9FA9AF', lw=0.7, ls=(0, (2, 3)))
ax.text(50.2, 47.3, 'Random chance (50%)', fontsize=7.0, color=GREY)

specs = [
    ('bcebm_full_equal', 'BC-Score (Default)', NAVY, 'o', '-', 70.5),
    ('bcebm_5comp_equal', 'BC-Score-5M', TEAL, 's', '--', 66.8),
    ('rosetta_dg', 'PyRosetta ΔG', GREY, 'D', '--', 63.4),
    ('supplied_sidechain_ebm', 'Supplied complex\nsidechain score', ORANGE, '^', '-.', 59.2),
    ('esm2_pll', 'ESM-2 log-likelihood', PURPLE, 'P', ':', 54.8),
    ('backbone_only_if', 'Backbone-only\ninverse folding', BLUE, 'v', '--', 50.2),
]

for key, label, color, marker, ls, label_y in specs:
    values = np.array([ep[k][key + '_win_rate'] for k in ks]) * 100
    ci = np.array([ep[k][key + '_cluster_ci95'] for k in ks]) * 100
    if key in ['bcebm_full_equal', 'rosetta_dg']:
        ax.fill_between(xs, ci[:, 0], ci[:, 1], color=color, alpha=0.09, lw=0)
    lw_v = 2.0 if key == 'bcebm_full_equal' else (1.5 if key == 'bcebm_5comp_equal' else 1.2)
    ax.plot(xs, values, color=color, marker=marker, ms=3.8, lw=lw_v, ls=ls, zorder=4)
    ax.plot([90, 93.8, 95.0], [values[-1], label_y, label_y], color=color, lw=0.6, clip_on=False)
    ax.text(
        95.6, label_y, label, ha='left', va='center', fontsize=7.2, color=color,
        weight='bold' if key.startswith('bcebm') else 'normal', clip_on=False, linespacing=1.1
    )

for x, k in zip(xs, ks):
    n_p = ep[k]['n_pairs']
    n_str = f"{n_p // 1000}k" if n_p >= 10000 else f"{n_p:,}"
    ax.text(x, -0.28, f"{ep[k]['n_targets']} ({n_str})", transform=ax.get_xaxis_transform(), ha='center', va='center', fontsize=6.6, color=GREY)

ax.text(47.2, -0.28, 'Targets (pairs):', transform=ax.get_xaxis_transform(), ha='right', va='center', fontsize=6.6, color=GREY)

fig.savefig(OUT / 'ext_fig3_same_epitope_pairs.pdf', bbox_inches='tight')
fig.savefig(OUT / 'ext_fig3_same_epitope_pairs.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print('Generated Extended Data Figure 3 (same-epitope pairwise discrimination).')
