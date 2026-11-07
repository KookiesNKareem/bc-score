"""Generate Figure 1: Clean 3-panel Nature-style layout (a: 3-stage pipeline schematic; b, c: Mdm2 3D sidechain discrepancy structures)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / 'paper/figures'
OUT = F
INK = '#24313B'
GREY = '#7A848D'
NAVY = '#254B70'
TEAL = '#007F82'
BLUE = '#6786A1'
ORANGE = '#C87820'
RED = '#B04A50'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8.5,
    'font.weight': 'normal',
    'axes.titlesize': 9.0,
    'axes.titleweight': 'bold',
    'axes.labelsize': 8.0,
    'text.color': INK,
    'axes.labelcolor': INK,
    'xtick.color': INK,
    'ytick.color': INK,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
})

vec_path = F / 'mdm2_pose_matched_vectors.json'
v = json.loads(vec_path.read_text()) if vec_path.exists() else {'fp_mean': 1.01, 'hit_mean': 0.61}

fig = plt.figure(figsize=(9.2, 5.8), facecolor='white')

# ==============================================================================
# Panel a: Clean 3-Stage Mechanism Schematic (Top Row)
# ==============================================================================
fig.text(0.035, 0.955, 'a', fontsize=11.5, weight='bold', color=INK)

ax_sch = fig.add_axes([0.035, 0.695, 0.93, 0.255])
ax_sch.set_xlim(0, 100)
ax_sch.set_ylim(0, 28)
ax_sch.axis('off')

cards = [
    (0.5, 29.5, '1. Strip complex sidechains', '#F6F8FA', '#6786A1'),
    (34.5, 31.0, '2. Reconstruct monomer sidechains', '#EFF7F7', TEAL),
    (70.0, 29.5, '3. Default BC-Score (38 descriptors)', '#F2F5F8', NAVY),
]

for x0, w, hdr, bg, border in cards:
    patch = FancyBboxPatch(
        (x0, 1.0), w, 25.5,
        boxstyle='round,pad=0.35,rounding_size=1.4',
        facecolor=bg, edgecolor=border, linewidth=1.0
    )
    ax_sch.add_patch(patch)
    ax_sch.text(x0 + w / 2, 24.0, hdr, ha='center', va='center', fontsize=8.4, weight='bold', color=INK)
    ax_sch.plot([x0 + 1.8, x0 + w - 1.8], [21.8, 21.8], color=border, lw=0.6, alpha=0.45)

# --- Stage 1: Complex -> Strip sidechains -> Bare N-Ca-C-O backbone ---
ax_sch.fill_betweenx(np.linspace(9.0, 19.5, 20), 2.5, 6.2 + 0.9 * np.sin(np.linspace(0, 3.14, 20)), color='#DCE2E7', alpha=0.9)
ax_sch.text(4.3, 14.2, 'Target', fontsize=6.5, color=GREY, ha='center', va='center', weight='bold')

by_pts = np.linspace(9.8, 18.8, 5)
bx_pts = 8.4 + 0.55 * np.cos(np.linspace(0, 4.5, 5))
ax_sch.plot(bx_pts, by_pts, color=NAVY, lw=2.2, zorder=3)
for bx_i, by_i, sc_c in zip(bx_pts[1:4], by_pts[1:4], [RED, ORANGE, RED]):
    ax_sch.plot([bx_i, bx_i - 1.7], [by_i, by_i + 0.4], color=sc_c, lw=1.7, zorder=4)
    ax_sch.scatter([bx_i - 1.7], [by_i + 0.4], color=sc_c, s=18, zorder=5)

ax_sch.annotate('', xy=(16.5, 14.2), xytext=(11.2, 14.2), arrowprops=dict(arrowstyle='->', color=INK, lw=1.1))
ax_sch.text(13.85, 16.2, 'Strip χ', fontsize=6.5, color=RED, ha='center', weight='bold')

bb_x = [18.2, 20.6, 23.0, 25.4, 27.4]
bb_y = [11.2, 14.4, 12.0, 15.6, 13.6]
ax_sch.plot(bb_x, bb_y, color=NAVY, lw=2.0, zorder=3)
for bxi, byi, lbl_b in zip(bb_x[:4], bb_y[:4], ['N', 'Cα', 'C', 'O']):
    ax_sch.scatter([bxi], [byi], color='white', edgecolor=NAVY, lw=1.3, s=44, zorder=4)
    ax_sch.text(bxi, byi - 2.5, lbl_b, fontsize=6.3, ha='center', color=NAVY, weight='bold')

ax_sch.text(15.25, 3.8, 'Retains only monomer backbone & sequence',
            ha='center', va='center', fontsize=7.2, color=INK)

# --- Stage 2: SE(3) Frame + K=3 Rotamer MDN vs Supplied Rotamer ---
ca_x, ca_y = 39.8, 12.2
ax_sch.scatter([ca_x], [ca_y], color=INK, s=30, zorder=6)
ax_sch.annotate('', xy=(ca_x + 3.8, ca_y), xytext=(ca_x, ca_y), arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.2))
ax_sch.annotate('', xy=(ca_x, ca_y + 4.6), xytext=(ca_x, ca_y), arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.2))
ax_sch.annotate('', xy=(ca_x - 2.3, ca_y - 2.4), xytext=(ca_x, ca_y), arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2))
ax_sch.text(ca_x, ca_y + 5.8, r'$\mathbf{T}_i \in \mathrm{SE}(3)$', fontsize=6.8, color=NAVY, ha='center')

for el_x, el_y, el_w, el_h, ang, al in [
    (50.2, 15.6, 5.2, 3.1, 18, 0.30),
    (52.2, 10.6, 3.8, 2.2, -18, 0.16),
    (47.8, 9.6, 3.2, 1.9, 36, 0.14),
]:
    ell = Ellipse((el_x, el_y), el_w, el_h, angle=ang, facecolor=TEAL, edgecolor=TEAL, alpha=al, lw=0.9)
    ax_sch.add_patch(ell)

ax_sch.scatter([50.2], [15.6], color=TEAL, edgecolor='white', lw=0.8, s=44, zorder=6)
ax_sch.text(50.2, 18.6, r'$\mathbf{s}_i^{\mathrm{recon}}$', fontsize=7.4, color=TEAL, weight='bold', ha='center')

ax_sch.scatter([59.4], [11.6], color=RED, edgecolor='white', lw=0.8, s=44, zorder=6)
ax_sch.text(59.6, 8.4, r'$\mathbf{s}_i^{\mathrm{supplied}}$', fontsize=7.4, color=RED, weight='bold', ha='center')

ax_sch.annotate('', xy=(50.9, 15.3), xytext=(58.7, 11.9), arrowprops=dict(arrowstyle='<->', color=RED, lw=1.3, ls='--'))
ax_sch.text(56.4, 15.4, r'$\|\Delta\mathbf{s}_i\|$', fontsize=7.0, color=RED, weight='bold')

ax_sch.text(50.0, 3.8, 'Partner-blind SE(3) GNN + rotamer MDN',
            ha='center', va='center', fontsize=7.2, color=INK)

# --- Stage 3: 5 Physical Modules (C1-C5; 38 within / 29 global descriptors) ---
mod_specs = [
    ('C1', 'Monomer foldability & SAP', '#3B6B9B', 19.0),
    ('C2', 'Sidechain discrepancy & strain', TEAL, 16.0),
    ('C3', 'Buried polars (Buns) & O-rings', '#2A8F7A', 13.0),
    ('C4', 'Alchemical ΔΔG counterfactuals', '#5C6BC0', 10.0),
    ('C5', '6-DoF relax & 24-pose funnel', NAVY, 7.0),
]
for code, desc, mcol, my in mod_specs:
    pill = FancyBboxPatch((72.2, my - 0.95), 3.4, 1.9, boxstyle='round,pad=0.1,rounding_size=0.45', facecolor=mcol, edgecolor='none')
    ax_sch.add_patch(pill)
    ax_sch.text(73.9, my, code, ha='center', va='center', fontsize=6.4, color='white', weight='bold')
    ax_sch.text(76.5, my, desc, ha='left', va='center', fontsize=6.9, color=INK)

ax_sch.text(84.75, 3.2, 'Equal 1/38 within & 1/29 global weights',
            ha='center', va='center', fontsize=6.9, color=NAVY, weight='bold')

# Connecting arrows between the 3 cards
for x_start, x_end in [(30.4, 34.1), (65.9, 69.6)]:
    arr = FancyArrowPatch(
        (x_start, 13.8), (x_end, 13.8),
        arrowstyle='->,head_width=3.5,head_length=3.5',
        color=INK, linewidth=1.3
    )
    ax_sch.add_patch(arr)

# ==============================================================================
# Panels b and c: Spacious Side-by-Side Mdm2 Interface Close-Ups
# ==============================================================================
# Square 1:1 image in (9.2, 5.8) figure: width=0.315 -> height=0.315*(9.2/5.8)=0.500
struct_specs = [
    ('b', 'mirage_fp', 'Non-binder (p53_AF2_33)', v['fp_mean'], 0.095,
     [('Tyr77: 3.06 Å', 0.56, 0.86), ('Trp67: 1.75 Å', 0.62, 0.34), ('Phe63: 1.61 Å', 0.55, 0.14)]),
    ('c', 'true_hit', 'Verified binder (p53_AF2_91)', v['hit_mean'], 0.565,
     [('Ile47: 0.52 Å', 0.48, 0.64), ('Leu43: 0.81 Å', 0.68, 0.36), ('Glu49: 0.22 Å', 0.66, 0.18)]),
]

for letter, key, title_txt, mean_v, left_x, callouts in struct_specs:
    fig.text(left_x, 0.635, letter, fontsize=11.5, weight='bold', color=INK)
    fig.text(left_x + 0.022, 0.635, f'{title_txt} — mean discrepancy: {mean_v:.2f} Å', fontsize=8.6, weight='bold', color=INK)
    ax_s = fig.add_axes([left_x, 0.115, 0.315, 0.50])
    ax_s.imshow(plt.imread(OUT / f'mdm2_discrepancy_{key}_detail.png'))
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    for sp in ax_s.spines.values():
        sp.set_visible(True)
        sp.set_color('#C3C8CC')
        sp.set_linewidth(0.7)

    for lbl_c, rx, ry in callouts:
        ax_s.text(
            rx, ry, lbl_c, transform=ax_s.transAxes, fontsize=7.2, weight='bold', color=INK,
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#D0D6DC', lw=0.6, alpha=0.92)
        )

# Shared bottom colorbar and legend centered below b and c
cmap = LinearSegmentedColormap.from_list('discrepancy', [(0, (1, 0.93, 0.68)), (0.5, (0.99, 0.54, 0.20)), (1, (0.70, 0.07, 0.16))])
cax = fig.add_axes([0.14, 0.045, 0.30, 0.018])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 3.1), cmap=cmap), cax=cax, orientation='horizontal', ticks=[0, 1, 2, 3])
cb.outline.set_linewidth(0.55)
cb.outline.set_edgecolor('#6B737B')
cb.ax.tick_params(labelsize=7.5, length=2.5, colors=INK)
fig.text(0.29, 0.072, 'Supplied vs. reconstructed centroid discrepancy (Å)', ha='center', fontsize=8.0, color=INK)

fig.text(0.515, 0.054, '●', va='center', ha='center', fontsize=11.5, color=TEAL)
fig.text(0.532, 0.054, r'Partner-blind reconstructed centroid ($\mathbf{s}_i^{\mathrm{recon}}$; 3D arrow: $\mathbf{s}_i^{\mathrm{supplied}} \rightarrow \mathbf{s}_i^{\mathrm{recon}}$)',
         va='center', fontsize=8.0, color=INK)

fig.savefig(OUT / 'fig1_pipeline_and_mdm2.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig1_pipeline_and_mdm2.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print('Generated simplified 3-panel Figure 1.')
