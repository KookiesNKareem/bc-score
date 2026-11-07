# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "pymol-open-source-whl",
# ]
# ///
"""Render Mdm2 interface sidechains colored by discrepancy with 3D displacement vectors to unbound reconstructed centroids."""
from pathlib import Path
import json, math, os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pymol
pymol.pymol_argv = ['pymol', '-cq']
pymol.finish_launching()
from pymol import cmd
from pymol.cgo import CYLINDER, CONE, SPHERE, COLOR

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / 'paper/figures'
OUT = F
v = json.loads((F / 'mdm2_pose_matched_vectors.json').read_text())
stops = [(0, (1, .93, .68)), (.5, (.99, .54, .20)), (1, (.70, .07, .16))]

def color(x):
    t = max(0, min(1, x / 3.1))
    left, right = (stops[0], stops[1]) if t <= .5 else (stops[1], stops[2])
    w = (t - left[0]) / (right[0] - left[0])
    return [a + (b - a) * w for a, b in zip(left[1], right[1])]

cmd.reinitialize()
cmd.bg_color('white')
for k, val in [
    ('orthoscopic', 1), ('depth_cue', 0), ('ray_opaque_background', 1),
    ('antialias', 2), ('spec_reflect', .12), ('ambient', .55),
    ('direct', .5), ('ray_shadows', 0), ('two_sided_lighting', 1),
    ('surface_quality', 1), ('cartoon_fancy_helices', 1)
]:
    cmd.set(k, val)

cmd.set_color('backbone', [.65, .68, .70])
cmd.set_color('pocket', [.90, .91, .92])
for key, name in [('mirage_fp', 'mdm2_mirage_fp_pose.pdb'), ('true_hit', 'mdm2_true_hit_pose.pdb')]:
    cmd.load(str(F / name), key)

cmd.hide('everything', 'all')
cmd.orient('mirage_fp')
cmd.turn('y', 18)
cmd.turn('x', 8)
cmd.zoom('mirage_fp', buffer=1.5)
overview = cmd.get_view()

cmd.turn('y', -18)
cmd.turn('x', 6)
cmd.pseudoatom('low', pos=[0, -8, 0])
cmd.pseudoatom('high', pos=[0, 7, 0])
cmd.zoom('low or high', buffer=8.5)
cmd.delete('low')
cmd.delete('high')
detail = cmd.get_view()

teal_unb = [0.00, 0.50, 0.51]
ink_vec = [0.14, 0.19, 0.23]

for key in ['mirage_fp', 'true_hit']:
    cmd.hide('everything', 'all')
    cmd.disable('all')
    cmd.enable(key)
    cmd.show('surface', f'{key} and chain B')
    cmd.color('pocket', f'{key} and chain B')
    cmd.show('cartoon', f'{key} and chain A')
    cmd.color('backbone', f'{key} and chain A')
    cmd.set('cartoon_transparency', 0, f'{key} and chain A')

    cgo_objs = []
    for i, r in enumerate(v[key]):
        sel = f"{key} and chain A and resi {r['res_idx_1based']} and not name N+C+O and not elem H"
        c = color(r['strain_A'])
        cmd.set_color(f'c_{key}_{i}', c)
        cmd.show('sticks', sel)
        cmd.color(f'c_{key}_{i}', sel)
        cmd.set('stick_radius', .22, sel)

        p1 = r['sc_gen']
        p2 = r['sc_unb']
        dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        # Reconstructed unbound centroid sphere (teal) and supplied centroid sphere
        cgo_objs.extend([COLOR, *c, SPHERE, p1[0], p1[1], p1[2], 0.30])
        cgo_objs.extend([COLOR, *teal_unb, SPHERE, p2[0], p2[1], p2[2], 0.34])
        if dist > 0.45:
            ux, uy, uz = dx / dist, dy / dist, dz / dist
            head_len = min(0.55, 0.42 * dist)
            mid = [p2[0] - ux * head_len, p2[1] - uy * head_len, p2[2] - uz * head_len]
            cgo_objs.extend([
                CYLINDER, p1[0], p1[1], p1[2], mid[0], mid[1], mid[2], 0.11,
                *c, *teal_unb,
                CONE, mid[0], mid[1], mid[2], p2[0], p2[1], p2[2], 0.26, 0.0,
                *teal_unb, *teal_unb, 1.0, 1.0
            ])

    cmd.set_view(overview)
    cmd.clip('slab', 120)
    cmd.png(str(OUT / f'mdm2_discrepancy_{key}_overview.png'), width=900, height=900, dpi=300, ray=1)

    cgo_name = f'vec_{key}'
    cmd.load_cgo(cgo_objs, cgo_name)
    cmd.set('cartoon_transparency', .82, f'{key} and chain A')
    cmd.set_view(detail)
    cmd.clip('slab', 120)
    cmd.png(str(OUT / f'mdm2_discrepancy_{key}_detail.png'), width=1500, height=1500, dpi=300, ray=1)
    cmd.delete(cgo_name)

cmd.quit()

