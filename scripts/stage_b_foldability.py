"""Thermodynamic Monomer Foldability (3D Inverse-Folding + Intra-Monomer 5-Bead Physics) & Interface Desolvation.

Implements the three co-folder-free, Rosetta-free physical & structural modules:
  - Fix A: BackboneInverseFoldingGNN
      Pure 3D backbone (N, CA, C, O + ideal tetrahedral virtual CB) SE(3)-invariant GNN
      trained strictly on SAbDab2 crystal monomers (zero Overath data) to predict P_3D(a_i | X_backbone).
      Computes:
        * Unbound monomer 3D inverse-folding log-likelihood (all, buried core, surface, secondary structure)
        * Bayesian foldability contrast: log P_3D(a_i | X_backbone) - log P_1D_ESM2(a_i | seq)
        * Interface complex-vs-monomer 3D inverse-folding gain:
          log P_3D(a_i | X_complex) - log P_3D(a_i | X_unbound)
  - Fix B: Intensive Intra-Monomer 5-Bead Physical Stability (compute_intra_monomer_5bead_physics)
      Computes per-residue intra-monomer steric strain, 5-bead van der Waals core packing,
      backbone N-O hydrogen-bond satisfaction, buried unsatisfied polar core penalty,
      hydrophobic core-vs-surface partitioning, SAP aggregation proxy, and electrostatics.
  - Fix C: Interface Buried Unsatisfied Polar Desolvation (compute_interface_desolvation_and_density)
      Penalizes interface backbone N/O and polar/charged sidechains that are buried by the partner
      without forming a complementary hydrogen bond or salt bridge.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stage_a_orientation import (
    MonomerFrameGNNLayer,
    PHYSCHEM_TABLE,
    build_local_frames,
    rbf_encode,
)

PROJECT = Path(__file__).resolve().parents[1]
SABDAB2_CACHE = PROJECT / "results" / "sabdab2_cluster_reps_cache.pt"
IF_CKPT_PATH = PROJECT / "results" / "backbone_inverse_folding_gnn.pt"


def construct_ideal_virtual_cb(n_pos: torch.Tensor, ca_pos: torch.Tensor, c_pos: torch.Tensor) -> torch.Tensor:
    """Construct ideal tetrahedral virtual CB strictly from backbone N, CA, C (zero GLY/sidechain leak)."""
    b = ca_pos - n_pos
    c = c_pos - ca_pos
    a = torch.cross(b, c, dim=-1)
    # Standard ideal tetrahedral CB coefficients relative to (a, b, c)
    v_cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_pos
    return v_cb


def _dihedral_sin_cos(
    p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (sin, cos) of dihedral angle between 4 points [K, 3]."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = F.normalize(b1, dim=-1, eps=1e-6)
    v = b0 - (b0 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm
    w = b2 - (b2 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1_norm, v, dim=-1) * w).sum(dim=-1)
    norm = torch.sqrt(x * x + y * y + 1e-8)
    return y / norm, x / norm


def compute_pure_backbone_geometry(
    beads: torch.Tensor,
    chain: torch.Tensor,
    residue_index: torch.Tensor,
    cutoff: float = 12.0,
) -> dict[str, torch.Tensor]:
    """Extract leak-free 3D backbone node features [N, 30] and edge features [E, 41] from N, CA, C, O."""
    n_pos = beads[:, 0]
    ca_pos = beads[:, 1]
    c_pos = beads[:, 2]
    o_pos = beads[:, 3]
    vcb_pos = construct_ideal_virtual_cb(n_pos, ca_pos, c_pos)
    N = len(ca_pos)
    dtype = ca_pos.dtype
    dev = ca_pos.device

    frames = build_local_frames(n_pos, ca_pos, c_pos)  # [N, 3, 3]
    u_vcb = F.normalize(vcb_pos - ca_pos, dim=-1, eps=1e-6)

    # 1. Dihedrals phi, psi, omega (6 dims) + bond angle & curvature (2 dims)
    phi_sin = torch.zeros((N,), dtype=dtype, device=dev)
    phi_cos = torch.zeros((N,), dtype=dtype, device=dev)
    psi_sin = torch.zeros((N,), dtype=dtype, device=dev)
    psi_cos = torch.zeros((N,), dtype=dtype, device=dev)
    omg_sin = torch.zeros((N,), dtype=dtype, device=dev)
    omg_cos = torch.zeros((N,), dtype=dtype, device=dev)

    if N >= 2:
        same_prev = (chain[1:] == chain[:-1]) & ((residue_index[1:] - residue_index[:-1]) == 1)
        idx_prev = torch.nonzero(same_prev, as_tuple=True)[0] + 1
        if idx_prev.numel() > 0:
            s_phi, c_phi = _dihedral_sin_cos(
                c_pos[idx_prev - 1], n_pos[idx_prev], ca_pos[idx_prev], c_pos[idx_prev]
            )
            phi_sin[idx_prev] = s_phi
            phi_cos[idx_prev] = c_phi

            s_omg, c_omg = _dihedral_sin_cos(
                ca_pos[idx_prev - 1], c_pos[idx_prev - 1], n_pos[idx_prev], ca_pos[idx_prev]
            )
            omg_sin[idx_prev] = s_omg
            omg_cos[idx_prev] = c_omg

            s_psi, c_psi = _dihedral_sin_cos(
                n_pos[idx_prev - 1], ca_pos[idx_prev - 1], c_pos[idx_prev - 1], n_pos[idx_prev]
            )
            psi_sin[idx_prev - 1] = s_psi
            psi_cos[idx_prev - 1] = c_psi

    u_nc = F.normalize(n_pos - ca_pos, dim=-1, eps=1e-6)
    u_cc = F.normalize(c_pos - ca_pos, dim=-1, eps=1e-6)
    tau_cos = (u_nc * u_cc).sum(dim=-1)

    curv_cos = torch.zeros((N,), dtype=dtype, device=dev)
    if N >= 3:
        v1 = F.normalize(ca_pos[1:-1] - ca_pos[:-2], dim=-1, eps=1e-6)
        v2 = F.normalize(ca_pos[2:] - ca_pos[1:-1], dim=-1, eps=1e-6)
        valid_mid = (chain[1:-1] == chain[:-2]) & (chain[2:] == chain[1:-1])
        curv_cos[1:-1] = torch.where(valid_mid, (v1 * v2).sum(dim=-1), torch.zeros_like(curv_cos[1:-1]))

    # 2. Intra-backbone N-O hydrogen bond geometry (6 dims)
    d_no = torch.cdist(n_pos, o_pos)  # [N_donor, N_acceptor]
    seq_sep_signed = residue_index[:, None] - residue_index[None, :]
    same_ch_mat = chain[:, None] == chain[None, :]

    def _hbond_score_at_offset(offset: int) -> torch.Tensor:
        m = same_ch_mat & (seq_sep_signed == offset)
        d_m = torch.where(m, d_no, torch.full_like(d_no, 15.0)).amin(dim=-1)
        return torch.exp(-0.5 * ((d_m - 2.95) / 0.45) ** 2)

    hb_i_minus_4 = _hbond_score_at_offset(4)   # alpha-helix N(i) -> O(i-4)
    hb_i_minus_3 = _hbond_score_at_offset(3)   # 3_10-helix N(i) -> O(i-3)
    hb_i_plus_4 = _hbond_score_at_offset(-4)
    hb_i_plus_3 = _hbond_score_at_offset(-3)

    long_mask = (~same_ch_mat) | (seq_sep_signed.abs() >= 5)
    d_no_long_n = torch.where(long_mask, d_no, torch.full_like(d_no, 15.0)).amin(dim=1)
    d_no_long_o = torch.where(long_mask, d_no, torch.full_like(d_no, 15.0)).amin(dim=0)
    hb_sheet_n = torch.exp(-0.5 * ((d_no_long_n - 2.90) / 0.45) ** 2)
    hb_sheet_o = torch.exp(-0.5 * ((d_no_long_o - 2.90) / 0.45) ** 2)

    # 3. Multi-scale CA & vCB burial (8 dims) + Half-Sphere Exposure & Conical Crowding (6 dims)
    dist_ca = torch.cdist(ca_pos, ca_pos)
    dist_vcb = torch.cdist(vcb_pos, vcb_pos)
    with torch.no_grad():
        ca_ng = ca_pos.detach()
        u_vcb_ng = u_vcb.detach()
        eye_mask = torch.eye(N, dtype=torch.bool, device=dev)
        dist_ca_m = dist_ca.detach().masked_fill(eye_mask, 1e6)
        dist_vcb_m = dist_vcb.detach().masked_fill(eye_mask, 1e6)

        b_ca6 = (dist_ca_m < 6.0).to(dtype).sum(dim=-1) / 8.0
        b_ca8 = (dist_ca_m < 8.0).to(dtype).sum(dim=-1) / 15.0
        b_ca10 = (dist_ca_m < 10.0).to(dtype).sum(dim=-1) / 25.0
        b_ca12 = (dist_ca_m < 12.0).to(dtype).sum(dim=-1) / 40.0

        b_cb6 = (dist_vcb_m < 6.0).to(dtype).sum(dim=-1) / 8.0
        b_cb8 = (dist_vcb_m < 8.0).to(dtype).sum(dim=-1) / 15.0
        b_cb10 = (dist_vcb_m < 10.0).to(dtype).sum(dim=-1) / 25.0
        b_cb12 = (dist_vcb_m < 12.0).to(dtype).sum(dim=-1) / 40.0

        vec_ij = ca_ng[None, :, :] - ca_ng[:, None, :]  # [N, N, 3]
        dir_ij = F.normalize(vec_ij, dim=-1, eps=1e-6)
        dot_vcb = torch.einsum("ijk,ik->ij", dir_ij, u_vcb_ng)  # [N, N]

        in8 = dist_ca_m < 8.0
        in12 = dist_ca_m < 12.0
        hse_up8 = (in8 & (dot_vcb > 0.0)).to(dtype).sum(dim=-1) / 10.0
        hse_dn8 = (in8 & (dot_vcb <= 0.0)).to(dtype).sum(dim=-1) / 10.0
        hse_up12 = (in12 & (dot_vcb > 0.0)).to(dtype).sum(dim=-1) / 20.0
        hse_dn12 = (in12 & (dot_vcb <= 0.0)).to(dtype).sum(dim=-1) / 20.0
        protrusion = (hse_dn12 - hse_up12) / (hse_dn12 + hse_up12 + 0.1)

        # Conical steric crowding directly in front of vCB (cos > 0.5 and dist_vcb < 7.0 A)
        cone_crowd = ((dist_vcb_m < 7.0) & (dot_vcb > 0.5)).to(dtype).sum(dim=-1) / 6.0

        is_n_term = torch.ones((N,), dtype=dtype, device=dev)
        is_c_term = torch.ones((N,), dtype=dtype, device=dev)
        if N >= 2:
            same_adj = (chain[1:] == chain[:-1]) & ((residue_index[1:] - residue_index[:-1]).abs() == 1)
            is_n_term[1:] = (~same_adj).to(dtype)
            is_c_term[:-1] = (~same_adj).to(dtype)

        src, dst = torch.nonzero(dist_ca_m < cutoff, as_tuple=True)

    node_feats = torch.stack(
        [
            phi_sin, phi_cos, psi_sin, psi_cos, omg_sin, omg_cos,
            tau_cos, curv_cos,
            hb_i_minus_4, hb_i_minus_3, hb_i_plus_4, hb_i_plus_3, hb_sheet_n, hb_sheet_o,
            b_ca6, b_ca8, b_ca10, b_ca12,
            b_cb6, b_cb8, b_cb10, b_cb12,
            hse_up8, hse_dn8, hse_up12, hse_dn12, protrusion, cone_crowd,
            is_n_term, is_c_term,
        ],
        dim=-1,
    )  # [N, 30]

    if src.numel() > 0:
        d_ca_e = dist_ca[src, dst]
        d_cb_e = dist_vcb[src, dst]
        rbf_ca = rbf_encode(d_ca_e, 0.0, cutoff, 16)
        rbf_cb = rbf_encode(d_cb_e, 0.0, cutoff, 8)
        rel_vec = F.normalize(ca_pos[dst] - ca_pos[src], dim=-1, eps=1e-6)
        dir_i = torch.einsum("nij,ni->nj", frames[src], rel_vec)
        dir_j = torch.einsum("nij,ni->nj", frames[dst], -rel_vec)
        q_ij = torch.einsum("nij,nik->njk", frames[src], frames[dst])
        rot_6d = q_ij[:, :, :2].reshape(-1, 6)

        same_chain = chain[src] == chain[dst]
        seq_diff = (residue_index[dst] - residue_index[src]).abs()
        seq_feats = torch.stack(
            [
                same_chain.to(dtype),
                (same_chain & (seq_diff == 1)).to(dtype),
                (same_chain & (seq_diff == 2)).to(dtype),
                (same_chain & (seq_diff <= 4)).to(dtype),
                torch.clamp(seq_diff.to(dtype) / 32.0, max=1.0) * same_chain.to(dtype),
            ],
            dim=-1,
        )
        edge_attr = torch.cat([rbf_ca, rbf_cb, dir_i, dir_j, rot_6d, seq_feats], dim=-1)  # [E, 41]
    else:
        edge_attr = torch.zeros((0, 41), dtype=dtype, device=dev)

    return {
        "node_feats": node_feats,
        "src": src,
        "dst": dst,
        "edge_attr": edge_attr,
        "burial_cb10": b_cb10,
        "protrusion": protrusion,
        "hb_helix": torch.maximum(hb_i_minus_4, hb_i_plus_4),
        "hb_sheet": torch.maximum(hb_sheet_n, hb_sheet_o),
        "frames": frames,
        "ca_pos": ca_pos,
    }


class BackboneInverseFoldingGNN(nn.Module):
    """3-Layer SE(3)-Invariant 3D Backbone Inverse-Folding GNN trained on SAbDab2 crystal monomers."""

    def __init__(self, hidden: int = 96, n_layers: int = 3):
        super().__init__()
        self.node_in = nn.Sequential(
            nn.Linear(30, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList([MonomerFrameGNNLayer(hidden, edge_dim=41) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden + 30, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 20),
        )

    def forward_from_geom(self, geom: dict[str, torch.Tensor]) -> torch.Tensor:
        dev = next(self.parameters()).device
        x0 = geom["node_feats"].to(dev)
        h = self.node_in(x0)
        src, dst, edge_attr = geom["src"].to(dev), geom["dst"].to(dev), geom["edge_attr"].to(dev)
        if src.numel() > 0:
            for layer in self.layers:
                h = layer(h, src, dst, edge_attr)
        logits = self.head(torch.cat([h, x0], dim=-1))
        return F.log_softmax(logits, dim=-1)  # [N, 20]

    def forward(
        self, beads: torch.Tensor, chain: torch.Tensor, residue_index: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        geom = compute_pure_backbone_geometry(beads, chain, residue_index)
        log_probs = self.forward_from_geom(geom)
        return log_probs, geom


def train_or_load_backbone_inverse_folding_gnn(force_retrain: bool = False) -> BackboneInverseFoldingGNN:
    """Train BackboneInverseFoldingGNN on SAbDab2 crystal monomers (zero Overath data) or load cached weights."""
    model = BackboneInverseFoldingGNN(hidden=96, n_layers=3)
    if IF_CKPT_PATH.exists() and not force_retrain:
        ckpt = torch.load(IF_CKPT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model

    print("Training 3D BackboneInverseFoldingGNN on SAbDab2 crystal monomers ...", flush=True)
    t0 = time.time()
    torch.manual_seed(42)
    cache = torch.load(SABDAB2_CACHE, map_location="cpu", weights_only=False)

    train_samples = []
    for cl_items in cache["train_clusters"].values():
        for comp in cl_items[:1]:
            for pref in ("rec", "lig"):
                beads = comp[f"{pref}_beads"]
                aa = comp[f"{pref}_aa"]
                chain = comp[f"{pref}_chain"]
                res_idx = comp[f"{pref}_residue_index"]
                if (aa < 20).sum() >= 20:
                    geom = compute_pure_backbone_geometry(beads, chain, res_idx)
                    geom["aa"] = aa
                    train_samples.append(geom)

    val_samples = []
    for comp in cache["test_complexes"][:40]:
        for pref in ("rec", "lig"):
            beads = comp[f"{pref}_beads"]
            aa = comp[f"{pref}_aa"]
            chain = comp[f"{pref}_chain"]
            res_idx = comp[f"{pref}_residue_index"]
            if (aa < 20).sum() >= 20:
                geom = compute_pure_backbone_geometry(beads, chain, res_idx)
                geom["aa"] = aa
                val_samples.append(geom)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=14, eta_min=2e-4)

    rng = np.random.default_rng(42)
    for epoch in range(1, 15):
        model.train()
        order = rng.permutation(len(train_samples))
        total_loss = 0.0
        total_tok = 0
        for start in range(0, len(order), 8):
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_cnt = 0
            for idx in order[start : start + 8]:
                g = train_samples[idx]
                valid = g["aa"] < 20
                log_p = model.forward_from_geom(g)
                loss = F.nll_loss(log_p[valid], g["aa"][valid], reduction="sum")
                batch_loss = batch_loss + loss
                batch_cnt += int(valid.sum().item())
            if batch_cnt > 0:
                (batch_loss / batch_cnt).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(batch_loss.item())
                total_tok += batch_cnt
        scheduler.step()

        if epoch in (1, 5, 10, 14):
            model.eval()
            with torch.no_grad():
                v_loss = 0.0
                v_correct = 0
                v_tok = 0
                for g in val_samples:
                    valid = g["aa"] < 20
                    log_p = model.forward_from_geom(g)
                    v_loss += float(F.nll_loss(log_p[valid], g["aa"][valid], reduction="sum").item())
                    v_correct += int((log_p[valid].argmax(dim=-1) == g["aa"][valid]).sum().item())
                    v_tok += int(valid.sum().item())
                ppl = math.exp(v_loss / max(v_tok, 1))
                rec_rate = v_correct / max(v_tok, 1)
                print(
                    f"  [IF-GNN Epoch {epoch:02d}/14] train_nll={total_loss/max(total_tok,1):.3f} | "
                    f"val_nll={v_loss/max(v_tok,1):.3f} val_ppl={ppl:.2f} recovery={rec_rate*100:.1f}% ({time.time()-t0:.1f}s)",
                    flush=True,
                )

    model.eval()
    IF_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, IF_CKPT_PATH)
    print(f"Saved trained BackboneInverseFoldingGNN to {IF_CKPT_PATH}", flush=True)
    return model


@torch.no_grad()
def compute_intra_monomer_5bead_physics(
    beads: torch.Tensor,
    aa: torch.Tensor,
    chain: torch.Tensor,
    residue_index: torch.Tensor,
) -> dict[str, float]:
    """Compute intensive per-residue intra-monomer 5-bead physical stability features (zero Rosetta)."""
    N = len(beads)
    L = float(max(N, 1))
    n_pos = beads[:, 0]
    ca_pos = beads[:, 1]
    c_pos = beads[:, 2]
    o_pos = beads[:, 3]
    sc_pos = beads[:, 4]

    phys = PHYSCHEM_TABLE[aa.clamp(0, 20)]  # [N, 6]: charge, hydro, pol, arom, h_don, h_acc
    charge = phys[:, 0]
    hydro = phys[:, 1]
    polarity = phys[:, 2]
    arom = phys[:, 3]
    h_don = phys[:, 4]
    h_acc = phys[:, 5]

    seq_sep = (residue_index[:, None] - residue_index[None, :]).abs()
    same_ch = chain[:, None] == chain[None, :]
    non_adj2 = (~same_ch) | (seq_sep >= 2)
    non_adj3 = (~same_ch) | (seq_sep >= 3)
    non_adj4 = (~same_ch) | (seq_sep >= 4)

    dist_ca = torch.cdist(ca_pos, ca_pos)
    dist_sc = torch.cdist(sc_pos, sc_pos)
    dist_no = torch.cdist(n_pos, o_pos)

    # 1. Intra-monomer 5-bead steric clashes (|i - j| >= 2, excluding H-bonded N-O)
    src2, dst2 = torch.nonzero(non_adj2 & (dist_ca < 10.0), as_tuple=True)
    if src2.numel() > 0:
        b_i = beads[src2]  # [P, 5, 3]
        b_j = beads[dst2]  # [P, 5, 3]
        d_5x5 = torch.cdist(b_i, b_j)  # [P, 5, 5]
        # Allow N(0)-O(3) and O(3)-N(0) down to 2.60 A (hydrogen bonds); other beads 3.15 A
        r0 = torch.full((1, 5, 5), 3.15, dtype=beads.dtype)
        r0[0, 0, 3] = 2.55
        r0[0, 3, 0] = 2.55
        # Adjacent |i - j| == 2 backbone N/CA/C can be ~2.9 A across turns
        is_sep2 = same_ch[src2, dst2] & (seq_sep[src2, dst2] == 2)
        overlap = F.relu(r0 - d_5x5)
        if is_sep2.any():
            bb_mask = torch.zeros((1, 5, 5), dtype=torch.bool)
            bb_mask[0, :4, :4] = True
            overlap = torch.where(is_sep2[:, None, None] & bb_mask, F.relu(2.70 - d_5x5), overlap)
        clash_pair = (overlap ** 2).sum(dim=(1, 2))
        intra_clash_total = 0.5 * float(clash_pair.sum().item())
        intra_severe_clash_frac = float((overlap.amax(dim=(1, 2)) > 0.35).to(torch.float32).mean().item())
    else:
        intra_clash_total = 0.0
        intra_severe_clash_frac = 0.0

    # 2. Intra-monomer 5-bead van der Waals packing well (|i - j| >= 4)
    src4, dst4 = torch.nonzero(non_adj4 & (dist_ca < 10.5), as_tuple=True)
    if src4.numel() > 0:
        d_5x5_4 = torch.cdist(beads[src4], beads[dst4])
        well = torch.exp(-0.5 * ((d_5x5_4 - 3.95) / 0.60) ** 2).sum(dim=(1, 2))
        intra_vdw_total = 0.5 * float(well.sum().item())
        d_sc4 = dist_sc[src4, dst4]
        sc_well = torch.exp(-0.5 * ((d_sc4 - 4.10) / 0.75) ** 2)
        h_prod = F.relu(hydro[src4]) * F.relu(hydro[dst4])
        intra_hydro_pack_total = 0.5 * float((sc_well * h_prod).sum().item())
    else:
        intra_vdw_total = 0.0
        intra_hydro_pack_total = 0.0

    # 3. Backbone N-O H-bond satisfaction & buried unsatisfied backbone polar penalty
    d_no_valid = torch.where(non_adj3, dist_no, torch.full_like(dist_no, 15.0))
    min_d_n_to_o = d_no_valid.amin(dim=1)  # nearest O for each N_i
    min_d_o_to_n = d_no_valid.amin(dim=0)  # nearest N for each O_j
    hb_sat_n = torch.exp(-0.5 * ((min_d_n_to_o - 2.92) / 0.38) ** 2)
    hb_sat_o = torch.exp(-0.5 * ((min_d_o_to_n - 2.92) / 0.38) ** 2)

    # Also allow sidechain H-bond acceptors/donors to satisfy backbone N/O
    d_n_sc = torch.where(non_adj2 & (h_acc[None, :] > 0.2), torch.cdist(n_pos, sc_pos), torch.full_like(dist_no, 15.0)).amin(dim=1)
    d_o_sc = torch.where(non_adj2 & (h_don[None, :] > 0.2), torch.cdist(o_pos, sc_pos), torch.full_like(dist_no, 15.0)).amin(dim=1)
    sat_n = torch.maximum(hb_sat_n, torch.exp(-0.5 * ((d_n_sc - 3.20) / 0.50) ** 2))
    sat_o = torch.maximum(hb_sat_o, torch.exp(-0.5 * ((d_o_sc - 3.20) / 0.50) ** 2))

    eye_m = torch.eye(N, dtype=torch.bool)
    burial_10 = (dist_ca.masked_fill(eye_m, 1e6) < 10.0).to(beads.dtype).sum(dim=-1)
    core_weight = torch.sigmoid((burial_10 - 14.0) / 3.0)

    unsat_bb_core = core_weight * ((1.0 - sat_n) + (1.0 - sat_o))

    # 4. Buried unsatisfied polar/charged sidechains & intra-monomer electrostatics
    d_sc_valid = dist_sc.masked_fill(~non_adj2, 15.0)
    polar_or_charged = (polarity > 0.35) | (charge.abs() > 0.35)
    has_polar_partner = (
        (d_sc_valid < 4.30) & polar_or_charged[None, :]
    ).any(dim=1) | (min_d_n_to_o < 3.40) | (min_d_o_to_n < 3.40)
    unsat_sc_core = core_weight * polar_or_charged.to(beads.dtype) * (~has_polar_partner).to(beads.dtype)

    if src2.numel() > 0:
        d_sc2 = dist_sc[src2, dst2].clamp_min(2.8)
        q_ij = charge[src2] * charge[dst2]
        w_cut = 0.5 * (1.0 + torch.cos(math.pi * (d_sc2 / 8.5).clamp(0.0, 1.0))) * (d_sc2 < 8.5).to(beads.dtype)
        intra_elec_total = 0.5 * float((q_ij / (d_sc2 * (1.0 + 0.25 * d_sc2)) * w_cut).sum().item())
    else:
        intra_elec_total = 0.0

    # 5. Hydrophobic core-vs-surface partitioning & SAP proxy
    surf_weight = 1.0 - core_weight
    core_hydro = float((core_weight * hydro).sum().item() / (core_weight.sum().item() + 1e-4))
    surf_hydro = float((surf_weight * hydro).sum().item() / (surf_weight.sum().item() + 1e-4))
    sap_per_res = float((surf_weight * F.relu(hydro)).mean().item())

    # 6. Globular compactness (Radius of gyration normalized by Flory scaling L^0.38)
    ca_cent = ca_pos.mean(dim=0, keepdim=True)
    rg = float(torch.sqrt(((ca_pos - ca_cent) ** 2).sum(dim=-1).mean() + 1e-6).item())
    rg_norm = rg / (2.2 * (L ** 0.38))

    # Composite physics-based monomer stability per residue (higher = more stable monomer)
    monomer_phys_stability = (
        +0.35 * (intra_vdw_total / L)
        +0.40 * (intra_hydro_pack_total / L)
        +0.50 * float((sat_n + sat_o).mean().item())
        +0.45 * (core_hydro - surf_hydro)
        -0.80 * (intra_clash_total / L)
        -0.60 * float(unsat_bb_core.mean().item())
        -0.60 * float(unsat_sc_core.mean().item())
        -0.30 * (intra_elec_total / L)
    )

    return {
        "mono_vdw_per_res": intra_vdw_total / L,
        "mono_hydro_pack_per_res": intra_hydro_pack_total / L,
        "mono_clash_per_res": intra_clash_total / L,
        "mono_severe_clash_frac": intra_severe_clash_frac,
        "mono_bb_hbond_sat": float(0.5 * (sat_n + sat_o).mean().item()),
        "mono_unsat_bb_core": float(unsat_bb_core.mean().item()),
        "mono_unsat_sc_core": float(unsat_sc_core.mean().item()),
        "mono_elec_per_res": intra_elec_total / L,
        "mono_hydro_core_minus_surf": core_hydro - surf_hydro,
        "mono_sap_per_res": sap_per_res,
        "mono_rg_norm": rg_norm,
        "mono_phys_stability_composite": monomer_phys_stability,
    }


@torch.no_grad()
def compute_interface_desolvation_and_density(
    b_beads: torch.Tensor,
    b_aa: torch.Tensor,
    t_beads: torch.Tensor,
    t_aa: torch.Tensor,
) -> dict[str, float]:
    """Compute buried unsatisfied polar/charged interface desolvation and contact density (Fix C)."""
    b_phys = PHYSCHEM_TABLE[b_aa.clamp(0, 20)]
    t_phys = PHYSCHEM_TABLE[t_aa.clamp(0, 20)]

    d_ca = torch.cdist(b_beads[:, 1], t_beads[:, 1])
    d_sc = torch.cdist(b_beads[:, 4], t_beads[:, 4])
    d_no_bt = torch.cdist(b_beads[:, 0], t_beads[:, 3])  # N_binder - O_target
    d_on_bt = torch.cdist(b_beads[:, 3], t_beads[:, 0])  # O_binder - N_target

    # Cross-interface burial per residue (number of partner CA within 8.5 A and SC within 5.5 A)
    b_buried_by_t = (d_ca < 8.5).to(b_beads.dtype).sum(dim=1) + 1.5 * (d_sc < 5.5).to(b_beads.dtype).sum(dim=1)
    t_buried_by_b = (d_ca < 8.5).to(b_beads.dtype).sum(dim=0) + 1.5 * (d_sc < 5.5).to(b_beads.dtype).sum(dim=0)

    b_intf = b_buried_by_t >= 2.0
    t_intf = t_buried_by_b >= 2.0
    n_intf_res = float(int(b_intf.sum().item()) + int(t_intf.sum().item())) + 1.0

    # Cross-interface polar/charged satisfaction
    b_pol = (b_phys[:, 2] > 0.35) | (b_phys[:, 0].abs() > 0.35)
    t_pol = (t_phys[:, 2] > 0.35) | (t_phys[:, 0].abs() > 0.35)

    # Sidechain polar/salt-bridge contact (< 4.3 A to opposite partner polar/charged SC or < 3.6 A to partner N/O)
    d_bsc_tn = torch.cdist(b_beads[:, 4], t_beads[:, 0])
    d_bsc_to = torch.cdist(b_beads[:, 4], t_beads[:, 3])
    d_tsc_bn = torch.cdist(t_beads[:, 4], b_beads[:, 0])
    d_tsc_bo = torch.cdist(t_beads[:, 4], b_beads[:, 3])

    b_sc_sat = (
        ((d_sc < 4.30) & t_pol[None, :]).any(dim=1)
        | (d_bsc_tn < 3.60).any(dim=1)
        | (d_bsc_to < 3.60).any(dim=1)
    )
    t_sc_sat = (
        ((d_sc < 4.30) & b_pol[:, None]).any(dim=0)
        | (d_tsc_bn < 3.60).any(dim=1)
        | (d_tsc_bo < 3.60).any(dim=1)
    )

    # Inter-partner backbone H-bonds
    b_bb_hb = (d_no_bt < 3.45).any(dim=1) | (d_on_bt < 3.45).any(dim=1) | (d_tsc_bn.T < 3.55).any(dim=1) | (d_tsc_bo.T < 3.55).any(dim=1)
    t_bb_hb = (d_no_bt < 3.45).any(dim=0) | (d_on_bt < 3.45).any(dim=0) | (d_bsc_tn.T < 3.55).any(dim=1) | (d_bsc_to.T < 3.55).any(dim=1)

    b_unsat_sc = b_intf & b_pol & (~b_sc_sat)
    t_unsat_sc = t_intf & t_pol & (~t_sc_sat)
    b_sat_sc = b_intf & b_pol & b_sc_sat
    t_sat_sc = t_intf & t_pol & t_sc_sat

    unsat_sc_weight = (
        (b_buried_by_t * b_unsat_sc.to(b_beads.dtype)).sum().item()
        + (t_buried_by_b * t_unsat_sc.to(b_beads.dtype)).sum().item()
    )
    sat_sc_weight = (
        (b_buried_by_t * b_sat_sc.to(b_beads.dtype)).sum().item()
        + (t_buried_by_b * t_sat_sc.to(b_beads.dtype)).sum().item()
    )
    sat_bb_count = float(int((b_intf & b_bb_hb).sum().item()) + int((t_intf & t_bb_hb).sum().item()))

    # Buried apolar contact reward at interface
    b_apol = F.relu(b_phys[:, 1])
    t_apol = F.relu(t_phys[:, 1])
    apolar_burial = float(
        ((b_buried_by_t * b_apol)[b_intf].sum() + (t_buried_by_b * t_apol)[t_intf].sum()).item()
    )

    return {
        "intf_unsat_polar_sc_count": float(int(b_unsat_sc.sum().item()) + int(t_unsat_sc.sum().item())),
        "intf_unsat_polar_sc_frac": float(int(b_unsat_sc.sum().item()) + int(t_unsat_sc.sum().item())) / n_intf_res,
        "intf_unsat_polar_weighted": float(unsat_sc_weight) / n_intf_res,
        "intf_sat_polar_weighted": float(sat_sc_weight) / n_intf_res,
        "intf_sat_bb_hbond_frac": sat_bb_count / n_intf_res,
        "intf_polar_satisfaction_net": (float(sat_sc_weight) + 2.0 * sat_bb_count - float(unsat_sc_weight)) / n_intf_res,
        "intf_apolar_burial_per_res": apolar_burial / n_intf_res,
    }


@torch.no_grad()
def compute_inverse_folding_features(
    if_model: BackboneInverseFoldingGNN,
    b_beads: torch.Tensor,
    b_aa: torch.Tensor,
    b_chain: torch.Tensor,
    t_beads: torch.Tensor,
    t_aa: torch.Tensor,
    t_chain: torch.Tensor,
    b_esm_nll_mean: float = 0.0,
) -> dict[str, float]:
    """Compute Fix A 3D Inverse-Folding Monomer & Interface Compatibility features."""
    Nb = len(b_beads)
    Nt = len(t_beads)
    b_res_idx = torch.arange(Nb, dtype=torch.long)
    t_res_idx = torch.arange(Nt, dtype=torch.long)

    # 1. Unbound binder monomer 3D inverse-folding log-probabilities
    log_p_b, geom_b = if_model(b_beads, b_chain, b_res_idx)
    valid_b = b_aa < 20
    b_aa_clamped = b_aa.clamp(0, 19)
    ll_b = log_p_b[torch.arange(Nb), b_aa_clamped]  # [Nb]
    ent_b = -(log_p_b.exp() * log_p_b).sum(dim=-1)  # [Nb]

    core_m = geom_b["burial_cb10"] > geom_b["burial_cb10"].median()
    surf_m = ~core_m
    ss_m = (geom_b["hb_helix"] > 0.35) | (geom_b["hb_sheet"] > 0.35)

    ll_all = float(ll_b[valid_b].mean().item())
    ll_core = float(ll_b[valid_b & core_m].mean().item()) if (valid_b & core_m).any() else ll_all
    ll_surf = float(ll_b[valid_b & surf_m].mean().item()) if (valid_b & surf_m).any() else ll_all
    ll_ss = float(ll_b[valid_b & ss_m].mean().item()) if (valid_b & ss_m).any() else ll_all
    ll_worst10 = float(torch.topk(ll_b[valid_b], k=max(1, int(0.10 * Nb)), largest=False).values.mean().item())

    # Sequence recovery rate on the binder monomer backbone
    pred_aa = log_p_b.argmax(dim=-1)
    recovery_all = float((pred_aa[valid_b] == b_aa_clamped[valid_b]).to(torch.float32).mean().item())
    recovery_core = (
        float((pred_aa[valid_b & core_m] == b_aa_clamped[valid_b & core_m]).to(torch.float32).mean().item())
        if (valid_b & core_m).any()
        else recovery_all
    )

    # 2. Bound complex vs unbound monomer 3D inverse-folding log-likelihood gain at the interface
    d_ca = torch.cdist(b_beads[:, 1], t_beads[:, 1])
    b_intf = d_ca.amin(dim=1) < 9.0
    t_near = d_ca.amin(dim=0) < 13.0  # crop target to interface neighborhood for fast complex GNN evaluation

    if b_intf.any() and t_near.any():
        t_sub_beads = t_beads[t_near]
        t_sub_chain = t_chain[t_near] + int(b_chain.max().item()) + 10
        t_sub_idx = t_res_idx[t_near]
        comp_beads = torch.cat([b_beads, t_sub_beads], dim=0)
        comp_chain = torch.cat([b_chain, t_sub_chain], dim=0)
        comp_idx = torch.cat([b_res_idx, t_sub_idx], dim=0)

        log_p_comp, _ = if_model(comp_beads, comp_chain, comp_idx)
        ll_b_comp = log_p_comp[:Nb][torch.arange(Nb), b_aa_clamped]
        delta_ll_intf = ll_b_comp[b_intf & valid_b] - ll_b[b_intf & valid_b]
        if_intf_gain_mean = float(delta_ll_intf.mean().item()) if delta_ll_intf.numel() > 0 else 0.0
        if_intf_gain_sum = float(delta_ll_intf.sum().item()) if delta_ll_intf.numel() > 0 else 0.0
        if_intf_comp_ll = float(ll_b_comp[b_intf & valid_b].mean().item()) if delta_ll_intf.numel() > 0 else ll_all
    else:
        if_intf_gain_mean = 0.0
        if_intf_gain_sum = 0.0
        if_intf_comp_ll = ll_all

    return {
        "if3d_binder_ll_all": ll_all,
        "if3d_binder_ll_core": ll_core,
        "if3d_binder_ll_surf": ll_surf,
        "if3d_binder_ll_secstruct": ll_ss,
        "if3d_binder_ll_worst10pct": ll_worst10,
        "if3d_binder_recovery_all": recovery_all,
        "if3d_binder_recovery_core": recovery_core,
        "if3d_binder_entropy": float(ent_b[valid_b].mean().item()),
        "if3d_bayesian_fold_contrast": ll_all + b_esm_nll_mean,  # log P_3D(seq|X) - log P_1D_ESM2(seq)
        "if3d_core_bayesian_contrast": ll_core + b_esm_nll_mean,
        "if3d_intf_complex_ll": if_intf_comp_ll,
        "if3d_intf_complex_minus_monomer_mean": if_intf_gain_mean,
        "if3d_intf_complex_minus_monomer_sum": if_intf_gain_sum,
    }


@torch.no_grad()
def compute_mechanical_stiffness_and_tertiary_tethering(
    b_beads: torch.Tensor,
    b_aa: torch.Tensor,
    t_beads: torch.Tensor,
    t_aa: torch.Tensor,
) -> dict[str, float]:
    """Compute local 11-pose 5-bead mechanical perturbation stiffness and tertiary core tethering."""
    Nb = len(b_beads)
    b_phys = PHYSCHEM_TABLE[b_aa.clamp(0, 20)]
    t_phys = PHYSCHEM_TABLE[t_aa.clamp(0, 20)]

    # 1. Tertiary Inter-Segment Monomer Core Tethering (|i - j| >= 9)
    ca_b = b_beads[:, 1]
    sc_b = b_beads[:, 4]
    d_ca_bb = torch.cdist(ca_b, ca_b)
    d_sc_bb = torch.cdist(sc_b, sc_b)
    idx_b = torch.arange(Nb, dtype=torch.long)
    seq_sep = (idx_b[:, None] - idx_b[None, :]).abs()
    tert_mask = seq_sep >= 9

    w_tert_sc = torch.sigmoid((6.5 - d_sc_bb) / 0.8) * tert_mask.to(b_beads.dtype)
    w_tert_ca = torch.sigmoid((9.0 - d_ca_bb) / 1.0) * tert_mask.to(b_beads.dtype)
    hydro_b = F.relu(b_phys[:, 1])
    tert_hydro_mat = w_tert_sc * (hydro_b[:, None] * hydro_b[None, :])

    tert_deg_per_res = w_tert_sc.sum(dim=1) + 0.5 * w_tert_ca.sum(dim=1)
    tert_hydro_per_res = tert_hydro_mat.sum(dim=1)

    win = min(12, max(4, Nb // 4))
    kernel = torch.ones((1, 1, win), dtype=b_beads.dtype) / float(win)
    if Nb >= win:
        win_tert = F.conv1d(tert_deg_per_res.view(1, 1, -1), kernel).view(-1)
        worst_win_tert = float(win_tert.min().item())
        mean_tert = float(tert_deg_per_res.mean().item())
    else:
        worst_win_tert = float(tert_deg_per_res.mean().item())
        mean_tert = worst_win_tert

    # 2. Local 11-Pose 5-Bead Mechanical Perturbation Stiffness at the Interface
    d_ca_bt0 = torch.cdist(b_beads[:, 1], t_beads[:, 1])
    b_near = d_ca_bt0.amin(dim=1) < 12.0
    t_near = d_ca_bt0.amin(dim=0) < 14.0
    if int(b_near.sum().item()) < 3:
        b_near = torch.ones(Nb, dtype=torch.bool)
    if int(t_near.sum().item()) < 3:
        t_near = torch.ones(len(t_beads), dtype=torch.bool)

    b_sub = b_beads[b_near]
    t_sub = t_beads[t_near]
    b_p = b_phys[b_near]
    t_p = t_phys[t_near]

    b_cent = b_sub[:, 1].mean(dim=0)
    t_cent = t_sub[:, 1].mean(dim=0)
    u_norm = F.normalize(b_cent - t_cent, dim=-1, eps=1e-6)
    e_arb = torch.tensor([1.0, 0.0, 0.0], dtype=b_beads.dtype)
    if u_norm[0].abs() > 0.8:
        e_arb = torch.tensor([0.0, 1.0, 0.0], dtype=b_beads.dtype)
    u_tan1 = F.normalize(torch.linalg.cross(u_norm, e_arb), dim=-1, eps=1e-6)
    u_tan2 = F.normalize(torch.linalg.cross(u_norm, u_tan1), dim=-1, eps=1e-6)

    shifts = torch.stack(
        [
            torch.zeros(3, dtype=b_beads.dtype),
            -0.60 * u_norm,
            -0.30 * u_norm,
            +0.30 * u_norm,
            +0.60 * u_norm,
            +1.00 * u_norm,
            +1.60 * u_norm,
            +0.50 * u_tan1,
            -0.50 * u_tan1,
            +0.50 * u_tan2,
            -0.50 * u_tan2,
        ],
        dim=0,
    )

    b_shifted = b_sub[None, :, :, :] + shifts[:, None, None, :]
    d_sc = torch.cdist(b_shifted[:, :, 4], t_sub[None, :, 4].expand(11, -1, -1))
    d_ca = torch.cdist(b_shifted[:, :, 1], t_sub[None, :, 1].expand(11, -1, -1))
    d_no = torch.cdist(b_shifted[:, :, 0], t_sub[None, :, 3].expand(11, -1, -1))
    d_on = torch.cdist(b_shifted[:, :, 3], t_sub[None, :, 0].expand(11, -1, -1))

    r_vdw = 3.80 + 0.35 * (b_p[:, 5][:, None] + t_p[:, 5][None, :])[None, :, :]
    x_sc = d_sc / r_vdw.clamp_min(3.0)
    e_clash = (
        F.relu(1.0 - x_sc).pow(2).sum(dim=(1, 2))
        + 2.0 * F.relu(3.4 - d_ca).pow(2).sum(dim=(1, 2))
    )
    e_attr = -(
        torch.exp(-0.5 * ((x_sc - 1.08) / 0.22) ** 2) * (d_sc < 8.0).to(b_beads.dtype)
    ).sum(dim=(1, 2))
    e_hb = -(
        torch.exp(-0.5 * ((d_no - 2.95) / 0.35) ** 2).sum(dim=(1, 2))
        + torch.exp(-0.5 * ((d_on - 2.95) / 0.35) ** 2).sum(dim=(1, 2))
    )
    q_ij = (b_p[:, 0][:, None] * t_p[:, 0][None, :])[None, :, :]
    e_elec = (
        q_ij / (d_sc.clamp_min(2.8) * (1.0 + 0.25 * d_sc.clamp_min(2.8))) * (d_sc < 8.5).to(b_beads.dtype)
    ).sum(dim=(1, 2))

    n_contact = float((d_ca[0] < 8.5).to(b_beads.dtype).sum().item()) + 1.0
    e_tot = (1.5 * e_clash + 1.0 * e_attr + 1.2 * e_hb + 0.8 * e_elec) / n_contact

    e0 = float(e_tot[0].item())
    e_push_03 = float(e_tot[2].item())
    e_pull_03 = float(e_tot[3].item())
    e_pull_10 = float(e_tot[5].item())
    e_pull_16 = float(e_tot[6].item())
    e_shear_mean = float(e_tot[7:11].mean().item())
    e_min_local = float(e_tot.min().item())

    return {
        "tert_deg_mean": mean_tert,
        "tert_worst_win": worst_win_tert,
        "tert_worst_over_mean": worst_win_tert / (mean_tert + 1e-4),
        "tert_hydro_mean": float(tert_hydro_per_res.mean().item()),
        "mech_e0_per_contact": e0,
        "mech_relax_drop": e0 - e_min_local,
        "mech_normal_grad": (e_pull_03 - e_push_03) / 0.60,
        "mech_normal_curv": (e_pull_03 - 2.0 * e0 + e_push_03) / (0.30 ** 2),
        "mech_pull_barrier": e_pull_16 - min(e0, e_pull_03),
        "mech_shear_stiffness": e_shear_mean - e0,
        "mech_clash_0": float((e_clash[0] / n_contact).item()),
        "mech_clash_relax_drop": float(((e_clash[0] - e_clash[1:5].min()) / n_contact).item()),
    }


@torch.no_grad()
def compute_corrected_physics_and_oring(
    b_beads: torch.Tensor,
    b_aa: torch.Tensor,
    t_beads: torch.Tensor,
    t_aa: torch.Tensor,
) -> dict[str, float]:
    """Compute true PHYSCHEM_TABLE donor/acceptor/aromatic physics and O-Ring hotspot organization."""
    # PHYSCHEM_TABLE columns: 0=CHARGE, 1=HYDROPATHY, 2=DONOR, 3=ACCEPTOR, 4=AROMATIC, 5=VOLUME
    b_phys = PHYSCHEM_TABLE[b_aa.clamp(0, 20)]
    t_phys = PHYSCHEM_TABLE[t_aa.clamp(0, 20)]

    b_q, b_hyd, b_don, b_acc, b_arom, _ = [b_phys[:, i] for i in range(6)]
    t_q, t_hyd, t_don, t_acc, t_arom, _ = [t_phys[:, i] for i in range(6)]

    Nb = len(b_beads)
    idx_b = torch.arange(Nb)
    non_adj2 = (idx_b[:, None] - idx_b[None, :]).abs() >= 2
    dist_ca_bb = torch.cdist(b_beads[:, 1], b_beads[:, 1])
    dist_sc_bb = torch.cdist(b_beads[:, 4], b_beads[:, 4])
    dist_no_bb = torch.cdist(b_beads[:, 0], b_beads[:, 3])

    min_d_n_o = dist_no_bb.masked_fill(~non_adj2, 15.0).amin(dim=1)
    min_d_o_n = dist_no_bb.T.masked_fill(~non_adj2, 15.0).amin(dim=1)
    hb_n = torch.exp(-0.5 * ((min_d_n_o - 2.95) / 0.45) ** 2)
    hb_o = torch.exp(-0.5 * ((min_d_o_n - 2.95) / 0.45) ** 2)

    d_n_sc = torch.where(
        non_adj2 & (b_acc[None, :] > 0.2),
        torch.cdist(b_beads[:, 0], b_beads[:, 4]),
        torch.full_like(dist_no_bb, 15.0),
    ).amin(dim=1)
    d_o_sc = torch.where(
        non_adj2 & (b_don[None, :] > 0.2),
        torch.cdist(b_beads[:, 3], b_beads[:, 4]),
        torch.full_like(dist_no_bb, 15.0),
    ).amin(dim=1)
    sat_n = torch.maximum(hb_n, torch.exp(-0.5 * ((d_n_sc - 3.20) / 0.50) ** 2))
    sat_o = torch.maximum(hb_o, torch.exp(-0.5 * ((d_o_sc - 3.20) / 0.50) ** 2))

    burial_10 = (dist_ca_bb.masked_fill(torch.eye(Nb, dtype=torch.bool), 1e6) < 10.0).to(b_beads.dtype).sum(dim=-1)
    core_w = torch.sigmoid((burial_10 - 14.0) / 3.0)

    b_pol_true = (b_don > 0.2) | (b_acc > 0.2) | (b_q.abs() > 0.2)
    sc_hb_partner = (
        (dist_sc_bb.masked_fill(~non_adj2, 15.0) < 4.20)
        & ((b_don[:, None] * b_acc[None, :] > 0.1) | (b_acc[:, None] * b_don[None, :] > 0.1))
    ).any(dim=1)
    unsat_sc_core_true = core_w * b_pol_true.to(b_beads.dtype) * (~(sc_hb_partner | (sat_n > 0.5) | (sat_o > 0.5))).to(b_beads.dtype)

    d_ca = torch.cdist(b_beads[:, 1], t_beads[:, 1])
    d_sc = torch.cdist(b_beads[:, 4], t_beads[:, 4])
    b_bur = (d_ca < 8.5).to(b_beads.dtype).sum(dim=1) + 1.5 * (d_sc < 5.5).to(b_beads.dtype).sum(dim=1)
    t_bur = (d_ca < 8.5).to(b_beads.dtype).sum(dim=0) + 1.5 * (d_sc < 5.5).to(b_beads.dtype).sum(dim=0)

    b_hotspot = b_bur >= 5.0
    b_rim = (b_bur >= 1.5) & (b_bur < 5.0)
    t_hotspot = t_bur >= 5.0
    t_rim = (t_bur >= 1.5) & (t_bur < 5.0)

    b_apolar_arom = F.relu(b_hyd) + 1.2 * b_arom
    t_apolar_arom = F.relu(t_hyd) + 1.2 * t_arom
    b_polar_rim = b_don + b_acc + b_q.abs()
    t_polar_rim = t_don + t_acc + t_q.abs()

    b_hot_score = float(b_apolar_arom[b_hotspot].mean().item()) if b_hotspot.any() else 0.0
    b_rim_score = float(b_polar_rim[b_rim].mean().item()) if b_rim.any() else 0.0
    t_hot_score = float(t_apolar_arom[t_hotspot].mean().item()) if t_hotspot.any() else 0.0
    t_rim_score = float(t_polar_rim[t_rim].mean().item()) if t_rim.any() else 0.0

    b_hotspot_arom = float(b_arom[b_hotspot].mean().item()) if b_hotspot.any() else 0.0
    b_nonintf_arom = float(b_arom[~(b_hotspot | b_rim)].mean().item()) if (~(b_hotspot | b_rim)).any() else 0.0

    w_sc_contact = torch.exp(-0.5 * ((d_sc - 3.60) / 0.65) ** 2) * (d_sc < 5.2).to(b_beads.dtype)
    da_pair = (b_don[:, None] * t_acc[None, :]) + (b_acc[:, None] * t_don[None, :])
    sb_pair = F.relu(-(b_q[:, None] * t_q[None, :]))
    arom_pair = (b_arom[:, None] * t_arom[None, :]) + 0.5 * (
        b_arom[:, None] * F.relu(t_hyd)[None, :] + F.relu(b_hyd)[:, None] * t_arom[None, :]
    )
    w_arom_contact = torch.exp(-0.5 * ((d_sc - 4.50) / 0.85) ** 2) * (d_sc < 6.5).to(b_beads.dtype)

    n_intf = float(int((b_hotspot | b_rim).sum().item()) + int((t_hotspot | t_rim).sum().item())) + 1.0
    true_sc_hbond = float((w_sc_contact * da_pair).sum().item()) / n_intf
    true_saltbridge = float((w_sc_contact * sb_pair).sum().item()) / n_intf
    true_arom_pack = float((w_arom_contact * arom_pair).sum().item()) / n_intf
    t_hot_sealed_frac = float((d_sc[:, t_hotspot].amin(dim=0) < 4.5).to(b_beads.dtype).mean().item()) if t_hotspot.any() else 0.0

    return {
        "mono_bb_hbond_sat_true": float(0.5 * (sat_n + sat_o).mean().item()),
        "mono_unsat_bb_core_true": float((core_w * ((1.0 - sat_n) + (1.0 - sat_o))).mean().item()),
        "mono_unsat_sc_core_true": float(unsat_sc_core_true.mean().item()),
        "oring_b_hotspot_apolar": b_hot_score,
        "oring_b_rim_polar": b_rim_score,
        "oring_b_contrast": b_hot_score + 0.5 * b_rim_score,
        "oring_hotspot_arom": b_hotspot_arom,
        "oring_arom_contrast": b_hotspot_arom - b_nonintf_arom,
        "oring_t_seal_frac": t_hot_sealed_frac,
        "intf_true_sc_hbond": true_sc_hbond,
        "intf_true_saltbridge": true_saltbridge,
        "intf_true_arom_pack": true_arom_pack,
    }

