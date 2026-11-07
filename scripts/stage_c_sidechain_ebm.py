"""General-Protein Sidechain Erasure, Unbound Sidechain Predictor, Subdomain EBM & Re-Docking Self-Consistency.

Addresses the two core gaps between static input_pdb scoring and full co-folders (Boltz-1 / AF3):
  1. Part 2 (General Protein Representation & Physics — Beyond Antibodies):
       - Trains on 450 general solved crystal proteins from `data/discotope_trainval/solved_pdbs`
         plus SAbDab2 crystal monomers (zero Overath data).
       - Trains:
           (a) `GeneralBackboneInverseFoldingGNN`: 3D backbone -> AA log-probabilities across all fold classes
               (alpha-helical bundles, beta-sheets, alpha/beta domains).
           (b) `UnboundSidechainPredictorGNN`: predicts natural unbound sidechain centroid positions
               `x_SC_hat` and conformational uncertainty `sigma_SC` strictly from pure backbone (N, CA, C, O, vCB)
               and amino acid sequence — completely erasing Rosetta/AF2's forced input_pdb sidechain packing.
           (c) `GeneralPPISubdomainEBM`: general secondary-structure & surface-geometry interface EBM trained
               on cleaved native crystal subdomains (helical hairpins, 3-helix bundles, beta-motifs packing
               against protein cores) + SAbDab2 crystal interfaces using sidechain-erased unbound beads.
  2. Part 1 (Sidechain-Erased Re-Docking Pose Self-Consistency — Defeating the Rosetta Mirage):
       - Erases input_pdb sidechains (`beads[:, 4]`) and substitutes predicted unbound sidechains `x_SC_hat`.
       - Measures Generator Rotamer Strain (`sc_intf_strain`, `mirage_energy_jump`, `mirage_clash_jump`).
       - Evaluates a 64-pose SE(3) re-docking + local relaxation ensemble per complex under sidechain-erased
         general PPI energy to compute `redock_boltzmann_rmsd`, `redock_local_drift_rmsd`,
         `redock_p_native_basin`, and `redock_global_margin`.
"""
from __future__ import annotations

import glob
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stage_b_foldability import (
    BackboneInverseFoldingGNN,
    compute_pure_backbone_geometry,
    construct_ideal_virtual_cb,
)
from stage_a_orientation import (
    MonomerFrameGNNLayer,
    PHYSCHEM_TABLE,
    build_local_frames,
    extract_monomer_geometry,
    rbf_encode,
)
AA3 = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, "GLN": 5, "GLU": 6,
    "GLY": 7, "HIS": 8, "ILE": 9, "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13,
    "PRO": 14, "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}

PROJECT = Path(__file__).resolve().parents[1]
SABDAB2_CACHE = PROJECT / "results" / "sabdab2_cluster_reps_cache.pt"
OVERATH_5BEAD_CACHE = PROJECT / "results" / "overath_5bead_struct_cache.pt"
OVERATH_FEAT_CACHE = PROJECT / "results" / "overath_ebm_features_cache.pt"
GEN_PROTEIN_CKPT = PROJECT / "results" / "general_protein_sc_and_ebm.pt"


_PRIMARY_TIP_ATOMS = {
    "ALA": ("CB",),
    "ARG": ("NH1", "NH2"),
    "ASN": ("OD1", "ND2"),
    "ASP": ("OD1", "OD2"),
    "CYS": ("SG",),
    "GLN": ("OE1", "NE2"),
    "GLU": ("OE1", "OE2"),
    "GLY": ("CA",),
    "HIS": ("ND1", "NE2"),
    "ILE": ("CD1",),
    "LEU": ("CD1", "CD2"),
    "LYS": ("NZ",),
    "MET": ("SD", "CE"),
    "PHE": ("CZ",),
    "PRO": ("CG", "CD"),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TRP": ("NE1", "CZ2"),
    "TYR": ("OH",),
    "VAL": ("CG1", "CG2"),
}


def extract_distal_functional_tips(residues: list[dict], beads: torch.Tensor) -> torch.Tensor:
    """Extract distal functional group tip coordinates [N, 3] from heavy-atom residue dicts."""
    tips = beads[:, 4].clone()
    for i, r in enumerate(residues):
        rname = r.get("res_name", "UNK")
        t_atoms = _PRIMARY_TIP_ATOMS.get(rname, ())
        atoms_dict = r.get("atoms", {})
        pts = [atoms_dict[a] for a in t_atoms if a in atoms_dict]
        if pts:
            tips[i] = torch.tensor(np.mean(pts, axis=0), dtype=beads.dtype)
    return tips


class UnboundSidechainPredictorGNN(nn.Module):
    """Predicts natural unbound sidechain centroid AND distal functional tip (in local frame) + uncertainties and K=3 rotamer mixture modes from backbone + sequence."""

    def __init__(self, hidden: int = 96, n_layers: int = 3, n_modes: int = 3):
        super().__init__()
        self.n_modes = n_modes
        # 30 pure backbone geom dims + 21 one-hot AA + 6 physchem = 57 dims
        self.node_in = nn.Sequential(
            nn.Linear(30 + 21 + 6, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList([MonomerFrameGNNLayer(hidden, edge_dim=41) for _ in range(n_layers)])
        # Outputs:
        #   0..2: 3D local vector u_sc_local (primary centroid) in backbone frame R_i
        #   3   : 1D log_sigma_cen
        #   4..6: 3D local vector u_tip_local (primary distal functional tip) in backbone frame R_i
        #   7   : 1D log_sigma_tip
        self.head = nn.Sequential(
            nn.Linear(hidden + 30 + 21 + 6, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 8),
        )
        # K=3 Mixture-of-Rotamers head: per mode k in {0,1,2} -> (3D delta_u_sc, 3D delta_u_tip, 1D logit_pi, 1D log_sigma_k) = 8 * K = 24 dims
        self.mdn_head = nn.Sequential(
            nn.Linear(hidden + 30 + 21 + 6, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 8 * n_modes),
        )

    def forward_from_geom(
        self,
        geom: dict[str, torch.Tensor],
        aa: torch.Tensor,
        return_tip: bool = False,
        return_mdn: bool = False,
    ) -> tuple:
        dev = next(self.parameters()).device
        aa_c = aa.to(dev).clamp(0, 20)
        nf = geom["node_feats"].to(dev)
        aa_oh = F.one_hot(aa_c, num_classes=21).to(nf.dtype)
        phys = PHYSCHEM_TABLE.to(dev)[aa_c].to(nf.dtype)
        x_in = torch.cat([nf, aa_oh, phys], dim=-1)
        h = self.node_in(x_in)
        src, dst, edge_attr = geom["src"].to(dev), geom["dst"].to(dev), geom["edge_attr"].to(dev)
        if src.numel() > 0:
            for layer in self.layers:
                h = layer(h, src, dst, edge_attr)
        h_cat = torch.cat([h, x_in], dim=-1)
        out = self.head(h_cat)
        u_local = out[:, :3]  # [N, 3] in local frame R_i
        sigma = F.softplus(out[:, 3]) + 0.15  # [N], minimum 0.15 A uncertainty
        u_tip_local = out[:, 4:7]  # [N, 3] in local frame R_i
        sigma_tip = F.softplus(out[:, 7]) + 0.15  # [N]

        # Transform local vectors into global coordinates: x = x_CA + R_i @ u_local
        # For GLY (aa == 5), both centroid and tip are identically CA
        is_gly = (aa_c == 5).unsqueeze(-1).to(u_local.dtype)
        u_local = u_local * (1.0 - is_gly)
        u_tip_local = u_tip_local * (1.0 - is_gly)
        R = geom["frames"].to(dev)  # [N, 3, 3]
        ca = geom["ca_pos"].to(dev)  # [N, 3]
        x_sc_hat = ca + torch.einsum("nij,nj->ni", R, u_local)
        if not return_tip and not return_mdn:
            return x_sc_hat, u_local, sigma
        x_tip_hat = ca + torch.einsum("nij,nj->ni", R, u_tip_local)
        if not return_mdn:
            return x_sc_hat, u_local, sigma, x_tip_hat, u_tip_local, sigma_tip

        mdn_raw = self.mdn_head(h_cat).view(-1, self.n_modes, 8)
        u_sc_modes = (u_local.unsqueeze(1) + 0.65 * mdn_raw[:, :, :3]) * (1.0 - is_gly.unsqueeze(1))  # [N, K, 3]
        u_tip_modes = (u_tip_local.unsqueeze(1) + 0.85 * mdn_raw[:, :, 3:6]) * (1.0 - is_gly.unsqueeze(1))  # [N, K, 3]
        pi_logits = mdn_raw[:, :, 6]  # [N, K]
        pi_modes = F.softmax(pi_logits, dim=-1)  # [N, K]
        sigma_modes = F.softplus(mdn_raw[:, :, 7]) + 0.15  # [N, K]

        x_sc_modes = ca.unsqueeze(1) + torch.einsum("nij,nkj->nki", R, u_sc_modes)  # [N, K, 3]
        x_tip_modes = ca.unsqueeze(1) + torch.einsum("nij,nkj->nki", R, u_tip_modes)  # [N, K, 3]
        mdn_dict = {
            "x_sc_modes": x_sc_modes,
            "x_tip_modes": x_tip_modes,
            "u_sc_modes": u_sc_modes,
            "u_tip_modes": u_tip_modes,
            "pi_modes": pi_modes,
            "sigma_modes": sigma_modes,
        }
        return x_sc_hat, u_local, sigma, x_tip_hat, u_tip_local, sigma_tip, mdn_dict


class GeneralPPISubdomainEBM(nn.Module):
    """General Protein-Protein Interface EBM using 2-point sidechain-erased geometry (centroid + distal functional tip) & bilateral Buns."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        # Node input: 30 pure backbone geom + 21 AA + 6 physchem = 57
        self.node_enc = nn.Sequential(
            nn.Linear(57, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.gnn_layers = nn.ModuleList([MonomerFrameGNNLayer(hidden, edge_dim=41) for _ in range(2)])

        # Cross-interface pair MLP:
        # h_b (64) + h_t (64) + 8 RBF(d_CA) + 8 RBF(d_SC_hat) + 6 RBF(d_tip_hat) + 6 RBF(d_NO) + 6 RBF(d_ON)
        # + 6 physchem pair products + 5 orientation cosines/cutoffs = 64 + 64 + 45 = 173
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 45, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 32),
            nn.SiLU(),
            nn.Linear(32, 2),  # [pair_energy, pair_gate_logit]
        )
        nn.init.normal_(self.pair_mlp[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.pair_mlp[-1].bias)

    def encode_partner(self, geom: dict[str, torch.Tensor], aa: torch.Tensor) -> torch.Tensor:
        dev = next(self.parameters()).device
        aa_c = aa.to(dev).clamp(0, 20)
        nf = geom["node_feats"].to(dev)
        aa_oh = F.one_hot(aa_c, num_classes=21).to(nf.dtype)
        phys = PHYSCHEM_TABLE.to(dev)[aa_c].to(nf.dtype)
        x_in = torch.cat([nf, aa_oh, phys], dim=-1)
        h = self.node_enc(x_in)
        src, dst, edge_attr = geom["src"].to(dev), geom["dst"].to(dev), geom["edge_attr"].to(dev)
        if src.numel() > 0:
            for layer in self.gnn_layers:
                h = layer(h, src, dst, edge_attr)
        return h

    def score_poses(
        self,
        b_beads_erased: torch.Tensor,  # [Nb, 5, 3] with beads[:, 4] = x_sc_hat
        b_aa: torch.Tensor,
        h_b: torch.Tensor,
        t_beads_erased: torch.Tensor,  # [Nt, 5, 3] with beads[:, 4] = x_sc_hat
        t_aa: torch.Tensor,
        h_t: torch.Tensor,
        rots: torch.Tensor,  # [B, 3, 3]
        trans: torch.Tensor,  # [B, 3]
        return_components: bool = False,
        b_tip_erased: torch.Tensor | None = None,  # [Nb, 3] distal functional tips
        t_tip_erased: torch.Tensor | None = None,  # [Nt, 3] distal functional tips
        b_sigma: torch.Tensor | None = None,  # [Nb] unbound sidechain uncertainty
        t_sigma: torch.Tensor | None = None,  # [Nt] unbound sidechain uncertainty
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Vectorized scoring of B rigid-body poses of binder relative to target."""
        dev = next(self.parameters()).device
        b_beads_erased = b_beads_erased.to(dev)
        t_beads_erased = t_beads_erased.to(dev)
        b_aa = b_aa.to(dev)
        t_aa = t_aa.to(dev)
        h_b = h_b.to(dev)
        h_t = h_t.to(dev)
        rots = rots.to(dev)
        trans = trans.to(dev)
        B = rots.shape[0]
        dtype = b_beads_erased.dtype
        phys_tbl = PHYSCHEM_TABLE.to(dev)
        b_phys = phys_tbl[b_aa.clamp(0, 20)].to(dtype)
        t_phys = phys_tbl[t_aa.clamp(0, 20)].to(dtype)

        if b_tip_erased is None:
            b_tip_erased = b_beads_erased[:, 4]
        else:
            b_tip_erased = b_tip_erased.to(dev)
        if t_tip_erased is None:
            t_tip_erased = t_beads_erased[:, 4]
        else:
            t_tip_erased = t_tip_erased.to(dev)
        if b_sigma is not None:
            b_sigma = b_sigma.to(dev)
        if t_sigma is not None:
            t_sigma = t_sigma.to(dev)

        # Transform binder beads [B, Nb, 5, 3] and distal functional tips [B, Nb, 3]
        b_cent = b_beads_erased[:, 1].mean(dim=0, keepdim=True)  # [1, 3]
        b_centered = b_beads_erased - b_cent[:, None, :]  # [Nb, 5, 3]
        b_posed = torch.einsum("bij,nkj->bnki", rots, b_centered) + b_cent[None, :, None, :] + trans[:, None, None, :]

        b_tip_cent = b_tip_erased - b_cent  # [Nb, 3]
        b_tip_posed = torch.einsum("bij,nj->bni", rots, b_tip_cent) + b_cent[None, :, :] + trans[:, None, :]

        d_ca = torch.cdist(b_posed[:, :, 1], t_beads_erased[None, :, 1].expand(B, -1, -1))  # [B, Nb, Nt]

        # Crop inactive target residues beyond 22.0 A (float32 underflow of all EBM kernels)
        if t_beads_erased.shape[0] > 48:
            t_keep = d_ca.amin(dim=(0, 1)) < 22.0
            if not bool(t_keep.all()):
                if not bool(t_keep.any()):
                    t_keep = torch.zeros_like(t_keep)
                    t_keep[0] = True
                d_ca = d_ca[:, :, t_keep]
                t_beads_erased = t_beads_erased[t_keep]
                t_tip_erased = t_tip_erased[t_keep]
                t_phys = t_phys[t_keep]
                h_t = h_t[t_keep]
                if t_sigma is not None:
                    t_sigma = t_sigma[t_keep]

        # When B > 4 on large receptors, evaluate active (pose, binder, target) pairs sparsely
        Nb = b_beads_erased.shape[0]
        Nt = t_beads_erased.shape[0]
        if B > 4 and Nt > 96:
            p_b, p_i, p_j = torch.where(d_ca < 22.0)
            if p_b.numel() == 0:
                z = torch.zeros(B, dtype=dtype, device=dev)
                if return_components:
                    return {
                        k: z
                        for k in (
                            "total", "clash", "vdw", "hb", "tip_hb", "elec",
                            "desolv", "buns_binder", "buns_target", "rot_ent", "neural",
                        )
                    }
                return z

            d_ca_p = d_ca[p_b, p_i, p_j]
            bp = b_posed[p_b, p_i]
            tp = t_beads_erased[p_j]
            btip = b_tip_posed[p_b, p_i]
            ttip = t_tip_erased[p_j]

            d_sc_p = torch.linalg.norm(bp[:, 4] - tp[:, 4], dim=-1)
            d_tip_p = torch.linalg.norm(btip - ttip, dim=-1)
            d_no_p = torch.linalg.norm(bp[:, 0] - tp[:, 3], dim=-1)
            d_on_p = torch.linalg.norm(bp[:, 3] - tp[:, 0], dim=-1)
            d_btip_tO_p = torch.linalg.norm(btip - tp[:, 3], dim=-1)
            d_bO_ttip_p = torch.linalg.norm(bp[:, 3] - ttip, dim=-1)

            bp_i = b_phys[p_i]
            tp_j = t_phys[p_j]
            r_vdw_p = (3.75 + 0.35 * (bp_i[:, 5] + tp_j[:, 5])).clamp_min(3.0)
            x_sc_p = d_sc_p / r_vdw_p

            if b_sigma is not None and t_sigma is not None:
                sig_pair_p = torch.sqrt(b_sigma.to(dtype)[p_i].pow(2) + t_sigma.to(dtype)[p_j].pow(2) + 0.04)
                clash_tol_p = 0.93 - 0.08 * (sig_pair_p - 0.35).clamp(0.0, 1.0)
                vdw_width_p = 0.21 + 0.10 * (sig_pair_p - 0.35).clamp(0.0, 1.0)
            else:
                clash_tol_p = 0.92
                vdw_width_p = 0.22

            clash_p = (
                1.2 * F.relu(clash_tol_p - x_sc_p).pow(2)
                + 2.5 * F.relu(3.45 - d_ca_p).pow(2)
                + 0.8 * F.relu(2.55 - d_tip_p).pow(2)
            )
            e_clash = torch.zeros(B, dtype=dtype, device=dev).index_add_(0, p_b, clash_p)

            w_vdw_p = torch.exp(-0.5 * ((x_sc_p - 1.06) / vdw_width_p) ** 2) * (d_sc_p < 8.0).to(dtype)
            b_ap_p = F.relu(bp_i[:, 1]) + bp_i[:, 4]
            t_ap_p = F.relu(tp_j[:, 1]) + tp_j[:, 4]
            e_vdw = -torch.zeros(B, dtype=dtype, device=dev).index_add_(0, p_b, w_vdw_p * (1.0 + 0.8 * (b_ap_p * t_ap_p)))

            bur_p = 0.5 * torch.sigmoid((5.4 - d_sc_p) / 0.6) + 0.5 * torch.sigmoid((4.8 - d_tip_p) / 0.5)
            b_buried = torch.zeros(B * Nb, dtype=dtype, device=dev).index_add_(0, p_b * Nb + p_i, bur_p).view(B, Nb)
            t_buried = torch.zeros(B * Nt, dtype=dtype, device=dev).index_add_(0, p_b * Nt + p_j, bur_p).view(B, Nt)
            pair_diel_p = 0.55 + 0.90 * torch.sigmoid(0.5 * (b_buried[p_b, p_i] + t_buried[p_b, p_j]) - 1.2)

            da_p = (bp_i[:, 2] * tp_j[:, 3]) + (bp_i[:, 3] * tp_j[:, 2])
            w_sc_hb_p = torch.exp(-0.5 * ((d_sc_p - 3.60) / 0.55) ** 2) * (d_sc_p < 5.0).to(dtype)
            w_tip_hb_p = torch.exp(-0.5 * ((d_tip_p - 3.05) / 0.45) ** 2) * (d_tip_p < 4.4).to(dtype)
            w_btip_tO_p = torch.exp(-0.5 * ((d_btip_tO_p - 2.95) / 0.40) ** 2) * (d_btip_tO_p < 4.1).to(dtype) * bp_i[:, 2]
            w_bO_ttip_p = torch.exp(-0.5 * ((d_bO_ttip_p - 2.95) / 0.40) ** 2) * (d_bO_ttip_p < 4.1).to(dtype) * tp_j[:, 2]

            tip_hb_p = ((w_tip_hb_p * da_p) + 0.75 * (w_btip_tO_p + w_bO_ttip_p)) * pair_diel_p
            e_tip_hb = -torch.zeros(B, dtype=dtype, device=dev).index_add_(0, p_b, tip_hb_p)

            hb_p = (
                torch.exp(-0.5 * ((d_no_p - 2.95) / 0.35) ** 2)
                + torch.exp(-0.5 * ((d_on_p - 2.95) / 0.35) ** 2)
                + 0.45 * (w_sc_hb_p * da_p)
            ) * pair_diel_p
            e_hb = -torch.zeros(B, dtype=dtype, device=dev).index_add_(0, p_b, hb_p) + 0.65 * e_tip_hb

            q_p = bp_i[:, 0] * tp_j[:, 0]
            d_ch_p = 0.5 * (d_sc_p + d_tip_p)
            d_ch_c = d_ch_p.clamp_min(2.8)
            elec_p = pair_diel_p * q_p / (d_ch_c * (1.0 + 0.25 * d_ch_c)) * (d_ch_p < 8.5).to(dtype)
            e_elec = torch.zeros(B, dtype=dtype, device=dev).index_add_(0, p_b, elec_p)

            b_pol = ((b_phys[:, 2] > 0.2) | (b_phys[:, 3] > 0.2) | (b_phys[:, 0].abs() > 0.2)).to(dtype)
            t_pol = ((t_phys[:, 2] > 0.2) | (t_phys[:, 3] > 0.2) | (t_phys[:, 0].abs() > 0.2)).to(dtype)

            sat_c = (0.5 * w_sc_hb_p + 0.8 * w_tip_hb_p) * (da_p + F.relu(-q_p)) + (d_no_p < 3.4).to(dtype) + (d_on_p < 3.4).to(dtype)
            b_sat = torch.zeros(B * Nb, dtype=dtype, device=dev).index_add_(0, p_b * Nb + p_i, sat_c + w_btip_tO_p).view(B, Nb).clamp(0.0, 1.0)
            t_sat = torch.zeros(B * Nt, dtype=dtype, device=dev).index_add_(0, p_b * Nt + p_j, sat_c + w_bO_ttip_p).view(B, Nt).clamp(0.0, 1.0)

            e_buns_binder = (b_buried * b_pol[None, :] * (1.0 - b_sat)).sum(dim=1) * 0.15
            e_buns_target = (t_buried * t_pol[None, :] * (1.0 - t_sat)).sum(dim=1) * 0.12
            e_desolv = e_buns_binder + e_buns_target

            if b_sigma is not None:
                b_ent_w = torch.log1p(F.relu(b_sigma.to(dtype) - 0.35))[None, :]
                e_rot_ent = (b_buried * b_ent_w * (1.0 - 0.75 * b_sat)).sum(dim=1) * 0.14
            else:
                e_rot_ent = (b_buried * F.relu(b_phys[:, 5] - 0.2)[None, :] * (1.0 - 0.75 * b_sat)).sum(dim=1) * 0.08
            if t_sigma is not None:
                t_ent_w = torch.log1p(F.relu(t_sigma.to(dtype) - 0.35))[None, :]
                e_rot_ent = e_rot_ent + (t_buried * t_ent_w * (1.0 - 0.75 * t_sat)).sum(dim=1) * 0.10

            m_n = d_ca_p < 10.5
            e_neural = torch.zeros(B, dtype=dtype, device=dev)
            if m_n.any():
                b_idx = p_b[m_n]
                i_idx = p_i[m_n]
                j_idx = p_j[m_n]
                dca_ij = d_ca_p[m_n]
                dsc_ij = d_sc_p[m_n]
                dtip_ij = d_tip_p[m_n]
                dno_ij = d_no_p[m_n]
                don_ij = d_on_p[m_n]
                bp_n = bp[m_n]
                tp_n = tp[m_n]
                btip_n = btip[m_n]
                ttip_n = ttip[m_n]

                u_b_sc = F.normalize(btip_n - bp_n[:, 1], dim=-1, eps=1e-6)
                u_t_sc = F.normalize(ttip_n - tp_n[:, 1], dim=-1, eps=1e-6)
                u_bt = F.normalize(tp_n[:, 1] - bp_n[:, 1], dim=-1, eps=1e-6)
                u_tip_vec = F.normalize(ttip_n - btip_n, dim=-1, eps=1e-6)

                cos1 = (u_b_sc * u_bt).sum(dim=-1, keepdim=True)
                cos2 = (u_t_sc * (-u_bt)).sum(dim=-1, keepdim=True)
                cos3 = (u_b_sc * u_t_sc).sum(dim=-1, keepdim=True)
                cos4 = (u_b_sc * u_tip_vec).sum(dim=-1, keepdim=True)
                w_cut = 0.5 * (1.0 + torch.cos(math.pi * (dca_ij / 10.5).clamp(0.0, 1.0))).unsqueeze(-1)

                phys_prod = bp_i[m_n] * tp_j[m_n]
                pair_in = torch.cat(
                    [
                        h_b[i_idx],
                        h_t[j_idx],
                        rbf_encode(dca_ij, 2.5, 10.5, 8),
                        rbf_encode(dsc_ij, 2.0, 9.5, 8),
                        rbf_encode(dtip_ij, 2.0, 8.5, 6),
                        rbf_encode(dno_ij, 2.2, 6.0, 6),
                        rbf_encode(don_ij, 2.2, 6.0, 6),
                        phys_prod,
                        cos1,
                        cos2,
                        cos3,
                        cos4,
                        w_cut,
                    ],
                    dim=-1,
                )
                out_ij = self.pair_mlp(pair_in)
                val_ij = torch.tanh(out_ij[:, 0]) * torch.sigmoid(out_ij[:, 1]) * w_cut.squeeze(-1)
                e_neural.index_add_(0, b_idx, val_ij)

            e_total = 0.45 * e_clash + 0.24 * e_vdw + 0.26 * e_hb + 0.20 * e_elec + 0.85 * e_desolv + 0.50 * e_rot_ent + 0.22 * e_neural
            if return_components:
                return {
                    "total": e_total,
                    "clash": e_clash,
                    "vdw": e_vdw,
                    "hb": e_hb,
                    "tip_hb": e_tip_hb,
                    "elec": e_elec,
                    "desolv": e_desolv,
                    "buns_binder": e_buns_binder,
                    "buns_target": e_buns_target,
                    "rot_ent": e_rot_ent,
                    "neural": e_neural,
                }
            return e_total

        d_sc = torch.cdist(b_posed[:, :, 4], t_beads_erased[None, :, 4].expand(B, -1, -1))  # [B, Nb, Nt]
        d_tip = torch.cdist(b_tip_posed, t_tip_erased[None, :, :].expand(B, -1, -1))  # [B, Nb, Nt]
        d_no = torch.cdist(b_posed[:, :, 0], t_beads_erased[None, :, 3].expand(B, -1, -1))  # [B, Nb, Nt]
        d_on = torch.cdist(b_posed[:, :, 3], t_beads_erased[None, :, 0].expand(B, -1, -1))  # [B, Nb, Nt]
        d_btip_tO = torch.cdist(b_tip_posed, t_beads_erased[None, :, 3].expand(B, -1, -1))  # [B, Nb, Nt]
        d_bO_ttip = torch.cdist(b_posed[:, :, 3], t_tip_erased[None, :, :].expand(B, -1, -1))  # [B, Nb, Nt]

        # 1. Calibrated physical energy with erased/predicted unbound centroids & distal functional tips
        # PHYSCHEM_TABLE: 0=CHARGE, 1=HYDROPATHY, 2=DONOR, 3=ACCEPTOR, 4=AROMATIC, 5=VOLUME
        r_vdw = (3.75 + 0.35 * (b_phys[:, 5][:, None] + t_phys[:, 5][None, :])).clamp_min(3.0)[None, :, :]
        x_sc = d_sc / r_vdw

        # Pairwise combined rotamer positional variance (soft Gaussian-convolved steric & VdW kernel)
        if b_sigma is not None and t_sigma is not None:
            sig_pair = torch.sqrt(b_sigma.to(dtype)[:, None].pow(2) + t_sigma.to(dtype)[None, :].pow(2) + 0.04)[None, :, :]
            clash_tol = (0.93 - 0.08 * (sig_pair - 0.35).clamp(0.0, 1.0))
            vdw_width = 0.21 + 0.10 * (sig_pair - 0.35).clamp(0.0, 1.0)
        else:
            clash_tol = 0.92
            vdw_width = 0.22

        # Soft steric clash (tolerates small rotamer uncertainty, penalizes backbone & deep SC overlap)
        e_clash = (
            1.2 * F.relu(clash_tol - x_sc).pow(2).sum(dim=(1, 2))
            + 2.5 * F.relu(3.45 - d_ca).pow(2).sum(dim=(1, 2))
            + 0.8 * F.relu(2.55 - d_tip).pow(2).sum(dim=(1, 2))
        )

        # Short-range VdW + Hydrophobic + Aromatic packing well
        w_vdw = torch.exp(-0.5 * ((x_sc - 1.06) / vdw_width) ** 2) * (d_sc < 8.0).to(dtype)
        b_apolar = F.relu(b_phys[:, 1]) + 1.0 * b_phys[:, 4]
        t_apolar = F.relu(t_phys[:, 1]) + 1.0 * t_phys[:, 4]
        apolar_prod = (1.0 + 0.8 * (b_apolar[:, None] * t_apolar[None, :]))[None, :, :]
        e_vdw = -(w_vdw * apolar_prod).sum(dim=(1, 2))

        # Bilateral Burial Depth for Dielectric Screening (Bogan & Thorn O-ring core vs wet solvent rim)
        b_buried = (0.5 * torch.sigmoid((5.4 - d_sc) / 0.6) + 0.5 * torch.sigmoid((4.8 - d_tip) / 0.5)).sum(dim=2)  # [B, Nb]
        t_buried = (0.5 * torch.sigmoid((5.4 - d_sc) / 0.6) + 0.5 * torch.sigmoid((4.8 - d_tip) / 0.5)).sum(dim=1)  # [B, Nt]
        pair_diel_weight = (0.55 + 0.90 * torch.sigmoid(0.5 * (b_buried[:, :, None] + t_buried[:, None, :]) - 1.2))  # [B, Nb, Nt]

        # Backbone + Distal Functional-Tip H-bonds (tip-tip and tip-backbone O), weighted by low-dielectric burial
        da_pair = ((b_phys[:, 2][:, None] * t_phys[:, 3][None, :]) + (b_phys[:, 3][:, None] * t_phys[:, 2][None, :]))[None, :, :]
        w_sc_hb = torch.exp(-0.5 * ((d_sc - 3.60) / 0.55) ** 2) * (d_sc < 5.0).to(dtype)
        w_tip_hb = torch.exp(-0.5 * ((d_tip - 3.05) / 0.45) ** 2) * (d_tip < 4.4).to(dtype)
        w_btip_tO = (
            torch.exp(-0.5 * ((d_btip_tO - 2.95) / 0.40) ** 2)
            * (d_btip_tO < 4.1).to(dtype)
            * b_phys[:, 2][None, :, None]
        )
        w_bO_ttip = (
            torch.exp(-0.5 * ((d_bO_ttip - 2.95) / 0.40) ** 2)
            * (d_bO_ttip < 4.1).to(dtype)
            * t_phys[:, 2][None, None, :]
        )
        e_tip_hb = -(((w_tip_hb * da_pair) + 0.75 * (w_btip_tO + w_bO_ttip)) * pair_diel_weight).sum(dim=(1, 2))
        e_hb = -(
            (torch.exp(-0.5 * ((d_no - 2.95) / 0.35) ** 2) * pair_diel_weight).sum(dim=(1, 2))
            + (torch.exp(-0.5 * ((d_on - 2.95) / 0.35) ** 2) * pair_diel_weight).sum(dim=(1, 2))
            + 0.45 * (w_sc_hb * da_pair * pair_diel_weight).sum(dim=(1, 2))
        ) + 0.65 * e_tip_hb

        q_ij = (b_phys[:, 0][:, None] * t_phys[:, 0][None, :])[None, :, :]
        d_charge = 0.5 * (d_sc + d_tip)
        e_elec = (
            pair_diel_weight * q_ij / (d_charge.clamp_min(2.8) * (1.0 + 0.25 * d_charge.clamp_min(2.8))) * (d_charge < 8.5).to(dtype)
        ).sum(dim=(1, 2))

        # Bilateral Buried Unsatisfied Polar (Buns) penalty on both Binder AND Target Epitope functional tips
        b_pol = ((b_phys[:, 2] > 0.2) | (b_phys[:, 3] > 0.2) | (b_phys[:, 0].abs() > 0.2)).to(dtype)
        t_pol = ((t_phys[:, 2] > 0.2) | (t_phys[:, 3] > 0.2) | (t_phys[:, 0].abs() > 0.2)).to(dtype)

        b_sat = (
            ((0.5 * w_sc_hb + 0.8 * w_tip_hb) * (da_pair + F.relu(-q_ij))).sum(dim=2)
            + w_btip_tO.sum(dim=2)
            + (d_no < 3.4).to(dtype).sum(dim=2)
            + (d_on < 3.4).to(dtype).sum(dim=2)
        ).clamp(0.0, 1.0)
        e_buns_binder = (b_buried * b_pol[None, :] * (1.0 - b_sat)).sum(dim=1) * 0.15

        t_sat = (
            ((0.5 * w_sc_hb + 0.8 * w_tip_hb) * (da_pair + F.relu(-q_ij))).sum(dim=1)
            + w_bO_ttip.sum(dim=1)
            + (d_no < 3.4).to(dtype).sum(dim=1)
            + (d_on < 3.4).to(dtype).sum(dim=1)
        ).clamp(0.0, 1.0)
        e_buns_target = (t_buried * t_pol[None, :] * (1.0 - t_sat)).sum(dim=1) * 0.12
        e_desolv = e_buns_binder + e_buns_target

        # Sidechain conformational entropy cost upon burial (penalizes burying high-sigma sidechains without H-bond/salt-bridge lock)
        if b_sigma is not None:
            b_ent_w = torch.log1p(F.relu(b_sigma.to(dtype) - 0.35))[None, :]
            e_rot_ent = (b_buried * b_ent_w * (1.0 - 0.75 * b_sat)).sum(dim=1) * 0.14
        else:
            e_rot_ent = (b_buried * F.relu(b_phys[:, 5] - 0.2)[None, :] * (1.0 - 0.75 * b_sat)).sum(dim=1) * 0.08
        if t_sigma is not None:
            t_ent_w = torch.log1p(F.relu(t_sigma.to(dtype) - 0.35))[None, :]
            e_rot_ent = e_rot_ent + (t_buried * t_ent_w * (1.0 - 0.75 * t_sat)).sum(dim=1) * 0.10

        # 2. Neural general-protein pair residual on active interface pairs (d_ca < 10.5 A)
        b_idx, i_idx, j_idx = torch.where(d_ca < 10.5)
        e_neural = torch.zeros(B, dtype=dtype, device=b_beads_erased.device)
        if b_idx.numel() > 0:
            dca_ij = d_ca[b_idx, i_idx, j_idx]
            dsc_ij = d_sc[b_idx, i_idx, j_idx]
            dtip_ij = d_tip[b_idx, i_idx, j_idx]
            dno_ij = d_no[b_idx, i_idx, j_idx]
            don_ij = d_on[b_idx, i_idx, j_idx]

            u_b_sc = F.normalize(b_tip_posed[b_idx, i_idx] - b_posed[b_idx, i_idx, 1], dim=-1, eps=1e-6)
            u_t_sc = F.normalize(t_tip_erased[j_idx] - t_beads_erased[j_idx, 1], dim=-1, eps=1e-6)
            u_bt = F.normalize(t_beads_erased[j_idx, 1] - b_posed[b_idx, i_idx, 1], dim=-1, eps=1e-6)
            u_tip_vec = F.normalize(t_tip_erased[j_idx] - b_tip_posed[b_idx, i_idx], dim=-1, eps=1e-6)

            cos1 = (u_b_sc * u_bt).sum(dim=-1, keepdim=True)
            cos2 = (u_t_sc * (-u_bt)).sum(dim=-1, keepdim=True)
            cos3 = (u_b_sc * u_t_sc).sum(dim=-1, keepdim=True)
            cos4 = (u_b_sc * u_tip_vec).sum(dim=-1, keepdim=True)
            w_cut = 0.5 * (1.0 + torch.cos(math.pi * (dca_ij / 10.5).clamp(0.0, 1.0))).unsqueeze(-1)

            phys_prod = b_phys[i_idx] * t_phys[j_idx]  # [E, 6]
            pair_in = torch.cat(
                [
                    h_b[i_idx],
                    h_t[j_idx],
                    rbf_encode(dca_ij, 2.5, 10.5, 8),
                    rbf_encode(dsc_ij, 2.0, 9.5, 8),
                    rbf_encode(dtip_ij, 2.0, 8.5, 6),
                    rbf_encode(dno_ij, 2.2, 6.0, 6),
                    rbf_encode(don_ij, 2.2, 6.0, 6),
                    phys_prod,
                    cos1,
                    cos2,
                    cos3,
                    cos4,
                    w_cut,
                ],
                dim=-1,
            )
            out_ij = self.pair_mlp(pair_in)
            val_ij = torch.tanh(out_ij[:, 0]) * torch.sigmoid(out_ij[:, 1]) * w_cut.squeeze(-1)
            e_neural.index_add_(0, b_idx, val_ij)

        e_total = 0.45 * e_clash + 0.24 * e_vdw + 0.26 * e_hb + 0.20 * e_elec + 0.85 * e_desolv + 0.50 * e_rot_ent + 0.22 * e_neural
        if return_components:
            return {
                "total": e_total,
                "clash": e_clash,
                "vdw": e_vdw,
                "hb": e_hb,
                "tip_hb": e_tip_hb,
                "elec": e_elec,
                "desolv": e_desolv,
                "buns_binder": e_buns_binder,
                "buns_target": e_buns_target,
                "rot_ent": e_rot_ent,
                "neural": e_neural,
            }
        return e_total


def _axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert [B, 3] axis-angle vectors to [B, 3, 3] rotation matrices (Rodrigues)."""
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp_min(1e-8)
    k = rotvec / theta
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    z = torch.zeros_like(kx)
    K = torch.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], dim=-1).reshape(-1, 3, 3)
    I = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).unsqueeze(0)
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    return I + sin_t * K + (1.0 - cos_t) * (K @ K)


def build_redocking_pose_ensemble(
    b_beads: torch.Tensor,
    t_beads: torch.Tensor,
    rng: np.random.Generator,
    n_global: int = 11,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build 24-pose ensemble: pose 0 (native x_0), poses 1..12 (local 6-DoF relaxation grid), poses 13..23 (global surface re-docking).

    Returns:
      rots: [24, 3, 3]
      trans: [24, 3]
      pose_rmsd: [24] exact C-alpha RMSD (in Angstroms) of each pose relative to x_0
    """
    b_beads = b_beads.detach().cpu()
    t_beads = t_beads.detach().cpu()
    dtype = b_beads.dtype
    b_ca = b_beads[:, 1]
    t_ca = t_beads[:, 1]
    b_cent = b_ca.mean(dim=0)
    t_cent = t_ca.mean(dim=0)

    # Interface normal & tangent basis
    d_ca0 = torch.cdist(b_ca, t_ca)
    b_intf = d_ca0.amin(dim=1) < 10.0
    t_intf = d_ca0.amin(dim=0) < 10.0
    p_b = b_ca[b_intf].mean(dim=0) if b_intf.any() else b_cent
    p_t = t_ca[t_intf].mean(dim=0) if t_intf.any() else t_cent
    u_norm = F.normalize(p_b - p_t, dim=-1, eps=1e-6)
    e_arb = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    if u_norm[0].abs() > 0.8:
        e_arb = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    u_tan1 = F.normalize(torch.linalg.cross(u_norm, e_arb), dim=-1, eps=1e-6)
    u_tan2 = F.normalize(torch.linalg.cross(u_norm, u_tan1), dim=-1, eps=1e-6)

    rots_list = [torch.eye(3, dtype=dtype)]
    trans_list = [torch.zeros(3, dtype=dtype)]

    # Poses 1..12: Local 6-DoF relaxation grid around x_0 (translations + rotations)
    for s_n in (-0.5, 0.6, 1.4, 2.4):
        rots_list.append(torch.eye(3, dtype=dtype))
        trans_list.append(s_n * u_norm)
    for s_t in (-1.0, 1.0):
        rots_list.append(torch.eye(3, dtype=dtype))
        trans_list.append(s_t * u_tan1)
        rots_list.append(torch.eye(3, dtype=dtype))
        trans_list.append(s_t * u_tan2)
    # 4 local tilt/twist rotations (-12 deg, +12 deg around u_tan1, u_tan2)
    for ang_deg in (-12.0, 12.0):
        rad = ang_deg * math.pi / 180.0
        for axis in (u_tan1, u_tan2):
            R_loc = _axis_angle_to_matrix((rad * axis).unsqueeze(0)).squeeze(0)
            rots_list.append(R_loc)
            trans_list.append(0.20 * u_norm)

    # Poses 25..63: Global target-surface ray-cast re-docking poses across the target
    # Compute outward target surface normals at surface residues
    d_tt = torch.cdist(t_ca, t_ca)
    w12 = (d_tt < 12.0).to(dtype)
    n_surf = F.normalize(t_ca - (w12 @ t_ca) / w12.sum(dim=-1, keepdim=True).clamp_min(1.0), dim=-1, eps=1e-6)
    burial_t = (d_tt < 10.0).to(dtype).sum(dim=-1)
    surf_idx = torch.where(burial_t <= torch.quantile(burial_t, 0.65))[0]
    if surf_idx.numel() == 0:
        surf_idx = torch.arange(len(t_ca))

    # Binder radius along approach direction
    b_para_vec = F.normalize(p_b - b_cent, dim=-1, eps=1e-6)
    b_rad = float(torch.norm(p_b - b_cent).item())

    chosen_surf = surf_idx[rng.choice(len(surf_idx), size=n_global, replace=True)]
    rand_rotvecs = torch.from_numpy(rng.normal(0.0, 1.2, size=(n_global, 3))).to(dtype)
    R_glob = _axis_angle_to_matrix(rand_rotvecs)

    for k in range(n_global):
        idx_t = int(chosen_surf[k].item())
        anchor_ca = t_ca[idx_t]
        out_n = n_surf[idx_t]
        R_k = R_glob[k]
        # Place binder so its rotated paratope faces anchor_ca at contact distance ~ (b_rad + 6.5 A)
        target_b_cent = anchor_ca + (b_rad + 6.5) * out_n
        t_k = target_b_cent - b_cent
        rots_list.append(R_k)
        trans_list.append(t_k)

    rots = torch.stack(rots_list, dim=0)  # [64, 3, 3]
    trans = torch.stack(trans_list, dim=0)  # [64, 3]

    # Compute exact C-alpha RMSD of each pose relative to x_0
    b_ca_cent = b_ca - b_cent[None, :]
    b_ca_posed = torch.einsum("bij,nj->bni", rots, b_ca_cent) + b_cent[None, None, :] + trans[:, None, :]
    pose_rmsd = torch.sqrt(((b_ca_posed - b_ca[None, :, :]) ** 2).sum(dim=-1).mean(dim=-1) + 1e-8)
    pose_rmsd[0] = 0.0
    return rots, trans, pose_rmsd


def load_general_crystal_corpus(max_solved_pdbs: int = 900) -> tuple[list[dict], list[dict], list[dict]]:
    """Load general solved crystal structures from data/discotope_trainval/solved_pdbs + SAbDab2 crystal chains."""
    t0 = time.time()
    slim_cache = PROJECT / "results" / "crystal_corpus_900_slim.pt"
    if slim_cache.exists() and max_solved_pdbs >= 900:
        d = torch.load(slim_cache, map_location="cpu", weights_only=False)
        train_mono = []
        for m in d["train_mono"]:
            beads = m["beads"].to(torch.float32)
            aa = m["aa"].to(torch.long)
            N_r = len(beads)
            geom = compute_pure_backbone_geometry(beads, torch.zeros(N_r, dtype=torch.long), torch.arange(N_r))
            geom["aa"] = aa
            geom["true_sc"] = m["true_sc"].to(torch.float32)
            geom["true_tip"] = m["true_tip"].to(torch.float32)
            geom["has_true_tip"] = bool(m.get("has_true_tip", False))
            geom["beads"] = beads
            train_mono.append(geom)
        val_mono = []
        for m in d["val_mono"]:
            beads = m["beads"].to(torch.float32)
            aa = m["aa"].to(torch.long)
            N_r = len(beads)
            geom = compute_pure_backbone_geometry(beads, torch.zeros(N_r, dtype=torch.long), torch.arange(N_r))
            geom["aa"] = aa
            geom["true_sc"] = m["true_sc"].to(torch.float32)
            geom["true_tip"] = m["true_tip"].to(torch.float32)
            geom["has_true_tip"] = bool(m.get("has_true_tip", False))
            geom["beads"] = beads
            val_mono.append(geom)
        sub_complexes = d["sub_complexes"]
        print(
            f"Loaded general crystal corpus from slim cache in {time.time() - t0:.1f}s: "
            f"{len(train_mono)} train monomers, {len(val_mono)} val monomers, "
            f"{len(sub_complexes)} crystal subdomain/PPI complexes.",
            flush=True,
        )
        return train_mono, val_mono, sub_complexes
    rng = np.random.default_rng(123)
    pdb_files = sorted(glob.glob(str(PROJECT / "data" / "discotope_trainval" / "solved_pdbs" / "*.pdb")))
    if len(pdb_files) > max_solved_pdbs:
        idx_sel = sorted(rng.choice(len(pdb_files), size=max_solved_pdbs, replace=False))
        pdb_files = [pdb_files[i] for i in idx_sel]

    monomers = []
    subdomain_complexes = []

    for p_path in pdb_files:
        try:
            g = extract_monomer_geometry(Path(p_path).read_text())
        except Exception:
            continue
        res = g["residues"]
        if len(res) < 55 or len(res) > 450:
            continue
        beads = torch.from_numpy(g["beads"]).to(torch.float32)
        aa = torch.tensor([AA3.get(r["res_name"], 20) for r in res], dtype=torch.long)
        if (aa < 20).sum() < 45:
            continue
        chain = torch.zeros(len(res), dtype=torch.long)
        res_idx = torch.arange(len(res), dtype=torch.long)
        geom = compute_pure_backbone_geometry(beads, chain, res_idx)
        geom["aa"] = aa
        geom["true_sc"] = beads[:, 4]
        geom["true_tip"] = extract_distal_functional_tips(res, beads)
        geom["has_true_tip"] = True
        geom["beads"] = beads
        monomers.append(geom)

        # Cleave contiguous secondary-structure subdomains (length 45..62) that pack against the rest of the crystal core
        N = len(res)
        if N >= 120 and len(subdomain_complexes) < 380:
            d_ca = torch.cdist(beads[:, 1], beads[:, 1])
            for start in (0, N - 55, N // 3):
                end = min(N, start + 52)
                if end - start < 42:
                    continue
                b_mask = torch.zeros(N, dtype=torch.bool)
                b_mask[start:end] = True
                # Leave a 6-residue buffer around cleavage sites so there is zero covalent backbone continuity
                buf_mask = torch.zeros(N, dtype=torch.bool)
                buf_mask[max(0, start - 6) : min(N, end + 6)] = True
                t_mask = ~buf_mask
                if int(t_mask.sum().item()) < 50:
                    continue
                n_contacts = int((d_ca[b_mask][:, t_mask] < 8.5).sum().item())
                if n_contacts >= 18:
                    subdomain_complexes.append(
                        {
                            "b_beads": beads[b_mask],
                            "b_aa": aa[b_mask],
                            "t_beads": beads[t_mask],
                            "t_aa": aa[t_mask],
                        }
                    )
                    break

    # Also add SAbDab2 antigen & antibody crystal monomers + complexes
    if SABDAB2_CACHE.exists():
        sab = torch.load(SABDAB2_CACHE, map_location="cpu", weights_only=False)
        for cl_items in list(sab["train_clusters"].values())[:240]:
            comp = cl_items[0]
            for pref in ("lig", "rec"):
                beads = comp[f"{pref}_beads"].to(torch.float32)
                aa = comp[f"{pref}_aa"].to(torch.long)
                chain = comp[f"{pref}_chain"].to(torch.long)
                res_idx = comp[f"{pref}_residue_index"].to(torch.long)
                if 40 <= len(beads) <= 450 and (aa < 20).sum() >= 35:
                    geom = compute_pure_backbone_geometry(beads, chain, res_idx)
                    geom["aa"] = aa
                    geom["true_sc"] = beads[:, 4]
                    geom["true_tip"] = beads[:, 4]
                    geom["has_true_tip"] = False
                    geom["beads"] = beads
                    monomers.append(geom)
            if len(subdomain_complexes) < 520:
                subdomain_complexes.append(
                    {
                        "b_beads": comp["rec_beads"].to(torch.float32),
                        "b_aa": comp["rec_aa"].to(torch.long),
                        "t_beads": comp["lig_beads"].to(torch.float32),
                        "t_aa": comp["lig_aa"].to(torch.long),
                    }
                )

    # Deterministic shuffle so val_mono has a representative mix of solved_pdbs (with true_tip) and SAbDab2
    perm_m = rng.permutation(len(monomers))
    monomers = [monomers[i] for i in perm_m]
    n_val = min(60, len(monomers) // 8)
    train_mono = monomers[:-n_val]
    val_mono = monomers[-n_val:]
    print(
        f"Loaded general crystal corpus in {time.time() - t0:.1f}s: "
        f"{len(train_mono)} train monomers, {len(val_mono)} val monomers, "
        f"{len(subdomain_complexes)} crystal subdomain/PPI complexes.",
        flush=True,
    )
    return train_mono, val_mono, subdomain_complexes


def build_inplace_mirage_sequences(
    b_aa: torch.Tensor, b_beads: torch.Tensor, t_beads: torch.Tensor, rng: np.random.Generator
) -> list[torch.Tensor]:
    """Generate 5 hard in-place (x_0) sequence mirage decoys on the exact same binder backbone pose."""
    d_ca0 = torch.cdist(b_beads[:, 1], t_beads[:, 1])
    min_d = d_ca0.amin(dim=1)
    intf_idx = torch.where(min_d < 9.0)[0]
    core_idx = torch.where(min_d < 6.8)[0]
    nonintf_idx = torch.where(min_d >= 9.5)[0]
    if intf_idx.numel() < 3:
        intf_idx = torch.argsort(min_d)[: min(8, len(b_aa))]
    if core_idx.numel() < 2:
        core_idx = intf_idx[: max(2, len(intf_idx) // 2)]

    # Mirage 1 & 2: Random permutations of interface sequence (destroys specific lock-and-key while preserving AA composition)
    aa_perm1 = b_aa.clone()
    p1 = intf_idx[rng.permutation(len(intf_idx))]
    aa_perm1[intf_idx] = b_aa[p1]

    aa_perm2 = b_aa.clone()
    p2 = intf_idx[rng.permutation(len(intf_idx))]
    aa_perm2[intf_idx] = b_aa[p2]

    # Mirage 3: Buried Unsatisfied Polar (Buns) injection into deep interface core
    # Replace core residues with charged/polar (ASP=3, GLU=6, LYS=11, ARG=1, ASN=2, GLN=5)
    polar_pool = torch.tensor([3, 6, 11, 1, 2, 5], dtype=torch.long)
    aa_buns = b_aa.clone()
    aa_buns[core_idx] = polar_pool[torch.arange(len(core_idx)) % len(polar_pool)]

    # Mirage 4: Hotspot-to-Alanine Cavity Shave (replace interface aromatics/hydrophobics with ALA=0)
    aa_ala = b_aa.clone()
    aa_ala[intf_idx] = 0  # ALA

    # Mirage 5: Core-Rim O-Ring Inversion + Charge Flip at interface
    aa_inv = b_aa.clone()
    if nonintf_idx.numel() >= core_idx.numel():
        aa_inv[core_idx] = b_aa[nonintf_idx[: len(core_idx)]]
    # Also flip charges on interface: ARG(1)/LYS(11) <-> ASP(3)/GLU(6)
    for idx_i in intf_idx.tolist():
        v = int(aa_inv[idx_i].item())
        if v in (1, 11):
            aa_inv[idx_i] = 3
        elif v in (3, 6):
            aa_inv[idx_i] = 1

    return [aa_perm1, aa_perm2, aa_buns, aa_ala, aa_inv]


def _compute_canonical_template_tables(train_mono: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute canonical rigid-template local-frame offsets u_sc[21, 3], u_tip[21, 3], and sigma[21] from crystal monomers."""
    u_sc_sum = torch.zeros((21, 3), dtype=torch.float32)
    u_tip_sum = torch.zeros((21, 3), dtype=torch.float32)
    u_sc_sq = torch.zeros((21,), dtype=torch.float32)
    counts = torch.zeros((21,), dtype=torch.float32)
    for g in train_mono:
        aa = g["aa"].detach().cpu().clamp(0, 20)
        R = g["frames"].detach().cpu()  # [N, 3, 3]
        ca = g["ca_pos"].detach().cpu()  # [N, 3]
        t_sc = g["true_sc"].detach().cpu()
        t_tip = g["true_tip"].detach().cpu()
        u_sc = torch.einsum("nij,ni->nj", R, t_sc - ca)
        u_tip = torch.einsum("nij,ni->nj", R, t_tip - ca)
        for a_idx in range(20):
            if a_idx == 5:  # GLY
                continue
            m = aa == a_idx
            if m.any():
                u_sc_sum[a_idx] += u_sc[m].sum(dim=0)
                u_tip_sum[a_idx] += u_tip[m].sum(dim=0)
                counts[a_idx] += float(m.sum().item())
    denom = counts.clamp_min(1.0).unsqueeze(-1)
    tmpl_u_sc = u_sc_sum / denom
    tmpl_u_tip = u_tip_sum / denom
    for g in train_mono:
        aa = g["aa"].detach().cpu().clamp(0, 20)
        R = g["frames"].detach().cpu()
        ca = g["ca_pos"].detach().cpu()
        t_sc = g["true_sc"].detach().cpu()
        u_sc = torch.einsum("nij,ni->nj", R, t_sc - ca)
        for a_idx in range(20):
            if a_idx == 5:
                continue
            m = aa == a_idx
            if m.any():
                u_sc_sq[a_idx] += ((u_sc[m] - tmpl_u_sc[a_idx]) ** 2).sum(dim=-1).sum()
    tmpl_sigma = torch.sqrt(u_sc_sq / counts.clamp_min(1.0) + 0.04).clamp_min(0.20)
    tmpl_sigma[5] = 0.15
    return tmpl_u_sc, tmpl_u_tip, tmpl_sigma


def train_or_load_general_models(
    force_retrain: bool = False,
) -> tuple[BackboneInverseFoldingGNN, UnboundSidechainPredictorGNN, GeneralPPISubdomainEBM]:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if_model = BackboneInverseFoldingGNN(hidden=96, n_layers=3).to(dev)
    sc_model = UnboundSidechainPredictorGNN(hidden=96, n_layers=3).to(dev)
    ebm_model = GeneralPPISubdomainEBM(hidden=64).to(dev)

    if GEN_PROTEIN_CKPT.exists():
        ckpt = torch.load(GEN_PROTEIN_CKPT, map_location=dev, weights_only=False)
        if "if_state" in ckpt:
            if_model.load_state_dict(ckpt["if_state"])
            if_model.eval()
        if ckpt.get("version", 1) == 4 and not force_retrain and "tmpl_u_sc" in ckpt:
            sc_model.load_state_dict(ckpt["sc_state"])
            ebm_model.load_state_dict(ckpt["ebm_state"])
            sc_model.tmpl_u_sc = ckpt["tmpl_u_sc"].to(dev)
            sc_model.tmpl_u_tip = ckpt["tmpl_u_tip"].to(dev)
            sc_model.tmpl_sigma = ckpt["tmpl_sigma"].to(dev)
            sc_model.eval()
            ebm_model.eval()
            return if_model, sc_model, ebm_model

    train_mono, val_mono, sub_complexes = load_general_crystal_corpus(max_solved_pdbs=900)
    tmpl_u_sc, tmpl_u_tip, tmpl_sigma = _compute_canonical_template_tables(train_mono)
    sc_model.tmpl_u_sc = tmpl_u_sc.to(dev)
    sc_model.tmpl_u_tip = tmpl_u_tip.to(dev)
    sc_model.tmpl_sigma = tmpl_sigma.to(dev)

    def _move_geom(g: dict) -> dict:
        return {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in g.items()}

    train_mono = [_move_geom(g) for g in train_mono]
    val_mono = [_move_geom(g) for g in val_mono]

    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)

    # 1. Train GeneralBackboneInverseFoldingGNN + 2-Point UnboundSidechainPredictorGNN (with K=3 MDN rotamer modes) on expanded crystal corpus
    print(f"Training BackboneInverseFoldingGNN & 2-Point UnboundSidechainPredictorGNN (with K=3 Rotamer MDN) on {dev} ...", flush=True)
    t0 = time.time()
    n_sc_epochs = 15
    opt_if = torch.optim.AdamW(if_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    opt_sc = torch.optim.AdamW(sc_model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    sched_if = torch.optim.lr_scheduler.CosineAnnealingLR(opt_if, T_max=n_sc_epochs, eta_min=1e-4)
    sched_sc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sc, T_max=n_sc_epochs, eta_min=1e-4)

    for epoch in range(1, n_sc_epochs + 1):
        if_model.train()
        sc_model.train()
        order = rng.permutation(len(train_mono))
        for start in range(0, len(order), 8):
            opt_if.zero_grad()
            opt_sc.zero_grad()
            loss_if_sum = 0.0
            loss_sc_sum = 0.0
            cnt_if = 0
            cnt_sc = 0
            for idx in order[start : start + 8]:
                g = train_mono[idx]
                aa = g["aa"]
                valid = aa < 20
                if not valid.any():
                    continue
                log_p = if_model.forward_from_geom(g)
                loss_if_sum = loss_if_sum + F.nll_loss(log_p[valid], aa[valid], reduction="sum")
                cnt_if += int(valid.sum().item())

                # Non-GLY sidechain centroid + distal functional tip + K=3 rotamer mixture modes
                non_gly = valid & (aa != 5)
                if non_gly.any():
                    x_sc_hat, _, sigma, x_tip_hat, _, sigma_tip, mdn_d = sc_model.forward_from_geom(
                        g, aa, return_tip=True, return_mdn=True
                    )
                    true_sc_ng = g["true_sc"][non_gly]
                    diff_sq = ((x_sc_hat[non_gly] - true_sc_ng) ** 2).sum(dim=-1)
                    sig_ng = sigma[non_gly]
                    l_sc = (0.5 * diff_sq / (sig_ng ** 2) + 1.5 * torch.log(sig_ng ** 2) + 1.0 * diff_sq).sum()

                    # K=3 Mixture-of-Rotamers loss (Mixture NLL + Winner-Take-All Diversity)
                    x_sc_m = mdn_d["x_sc_modes"][non_gly]  # [M, K, 3]
                    pi_m = mdn_d["pi_modes"][non_gly].clamp_min(1e-5)  # [M, K]
                    sig_m = mdn_d["sigma_modes"][non_gly]  # [M, K]
                    d_modes_sq = ((x_sc_m - true_sc_ng.unsqueeze(1)) ** 2).sum(dim=-1)  # [M, K]
                    if g.get("has_true_tip", False):
                        true_tip_ng = g["true_tip"][non_gly]
                        x_tip_m = mdn_d["x_tip_modes"][non_gly]
                        d_tip_modes_sq = ((x_tip_m - true_tip_ng.unsqueeze(1)) ** 2).sum(dim=-1)
                        d_modes_tot = d_modes_sq + 0.65 * d_tip_modes_sq
                    else:
                        d_modes_tot = d_modes_sq
                    log_p_modes = torch.log(pi_m) - 0.5 * d_modes_tot / (sig_m ** 2) - 1.5 * torch.log(sig_m ** 2)
                    l_mdn_nll = -torch.logsumexp(log_p_modes, dim=-1).sum()
                    l_mdn_wta = d_modes_tot.amin(dim=-1).sum()
                    l_sc = l_sc + 0.35 * (l_mdn_nll + l_mdn_wta)

                    if g.get("has_true_tip", False):
                        diff_tip_sq = ((x_tip_hat[non_gly] - g["true_tip"][non_gly]) ** 2).sum(dim=-1)
                        sig_t_ng = sigma_tip[non_gly]
                        ca_ng = g["ca_pos"][non_gly]
                        u_tip_pred = F.normalize(x_tip_hat[non_gly] - ca_ng, dim=-1, eps=1e-6)
                        u_tip_true = F.normalize(g["true_tip"][non_gly] - ca_ng, dim=-1, eps=1e-6)
                        l_dir = (1.0 - (u_tip_pred * u_tip_true).sum(dim=-1)).sum()
                        l_tip = (
                            0.5 * diff_tip_sq / (sig_t_ng ** 2)
                            + 1.5 * torch.log(sig_t_ng ** 2)
                            + 1.0 * diff_tip_sq
                        ).sum() + 0.35 * l_dir
                        l_sc = l_sc + 0.85 * l_tip
                    loss_sc_sum = loss_sc_sum + l_sc
                    cnt_sc += int(non_gly.sum().item())

            if cnt_if > 0:
                (loss_if_sum / cnt_if).backward()
                torch.nn.utils.clip_grad_norm_(if_model.parameters(), 1.0)
                opt_if.step()
            if cnt_sc > 0:
                (loss_sc_sum / cnt_sc).backward()
                torch.nn.utils.clip_grad_norm_(sc_model.parameters(), 1.0)
                opt_sc.step()
        sched_if.step()
        sched_sc.step()

        if epoch in (1, 5, 10, n_sc_epochs):
            if_model.eval()
            sc_model.eval()
            with torch.no_grad():
                v_nll, v_rec, v_rmsd, v_tip_rmsd, n_tok, n_ng, n_tip = 0.0, 0, 0.0, 0.0, 0, 0, 0
                for g in val_mono:
                    aa = g["aa"]
                    valid = aa < 20
                    lp = if_model.forward_from_geom(g)
                    v_nll += float(F.nll_loss(lp[valid], aa[valid], reduction="sum").item())
                    v_rec += int((lp[valid].argmax(dim=-1) == aa[valid]).sum().item())
                    n_tok += int(valid.sum().item())
                    non_gly = valid & (aa != 5)
                    x_hat, _, _, x_tip_hat, _, _ = sc_model.forward_from_geom(g, aa, return_tip=True)
                    v_rmsd += float(torch.sqrt(((x_hat[non_gly] - g["true_sc"][non_gly]) ** 2).sum(dim=-1) + 1e-8).sum().item())
                    n_ng += int(non_gly.sum().item())
                    if g.get("has_true_tip", False):
                        v_tip_rmsd += float(torch.sqrt(((x_tip_hat[non_gly] - g["true_tip"][non_gly]) ** 2).sum(dim=-1) + 1e-8).sum().item())
                        n_tip += int(non_gly.sum().item())
                print(
                    f"  [Epoch {epoch:2d}/{n_sc_epochs}] Gen-IF Val Recovery={v_rec / max(n_tok, 1):.3f} "
                    f"(Perplexity={math.exp(v_nll / max(n_tok, 1)):.2f}) | "
                    f"SC Centroid MAE={v_rmsd / max(n_ng, 1):.3f} A | "
                    f"Distal Tip MAE={v_tip_rmsd / max(n_tip, 1):.3f} A ({time.time() - t0:.1f}s)",
                    flush=True,
                )

    # 2. Train GeneralPPISubdomainEBM on sidechain-erased crystal interfaces with BOTH Rigid-Body & Hard In-Place Mirage Decoys
    print("Training GeneralPPISubdomainEBM on crystal interfaces (Rigid-Body Decoys + Hard In-Place x_0 Mirage Decoys) ...", flush=True)
    sc_model.eval()
    prepared_sub = []
    with torch.no_grad():
        for comp in sub_complexes:
            b_beads = comp["b_beads"].to(dev)
            b_aa = comp["b_aa"].to(dev)
            t_beads = comp["t_beads"].to(dev)
            t_aa = comp["t_aa"].to(dev)
            b_geom = compute_pure_backbone_geometry(
                b_beads, torch.zeros(len(b_beads), dtype=torch.long, device=dev), torch.arange(len(b_beads), device=dev)
            )
            t_geom = compute_pure_backbone_geometry(
                t_beads, torch.zeros(len(t_beads), dtype=torch.long, device=dev), torch.arange(len(t_beads), device=dev)
            )
            b_sc_hat, _, b_sig, b_tip_hat, _, b_sig_tip = sc_model.forward_from_geom(b_geom, b_aa, return_tip=True)
            t_sc_hat, _, t_sig, t_tip_hat, _, t_sig_tip = sc_model.forward_from_geom(t_geom, t_aa, return_tip=True)
            b_erased = b_beads.clone()
            b_erased[:, 4] = b_sc_hat
            t_erased = t_beads.clone()
            t_erased[:, 4] = t_sc_hat
            rots, trans, pose_rmsd = build_redocking_pose_ensemble(b_erased, t_erased, rng, n_global=15)
            rots, trans, pose_rmsd = rots.to(dev), trans.to(dev), pose_rmsd.to(dev)

            # Build 5 hard in-place (x_0) sequence mirage decoys on the exact same binder backbone
            mirage_aas = build_inplace_mirage_sequences(b_aa.cpu(), b_beads.cpu(), t_beads.cpu(), rng)
            mirage_packs = []
            for m_aa_cpu in mirage_aas:
                m_aa = m_aa_cpu.to(dev)
                m_sc, _, m_sig, m_tip, _, m_sig_tip = sc_model.forward_from_geom(b_geom, m_aa, return_tip=True)
                m_erased = b_beads.clone()
                m_erased[:, 4] = m_sc
                mirage_packs.append({"aa": m_aa, "beads": m_erased, "tip": m_tip, "sigma": 0.5 * (m_sig + m_sig_tip)})

            prepared_sub.append(
                {
                    "b_erased": b_erased,
                    "b_tip": b_tip_hat,
                    "b_sigma": 0.5 * (b_sig + b_sig_tip),
                    "b_aa": b_aa,
                    "b_geom": b_geom,
                    "t_erased": t_erased,
                    "t_tip": t_tip_hat,
                    "t_sigma": 0.5 * (t_sig + t_sig_tip),
                    "t_aa": t_aa,
                    "t_geom": t_geom,
                    "rots": rots,
                    "trans": trans,
                    "pose_rmsd": pose_rmsd,
                    "mirages": mirage_packs,
                }
            )

    I_rot1 = torch.eye(3, dtype=torch.float32, device=dev).unsqueeze(0)
    Z_tr1 = torch.zeros((1, 3), dtype=torch.float32, device=dev)
    n_ebm_epochs = 12
    opt_ebm = torch.optim.AdamW(ebm_model.parameters(), lr=2.0e-3, weight_decay=1e-4)
    sched_ebm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ebm, T_max=n_ebm_epochs, eta_min=1e-4)
    for epoch in range(1, n_ebm_epochs + 1):
        ebm_model.train()
        order = rng.permutation(len(prepared_sub))
        tot_loss = 0.0
        top1_pose_hits = 0
        top1_mirage_hits = 0
        for idx in order:
            item = prepared_sub[idx]
            opt_ebm.zero_grad()
            h_b = ebm_model.encode_partner(item["b_geom"], item["b_aa"])
            h_t = ebm_model.encode_partner(item["t_geom"], item["t_aa"])
            energies = ebm_model.score_poses(
                item["b_erased"],
                item["b_aa"],
                h_b,
                item["t_erased"],
                item["t_aa"],
                h_t,
                item["rots"],
                item["trans"],
                b_tip_erased=item["b_tip"],
                t_tip_erased=item["t_tip"],
                b_sigma=item["b_sigma"],
                t_sigma=item["t_sigma"],
            )
            # Score the 5 in-place (x_0) sequence mirage decoys
            e_mirages = []
            for m_pack in item["mirages"]:
                h_bm = ebm_model.encode_partner(item["b_geom"], m_pack["aa"])
                em = ebm_model.score_poses(
                    m_pack["beads"],
                    m_pack["aa"],
                    h_bm,
                    item["t_erased"],
                    item["t_aa"],
                    h_t,
                    I_rot1,
                    Z_tr1,
                    b_tip_erased=m_pack["tip"],
                    t_tip_erased=item["t_tip"],
                    b_sigma=m_pack["sigma"],
                    t_sigma=item["t_sigma"],
                )[0]
                e_mirages.append(em)
            e_mir = torch.stack(e_mirages, dim=0)

            decoy_m = item["pose_rmsd"] > 2.0
            e0 = energies[0]
            e_dec = energies[decoy_m]
            all_neg = torch.cat([e_dec, e_mir], dim=0)
            logits = torch.cat([-e0.unsqueeze(0), -all_neg], dim=0) / 0.50
            loss_nce = -F.log_softmax(logits, dim=0)[0]
            hard_pose_min = e_dec.min()
            hard_mir_min = e_mir.min()
            loss_margin = 0.5 * F.relu(e0 - hard_pose_min + 0.50) + 0.6 * F.relu(e0 - hard_mir_min + 0.35)
            loss = loss_nce + loss_margin
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ebm_model.parameters(), 1.0)
            opt_ebm.step()
            tot_loss += float(loss.item())
            if float(e0.item()) < float(hard_pose_min.item()):
                top1_pose_hits += 1
            if float(e0.item()) < float(hard_mir_min.item()):
                top1_mirage_hits += 1
        sched_ebm.step()
        if epoch in (1, 4, 8, n_ebm_epochs):
            n_sub = max(len(prepared_sub), 1)
            print(
                f"  [EBM Epoch {epoch:2d}/{n_ebm_epochs}] Loss={tot_loss / n_sub:.4f} | "
                f"Native < Pose Decoys={top1_pose_hits}/{n_sub} ({100.0 * top1_pose_hits / n_sub:.1f}%) | "
                f"Native < In-Place Mirages={top1_mirage_hits}/{n_sub} ({100.0 * top1_mirage_hits / n_sub:.1f}%)",
                flush=True,
            )

    if_model.eval()
    sc_model.eval()
    ebm_model.eval()
    torch.save(
        {
            "version": 4,
            "if_state": {k: v.cpu() for k, v in if_model.state_dict().items()},
            "sc_state": {k: v.cpu() for k, v in sc_model.state_dict().items()},
            "ebm_state": {k: v.cpu() for k, v in ebm_model.state_dict().items()},
            "tmpl_u_sc": sc_model.tmpl_u_sc.cpu(),
            "tmpl_u_tip": sc_model.tmpl_u_tip.cpu(),
            "tmpl_sigma": sc_model.tmpl_sigma.cpu(),
        },
        GEN_PROTEIN_CKPT,
    )
    print(f"Saved general protein v4 models to {GEN_PROTEIN_CKPT}", flush=True)
    return if_model, sc_model, ebm_model


def extract_sidechain_erased_redocking_features(force_retrain: bool = False) -> list[dict]:
    """Run 2-Point Sidechain Erasure, In-Place Alchemical Specificity, and 24-Pose SE(3) Re-Docking across all 3,532 Overath complexes."""
    if_model, sc_model, ebm_model = train_or_load_general_models(force_retrain=force_retrain)

    with torch.no_grad():
        return _extract_sidechain_erased_redocking_features_nograd(if_model, sc_model, ebm_model)


@torch.no_grad()
def _extract_sidechain_erased_redocking_features_nograd(
    if_model: BackboneInverseFoldingGNN,
    sc_model: UnboundSidechainPredictorGNN,
    ebm_model: GeneralPPISubdomainEBM,
) -> list[dict]:
    dev = next(sc_model.parameters()).device
    tmpl_u_sc = sc_model.tmpl_u_sc.to(dev)
    tmpl_u_tip = sc_model.tmpl_u_tip.to(dev)
    tmpl_sigma = sc_model.tmpl_sigma.to(dev)
    phys_tbl = PHYSCHEM_TABLE.to(dev)

    cache_5b = torch.load(OVERATH_5BEAD_CACHE, map_location="cpu", weights_only=False)
    targets_cache = cache_5b["targets"]
    binders_cache = cache_5b["binders"]
    feat_rows = torch.load(OVERATH_FEAT_CACHE, map_location="cpu", weights_only=False)

    t0 = time.time()
    target_seq_cache = {}
    contact_counts = {}
    target_geom_props = {}
    for b_item in binders_cache:
        bb = b_item["beads"].to(dev, dtype=torch.float32)
        tv = targets_cache[b_item["t_hash"]]
        t_beads0 = tv["beads"].to(dev, dtype=torch.float32)
        t_aa0 = tv["aa"].to(dev, dtype=torch.long)
        seq_key = (b_item["target_id"], tuple(tv["aa"].tolist()))
        d_ca_bt = torch.cdist(bb[:, 1], t_beads0[:, 1])
        t_cont = (d_ca_bt.amin(dim=0) < 8.5).to(torch.float32)
        if seq_key not in target_seq_cache:
            t_chain0 = tv["chain"].to(dev, dtype=torch.long)
            with torch.enable_grad():
                tb_g = t_beads0.clone().requires_grad_(True)
                t_geom0 = compute_pure_backbone_geometry(tb_g, t_chain0, torch.arange(len(t_beads0), device=dev))
                lp_t0 = if_model.forward_from_geom(t_geom0)
                ll_t0_sum = lp_t0[torch.arange(len(t_beads0), device=dev), t_aa0.clamp(0, 19)].sum()
                grad_t0 = torch.autograd.grad(ll_t0_sum, tb_g)[0].detach()
                f_t_apo = torch.norm(grad_t0[:, :4].reshape(len(t_beads0), -1), dim=-1)
                lp_t0 = lp_t0.detach()
                t_geom0 = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in t_geom0.items()}
            _, u_local_t, t_sigma, _, u_tip_local_t, t_sigma_tip = sc_model.forward_from_geom(
                t_geom0, t_aa0, return_tip=True
            )
            h_t = ebm_model.encode_partner(t_geom0, t_aa0)
            t_aa_c0 = t_aa0.clamp(0, 20)
            target_seq_cache[seq_key] = {
                "u_local_t": u_local_t,
                "t_sigma": t_sigma,
                "u_tip_local_t": u_tip_local_t,
                "t_sigma_tip": t_sigma_tip,
                "u_tmpl_t": tmpl_u_sc[t_aa_c0],
                "u_tip_tmpl_t": tmpl_u_tip[t_aa_c0],
                "t_sigma_tmpl": tmpl_sigma[t_aa_c0],
                "h_t": h_t,
                "lp_t_apo": lp_t0,
                "f_t_apo": f_t_apo,
            }
            contact_counts[seq_key] = t_cont.clone()
            t_ca = t_beads0[:, 1]
            d_tt = torch.cdist(t_ca, t_ca).masked_fill(torch.eye(len(t_ca), dtype=torch.bool, device=dev), 1e6)
            bur6 = (d_tt < 6.0).to(torch.float32).sum(dim=-1)
            bur14 = (d_tt < 14.0).to(torch.float32).sum(dim=-1)
            t_phys = phys_tbl[t_aa_c0]
            target_geom_props[seq_key] = {
                "concavity": bur14 / (bur6 + 2.0),
                "hydro": F.relu(t_phys[:, 1]),
                "arom": t_phys[:, 4],
            }
        else:
            contact_counts[seq_key] += t_cont
    for k in contact_counts:
        contact_counts[k] = contact_counts[k] / (contact_counts[k].max() + 1e-5)
    print(f"Pre-encoded {len(target_seq_cache)} unique target constructs on {dev} in {time.time() - t0:.2f}s.", flush=True)

    rng = np.random.default_rng(2026)
    print(f"Computing 2-Point Sidechain-Erased Re-Docking & Alchemical Specificity on {len(binders_cache)} Overath complexes ({dev}) ...", flush=True)

    I_rot = torch.eye(3, dtype=torch.float32, device=dev).unsqueeze(0)
    Z_tr = torch.zeros((1, 3), dtype=torch.float32, device=dev)

    for idx, (d, row) in enumerate(zip(binders_cache, feat_rows)):
        b_beads = d["beads"].to(dev, dtype=torch.float32)
        b_aa = d["aa"].to(dev, dtype=torch.long)
        b_chain = d["chain"].to(dev, dtype=torch.long)
        tv = targets_cache[d["t_hash"]]
        t_beads = tv["beads"].to(dev, dtype=torch.float32)
        t_aa = tv["aa"].to(dev, dtype=torch.long)
        seq_key = (d["target_id"], tuple(tv["aa"].tolist()))
        t_info = target_seq_cache[seq_key]
        h_t = t_info["h_t"]

        R_t = build_local_frames(t_beads[:, 0], t_beads[:, 1], t_beads[:, 2])
        t_sc_hat = t_beads[:, 1] + torch.einsum("nij,nj->ni", R_t, t_info["u_local_t"])
        t_tip_hat = t_beads[:, 1] + torch.einsum("nij,nj->ni", R_t, t_info["u_tip_local_t"])
        t_erased = t_beads.clone()
        t_erased[:, 4] = t_sc_hat

        t_sc_tmpl = t_beads[:, 1] + torch.einsum("nij,nj->ni", R_t, t_info["u_tmpl_t"])
        t_tip_tmpl = t_beads[:, 1] + torch.einsum("nij,nj->ni", R_t, t_info["u_tip_tmpl_t"])
        t_tmpl = t_beads.clone()
        t_tmpl[:, 4] = t_sc_tmpl

        Nb = len(b_beads)
        valid_b = b_aa < 20
        b_aa_c = b_aa.clamp(0, 19)
        with torch.enable_grad():
            bb_g = b_beads.clone().requires_grad_(True)
            b_geom_g = compute_pure_backbone_geometry(bb_g, b_chain, torch.arange(Nb, device=dev))
            log_p_b_g = if_model.forward_from_geom(b_geom_g)
            ll_b_g = log_p_b_g[torch.arange(Nb, device=dev), b_aa_c]
            grad_b_apo = torch.autograd.grad(ll_b_g.sum(), bb_g)[0].detach()
            f_b_apo = torch.norm(grad_b_apo[:, :4].reshape(Nb, -1), dim=-1)
            ll_b = ll_b_g.detach()
            b_geom = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in b_geom_g.items()}

        # 1. General-Protein 3D Inverse Folding (trained on expanded crystal structures + SAbDab2)
        core_m = b_geom["burial_cb10"] > b_geom["burial_cb10"].median()
        gen_if_ll_all = float(ll_b[valid_b].mean().item())
        gen_if_ll_core = float(ll_b[valid_b & core_m].mean().item()) if (valid_b & core_m).any() else gen_if_ll_all
        b_esm_nll = float(row["features"].get("esm_binder_all_nll", 0.0))

        d_ca0 = torch.cdist(b_beads[:, 1], t_beads[:, 1])
        b_intf = d_ca0.amin(dim=1) < 9.0
        gen_if_ll_intf = float(ll_b[valid_b & b_intf].mean().item()) if (valid_b & b_intf).any() else gen_if_ll_all
        gen_if_intf_minus_core = gen_if_ll_intf - gen_if_ll_core

        # Weakest-link sliding-window monomer & interface backbone-sequence foldability (5-residue windows)
        if Nb >= 5:
            win5_ll = F.avg_pool1d(ll_b.view(1, 1, -1), kernel_size=5, stride=1).view(-1)
            gen_if_worst_win5 = float(win5_ll.min().item())
            intf_win5 = F.avg_pool1d(b_intf.to(torch.float32).view(1, 1, -1), kernel_size=5, stride=1).view(-1) > 0.39
            gen_if_worst_intf_win5 = float(win5_ll[intf_win5].min().item()) if intf_win5.any() else gen_if_worst_win5
        else:
            gen_if_worst_win5 = gen_if_ll_all
            gen_if_worst_intf_win5 = gen_if_ll_intf

        # 2. 2-Point Sidechain Erasure (Centroid + Distal Functional Tip + K=3 Rotamer Mixture Modes) & Generator Rotamer Strain
        b_sc_hat, _, b_sigma, b_tip_hat, _, b_sigma_tip, mdn_b = sc_model.forward_from_geom(
            b_geom, b_aa, return_tip=True, return_mdn=True
        )
        b_erased = b_beads.clone()
        b_erased[:, 4] = b_sc_hat

        # Holo vs. Apo Joint Complex 3D Inverse Folding, Rotamer Reorganization, and Autograd Backbone Restoring Force
        t_near_idx = torch.where(d_ca0.amin(dim=0) < 15.0)[0]
        if t_near_idx.numel() < 5:
            t_near_idx = torch.argsort(d_ca0.amin(dim=0))[: min(25, len(t_beads))]
        Nt_n = len(t_near_idx)
        comp_aa = torch.cat([b_aa, t_aa[t_near_idx]], dim=0)
        comp_chain = torch.cat([torch.zeros(Nb, dtype=torch.long, device=dev), torch.ones(Nt_n, dtype=torch.long, device=dev)], dim=0)
        comp_ridx = torch.cat([torch.arange(Nb, device=dev), t_near_idx + 1000], dim=0)
        with torch.enable_grad():
            comp_beads_g = torch.cat([b_beads, t_beads[t_near_idx]], dim=0).clone().requires_grad_(True)
            cg_g = compute_pure_backbone_geometry(comp_beads_g, comp_chain, comp_ridx)
            lp_comp_g = if_model.forward_from_geom(cg_g)
            ll_comp_vec = lp_comp_g[torch.arange(Nb + Nt_n, device=dev), comp_aa.clamp(0, 19)]
            grad_comp = torch.autograd.grad(ll_comp_vec.sum(), comp_beads_g)[0].detach()
            f_comp = torch.norm(grad_comp[:, :4].reshape(Nb + Nt_n, -1), dim=-1)
            ll_b_holo = ll_comp_vec[:Nb].detach()
            ll_t_holo = ll_comp_vec[Nb:].detach()
            cg = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in cg_g.items()}
        sc_comp, _, sig_comp, _, _, _ = sc_model.forward_from_geom(cg, comp_aa, return_tip=True)
        ta_n = t_aa[t_near_idx].clamp(0, 19)
        ll_t_apo = t_info["lp_t_apo"][t_near_idx, ta_n]
        t_intf_n = (d_ca0[:, t_near_idx].amin(dim=0) < 9.0) & (t_aa[t_near_idx] < 20)
        b_intf_v = b_intf & valid_b
        holo_if_b_intf = float(ll_b_holo[b_intf_v].mean().item()) if b_intf_v.any() else float(ll_b_holo.mean().item())
        holo_minus_apo_if_b = float((ll_b_holo[b_intf_v] - ll_b[b_intf_v]).mean().item()) if b_intf_v.any() else 0.0
        holo_if_t_intf = float(ll_t_holo[t_intf_n].mean().item()) if t_intf_n.any() else float(ll_t_holo.mean().item())
        holo_minus_apo_if_t = float((ll_t_holo[t_intf_n] - ll_t_apo[t_intf_n]).mean().item()) if t_intf_n.any() else 0.0
        holo_if_joint_gain = holo_minus_apo_if_b + holo_minus_apo_if_t
        b_ng_i = b_intf_v & (b_aa != 5)
        t_ng_i = t_intf_n & (t_aa[t_near_idx] != 5)
        sc_holo_dist_b = float(torch.norm(b_beads[b_ng_i, 4] - sc_comp[:Nb][b_ng_i], dim=-1).mean().item()) if b_ng_i.any() else 0.0
        sc_holo_minus_apo_shift_b = float(torch.norm(sc_comp[:Nb][b_ng_i] - b_sc_hat[b_ng_i], dim=-1).mean().item()) if b_ng_i.any() else 0.0
        sc_holo_dist_t = float(torch.norm(t_beads[t_near_idx][t_ng_i, 4] - sc_comp[Nb:][t_ng_i], dim=-1).mean().item()) if t_ng_i.any() else 0.0
        sc_holo_ent_drop_b = float((b_sigma[b_ng_i] - sig_comp[:Nb][b_ng_i]).mean().item()) if b_ng_i.any() else 0.0
        bb_force_norm_all = float(f_b_apo.mean().item())
        bb_force_norm_intf = float(f_b_apo[b_intf_v].mean().item()) if b_intf_v.any() else bb_force_norm_all
        bb_force_core = float(f_b_apo[core_m].mean().item()) if core_m.any() else bb_force_norm_all
        bb_force_intf_minus_core = bb_force_norm_intf - bb_force_core
        bb_force_holo_b_intf = float(f_comp[:Nb][b_intf_v].mean().item()) if b_intf_v.any() else bb_force_norm_intf
        bb_force_holo_minus_apo_b = bb_force_holo_b_intf - bb_force_norm_intf
        bb_force_holo_minus_apo_t = float((f_comp[Nb:][t_intf_n] - t_info["f_t_apo"][t_near_idx][t_intf_n]).mean().item()) if t_intf_n.any() else 0.0

        # Also construct canonical rigid-template sidechain beads (context-blind reconstruction baseline)
        R_b = b_geom["frames"]
        b_aa_c20 = b_aa.clamp(0, 20)
        b_sc_tmpl = b_beads[:, 1] + torch.einsum("nij,nj->ni", R_b, tmpl_u_sc[b_aa_c20])
        b_tip_tmpl = b_beads[:, 1] + torch.einsum("nij,nj->ni", R_b, tmpl_u_tip[b_aa_c20])
        b_sig_tmpl = tmpl_sigma[b_aa_c20]
        b_tmpl = b_beads.clone()
        b_tmpl[:, 4] = b_sc_tmpl

        non_gly = valid_b & (b_aa != 5)
        sc_dist = torch.sqrt(((b_beads[:, 4] - b_sc_hat) ** 2).sum(dim=-1) + 1e-8)
        sc_z = sc_dist / b_sigma.clamp_min(0.15)
        sc_dist_tmpl_vec = torch.sqrt(((b_beads[:, 4] - b_sc_tmpl) ** 2).sum(dim=-1) + 1e-8)
        intf_ng = b_intf & non_gly
        nonintf_ng = (~b_intf) & non_gly

        # Multi-rotamer minimum-mode distance & Boltzmann mixture NLL strain + Angular vs Radial decomposition
        x_sc_modes = mdn_b["x_sc_modes"]  # [Nb, K, 3]
        pi_modes = mdn_b["pi_modes"].clamp_min(1e-5)  # [Nb, K]
        sig_modes = mdn_b["sigma_modes"].clamp_min(0.15)  # [Nb, K]
        d_modes = torch.sqrt(((b_beads[:, 4].unsqueeze(1) - x_sc_modes) ** 2).sum(dim=-1) + 1e-8)  # [Nb, K]
        sc_min_mode_dist = d_modes.amin(dim=-1)  # [Nb]
        log_p_m = torch.log(pi_modes) - 0.5 * (d_modes / sig_modes).pow(2) - torch.log(sig_modes)
        sc_mdn_nll = -torch.logsumexp(log_p_m, dim=-1)  # [Nb]

        v_sup = b_beads[:, 4] - b_beads[:, 1]
        v_rec = b_sc_hat - b_beads[:, 1]
        r_sup = torch.linalg.norm(v_sup, dim=-1).clamp_min(1e-5)
        r_rec = torch.linalg.norm(v_rec, dim=-1).clamp_min(1e-5)
        sc_radial_strain = (r_sup - r_rec).abs()
        cos_ang = ((v_sup * v_rec).sum(dim=-1) / (r_sup * r_rec)).clamp(-1.0, 1.0)
        sc_angular_strain = 1.0 - cos_ang

        if intf_ng.any():
            sc_intf_min_mode_dist = float(sc_min_mode_dist[intf_ng].mean().item())
            sc_intf_mdn_nll = float(sc_mdn_nll[intf_ng].mean().item())
            sc_intf_radial_strain = float(sc_radial_strain[intf_ng].mean().item())
            sc_intf_angular_strain = float(sc_angular_strain[intf_ng].mean().item())
            sc_intf_dist_tmpl = float(sc_dist_tmpl_vec[intf_ng].mean().item())
        else:
            sc_intf_min_mode_dist = float(sc_min_mode_dist[non_gly].mean().item())
            sc_intf_mdn_nll = 0.0
            sc_intf_radial_strain = 0.0
            sc_intf_angular_strain = 0.0
            sc_intf_dist_tmpl = float(sc_dist_tmpl_vec[non_gly].mean().item())

        sc_intf_dist = float(sc_dist[intf_ng].mean().item()) if intf_ng.any() else float(sc_dist[non_gly].mean().item())
        sc_nonintf_dist = float(sc_dist[nonintf_ng].mean().item()) if nonintf_ng.any() else sc_intf_dist
        sc_intf_z_strain = float(sc_z[intf_ng].mean().item()) if intf_ng.any() else 0.0

        # Hotspot-concentrated sidechain discrepancy (Top-3 & Deep-Core contact-weighted)
        if intf_ng.any():
            k_hot = min(3, int(intf_ng.sum().item()))
            sc_intf_top3_dist = float(torch.topk(sc_dist[intf_ng], k=k_hot).values.mean().item())
            sc_intf_top3_z_strain = float(torch.topk(sc_z[intf_ng], k=k_hot).values.mean().item())
            sc_intf_top3_dist_tmpl = float(torch.topk(sc_dist_tmpl_vec[intf_ng], k=k_hot).values.mean().item())
            d_min_b = d_ca0.amin(dim=1)
            w_core = torch.sigmoid((7.5 - d_min_b[intf_ng]) / 0.8) + 1e-4
            sc_core_weighted_dist = float(((sc_dist[intf_ng] * w_core).sum() / w_core.sum()).item())
            sc_core_weighted_z = float(((sc_z[intf_ng] * w_core).sum() / w_core.sum()).item())
            sc_core_weighted_dist_tmpl = float(((sc_dist_tmpl_vec[intf_ng] * w_core).sum() / w_core.sum()).item())
        else:
            sc_intf_top3_dist = sc_intf_dist
            sc_intf_top3_z_strain = sc_intf_z_strain
            sc_intf_top3_dist_tmpl = sc_intf_dist_tmpl
            sc_core_weighted_dist = sc_intf_dist
            sc_core_weighted_z = sc_intf_z_strain
            sc_core_weighted_dist_tmpl = sc_intf_dist_tmpl

        # Target-side epitope sidechain discrepancy
        t_valid_ng = (t_aa < 20) & (t_aa != 5)
        t_intf_mask = (d_ca0.amin(dim=0) < 9.0) & t_valid_ng
        t_sc_dist = torch.sqrt(((t_beads[:, 4] - t_sc_hat) ** 2).sum(dim=-1) + 1e-8)
        t_sc_dist_tmpl = torch.sqrt(((t_beads[:, 4] - t_sc_tmpl) ** 2).sum(dim=-1) + 1e-8)
        t_sc_z = t_sc_dist / t_info["t_sigma"].clamp_min(0.15)
        if t_intf_mask.any():
            sc_target_intf_dist = float(t_sc_dist[t_intf_mask].mean().item())
            sc_target_intf_z_strain = float(t_sc_z[t_intf_mask].mean().item())
            sc_target_intf_dist_tmpl = float(t_sc_dist_tmpl[t_intf_mask].mean().item())
        else:
            sc_target_intf_dist = 0.0
            sc_target_intf_z_strain = 0.0
            sc_target_intf_dist_tmpl = 0.0

        # Conformational entropy cost of burying high-uncertainty flexible sidechains at the interface
        b_sig_comb = 0.5 * (b_sigma + b_sigma_tip)
        t_sig_comb = 0.5 * (t_info["t_sigma"] + t_info["t_sigma_tip"])
        b_rot_ent = torch.log1p(F.relu(b_sig_comb - 0.30) / 0.25)
        sc_intf_rot_entropy_cost = float(b_rot_ent[intf_ng].mean().item()) if intf_ng.any() else 0.0

        # 3. 24-Pose SE(3) Re-Docking & Local Relaxation Ensemble under Supplied vs Template vs 2-Point GNN Sidechains
        h_b = ebm_model.encode_partner(b_geom, b_aa)
        rots_cpu, trans_cpu, pose_rmsd_cpu = build_redocking_pose_ensemble(b_erased, t_erased, rng, n_global=11)
        rots, trans, pose_rmsd = rots_cpu.to(dev), trans_cpu.to(dev), pose_rmsd_cpu.to(dev)

        # (A) Supplied sidechains (no sidechain erasure) across all 24 poses
        comp_input_sc = ebm_model.score_poses(
            b_beads,
            b_aa,
            h_b,
            t_beads,
            t_aa,
            h_t,
            rots,
            trans,
            return_components=True,
            b_sigma=b_sig_comb,
            t_sigma=t_sig_comb,
        )
        # (B) Canonical Rigid-Template sidechains across all 24 poses
        comp_tmpl = ebm_model.score_poses(
            b_tmpl,
            b_aa,
            h_b,
            t_tmpl,
            t_aa,
            h_t,
            rots,
            trans,
            return_components=True,
            b_tip_erased=b_tip_tmpl,
            t_tip_erased=t_tip_tmpl,
            b_sigma=b_sig_tmpl,
            t_sigma=t_info["t_sigma_tmpl"],
        )
        # (C) 2-Point GNN Reconstructed sidechains (Ours) across all 24 poses
        comp_erased = ebm_model.score_poses(
            b_erased,
            b_aa,
            h_b,
            t_erased,
            t_aa,
            h_t,
            rots,
            trans,
            return_components=True,
            b_tip_erased=b_tip_hat,
            t_tip_erased=t_tip_hat,
            b_sigma=b_sig_comb,
            t_sigma=t_sig_comb,
        )

        e_tot = comp_erased["total"]  # [24]
        e_tot_in = comp_input_sc["total"]  # [24]
        e_tot_tm = comp_tmpl["total"]  # [24]

        e0_erased = float(e_tot[0].item())
        e0_input = float(e_tot_in[0].item())
        e0_tmpl = float(e_tot_tm[0].item())

        mirage_energy_jump = e0_erased - e0_input
        mirage_energy_jump_tmpl = e0_tmpl - e0_input
        mirage_clash_jump = float((comp_erased["clash"][0] - comp_input_sc["clash"][0]).item())
        mirage_hb_loss = float((comp_erased["hb"][0] - comp_input_sc["hb"][0]).item())

        # 5. In-Place Alchemical Counterfactuals at x_0 (Sequence Permutation Specificity & Alanine Scan)
        mirage_aas = build_inplace_mirage_sequences(b_aa.cpu(), b_beads.cpu(), t_beads.cpu(), rng)
        e_mir_list = []
        e_mir_tmpl_list = []
        for m_aa_cpu in mirage_aas:
            m_aa = m_aa_cpu.to(dev)
            m_sc, _, m_sig_c, m_tip, _, m_sig_t = sc_model.forward_from_geom(b_geom, m_aa, return_tip=True)
            m_erased = b_beads.clone()
            m_erased[:, 4] = m_sc
            h_bm = ebm_model.encode_partner(b_geom, m_aa)
            em_val = float(
                ebm_model.score_poses(
                    m_erased,
                    m_aa,
                    h_bm,
                    t_erased,
                    t_aa,
                    h_t,
                    I_rot,
                    Z_tr,
                    b_tip_erased=m_tip,
                    t_tip_erased=t_tip_hat,
                    b_sigma=0.5 * (m_sig_c + m_sig_t),
                    t_sigma=t_sig_comb,
                )[0].item()
            )
            e_mir_list.append(em_val)

            m_aa_c20 = m_aa.clamp(0, 20)
            m_sc_tm = b_beads[:, 1] + torch.einsum("nij,nj->ni", R_b, tmpl_u_sc[m_aa_c20])
            m_tip_tm = b_beads[:, 1] + torch.einsum("nij,nj->ni", R_b, tmpl_u_tip[m_aa_c20])
            m_tmpl = b_beads.clone()
            m_tmpl[:, 4] = m_sc_tm
            em_tm_val = float(
                ebm_model.score_poses(
                    m_tmpl,
                    m_aa,
                    h_bm,
                    t_tmpl,
                    t_aa,
                    h_t,
                    I_rot,
                    Z_tr,
                    b_tip_erased=m_tip_tm,
                    t_tip_erased=t_tip_tmpl,
                    b_sigma=tmpl_sigma[m_aa_c20],
                    t_sigma=t_info["t_sigma_tmpl"],
                )[0].item()
            )
            e_mir_tmpl_list.append(em_tm_val)

        # mirage_aas: [perm1, perm2, buns, ala, inv]
        e_perm_mean = 0.5 * (e_mir_list[0] + e_mir_list[1])
        e_all_mir_mean = float(np.mean(e_mir_list))
        e_all_mir_std = float(np.std(e_mir_list)) + 0.15
        alchem_seq_margin = e0_erased - e_perm_mean
        alchem_seq_zscore = (e0_erased - e_all_mir_mean) / e_all_mir_std
        alchem_buns_mirage_gap = e0_erased - e_mir_list[2]
        alchem_ala_scan_ddg = e0_erased - e_mir_list[3]

        e_tm_all_mean = float(np.mean(e_mir_tmpl_list))
        e_tm_all_std = float(np.std(e_mir_tmpl_list)) + 0.15
        alchem_seq_zscore_tmpl = (e0_tmpl - e_tm_all_mean) / e_tm_all_std
        alchem_buns_mirage_gap_tmpl = e0_tmpl - e_mir_tmpl_list[2]

        # Local relaxation (poses 0..12)
        e_local = e_tot[:13]
        rmsd_local = pose_rmsd[:13]
        best_loc_idx = int(torch.argmin(e_local).item())
        local_drift_rmsd = float(rmsd_local[best_loc_idx].item())
        local_relax_drop = e0_erased - float(e_local[best_loc_idx].item())

        local_drift_rmsd_input = float(rmsd_local[int(torch.argmin(e_tot_in[:13]).item())].item())
        local_drift_rmsd_tmpl = float(rmsd_local[int(torch.argmin(e_tot_tm[:13]).item())].item())

        # Global re-docking (all 24 poses)
        best_glob_idx = int(torch.argmin(e_tot).item())
        redock_top1_rmsd = float(pose_rmsd[best_glob_idx].item())

        # Boltzmann-expected RMSD & native basin probability at tau = 0.5
        for tau_name, tau in (("05", 0.50), ("10", 1.00)):
            w_boltz = F.softmax(-e_tot / tau, dim=0)
            boltz_rmsd = float((w_boltz * pose_rmsd).sum().item())
            p_native = float(w_boltz[pose_rmsd < 2.5].sum().item())
            row["features"][f"redock_boltz_rmsd_t{tau_name}"] = boltz_rmsd
            row["features"][f"redock_p_native_t{tau_name}"] = p_native

        w_boltz_in = F.softmax(-e_tot_in / 0.50, dim=0)
        redock_boltz_rmsd_t05_input = float((w_boltz_in * pose_rmsd).sum().item())
        redock_p_native_t05_input = float(w_boltz_in[pose_rmsd < 2.5].sum().item())

        w_boltz_tm = F.softmax(-e_tot_tm / 0.50, dim=0)
        redock_boltz_rmsd_t05_tmpl = float((w_boltz_tm * pose_rmsd).sum().item())
        redock_p_native_t05_tmpl = float(w_boltz_tm[pose_rmsd < 2.5].sum().item())

        far_m = pose_rmsd > 5.0
        e_far_min = float(e_tot[far_m].min().item()) if far_m.any() else float(e_tot[1:].min().item())
        e_far_mean = float(e_tot[far_m].mean().item()) if far_m.any() else float(e_tot[1:].mean().item())
        e_far_std = float(e_tot[far_m].std().item()) + 1e-4

        n_intf_c = float((d_ca0 < 8.5).sum().item()) + 1.0

        # 6. Interface Secondary-Structure Pre-Organization & Unsupervised Target Epitope Pocket
        b_ca = b_beads[:, 1]
        d13 = torch.zeros(Nb, device=dev)
        d14 = torch.zeros(Nb, device=dev)
        if Nb > 3:
            d13[:-3] = torch.norm(b_ca[3:] - b_ca[:-3], dim=-1)
        if Nb > 4:
            d14[:-4] = torch.norm(b_ca[4:] - b_ca[:-4], dim=-1)
        is_helix_raw = (d13 > 4.6) & (d13 < 5.8) & (d14 > 5.4) & (d14 < 6.9)
        is_sheet_raw = (d13 > 8.6) & (d13 < 11.4)
        is_helix = is_helix_raw.clone()
        is_sheet = is_sheet_raw.clone()
        for s in (1, 2):
            if Nb > s:
                is_helix[s:] |= is_helix_raw[:-s]
                is_sheet[s:] |= is_sheet_raw[:-s]
        is_sheet = is_sheet & (~is_helix)
        is_loop = (~is_helix) & (~is_sheet)

        ss_helix_all = float(is_helix.to(torch.float32).mean().item())
        ss_sheet_all = float(is_sheet.to(torch.float32).mean().item())
        if b_intf.any():
            ss_helix_intf = float(is_helix[b_intf].to(torch.float32).mean().item())
            ss_sheet_intf = float(is_sheet[b_intf].to(torch.float32).mean().item())
            ss_loop_intf = float(is_loop[b_intf].to(torch.float32).mean().item())
        else:
            ss_helix_intf = ss_helix_all
            ss_sheet_intf = ss_sheet_all
            ss_loop_intf = 1.0 - ss_helix_all - ss_sheet_all

        max_h_run = 0
        cur_run = 0
        for flag in (is_helix & b_intf).tolist():
            if flag:
                cur_run += 1
                if cur_run > max_h_run:
                    max_h_run = cur_run
            else:
                cur_run = 0

        t_intf = d_ca0.amin(dim=0) < 9.0
        t_props = target_geom_props[seq_key]
        t_cons = contact_counts[seq_key]
        t_phys_all = phys_tbl[t_aa.clamp(0, 20)]
        t_pol_vec = ((t_phys_all[:, 2] > 0.2) | (t_phys_all[:, 3] > 0.2) | (t_phys_all[:, 0].abs() > 0.2)).to(torch.float32)
        if t_intf.any():
            epi_concavity = float(t_props["concavity"][t_intf].mean().item())
            epi_hydro_patch = float(t_props["hydro"][t_intf].mean().item())
            epi_arom_patch = float(t_props["arom"][t_intf].mean().item())
            epi_consensus = float(t_cons[t_intf].mean().item())
            epi_polar_frac = float(t_pol_vec[t_intf].mean().item())
            epi_hydro_to_polar = (epi_hydro_patch + 0.5 * epi_arom_patch) / (epi_polar_frac + 0.25)
            # Curvature complementarity: binder protruding (low monomer burial) plugging into concave receptor pocket
            b_protrude = 1.0 / (b_geom["burial_cb10"][b_intf] + 1.0) if b_intf.any() else torch.ones(1, device=dev)
            epi_curvature_match = float(b_protrude.mean().item()) * epi_concavity
        else:
            epi_concavity = float(t_props["concavity"].mean().item())
            epi_hydro_patch = 0.0
            epi_arom_patch = 0.0
            epi_consensus = 0.0
            epi_hydro_to_polar = 0.0
            epi_curvature_match = 0.0

        buns_b0 = float(comp_erased["buns_binder"][0].item())
        buns_t0 = float(comp_erased["buns_target"][0].item())
        buns_in_tot = float((comp_input_sc["buns_binder"][0] + comp_input_sc["buns_target"][0]).item())
        buns_tm_tot = float((comp_tmpl["buns_binder"][0] + comp_tmpl["buns_target"][0]).item())

        row["features"].update(
            {
                "gen_if_ll_all": gen_if_ll_all,
                "gen_if_ll_core": gen_if_ll_core,
                "gen_if_core_bayesian": gen_if_ll_core + b_esm_nll,
                "gen_if_ll_intf": gen_if_ll_intf,
                "gen_if_intf_minus_core": gen_if_intf_minus_core,
                "gen_if_worst_win5": gen_if_worst_win5,
                "gen_if_worst_intf_win5": gen_if_worst_intf_win5,
                "sc_intf_dist": sc_intf_dist,
                "sc_intf_minus_nonintf_dist": sc_intf_dist - sc_nonintf_dist,
                "sc_intf_z_strain": sc_intf_z_strain,
                "sc_intf_min_mode_dist": sc_intf_min_mode_dist,
                "sc_intf_mdn_nll": sc_intf_mdn_nll,
                "sc_intf_radial_strain": sc_intf_radial_strain,
                "sc_intf_angular_strain": sc_intf_angular_strain,
                "sc_intf_top3_dist": sc_intf_top3_dist,
                "sc_intf_top3_z_strain": sc_intf_top3_z_strain,
                "sc_core_weighted_dist": sc_core_weighted_dist,
                "sc_core_weighted_z": sc_core_weighted_z,
                "sc_target_intf_dist": sc_target_intf_dist,
                "sc_target_intf_z_strain": sc_target_intf_z_strain,
                "sc_intf_rot_entropy_cost": sc_intf_rot_entropy_cost,
                "sc_intf_dist_tmpl": sc_intf_dist_tmpl,
                "sc_intf_top3_dist_tmpl": sc_intf_top3_dist_tmpl,
                "sc_core_weighted_dist_tmpl": sc_core_weighted_dist_tmpl,
                "sc_target_intf_dist_tmpl": sc_target_intf_dist_tmpl,
                "mirage_energy_jump": mirage_energy_jump / n_intf_c,
                "mirage_energy_jump_tmpl": mirage_energy_jump_tmpl / n_intf_c,
                "mirage_clash_jump": mirage_clash_jump / n_intf_c,
                "mirage_hb_loss": mirage_hb_loss / n_intf_c,
                "gen_ebm_e0_erased": e0_erased,
                "gen_ebm_e0_input": e0_input,
                "gen_ebm_e0_tmpl": e0_tmpl,
                "gen_ebm_e0_per_contact": e0_erased / n_intf_c,
                "gen_ebm_neural_e0": float(comp_erased["neural"][0].item()),
                "gen_ebm_vdw_e0": float(comp_erased["vdw"][0].item()) / n_intf_c,
                "gen_ebm_hb_e0": float(comp_erased["hb"][0].item()) / n_intf_c,
                "gen_ebm_tip_hb_e0": float(comp_erased["tip_hb"][0].item()) / n_intf_c,
                "gen_ebm_desolv_e0": float(comp_erased["desolv"][0].item()) / n_intf_c,
                "gen_ebm_rot_ent_e0": float(comp_erased["rot_ent"][0].item()) / n_intf_c,
                "gen_ebm_buns_binder_e0": buns_b0 / n_intf_c,
                "gen_ebm_buns_target_e0": buns_t0 / n_intf_c,
                "gen_ebm_buns_total_e0": (buns_b0 + buns_t0) / n_intf_c,
                "gen_ebm_buns_total_input": buns_in_tot / n_intf_c,
                "gen_ebm_buns_total_tmpl": buns_tm_tot / n_intf_c,
                "alchem_seq_margin": alchem_seq_margin,
                "alchem_seq_zscore": alchem_seq_zscore,
                "alchem_buns_mirage_gap": alchem_buns_mirage_gap,
                "alchem_ala_scan_ddg": alchem_ala_scan_ddg,
                "alchem_seq_zscore_tmpl": alchem_seq_zscore_tmpl,
                "alchem_buns_mirage_gap_tmpl": alchem_buns_mirage_gap_tmpl,
                "redock_local_drift_rmsd": local_drift_rmsd,
                "redock_local_drift_rmsd_input": local_drift_rmsd_input,
                "redock_local_drift_rmsd_tmpl": local_drift_rmsd_tmpl,
                "redock_boltz_rmsd_t05_input": redock_boltz_rmsd_t05_input,
                "redock_p_native_t05_input": redock_p_native_t05_input,
                "redock_boltz_rmsd_t05_tmpl": redock_boltz_rmsd_t05_tmpl,
                "redock_p_native_t05_tmpl": redock_p_native_t05_tmpl,
                "redock_local_relax_drop": local_relax_drop,
                "redock_top1_rmsd": redock_top1_rmsd,
                "redock_global_margin": float(e_local.min().item()) - e_far_min,
                "redock_global_funnel_gap": float(e_local.min().item()) - e_far_mean,
                "redock_global_zscore": (float(e_local.min().item()) - e_far_mean) / e_far_std,
                "ss_helix_all": ss_helix_all,
                "ss_sheet_all": ss_sheet_all,
                "ss_helix_intf": ss_helix_intf,
                "ss_sheet_intf": ss_sheet_intf,
                "ss_loop_intf": ss_loop_intf,
                "ss_max_helix_intf_run": float(max_h_run),
                "epi_concavity": epi_concavity,
                "epi_hydro_patch": epi_hydro_patch,
                "epi_arom_patch": epi_arom_patch,
                "epi_consensus_density": epi_consensus,
                "epi_hydro_to_polar": epi_hydro_to_polar,
                "epi_curvature_match": epi_curvature_match,
                "holo_if_b_intf": holo_if_b_intf,
                "holo_minus_apo_if_b": holo_minus_apo_if_b,
                "holo_if_t_intf": holo_if_t_intf,
                "holo_minus_apo_if_t": holo_minus_apo_if_t,
                "holo_if_joint_gain": holo_if_joint_gain,
                "sc_holo_dist_b": sc_holo_dist_b,
                "sc_holo_minus_apo_shift_b": sc_holo_minus_apo_shift_b,
                "sc_holo_dist_t": sc_holo_dist_t,
                "sc_holo_ent_drop_b": sc_holo_ent_drop_b,
                "bb_force_norm_all": bb_force_norm_all,
                "bb_force_norm_intf": bb_force_norm_intf,
                "bb_force_intf_minus_core": bb_force_intf_minus_core,
                "bb_force_holo_b_intf": bb_force_holo_b_intf,
                "bb_force_holo_minus_apo_b": bb_force_holo_minus_apo_b,
                "bb_force_holo_minus_apo_t": bb_force_holo_minus_apo_t,
            }
        )

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(binders_cache):
            el = time.time() - t0
            print(f"  [{idx + 1:4d}/{len(binders_cache)}] ({(idx + 1) / el:.1f} complexes/s, {el:.1f}s)", flush=True)

    torch.save(feat_rows, OVERATH_FEAT_CACHE)
    print(f"Updated {OVERATH_FEAT_CACHE} with {len(feat_rows[0]['features'])} total features.", flush=True)
    return update_v6_3d_physics_features(if_model=if_model, sc_model=sc_model, ebm_model=ebm_model)


def update_v6_3d_physics_features(
    if_model: BackboneInverseFoldingGNN | None = None,
    sc_model: UnboundSidechainPredictorGNN | None = None,
    ebm_model: GeneralPPISubdomainEBM | None = None,
) -> list[dict]:
    """Compute v6 3D physical features (2-point bilateral-dielectric electrostatics, holo sidechain repacking EBM, and continuous 6-DoF autograd relaxation)."""
    t0 = time.time()
    if if_model is None or sc_model is None or ebm_model is None:
        if_model, sc_model, ebm_model = train_or_load_general_models(force_retrain=False)
    dev = torch.device("cpu")
    if_model.to(dev).eval()
    sc_model.to(dev).eval()
    ebm_model.to(dev).eval()
    phys_tbl = PHYSCHEM_TABLE.to(dev)

    cache_5b = torch.load(OVERATH_5BEAD_CACHE, map_location="cpu", weights_only=False)
    targets_cache = cache_5b["targets"]
    binders_cache = cache_5b["binders"]
    feat_rows = torch.load(OVERATH_FEAT_CACHE, map_location="cpu", weights_only=False)

    target_seq_cache = {}
    for b_item in binders_cache:
        tv = targets_cache[b_item["t_hash"]]
        seq_key = (b_item["target_id"], tuple(tv["aa"].tolist()))
        if seq_key not in target_seq_cache:
            t_beads0 = tv["beads"].to(torch.float32)
            t_aa0 = tv["aa"].to(torch.long)
            t_chain0 = tv["chain"].to(torch.long)
            with torch.no_grad():
                t_geom0 = compute_pure_backbone_geometry(t_beads0, t_chain0, torch.arange(len(t_beads0)))
                _, u_local_t, t_sigma, _, u_tip_local_t, t_sigma_tip = sc_model.forward_from_geom(t_geom0, t_aa0, return_tip=True)
                h_t = ebm_model.encode_partner(t_geom0, t_aa0)
            target_seq_cache[seq_key] = {
                "u_local_t": u_local_t,
                "t_sigma": t_sigma,
                "u_tip_local_t": u_tip_local_t,
                "t_sigma_tip": t_sigma_tip,
                "h_t": h_t,
            }

    I_rot = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    Z_tr = torch.zeros((1, 3), dtype=torch.float32)

    for idx, (d, row) in enumerate(zip(binders_cache, feat_rows)):
        b_beads = d["beads"].to(torch.float32)
        b_aa = d["aa"].to(torch.long)
        b_chain = d["chain"].to(torch.long)
        tv = targets_cache[d["t_hash"]]
        t_beads = tv["beads"].to(torch.float32)
        t_aa = tv["aa"].to(torch.long)
        seq_key = (d["target_id"], tuple(tv["aa"].tolist()))
        t_info = target_seq_cache[seq_key]

        d_ca0 = torch.cdist(b_beads[:, 1], t_beads[:, 1])
        t_near_idx = torch.where(d_ca0.amin(dim=0) < 15.0)[0]
        if t_near_idx.numel() < 5:
            t_near_idx = torch.argsort(d_ca0.amin(dim=0))[: min(25, len(t_beads))]

        t_b_n = t_beads[t_near_idx]
        t_a_n = t_aa[t_near_idx]
        h_t_n = t_info["h_t"][t_near_idx]
        R_tn = build_local_frames(t_b_n[:, 0], t_b_n[:, 1], t_b_n[:, 2])
        t_sc_n = t_b_n[:, 1] + torch.einsum("nij,nj->ni", R_tn, t_info["u_local_t"][t_near_idx])
        t_tip_n = t_b_n[:, 1] + torch.einsum("nij,nj->ni", R_tn, t_info["u_tip_local_t"][t_near_idx])
        t_sig_n = 0.5 * (t_info["t_sigma"][t_near_idx] + t_info["t_sigma_tip"][t_near_idx])
        t_erased_n = t_b_n.clone()
        t_erased_n[:, 4] = t_sc_n

        Nb = len(b_beads)
        Nt_n = len(t_near_idx)
        with torch.no_grad():
            b_geom = compute_pure_backbone_geometry(b_beads, b_chain, torch.arange(Nb))
            b_sc_hat, _, b_sigma, b_tip_hat, _, b_sigma_tip = sc_model.forward_from_geom(b_geom, b_aa, return_tip=True)
            b_sig_comb = 0.5 * (b_sigma + b_sigma_tip)
            b_erased = b_beads.clone()
            b_erased[:, 4] = b_sc_hat
            h_b = ebm_model.encode_partner(b_geom, b_aa)

            comp_aa = torch.cat([b_aa, t_a_n], dim=0)
            comp_chain = torch.cat([torch.zeros(Nb, dtype=torch.long), torch.ones(Nt_n, dtype=torch.long)], dim=0)
            comp_ridx = torch.cat([torch.arange(Nb), t_near_idx + 1000], dim=0)
            comp_beads = torch.cat([b_beads, t_b_n], dim=0)
            cg = compute_pure_backbone_geometry(comp_beads, comp_chain, comp_ridx)
            sc_comp, _, sig_comp, tip_comp, _, sig_tip_comp = sc_model.forward_from_geom(cg, comp_aa, return_tip=True)
            sig_comp_comb = 0.5 * (sig_comp + sig_tip_comp)

            b_holo = b_beads.clone()
            b_holo[:, 4] = sc_comp[:Nb]
            t_holo_n = t_b_n.clone()
            t_holo_n[:, 4] = sc_comp[Nb:]

            c_apo = ebm_model.score_poses(
                b_erased, b_aa, h_b, t_erased_n, t_a_n, h_t_n, I_rot, Z_tr,
                return_components=True, b_tip_erased=b_tip_hat, t_tip_erased=t_tip_n,
                b_sigma=b_sig_comb, t_sigma=t_sig_n,
            )
            c_holo = ebm_model.score_poses(
                b_holo, b_aa, h_b, t_holo_n, t_a_n, h_t_n, I_rot, Z_tr,
                return_components=True, b_tip_erased=tip_comp[:Nb], t_tip_erased=tip_comp[Nb:],
                b_sigma=sig_comp_comb[:Nb], t_sigma=sig_comp_comb[Nb:],
            )

            b_phys = phys_tbl[b_aa.clamp(0, 20)]
            t_phys = phys_tbl[t_a_n.clamp(0, 20)]
            d_sc = torch.cdist(b_sc_hat, t_sc_n)
            d_tip = torch.cdist(b_tip_hat, t_tip_n)
            b_bur = (0.5 * torch.sigmoid((5.4 - d_sc) / 0.6) + 0.5 * torch.sigmoid((4.8 - d_tip) / 0.5)).sum(dim=1)
            t_bur = (0.5 * torch.sigmoid((5.4 - d_sc) / 0.6) + 0.5 * torch.sigmoid((4.8 - d_tip) / 0.5)).sum(dim=0)
            p_diel = 0.55 + 0.90 * torch.sigmoid(0.5 * (b_bur[:, None] + t_bur[None, :]) - 1.2)
            q_ij = b_phys[:, 0][:, None] * t_phys[:, 0][None, :]
            d_ch = 0.5 * (d_sc + d_tip)
            coul_ij = p_diel * q_ij / (d_ch.clamp_min(2.8) * (1.0 + 0.25 * d_ch.clamp_min(2.8))) * (d_ch < 8.5).to(torch.float32)
            elec_attr = float(F.relu(-coul_ij).sum().item())
            elec_repul = float(F.relu(coul_ij).sum().item())
            elec_repul_max = float(F.relu(coul_ij).max().item()) if coul_ij.numel() > 0 else 0.0

        with torch.enable_grad():
            w_vec = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
            t_vec = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
            R_pose = _axis_angle_to_matrix(w_vec)
            e0_g = ebm_model.score_poses(
                b_erased, b_aa, h_b, t_erased_n, t_a_n, h_t_n, R_pose, t_vec,
                return_components=False, b_tip_erased=b_tip_hat, t_tip_erased=t_tip_n,
                b_sigma=b_sig_comb, t_sigma=t_sig_n,
            )[0]
            g_w, g_t = torch.autograd.grad(e0_g, [w_vec, t_vec])
            g_w = g_w[0].detach()
            g_t = g_t[0].detach()

        with torch.no_grad():
            f_tr_norm = float(torch.norm(g_t).item())
            f_rot_norm = float(torch.norm(g_w).item())
            b_intf = d_ca0.amin(dim=1) < 10.0
            p_b = b_beads[b_intf, 1].mean(dim=0) if b_intf.any() else b_beads[:, 1].mean(dim=0)
            p_t = t_b_n[:, 1].mean(dim=0)
            u_norm = F.normalize(p_b - p_t, dim=-1, eps=1e-6)
            norm_repul = float((-torch.dot(g_t, u_norm)).item())

            u_t_step = -g_t / (torch.norm(g_t) + 1e-6)
            u_w_step = -g_w / (torch.norm(g_w) + 1e-6)
            alphas = torch.tensor([0.20, 0.45, 0.85], dtype=torch.float32)
            t_steps = alphas[:, None] * u_t_step[None, :]
            w_steps = (0.35 * alphas)[:, None] * u_w_step[None, :]
            R_steps = _axis_angle_to_matrix(w_steps)
            e_steps = ebm_model.score_poses(
                b_erased, b_aa, h_b, t_erased_n, t_a_n, h_t_n, R_steps, t_steps,
                return_components=False, b_tip_erased=b_tip_hat, t_tip_erased=t_tip_n,
                b_sigma=b_sig_comb, t_sigma=t_sig_n,
            )
            e_holo_steps = ebm_model.score_poses(
                b_holo, b_aa, h_b, t_holo_n, t_a_n, h_t_n, R_steps, t_steps,
                return_components=False, b_tip_erased=tip_comp[:Nb], t_tip_erased=tip_comp[Nb:],
                b_sigma=sig_comp_comb[:Nb], t_sigma=sig_comp_comb[Nb:],
            )
            e0_val = float(c_apo["total"][0].item())
            e_holo_val = float(c_holo["total"][0].item())
            e_min_step = float(min(e0_val, float(e_steps.min().item())))
            e_holo_min_step = float(min(e_holo_val, float(e_holo_steps.min().item())))
            cont_relax_drop = e0_val - e_min_step

        n_intf_c = float((d_ca0 < 8.5).sum().item()) + 1.0
        row["features"].update(
            {
                "gen_ebm_elec_e0": float(c_apo["elec"][0].item()),
                "gen_ebm_elec_pc": float(c_apo["elec"][0].item()) / n_intf_c,
                "gen_ebm_elec_attr_e0": elec_attr,
                "gen_ebm_elec_attr_pc": elec_attr / n_intf_c,
                "gen_ebm_elec_repul_e0": elec_repul,
                "gen_ebm_elec_repul_pc": elec_repul / n_intf_c,
                "gen_ebm_elec_repul_max": elec_repul_max,
                "ebm_holo_e0": e_holo_val,
                "ebm_holo_e0_pc": e_holo_val / n_intf_c,
                "ebm_holo_sc_gain_raw": e0_val - e_holo_val,
                "ebm_holo_sc_gain": (e0_val - e_holo_val) / n_intf_c,
                "ebm_holo_hb_gain": float((c_apo["hb"][0] - c_holo["hb"][0]).item()) / n_intf_c,
                "ebm_holo_clash_drop": float((c_apo["clash"][0] - c_holo["clash"][0]).item()) / n_intf_c,
                "ebm_holo_buns_drop": float((c_apo["desolv"][0] - c_holo["desolv"][0]).item()) / n_intf_c,
                "ebm_force_tr_norm": f_tr_norm,
                "ebm_force_rot_norm": f_rot_norm,
                "ebm_normal_repulsion_force": norm_repul,
                "ebm_normal_repulsion_pc": norm_repul / n_intf_c,
                "ebm_cont_relax_drop": cont_relax_drop,
                "ebm_cont_relax_emin": e_min_step,
                "ebm_cont_relax_emin_pc": e_min_step / n_intf_c,
                "ebm_holo_cont_relax_emin": e_holo_min_step,
                "ebm_holo_cont_relax_emin_pc": e_holo_min_step / n_intf_c,
            }
        )

    torch.save(feat_rows, OVERATH_FEAT_CACHE)
    print(
        f"Updated {OVERATH_FEAT_CACHE} with v6 3D physics ({len(feat_rows[0]['features'])} total features) in {time.time() - t0:.1f}s.",
        flush=True,
    )
    return feat_rows


if __name__ == "__main__":
    extract_sidechain_erased_redocking_features(force_retrain=True)
