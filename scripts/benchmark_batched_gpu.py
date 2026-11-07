"""Batched End-to-End NVIDIA A100 GPU Benchmark for the Locked v7 BC-EBM Pipeline.

Executes all 5 passes of the locked v7 BC-EBM feature extraction pipeline in batched
GPU mode on an NVIDIA A100 GPU:
  - Parse: Multi-threaded 5-bead PDB coordinate extraction + host-to-device transfer
  - Pass 1 (Stage A): Batched ESM-2 8M frustration + disjoint-batch MonomerFrameGNN +
    batched 16-pose surface-packed ray-casting & UnboundOrientationPairwiseEBM scoring
  - Pass 2 (Stage B): Disjoint-batch 3D Inverse-Folding GNN (apo + holo) + 3D-batched
    5-bead monomer physics, interface desolvation, O-ring, and 11-pose mechanical stiffness
  - Pass 3 (Stage C1/C2): Disjoint-batch 2-point sidechain & K=3 rotamer MDN GNN +
    disjoint-batch GeneralPPISubdomainEBM partner encoder + batched 24-pose re-docking
    ensemble & alchemical counterfactual scoring
  - Pass 4 (Stage C2 autograd): Disjoint-batch pure backbone geometry + 3D IF GNN forward
    + single batched autograd backward pass for apo & holo backbone restoring forces
  - Pass 5 (Stage C3): Batched continuous 6-DoF rigid-body autograd gradient + batched
    4-pose holo line-search relaxation
"""
from __future__ import annotations

import gzip
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

import subprocess

if Path("/content/bcebm_code_ckpts.tar.gz").exists() and not Path("/content/scripts").exists():
    subprocess.run("tar -xzf /content/bcebm_code_ckpts.tar.gz -C /content", shell=True, check=True)
if Path("/content/bcebm_part_aa").exists() and not Path("/content/bcebm_compact.pkl.gz").exists():
    subprocess.run("cat /content/bcebm_part_a* > /content/bcebm_compact.pkl.gz", shell=True, check=True)

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path("/content/scripts")
PROJECT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if "/content/scripts" not in sys.path and Path("/content/scripts").exists():
    sys.path.insert(0, "/content/scripts")
if "/content" not in sys.path and Path("/content/scripts").exists():
    sys.path.insert(0, "/content")

from esm2_frustration import MODEL_ID, _AA1_LIST
from stage_c_sidechain_ebm import (
    _axis_angle_to_matrix,
    build_local_frames,
    construct_ideal_virtual_cb,
    train_or_load_general_models,
)
from stage_b_foldability import (
    _dihedral_sin_cos,
    train_or_load_backbone_inverse_folding_gnn,
)
from stage_a_orientation import (
    NUM_REGIONS,
    PHYSCHEM_TABLE,
    REGION_AB_CDR1,
    REGION_AB_CDR2,
    REGION_AB_CDRH3,
    REGION_AB_CDRL3,
    REGION_AB_CONST,
    REGION_AB_VFW,
    UnboundOrientationPairwiseEBM,
    compute_antibody_paratope_geometry,
    extract_monomer_geometry,
    rbf_encode,
)
from pretrain_orientation_ebm import axis_angle_to_matrix
AA3 = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, "GLN": 5, "GLU": 6,
    "GLY": 7, "HIS": 8, "ILE": 9, "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13,
    "PRO": 14, "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}


def compute_pure_backbone_geometry_disjoint(
    beads: torch.Tensor,
    chain: torch.Tensor,
    residue_index: torch.Tensor,
    batch_id: torch.Tensor,
    cutoff: float = 12.0,
) -> dict[str, torch.Tensor]:
    """Exact disjoint-batch version of compute_pure_backbone_geometry on GPU."""
    n_pos = beads[:, 0]
    ca_pos = beads[:, 1]
    c_pos = beads[:, 2]
    o_pos = beads[:, 3]
    vcb_pos = construct_ideal_virtual_cb(n_pos, ca_pos, c_pos)
    N = len(ca_pos)
    dtype = ca_pos.dtype
    dev = ca_pos.device

    frames = build_local_frames(n_pos, ca_pos, c_pos)
    u_vcb = F.normalize(vcb_pos - ca_pos, dim=-1, eps=1e-6)

    phi_sin = torch.zeros((N,), dtype=dtype, device=dev)
    phi_cos = torch.zeros((N,), dtype=dtype, device=dev)
    psi_sin = torch.zeros((N,), dtype=dtype, device=dev)
    psi_cos = torch.zeros((N,), dtype=dtype, device=dev)
    omg_sin = torch.zeros((N,), dtype=dtype, device=dev)
    omg_cos = torch.zeros((N,), dtype=dtype, device=dev)

    if N >= 2:
        same_prev = (
            (batch_id[1:] == batch_id[:-1])
            & (chain[1:] == chain[:-1])
            & ((residue_index[1:] - residue_index[:-1]) == 1)
        )
        idx_prev = torch.nonzero(same_prev, as_tuple=True)[0] + 1
        if idx_prev.numel() > 0:
            s_phi, c_phi = _dihedral_sin_cos(c_pos[idx_prev - 1], n_pos[idx_prev], ca_pos[idx_prev], c_pos[idx_prev])
            phi_sin[idx_prev] = s_phi
            phi_cos[idx_prev] = c_phi

            s_omg, c_omg = _dihedral_sin_cos(ca_pos[idx_prev - 1], c_pos[idx_prev - 1], n_pos[idx_prev], ca_pos[idx_prev])
            omg_sin[idx_prev] = s_omg
            omg_cos[idx_prev] = c_omg

            s_psi, c_psi = _dihedral_sin_cos(n_pos[idx_prev - 1], ca_pos[idx_prev - 1], c_pos[idx_prev - 1], n_pos[idx_prev])
            psi_sin[idx_prev - 1] = s_psi
            psi_cos[idx_prev - 1] = c_psi

    u_nc = F.normalize(n_pos - ca_pos, dim=-1, eps=1e-6)
    u_cc = F.normalize(c_pos - ca_pos, dim=-1, eps=1e-6)
    tau_cos = (u_nc * u_cc).sum(dim=-1)

    curv_cos = torch.zeros((N,), dtype=dtype, device=dev)
    if N >= 3:
        v1 = F.normalize(ca_pos[1:-1] - ca_pos[:-2], dim=-1, eps=1e-6)
        v2 = F.normalize(ca_pos[2:] - ca_pos[1:-1], dim=-1, eps=1e-6)
        valid_mid = (
            (batch_id[1:-1] == batch_id[:-2])
            & (batch_id[2:] == batch_id[1:-1])
            & (chain[1:-1] == chain[:-2])
            & (chain[2:] == chain[1:-1])
        )
        curv_cos[1:-1] = torch.where(valid_mid, (v1 * v2).sum(dim=-1), torch.zeros_like(curv_cos[1:-1]))

    with torch.no_grad():
        same_struct = batch_id[:, None] == batch_id[None, :]
        d_no_ng = torch.cdist(n_pos.detach(), o_pos.detach())
        seq_sep_signed = residue_index[:, None] - residue_index[None, :]
        same_ch_mat = same_struct & (chain[:, None] == chain[None, :])

        def _hb_argmin_at_offset(offset: int):
            m = same_ch_mat & (seq_sep_signed == offset)
            d_m = torch.where(m, d_no_ng, 15.0)
            val, j_idx = d_m.min(dim=-1)
            return j_idx, val < 14.999

        j_m4, v_m4 = _hb_argmin_at_offset(4)
        j_m3, v_m3 = _hb_argmin_at_offset(3)
        j_p4, v_p4 = _hb_argmin_at_offset(-4)
        j_p3, v_p3 = _hb_argmin_at_offset(-3)

        long_mask = same_struct & ((~same_ch_mat) | (seq_sep_signed.abs() >= 5))
        d_long = torch.where(long_mask, d_no_ng, 15.0)
        val_ln, j_ln = d_long.min(dim=1)
        val_lo, i_lo = d_long.min(dim=0)
        v_ln = val_ln < 14.999
        v_lo = val_lo < 14.999
        del d_no_ng, d_long, long_mask, same_ch_mat, seq_sep_signed

        ca_ng = ca_pos.detach()
        u_vcb_ng = u_vcb.detach()
        eye_or_diff = (~same_struct) | torch.eye(N, dtype=torch.bool, device=dev)
        dist_ca_m = torch.cdist(ca_ng, ca_ng).masked_fill(eye_or_diff, 1e6)
        dist_vcb_m = torch.cdist(vcb_pos.detach(), vcb_pos.detach()).masked_fill(eye_or_diff, 1e6)
        del eye_or_diff, same_struct

        b_ca6 = (dist_ca_m < 6.0).to(dtype).sum(dim=-1) / 8.0
        b_ca8 = (dist_ca_m < 8.0).to(dtype).sum(dim=-1) / 15.0
        b_ca10 = (dist_ca_m < 10.0).to(dtype).sum(dim=-1) / 25.0
        b_ca12 = (dist_ca_m < 12.0).to(dtype).sum(dim=-1) / 40.0

        b_cb6 = (dist_vcb_m < 6.0).to(dtype).sum(dim=-1) / 8.0
        b_cb8 = (dist_vcb_m < 8.0).to(dtype).sum(dim=-1) / 15.0
        b_cb10 = (dist_vcb_m < 10.0).to(dtype).sum(dim=-1) / 25.0
        b_cb12 = (dist_vcb_m < 12.0).to(dtype).sum(dim=-1) / 40.0

        src, dst = torch.nonzero(dist_ca_m < cutoff, as_tuple=True)
        d_ca_sub = dist_ca_m[src, dst]
        d_vcb_sub = dist_vcb_m[src, dst]
        del dist_ca_m, dist_vcb_m
        dir_sub = F.normalize(ca_ng[dst] - ca_ng[src], dim=-1, eps=1e-6)
        dot_sub = (dir_sub * u_vcb_ng[src]).sum(dim=-1)

        in8_e = d_ca_sub < 8.0
        hse_up8 = torch.zeros((N,), dtype=dtype, device=dev).index_add_(0, src, (in8_e & (dot_sub > 0.0)).to(dtype)) / 10.0
        hse_dn8 = torch.zeros((N,), dtype=dtype, device=dev).index_add_(0, src, (in8_e & (dot_sub <= 0.0)).to(dtype)) / 10.0
        hse_up12 = torch.zeros((N,), dtype=dtype, device=dev).index_add_(0, src, (dot_sub > 0.0).to(dtype)) / 20.0
        hse_dn12 = torch.zeros((N,), dtype=dtype, device=dev).index_add_(0, src, (dot_sub <= 0.0).to(dtype)) / 20.0
        protrusion = (hse_dn12 - hse_up12) / (hse_dn12 + hse_up12 + 0.1)
        cone_crowd = torch.zeros((N,), dtype=dtype, device=dev).index_add_(0, src, ((d_vcb_sub < 7.0) & (dot_sub > 0.5)).to(dtype)) / 6.0

        is_n_term = torch.ones((N,), dtype=dtype, device=dev)
        is_c_term = torch.ones((N,), dtype=dtype, device=dev)
        if N >= 2:
            same_adj = (
                (batch_id[1:] == batch_id[:-1])
                & (chain[1:] == chain[:-1])
                & ((residue_index[1:] - residue_index[:-1]).abs() == 1)
            )
            is_n_term[1:] = (~same_adj).to(dtype)
            is_c_term[:-1] = (~same_adj).to(dtype)

    def _eval_hb(j_idx: torch.Tensor, valid_m: torch.Tensor, mu: float, sig: float) -> torch.Tensor:
        d = torch.linalg.norm(n_pos - o_pos[j_idx], dim=-1)
        d = torch.where(valid_m, d, torch.full_like(d, 15.0))
        return torch.exp(-0.5 * ((d - mu) / sig) ** 2)

    hb_i_minus_4 = _eval_hb(j_m4, v_m4, 2.95, 0.45)
    hb_i_minus_3 = _eval_hb(j_m3, v_m3, 2.95, 0.45)
    hb_i_plus_4 = _eval_hb(j_p4, v_p4, 2.95, 0.45)
    hb_i_plus_3 = _eval_hb(j_p3, v_p3, 2.95, 0.45)

    d_no_long_n = torch.where(v_ln, torch.linalg.norm(n_pos - o_pos[j_ln], dim=-1), torch.full((N,), 15.0, dtype=dtype, device=dev))
    d_no_long_o = torch.where(v_lo, torch.linalg.norm(n_pos[i_lo] - o_pos, dim=-1), torch.full((N,), 15.0, dtype=dtype, device=dev))
    hb_sheet_n = torch.exp(-0.5 * ((d_no_long_n - 2.90) / 0.45) ** 2)
    hb_sheet_o = torch.exp(-0.5 * ((d_no_long_o - 2.90) / 0.45) ** 2)

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
    )

    if src.numel() > 0:
        d_ca_e = torch.linalg.norm(ca_pos[src] - ca_pos[dst], dim=-1)
        d_cb_e = torch.linalg.norm(vcb_pos[src] - vcb_pos[dst], dim=-1)
        rbf_ca = rbf_encode(d_ca_e, 0.0, cutoff, 16)
        rbf_cb = rbf_encode(d_cb_e, 0.0, cutoff, 8)
        rel_vec = F.normalize(ca_pos[dst] - ca_pos[src], dim=-1, eps=1e-6)
        dir_i = torch.einsum("nij,ni->nj", frames[src], rel_vec)
        dir_j = torch.einsum("nij,ni->nj", frames[dst], -rel_vec)
        q_ij = torch.einsum("nij,nik->njk", frames[src], frames[dst])
        rot_6d = q_ij[:, :, :2].reshape(-1, 6)

        same_chain = (batch_id[src] == batch_id[dst]) & (chain[src] == chain[dst])
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
        edge_attr = torch.cat([rbf_ca, rbf_cb, dir_i, dir_j, rot_6d, seq_feats], dim=-1)
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


@torch.no_grad()
def extract_esm_batched_gpu(
    seq_items: list[tuple[tuple, torch.Tensor, torch.Tensor]],
    tok,
    esm,
    aa_tok_ids_t: torch.Tensor,
    phys_tbl_dev: torch.Tensor,
    dev: torch.device,
    batch_size: int = 64,
) -> dict[tuple, dict[str, torch.Tensor]]:
    """Exact batched GPU version of extract_partner_esm across unique (aa, chain) sequences."""
    sub_tasks = []
    results = {}
    for key, aa, chain in seq_items:
        n_res = len(aa)
        aa_list = aa.tolist()
        ch_list = chain.tolist()
        seq_chars = [_AA1_LIST[i] if i < 20 else "X" for i in aa_list]
        results[key] = {
            "esm_hidden": torch.zeros((n_res, 320), dtype=torch.float16, device=dev),
            "esm_frust": torch.zeros((n_res, 6), dtype=torch.float32, device=dev),
        }
        start = 0
        while start < n_res:
            end = start + 1
            while end < n_res and ch_list[end] == ch_list[start]:
                end += 1
            sub_seq = "".join(seq_chars[start:end])
            sub_tasks.append((key, start, end, sub_seq, aa[start:end].to(dev)))
            start = end

    sub_tasks.sort(key=lambda x: len(x[3]))
    abs_q = phys_tbl_dev[:, 0].abs()

    for b_start in range(0, len(sub_tasks), batch_size):
        batch = sub_tasks[b_start : b_start + batch_size]
        seqs = [x[3] for x in batch]
        inputs = tok(seqs, return_tensors="pt", padding=True, add_special_tokens=True).to(dev)
        out = esm(**inputs, output_hidden_states=True)
        h_all = out.hidden_states[-1].to(torch.float16)
        logits_all = out.logits[:, :, aa_tok_ids_t]
        probs_all = F.softmax(logits_all, dim=-1)
        log_probs_all = F.log_softmax(logits_all, dim=-1)
        ent_all = -(probs_all * log_probs_all).sum(dim=-1)
        exp_phys_all = probs_all @ phys_tbl_dev
        exp_abs_q_all = probs_all @ abs_q

        for i, (key, start, end, sub_seq, aa_sub) in enumerate(batch):
            L = end - start
            h_sub = h_all[i, 1 : 1 + L]
            lp_sub = log_probs_all[i, 1 : 1 + L]
            ent_sub = ent_all[i, 1 : 1 + L]
            ep_sub = exp_phys_all[i, 1 : 1 + L]
            eq_sub = exp_abs_q_all[i, 1 : 1 + L]

            valid_aa = aa_sub < 20
            aa_c = aa_sub.clamp(0, 19)
            nll = torch.where(
                valid_aa,
                -lp_sub[torch.arange(L, device=dev), aa_c],
                torch.full((L,), 3.0, dtype=torch.float32, device=dev),
            )
            act_p = torch.where(
                valid_aa.unsqueeze(-1),
                phys_tbl_dev[aa_c],
                torch.zeros((L, 6), dtype=torch.float32, device=dev),
            )
            frust_sub = torch.stack(
                [
                    nll,
                    ent_sub,
                    act_p[:, 1] - ep_sub[:, 1],
                    act_p[:, 4] - ep_sub[:, 4],
                    act_p[:, 0].abs() - eq_sub,
                    ep_sub[:, 1],
                ],
                dim=-1,
            )
            results[key]["esm_hidden"][start:end] = h_sub
            results[key]["esm_frust"][start:end] = frust_sub

    return results


@torch.no_grad()
def compute_monomer_features_disjoint_a(
    ebm_a: UnboundOrientationPairwiseEBM,
    beads: torch.Tensor,
    cb: torch.Tensor,
    aa: torch.Tensor,
    role: torch.Tensor,
    region: torch.Tensor,
    chain: torch.Tensor,
    residue_index: torch.Tensor,
    batch_id: torch.Tensor,
):
    """Exact disjoint-batch version of ebm_a._compute_monomer_features on GPU."""
    n_pos = beads[:, 0]
    ca_pos = beads[:, 1]
    c_pos = beads[:, 2]
    o_pos = beads[:, 3]
    sc_pos = beads[:, 4]
    N = len(ca_pos)
    dev = ca_pos.device
    dtype = ca_pos.dtype

    frames = build_local_frames(n_pos, ca_pos, c_pos)
    sc_vec = F.normalize(sc_pos - ca_pos, dim=-1, eps=1e-6)
    cb_vec = F.normalize(cb - ca_pos, dim=-1, eps=1e-6)
    o_vec = F.normalize(o_pos - c_pos, dim=-1, eps=1e-6)
    sc_local = torch.einsum("nij,ni->nj", frames, sc_pos - ca_pos) / 5.0

    same_struct = batch_id[:, None] == batch_id[None, :]
    dist_ca = torch.cdist(ca_pos, ca_pos)
    eye_or_diff = (~same_struct) | torch.eye(N, device=dev, dtype=torch.bool)
    dist_masked = dist_ca.masked_fill(eye_or_diff, 1e6)

    burial_6 = (dist_masked < 6.0).sum(dim=-1, keepdim=True).to(dtype) / 10.0
    burial_8 = (dist_masked < 8.0).sum(dim=-1, keepdim=True).to(dtype) / 20.0
    burial_12 = (dist_masked < 12.0).sum(dim=-1, keepdim=True).to(dtype) / 50.0

    src, dst = torch.nonzero(dist_masked < ebm_a.monomer_cutoff, as_tuple=True)
    rel_unit_sub = F.normalize(ca_pos[dst] - ca_pos[src], dim=-1, eps=1e-6)
    dot_cb_sub = (rel_unit_sub * cb_vec[src]).sum(dim=-1)
    hse_up = torch.zeros((N, 1), device=dev, dtype=dtype).index_add_(0, src, (dot_cb_sub > 0.0).to(dtype).unsqueeze(-1)) / 25.0
    hse_down = torch.zeros((N, 1), device=dev, dtype=dtype).index_add_(0, src, (dot_cb_sub <= 0.0).to(dtype).unsqueeze(-1)) / 25.0
    protrusion = (hse_down - hse_up) / (hse_down + hse_up + 0.1)

    curvature = torch.zeros((N, 1), device=dev, dtype=dtype)
    if N >= 3:
        same_prev = (batch_id[1:-1] == batch_id[:-2]) & (chain[1:-1] == chain[:-2])
        same_next = (batch_id[1:-1] == batch_id[2:]) & (chain[1:-1] == chain[2:])
        valid_turn = (same_prev & same_next).to(dtype).unsqueeze(-1)
        sec_diff = ca_pos[2:] - 2.0 * ca_pos[1:-1] + ca_pos[:-2]
        curvature[1:-1] = (torch.linalg.norm(sec_diff, dim=-1, keepdim=True) / 4.0).clamp(0.0, 2.0) * valid_turn

    aa_clamped = aa.clamp(0, 20)
    aa_oh = F.one_hot(aa_clamped, 21).to(dtype)
    role_oh = F.one_hot(role.clamp(0, 1), 2).to(dtype)
    region_oh = F.one_hot(region.clamp(0, NUM_REGIONS - 1), NUM_REGIONS).to(dtype)
    physchem = PHYSCHEM_TABLE.to(dev)[aa_clamped]

    node_feats = torch.cat(
        [aa_oh, role_oh, region_oh, physchem, sc_local, burial_6, burial_8, burial_12, hse_up, hse_down, protrusion, curvature],
        dim=-1,
    )
    h = ebm_a.node_in(node_feats)

    if src.numel() > 0:
        d_ca = dist_ca[src, dst]
        d_sc = torch.linalg.norm(sc_pos[dst] - sc_pos[src], dim=-1)
        rbf_ca = rbf_encode(d_ca, 0.0, ebm_a.monomer_cutoff, 16)
        rbf_sc = rbf_encode(d_sc, 0.0, ebm_a.monomer_cutoff, 8)
        dir_i = torch.einsum("nij,ni->nj", frames[src], rel_unit_sub)
        dir_j = torch.einsum("nij,ni->nj", frames[dst], -rel_unit_sub)
        q_ij = torch.einsum("nij,nik->njk", frames[src], frames[dst])
        rot_6d = q_ij[:, :, :2].reshape(-1, 6)

        same_chain = (batch_id[src] == batch_id[dst]) & (chain[src] == chain[dst])
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
        edge_attr = torch.cat([rbf_ca, rbf_sc, dir_i, dir_j, rot_6d, seq_feats], dim=-1)
        for layer in ebm_a.monomer_layers:
            h = layer(h, src, dst, edge_attr)

    return h, frames, sc_vec, o_vec, protrusion.squeeze(-1)


@torch.no_grad()
def generate_surface_packed_sabdab2_pool_batched_gpu(
    rec_beads_list: list[torch.Tensor],
    rec_region_list: list[torch.Tensor],
    lig_beads_list: list[torch.Tensor],
    rng: np.random.Generator,
    n_poses: int = 16,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Batched GPU version of generate_surface_packed_sabdab2_pool across M complexes."""
    dev = rec_beads_list[0].device
    dtype = torch.float32
    M = len(rec_beads_list)
    n_near = 6
    n_cdr_decoy = n_poses - 1 - n_near - 6
    decoy_start = n_near + 1
    b_dec = n_poses - decoy_start

    # Pre-draw RNG values in exact complex & pose order on CPU
    rots_np = np.zeros((M, n_poses, 3, 3), dtype=np.float32)
    rots_np[:, 0] = np.eye(3, dtype=np.float32)
    shifts_np = np.zeros((M, n_poses, 3), dtype=np.float32)
    target_d_np = np.zeros((M, b_dec), dtype=np.float32)

    for m in range(M):
        for p in range(1, n_poses):
            axis = rng.normal(size=3).astype(np.float32)
            axis /= max(np.linalg.norm(axis), 1e-6)
            if p <= n_near:
                ang = math.radians(float(rng.uniform(2.5, 7.8)))
                vec = rng.normal(0.0, 0.50, size=3).astype(np.float32)
            elif p <= n_near + n_cdr_decoy:
                ang = math.radians(float(rng.uniform(32.0, 178.0)))
                vec = rng.normal(0.0, 3.2, size=3).astype(np.float32)
            else:
                ang = math.radians(float(rng.uniform(45.0, 180.0)))
                vec = rng.normal(0.0, 4.0, size=3).astype(np.float32)
            rots_np[m, p] = axis_angle_to_matrix(axis, ang)
            shifts_np[m, p] = vec
        target_d_np[m] = rng.uniform(3.42, 4.12, size=b_dec).astype(np.float32)

    rots_all = torch.from_numpy(rots_np).to(dev)
    shifts_all = torch.from_numpy(shifts_np).to(dev)
    target_d_all = torch.from_numpy(target_d_np).to(dev)

    rots_out = []
    trans_out = []
    for m in range(M):
        rec_beads = rec_beads_list[m]
        rec_region = rec_region_list[m]
        lig_beads0 = lig_beads_list[m]
        rec_ca = rec_beads[:, 1]
        lig_ca = lig_beads0[:, 1]

        ag_center = lig_ca.mean(dim=0)
        cdr_mask = (
            (rec_region == REGION_AB_CDR1)
            | (rec_region == REGION_AB_CDR2)
            | (rec_region == REGION_AB_CDRL3)
            | (rec_region == REGION_AB_CDRH3)
        )
        cdr_center = rec_ca[cdr_mask].mean(dim=0) if cdr_mask.any() else rec_ca.mean(dim=0)
        fw_mask = ~cdr_mask
        fw_center = rec_ca[fw_mask].mean(dim=0) if fw_mask.any() else rec_ca.mean(dim=0)
        u_axis = F.normalize(ag_center - cdr_center, dim=-1, eps=1e-6)

        rots_t = rots_all[m]
        sh_m = shifts_all[m]
        rot_ag = torch.einsum("pij,j->pi", rots_t, ag_center)

        trans_t = torch.zeros((n_poses, 3), dtype=dtype, device=dev)
        # Near poses 1..6
        trans_t[1 : n_near + 1] = ag_center[None, :] - rot_ag[1 : n_near + 1] + sh_m[1 : n_near + 1]
        # CDR decoys 7..9
        lat_cdr = sh_m[n_near + 1 : n_near + 1 + n_cdr_decoy]
        lat_cdr = lat_cdr - (lat_cdr * u_axis[None, :]).sum(dim=-1, keepdim=True) * u_axis[None, :]
        trans_t[n_near + 1 : n_near + 1 + n_cdr_decoy] = cdr_center[None, :] + lat_cdr - rot_ag[n_near + 1 : n_near + 1 + n_cdr_decoy]
        # FW decoys 10..15
        lat_fw = sh_m[n_near + 1 + n_cdr_decoy :]
        trans_t[n_near + 1 + n_cdr_decoy :] = fw_center[None, :] + lat_fw - rot_ag[n_near + 1 + n_cdr_decoy :]

        if b_dec > 0:
            rots_d = rots_t[decoy_start:]
            trans_d = trans_t[decoy_start:]
            rec_2b = rec_beads[:, [1, 4], :]
            lig_2b = torch.einsum("nki,bji->bnkj", lig_beads0[:, [1, 4], :], rots_d) + trans_d[:, None, None, :]
            rec_ca_perp = rec_ca - (rec_ca @ u_axis)[:, None] * u_axis
            lig_ca_perp = lig_2b[:, :, 0] - torch.einsum("bni,i->bn", lig_2b[:, :, 0], u_axis)[:, :, None] * u_axis
            d_ca_perp = torch.cdist(rec_ca_perp.unsqueeze(0).expand(b_dec, -1, -1), lig_ca_perp)

            b_idx, r_idx, l_idx = torch.nonzero(d_ca_perp < 6.8, as_tuple=True)
            if len(b_idx) > 0:
                ab_b = rec_2b[r_idx]
                ag_b = lig_2b[b_idx, l_idx]
                diff = ag_b[:, None, :, :] - ab_b[:, :, None, :]
                z_diff = torch.einsum("pijk,k->pij", diff, u_axis)
                r_perp_sq = (diff ** 2).sum(dim=-1) - z_diff ** 2

                target_d = target_d_all[m]
                td_pair = target_d[b_idx][:, None, None]
                hit_mask = r_perp_sq < (td_pair ** 2)
                s_req = torch.where(
                    hit_mask,
                    -z_diff + torch.sqrt((td_pair ** 2 - r_perp_sq).clamp_min(0.0)),
                    torch.full_like(z_diff, -1e4),
                )
                s_pair = s_req.amax(dim=(1, 2))
                s_pose = torch.full((b_dec,), -1e4, dtype=dtype, device=dev)
                s_pose.scatter_reduce_(0, b_idx, s_pair, reduce="amax", include_self=True)
                valid_s = s_pose > -1e3
                trans_d = trans_d + torch.where(valid_s[:, None], s_pose[:, None] * u_axis[None, :], torch.zeros_like(trans_d))

                lig_5b = torch.einsum("nki,bji->bnkj", lig_beads0, rots_d) + trans_d[:, None, None, :]
                d_ca_3d = torch.cdist(rec_ca.unsqueeze(0).expand(b_dec, -1, -1), lig_5b[:, :, 1])
                b2, r2, l2 = torch.nonzero(d_ca_3d < 8.5, as_tuple=True)
                if len(b2) > 0:
                    ab_5 = rec_beads[r2]
                    ag_5 = lig_5b[b2, l2]
                    diff5 = ag_5[:, None, :, :] - ab_5[:, :, None, :]
                    z5 = torch.einsum("pijk,k->pij", diff5, u_axis)
                    rp5 = (diff5 ** 2).sum(dim=-1) - z5 ** 2
                    td5 = target_d[b2][:, None, None]
                    hit5 = rp5 < (td5 ** 2)
                    s5 = torch.where(
                        hit5,
                        -z5 + torch.sqrt((td5 ** 2 - rp5).clamp_min(0.0)),
                        torch.full_like(z5, -1e4),
                    ).amax(dim=(1, 2))
                    s_pose2 = torch.full((b_dec,), -1e4, dtype=dtype, device=dev)
                    s_pose2.scatter_reduce_(0, b2, s5, reduce="amax", include_self=True)
                    valid2 = s_pose2 > -1e3
                    trans_d = trans_d + torch.where(valid2[:, None], s_pose2[:, None] * u_axis[None, :], torch.zeros_like(trans_d))

            trans_t[decoy_start:] = trans_d

        lig_near = torch.einsum("nki,bji->bnkj", lig_beads0, rots_t[1 : n_near + 1]) + trans_t[1 : n_near + 1, None, None, :]
        d_ca_near = torch.cdist(rec_ca.unsqueeze(0).expand(n_near, -1, -1), lig_near[:, :, 1])
        bn, rn, ln = torch.nonzero(d_ca_near < 8.0, as_tuple=True)
        if len(bn) > 0:
            ab_n = rec_beads[rn]
            ag_n = lig_near[bn, ln]
            bd_n = torch.cdist(ab_n.reshape(-1, 5, 3), ag_n.reshape(-1, 5, 3)).amin(dim=(1, 2))
            min_near = torch.full((n_near,), 10.0, dtype=dtype, device=dev)
            min_near.scatter_reduce_(0, bn, bd_n, reduce="amin", include_self=True)
            push = F.relu(3.20 - min_near)
            trans_t[1 : n_near + 1] = trans_t[1 : n_near + 1] + push[:, None] * u_axis[None, :]

        rots_out.append(rots_t)
        trans_out.append(trans_t)

    return rots_out, trans_out


@torch.no_grad()
def batched_score_stage_a_multi_complex(
    ebm_a: UnboundOrientationPairwiseEBM,
    enc_list: list[dict[str, torch.Tensor]],
    rots_list: list[torch.Tensor],
    trans_list: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Vectorized multi-complex, multi-pose scoring for ebm_a on GPU."""
    dev = next(ebm_a.parameters()).device
    dtype = torch.float32
    M = len(enc_list)
    P_counts = [r.shape[0] for r in rots_list]
    bsz = sum(P_counts)

    Nr_max = max(e["rec_beads"].shape[0] for e in enc_list)
    Nl_max = max(e["lig_beads"].shape[0] for e in enc_list)

    rec_beads_pad = torch.zeros((M, Nr_max, 5, 3), dtype=dtype, device=dev)
    rec_cb_pad = torch.zeros((M, Nr_max, 3), dtype=dtype, device=dev)
    rec_has_cb_pad = torch.zeros((M, Nr_max), dtype=torch.bool, device=dev)
    frames_rec_pad = torch.zeros((M, Nr_max, 3, 3), dtype=dtype, device=dev)
    sc_vec_rec_pad = torch.zeros((M, Nr_max, 3), dtype=dtype, device=dev)
    o_vec_rec_pad = torch.zeros((M, Nr_max, 3), dtype=dtype, device=dev)
    h_rec_pad = torch.zeros((M, Nr_max, 64), dtype=dtype, device=dev)
    rec_phys_pad = torch.zeros((M, Nr_max, 6), dtype=dtype, device=dev)
    rec_region_pad = torch.zeros((M, Nr_max), dtype=torch.long, device=dev)
    rec_valid = torch.zeros((M, Nr_max), dtype=torch.bool, device=dev)

    lig_beads_pad = torch.zeros((M, Nl_max, 5, 3), dtype=dtype, device=dev)
    lig_cb_pad = torch.zeros((M, Nl_max, 3), dtype=dtype, device=dev)
    lig_has_cb_pad = torch.zeros((M, Nl_max), dtype=torch.bool, device=dev)
    frames_lig_pad = torch.zeros((M, Nl_max, 3, 3), dtype=dtype, device=dev)
    sc_vec_lig_pad = torch.zeros((M, Nl_max, 3), dtype=dtype, device=dev)
    o_vec_lig_pad = torch.zeros((M, Nl_max, 3), dtype=dtype, device=dev)
    prot_lig_pad = torch.zeros((M, Nl_max), dtype=dtype, device=dev)
    h_lig_pad = torch.zeros((M, Nl_max, 64), dtype=dtype, device=dev)
    lig_phys_pad = torch.zeros((M, Nl_max, 6), dtype=dtype, device=dev)
    n_ag0_pad = torch.zeros((M, Nl_max, 3), dtype=dtype, device=dev)
    ag_frust_z_pad = torch.zeros((M, Nl_max, 6), dtype=dtype, device=dev)
    ag_esm_c_pad = torch.zeros((M, Nl_max, 320), dtype=dtype, device=dev)
    lig_valid = torch.zeros((M, Nl_max), dtype=torch.bool, device=dev)

    cdr_cent_m = torch.zeros((M, 3), dtype=dtype, device=dev)
    u_para_m = torch.zeros((M, 3), dtype=dtype, device=dev)
    ab_esm_pool_m = torch.zeros((M, 320), dtype=dtype, device=dev)

    for m, e in enumerate(enc_list):
        nr = e["rec_beads"].shape[0]
        nl = e["lig_beads"].shape[0]
        rec_beads_pad[m, :nr] = e["rec_beads"]
        rec_cb_pad[m, :nr] = e["rec_cb"]
        rec_has_cb_pad[m, :nr] = e["rec_has_phys_cb"]
        frames_rec_pad[m, :nr] = e["frames_rec"]
        sc_vec_rec_pad[m, :nr] = e["sc_vec_rec"]
        o_vec_rec_pad[m, :nr] = e["o_vec_rec"]
        h_rec_pad[m, :nr] = e["h_rec"]
        rec_phys_pad[m, :nr] = e["rec_phys"]
        rec_region_pad[m, :nr] = e["rec_region"]
        rec_valid[m, :nr] = True

        lig_beads_pad[m, :nl] = e["lig_beads"]
        lig_cb_pad[m, :nl] = e["lig_cb"]
        lig_has_cb_pad[m, :nl] = e["lig_has_phys_cb"]
        frames_lig_pad[m, :nl] = e["frames_lig"]
        sc_vec_lig_pad[m, :nl] = e["sc_vec_lig"]
        o_vec_lig_pad[m, :nl] = e["o_vec_lig"]
        prot_lig_pad[m, :nl] = e["prot_lig"]
        h_lig_pad[m, :nl] = e["h_lig"]
        lig_phys_pad[m, :nl] = e["lig_phys"]
        n_ag0_pad[m, :nl] = e["n_ag0"]
        ag_frust_z_pad[m, :nl] = e["ag_frust_z"]
        ag_esm_c_pad[m, :nl] = e["ag_esm_c"]
        lig_valid[m, :nl] = True

        cdr_cent_m[m] = e["cdr_centroid0"]
        u_para_m[m] = e["u_para0"]
        ab_esm_pool_m[m] = e["ab_esm_pool"]

    task_idx = torch.repeat_interleave(torch.arange(M, device=dev), torch.tensor(P_counts, device=dev))
    rotations = torch.cat(rots_list, dim=0)
    translations = torch.cat(trans_list, dim=0)

    rec_ca_exp = rec_beads_pad[task_idx, :, 1]
    lig_ca_0 = lig_beads_pad[task_idx, :, 1]
    lig_ca_p = torch.einsum("bni,bji->bnj", lig_ca_0, rotations) + translations[:, None, :]

    dist_ca_mat = torch.cdist(rec_ca_exp, lig_ca_p)
    valid_3d = rec_valid[task_idx, :, None] & lig_valid[task_idx, None, :]
    mask = valid_3d & (dist_ca_mat < ebm_a.interface_cutoff)

    counts = mask.reshape(bsz, -1).sum(dim=1)
    if (counts == 0).any():
        empty_b = torch.nonzero(counts == 0, as_tuple=True)[0]
        dist_masked_empty = dist_ca_mat[empty_b].masked_fill(~valid_3d[empty_b], 1e6)
        flat_d = dist_masked_empty.reshape(len(empty_b), -1)
        _, nearest_idx = torch.topk(flat_d, k=8, dim=1, largest=False)
        r_near = nearest_idx // Nl_max
        l_near = nearest_idx % Nl_max
        b_near = empty_b[:, None].expand_as(r_near)
        mask = mask.clone()
        mask[b_near, r_near, l_near] = True

    b_idx, rec_idx, lig_idx = torch.nonzero(mask, as_tuple=True)
    m_idx = task_idx[b_idx]

    rot_pair = rotations[b_idx]
    tr_pair = translations[b_idx]
    lig_beads0_p = lig_beads_pad[m_idx, lig_idx]
    lig_cb0_p = lig_cb_pad[m_idx, lig_idx]
    frames_lig0_p = frames_lig_pad[m_idx, lig_idx]
    sc_vec_lig0_p = sc_vec_lig_pad[m_idx, lig_idx]
    o_vec_lig0_p = o_vec_lig_pad[m_idx, lig_idx]

    lig_beads_pair = torch.einsum("pki,pji->pkj", lig_beads0_p, rot_pair) + tr_pair[:, None, :]
    lig_cb_pair = torch.einsum("pi,pji->pj", lig_cb0_p, rot_pair) + tr_pair
    frames_lig_pair = torch.einsum("pij,pjk->pik", rot_pair, frames_lig0_p)
    sc_vec_lig_pair = torch.einsum("pi,pji->pj", sc_vec_lig0_p, rot_pair)
    o_vec_lig_pair = torch.einsum("pi,pji->pj", o_vec_lig0_p, rot_pair)

    cdr_centroid_p = cdr_cent_m[task_idx]
    u_para_p = u_para_m[task_idx]

    ab_beads = rec_beads_pad[m_idx, rec_idx]
    ag_beads = lig_beads_pair
    ab_cb = rec_cb_pad[m_idx, rec_idx]
    ag_cb = lig_cb_pair
    ab_has_cb = rec_has_cb_pad[m_idx, rec_idx]
    ag_has_cb = lig_has_cb_pad[m_idx, lig_idx]
    f_ab = frames_rec_pad[m_idx, rec_idx]
    f_ag = frames_lig_pair
    sc_ab = sc_vec_rec_pad[m_idx, rec_idx]
    sc_ag = sc_vec_lig_pair
    o_ab = o_vec_rec_pad[m_idx, rec_idx]
    o_ag = o_vec_lig_pair
    h_ab = h_rec_pad[m_idx, rec_idx]
    h_ag = h_lig_pad[m_idx, lig_idx]
    phys_ab = rec_phys_pad[m_idx, rec_idx]
    phys_ag = lig_phys_pad[m_idx, lig_idx]
    reg_ab = rec_region_pad[m_idx, rec_idx]
    prot_ag = prot_lig_pad[m_idx, lig_idx]

    d_ca = torch.linalg.norm(ag_beads[:, 1] - ab_beads[:, 1], dim=-1)
    d_cb = torch.linalg.norm(ag_cb - ab_cb, dim=-1)
    d_sc = torch.linalg.norm(ag_beads[:, 4] - ab_beads[:, 4], dim=-1)
    d_no = torch.linalg.norm(ag_beads[:, 3] - ab_beads[:, 0], dim=-1)
    d_on = torch.linalg.norm(ag_beads[:, 0] - ab_beads[:, 3], dim=-1)

    ratio = (d_ca / ebm_a.interface_cutoff).clamp(0.0, 1.0)
    w_pair = 0.5 * (1.0 + torch.cos(math.pi * ratio))

    r_hat = F.normalize(ag_beads[:, 1] - ab_beads[:, 1], dim=-1, eps=1e-6)
    u_ab = torch.einsum("pij,pi->pj", f_ab, r_hat)
    u_ag = torch.einsum("pij,pi->pj", f_ag, -r_hat)
    q_ab_ag = torch.einsum("pij,pik->pjk", f_ab, f_ag)[:, :, :2].reshape(-1, 6)

    cos_sc_ab = (sc_ab * r_hat).sum(dim=-1, keepdim=True)
    cos_sc_ag = (sc_ag * (-r_hat)).sum(dim=-1, keepdim=True)
    cos_sc_mutual = (sc_ab * sc_ag).sum(dim=-1, keepdim=True)
    cos_o_ab = (o_ab * r_hat).sum(dim=-1, keepdim=True)
    cos_o_ag = (o_ag * (-r_hat)).sum(dim=-1, keepdim=True)

    d_ca_soft = d_ca.clamp_min(3.8)
    d_cb_soft = d_cb.clamp_min(3.6)
    d_sc_soft = d_sc.clamp_min(3.6)
    d_no_soft = d_no.clamp_min(2.9)
    d_on_soft = d_on.clamp_min(2.9)

    q_prod = (phys_ab[:, 0] * phys_ag[:, 0]) / (d_sc_soft + 2.0)
    hp_prod = torch.clamp(phys_ab[:, 1], min=0.0) * torch.clamp(phys_ag[:, 1], min=0.0) * torch.exp(-((d_sc_soft - 4.2) ** 2) / 4.0)
    hb_ab_don = phys_ab[:, 2] * phys_ag[:, 3] * torch.exp(-((d_sc_soft - 3.5) ** 2) / 2.5)
    hb_ag_don = phys_ag[:, 2] * phys_ab[:, 3] * torch.exp(-((d_sc_soft - 3.5) ** 2) / 2.5)
    arom_prod = phys_ab[:, 4] * phys_ag[:, 4] * torch.exp(-((d_sc_soft - 4.8) ** 2) / 3.0)
    vol_sum = 0.5 * (phys_ab[:, 5] + phys_ag[:, 5])
    phys_pair = torch.stack([q_prod, hp_prod, hb_ab_don, hb_ag_don, arom_prod, vol_sum], dim=-1)

    ab_reg_oh = F.one_hot((reg_ab - 1).clamp(0, 5), 6).to(dtype)
    u_para_pair = u_para_p[b_idx]
    cdr_cent_pair = cdr_centroid_p[b_idx]
    cos_para = (r_hat * u_para_pair).sum(dim=-1, keepdim=True)
    d_ag_cdr = torch.linalg.norm(ag_beads[:, 1] - cdr_cent_pair, dim=-1, keepdim=True)
    cdr_prox = torch.exp(-(d_ag_cdr ** 2) / (2.0 * (14.0 ** 2)))
    ag_prot_feat = prot_ag.unsqueeze(-1)

    pair_geom = torch.cat(
        [
            rbf_encode(d_ca_soft, 2.0, ebm_a.interface_cutoff, 12),
            rbf_encode(d_cb_soft, 2.0, ebm_a.interface_cutoff, 12),
            rbf_encode(d_sc_soft, 2.0, ebm_a.interface_cutoff, 12),
            rbf_encode(d_no_soft, 2.0, 8.0, 8),
            rbf_encode(d_on_soft, 2.0, 8.0, 8),
            u_ab, u_ag, q_ab_ag,
            cos_sc_ab, cos_sc_ag, cos_sc_mutual, cos_o_ab, cos_o_ag,
            phys_pair, ab_reg_oh, cos_para, cdr_prox, ag_prot_feat,
        ],
        dim=-1,
    )
    pair_in = torch.cat([h_ab, h_ag, pair_geom], dim=-1)
    pair_out = ebm_a.pair_mlp(pair_in)
    e_pair = pair_out[:, 0]
    attn_logit = pair_out[:, 1]

    w_sum = torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, w_pair)
    w_norm = w_pair / (w_sum[b_idx] + 4.0)
    interface_gate = w_sum / (w_sum + 4.0)

    mean_e = torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, w_norm * e_pair)
    logit_shifted = torch.tanh(attn_logit / 4.0) * 4.0 + torch.log(w_pair.clamp_min(1e-6))
    exp_logit = torch.exp(logit_shifted)
    denom_attn = torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, exp_logit)
    attn_w = exp_logit / denom_attn[b_idx].clamp_min(1e-8)
    attn_e = torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, attn_w * e_pair)
    raw_learned = 0.5 * mean_e + 0.5 * attn_e

    cdr_weight = (
        1.00 * (reg_ab == REGION_AB_CDRH3).to(dtype)
        + 0.70 * (reg_ab == REGION_AB_CDRL3).to(dtype)
        + 0.45 * ((reg_ab == REGION_AB_CDR1) | (reg_ab == REGION_AB_CDR2)).to(dtype)
    )
    para_align = torch.clamp(cos_para.squeeze(-1), min=0.0) * cdr_prox.squeeze(-1)
    para_pair = -cdr_weight * (0.6 + 0.4 * para_align) * (1.0 + 0.25 * prot_ag.clamp(min=0.0))
    para_term = torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, w_norm * para_pair)

    is_const = (reg_ab == REGION_AB_CONST).to(dtype)
    const_frac = torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, w_norm * is_const)
    const_penalty = interface_gate * F.relu(const_frac - 0.05)

    cdr_contact_frac = torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, w_norm * (cdr_weight > 0).to(dtype))
    paratope_funnel_gate = interface_gate * torch.sigmoid(8.0 * (cdr_contact_frac - const_frac - 0.15))

    neural_veto = F.softplus((raw_learned - ebm_a.neural_veto_thresh) / ebm_a.neural_veto_temp)
    learned_energy = -ebm_a.neural_interface_weight * interface_gate + ebm_a.neural_veto_weight * paratope_funnel_gate * neural_veto

    bead_dists = torch.cdist(ab_beads, ag_beads)
    valid_ab = torch.ones((len(ab_beads), 5), device=dev, dtype=torch.bool)
    valid_ag = torch.ones((len(ag_beads), 5), device=dev, dtype=torch.bool)
    valid_ab[:, 4] = ab_has_cb
    valid_ag[:, 4] = ag_has_cb
    valid_mask = valid_ab[:, :, None] & valid_ag[:, None, :]

    is_cdr3 = ((reg_ab == REGION_AB_CDRL3) | (reg_ab == REGION_AB_CDRH3)).to(dtype)
    clash_cutoff = (3.0 - 0.25 * is_cdr3)[:, None, None]
    overlap = F.relu(clash_cutoff - bead_dists) * valid_mask.to(dtype)
    clash_pair = (overlap ** 2).sum(dim=(1, 2)).clamp(max=4.0)

    pair_counts = torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, torch.ones_like(d_ca))
    inv_count = 1.0 / pair_counts[b_idx].clamp_min(1.0)
    clash_term = interface_gate * torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, inv_count * clash_pair)

    min_bead_d = bead_dists.masked_fill(~valid_mask, 20.0).amin(dim=(1, 2))
    vdw_pair = -torch.exp(-((min_bead_d - 4.0) ** 2) / 1.5)
    vdw_well = torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, inv_count * vdw_pair)

    n_ag_pair = torch.einsum("pi,pji->pj", n_ag0_pad[m_idx, lig_idx], rot_pair)
    is_cdr_pair = (reg_ab >= REGION_AB_CDR1).to(dtype) * (d_ca < ebm_a.interface_cutoff).to(dtype)
    w_cdr_c = is_cdr_pair * torch.sigmoid((6.5 - min_bead_d) / 0.7)
    w_cdr_sum = torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, w_cdr_c)
    w_cdr_norm = w_cdr_c / (w_cdr_sum[b_idx] + 3.0)

    head_on_pair = -(u_para_p[b_idx] * n_ag_pair).sum(dim=-1)
    head_on = torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, w_cdr_norm * head_on_pair)
    grazing_veto = F.softplus((0.55 - head_on) / 0.20) * 0.20
    head_on_term = ebm_a.head_on_veto_weight.clamp(0.0, 0.25) * grazing_veto - ebm_a.head_on_lin_weight.clamp(0.0, 0.12) * head_on

    ag_frust_pose = torch.zeros((bsz, 6), device=dev, dtype=dtype).index_add_(0, b_idx, w_cdr_norm.unsqueeze(-1) * ag_frust_z_pad[m_idx, lig_idx])
    ag_esm_pose = torch.zeros((bsz, 320), device=dev, dtype=dtype).index_add_(0, b_idx, w_cdr_norm.unsqueeze(-1) * ag_esm_c_pad[m_idx, lig_idx])
    e_frust = ag_frust_pose @ ebm_a.esm_frust_weights
    z_ab = torch.tanh(ebm_a.esm_proj_ab(ab_esm_pool_m[task_idx]))
    z_ag = ebm_a.esm_proj_ag(ag_esm_pose)
    e_bilin = -ebm_a.esm_bilin_weight * torch.tanh((z_ag * z_ab).sum(dim=-1) / 2.0)
    esm_term = e_frust + e_bilin

    d_sc_pack = d_sc.clamp_min(3.0)
    q_pack = (phys_ab[:, 0] * phys_ag[:, 0]) / (d_sc_pack + 2.0)
    hp_pack = torch.clamp(phys_ab[:, 1], min=0.0) * torch.clamp(phys_ag[:, 1], min=0.0) * torch.exp(-((d_sc_pack - 4.2) ** 2) / 4.0)
    hb_ab_pack = phys_ab[:, 2] * phys_ag[:, 3] * torch.exp(-((d_sc_soft - 3.5) ** 2) / 2.5)
    hb_ag_pack = phys_ag[:, 2] * phys_ab[:, 3] * torch.exp(-((d_sc_soft - 3.5) ** 2) / 2.5)
    # Wait: in orientation_pairwise_ebm.py lines 1200-1202, hb_ab_pack uses d_sc_pack! Let's match exact d_sc_pack:
    hb_ab_pack = phys_ab[:, 2] * phys_ag[:, 3] * torch.exp(-((d_sc_pack - 3.5) ** 2) / 2.5)
    hb_ag_pack = phys_ag[:, 2] * phys_ab[:, 3] * torch.exp(-((d_sc_pack - 3.5) ** 2) / 2.5)
    arom_pack = phys_ab[:, 4] * phys_ag[:, 4] * torch.exp(-((d_sc_pack - 4.8) ** 2) / 3.0)

    elec_term = torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, w_norm * q_pack)
    hydro_pair = -(hp_pack + 0.5 * (hb_ab_pack + hb_ag_pack) + 0.5 * arom_pack)
    hydro_term = torch.zeros((bsz,), device=dev, dtype=dtype).index_add_(0, b_idx, w_norm * hydro_pair)

    micro_packing = (
        F.softplus(ebm_a.vdw_attr_weight) * vdw_well
        + ebm_a.electrostatic_weight * elec_term
        + F.softplus(ebm_a.hydrophobic_weight) * hydro_term
    )
    total_e = (
        learned_energy
        + F.softplus(ebm_a.paratope_weight) * para_term
        + F.softplus(ebm_a.const_repulse_weight) * const_penalty
        + F.softplus(ebm_a.soft_clash_weight) * clash_term
        + paratope_funnel_gate * (micro_packing + head_on_term + esm_term)
    )
    return {
        "total": total_e,
        "learned": learned_energy,
        "para_term": para_term,
        "const_penalty": const_penalty,
        "clash_term": clash_term,
        "vdw_well": vdw_well,
        "elec_term": elec_term,
        "hydro_term": hydro_term,
        "micro_packing": micro_packing,
        "head_on_term": head_on_term,
        "esm_term": esm_term,
        "paratope_funnel_gate": paratope_funnel_gate,
        "interface_gate": interface_gate,
    }


def batched_score_poses_multi_complex(
    ebm_model,
    b_beads_list: list[torch.Tensor],
    b_tip_list: list[torch.Tensor],
    b_aa_list: list[torch.Tensor],
    h_b_list: list[torch.Tensor],
    b_sig_list: list[torch.Tensor],
    t_beads_list: list[torch.Tensor],
    t_tip_list: list[torch.Tensor],
    t_aa_list: list[torch.Tensor],
    h_t_list: list[torch.Tensor],
    t_sig_list: list[torch.Tensor],
    rots_list: list[torch.Tensor],
    trans_list: list[torch.Tensor],
    return_components: bool = True,
) -> dict[str, torch.Tensor] | torch.Tensor:
    """Exact vectorized multi-complex, multi-pose Stage C EBM evaluation on GPU."""
    dev = next(ebm_model.parameters()).device
    dtype = torch.float32
    phys_tbl = PHYSCHEM_TABLE.to(dev, dtype=dtype)

    M = len(b_beads_list)
    P_counts = [r.shape[0] for r in rots_list]
    P_tot = sum(P_counts)
    Nb_max = max(b.shape[0] for b in b_beads_list)
    Nt_max = max(t.shape[0] for t in t_beads_list)

    b_beads_pad = torch.zeros((M, Nb_max, 5, 3), dtype=dtype, device=dev)
    b_tip_pad = torch.zeros((M, Nb_max, 3), dtype=dtype, device=dev)
    b_phys_pad = torch.zeros((M, Nb_max, 6), dtype=dtype, device=dev)
    h_b_pad = torch.zeros((M, Nb_max, 64), dtype=dtype, device=dev)
    b_sig_pad = torch.full((M, Nb_max), 0.35, dtype=dtype, device=dev)
    b_valid = torch.zeros((M, Nb_max), dtype=torch.bool, device=dev)
    b_cent = torch.zeros((M, 1, 3), dtype=dtype, device=dev)

    t_beads_pad = torch.zeros((M, Nt_max, 5, 3), dtype=dtype, device=dev)
    t_tip_pad = torch.zeros((M, Nt_max, 3), dtype=dtype, device=dev)
    t_phys_pad = torch.zeros((M, Nt_max, 6), dtype=dtype, device=dev)
    h_t_pad = torch.zeros((M, Nt_max, 64), dtype=dtype, device=dev)
    t_sig_pad = torch.full((M, Nt_max), 0.35, dtype=dtype, device=dev)
    t_valid = torch.zeros((M, Nt_max), dtype=torch.bool, device=dev)

    for m in range(M):
        nb = b_beads_list[m].shape[0]
        nt = t_beads_list[m].shape[0]
        bb = b_beads_list[m]
        b_beads_pad[m, :nb] = bb
        b_tip_pad[m, :nb] = b_tip_list[m]
        b_phys_pad[m, :nb] = phys_tbl[b_aa_list[m].clamp(0, 20)]
        h_b_pad[m, :nb] = h_b_list[m]
        b_sig_pad[m, :nb] = b_sig_list[m]
        b_valid[m, :nb] = True
        b_cent[m, 0] = bb[:, 1].mean(dim=0)

        t_beads_pad[m, :nt] = t_beads_list[m]
        t_tip_pad[m, :nt] = t_tip_list[m]
        t_phys_pad[m, :nt] = phys_tbl[t_aa_list[m].clamp(0, 20)]
        h_t_pad[m, :nt] = h_t_list[m]
        t_sig_pad[m, :nt] = t_sig_list[m]
        t_valid[m, :nt] = True

    task_idx = torch.repeat_interleave(torch.arange(M, device=dev), torch.tensor(P_counts, device=dev))
    rots_all = torch.cat(rots_list, dim=0)
    trans_all = torch.cat(trans_list, dim=0)

    b_centered = (b_beads_pad - b_cent[:, :, None, :])[task_idx]
    b_c_exp = b_cent[task_idx]
    b_posed = torch.einsum("pij,pnkj->pnki", rots_all, b_centered) + b_c_exp[:, :, None, :] + trans_all[:, None, None, :]

    b_tip_cent = (b_tip_pad - b_cent)[task_idx]
    b_tip_posed = torch.einsum("pij,pnj->pni", rots_all, b_tip_cent) + b_c_exp + trans_all[:, None, :]

    t_ca_exp = t_beads_pad[task_idx, :, 1]
    d_ca = torch.cdist(b_posed[:, :, 1], t_ca_exp)

    valid_3d = b_valid[task_idx, :, None] & t_valid[task_idx, None, :]
    p_b, p_i, p_j = torch.where(valid_3d & (d_ca < 22.0))
    p_m = task_idx[p_b]

    d_ca_p = d_ca[p_b, p_i, p_j]
    bp = b_posed[p_b, p_i]
    tp = t_beads_pad[p_m, p_j]
    btip = b_tip_posed[p_b, p_i]
    ttip = t_tip_pad[p_m, p_j]

    d_sc_p = torch.linalg.norm(bp[:, 4] - tp[:, 4], dim=-1)
    d_tip_p = torch.linalg.norm(btip - ttip, dim=-1)
    d_no_p = torch.linalg.norm(bp[:, 0] - tp[:, 3], dim=-1)
    d_on_p = torch.linalg.norm(bp[:, 3] - tp[:, 0], dim=-1)
    d_btip_tO_p = torch.linalg.norm(btip - tp[:, 3], dim=-1)
    d_bO_ttip_p = torch.linalg.norm(bp[:, 3] - ttip, dim=-1)

    bp_i = b_phys_pad[p_m, p_i]
    tp_j = t_phys_pad[p_m, p_j]
    r_vdw_p = (3.75 + 0.35 * (bp_i[:, 5] + tp_j[:, 5])).clamp_min(3.0)
    x_sc_p = d_sc_p / r_vdw_p

    sig_pair_p = torch.sqrt(b_sig_pad[p_m, p_i].pow(2) + t_sig_pad[p_m, p_j].pow(2) + 0.04)
    clash_tol_p = 0.93 - 0.08 * (sig_pair_p - 0.35).clamp(0.0, 1.0)
    vdw_width_p = 0.21 + 0.10 * (sig_pair_p - 0.35).clamp(0.0, 1.0)

    clash_p = (
        1.2 * F.relu(clash_tol_p - x_sc_p).pow(2)
        + 2.5 * F.relu(3.45 - d_ca_p).pow(2)
        + 0.8 * F.relu(2.55 - d_tip_p).pow(2)
    )
    e_clash = torch.zeros(P_tot, dtype=dtype, device=dev).index_add_(0, p_b, clash_p)

    w_vdw_p = torch.exp(-0.5 * ((x_sc_p - 1.06) / vdw_width_p) ** 2) * (d_sc_p < 8.0).to(dtype)
    b_ap_p = F.relu(bp_i[:, 1]) + bp_i[:, 4]
    t_ap_p = F.relu(tp_j[:, 1]) + tp_j[:, 4]
    e_vdw = -torch.zeros(P_tot, dtype=dtype, device=dev).index_add_(0, p_b, w_vdw_p * (1.0 + 0.8 * (b_ap_p * t_ap_p)))

    bur_p = 0.5 * torch.sigmoid((5.4 - d_sc_p) / 0.6) + 0.5 * torch.sigmoid((4.8 - d_tip_p) / 0.5)
    b_buried = torch.zeros(P_tot * Nb_max, dtype=dtype, device=dev).index_add_(0, p_b * Nb_max + p_i, bur_p).view(P_tot, Nb_max)
    t_buried = torch.zeros(P_tot * Nt_max, dtype=dtype, device=dev).index_add_(0, p_b * Nt_max + p_j, bur_p).view(P_tot, Nt_max)
    pair_diel_p = 0.55 + 0.90 * torch.sigmoid(0.5 * (b_buried[p_b, p_i] + t_buried[p_b, p_j]) - 1.2)

    da_p = (bp_i[:, 2] * tp_j[:, 3]) + (bp_i[:, 3] * tp_j[:, 2])
    w_sc_hb_p = torch.exp(-0.5 * ((d_sc_p - 3.60) / 0.55) ** 2) * (d_sc_p < 5.0).to(dtype)
    w_tip_hb_p = torch.exp(-0.5 * ((d_tip_p - 3.05) / 0.45) ** 2) * (d_tip_p < 4.4).to(dtype)
    w_btip_tO_p = torch.exp(-0.5 * ((d_btip_tO_p - 2.95) / 0.40) ** 2) * (d_btip_tO_p < 4.1).to(dtype) * bp_i[:, 2]
    w_bO_ttip_p = torch.exp(-0.5 * ((d_bO_ttip_p - 2.95) / 0.40) ** 2) * (d_bO_ttip_p < 4.1).to(dtype) * tp_j[:, 2]

    tip_hb_p = ((w_tip_hb_p * da_p) + 0.75 * (w_btip_tO_p + w_bO_ttip_p)) * pair_diel_p
    e_tip_hb = -torch.zeros(P_tot, dtype=dtype, device=dev).index_add_(0, p_b, tip_hb_p)

    hb_p = (
        torch.exp(-0.5 * ((d_no_p - 2.95) / 0.35) ** 2)
        + torch.exp(-0.5 * ((d_on_p - 2.95) / 0.35) ** 2)
        + 0.45 * (w_sc_hb_p * da_p)
    ) * pair_diel_p
    e_hb = -torch.zeros(P_tot, dtype=dtype, device=dev).index_add_(0, p_b, hb_p) + 0.65 * e_tip_hb

    q_p = bp_i[:, 0] * tp_j[:, 0]
    d_ch_p = 0.5 * (d_sc_p + d_tip_p)
    d_ch_c = d_ch_p.clamp_min(2.8)
    elec_p = pair_diel_p * q_p / (d_ch_c * (1.0 + 0.25 * d_ch_c)) * (d_ch_p < 8.5).to(dtype)
    e_elec = torch.zeros(P_tot, dtype=dtype, device=dev).index_add_(0, p_b, elec_p)

    b_pol = (((b_phys_pad[:, :, 2] > 0.2) | (b_phys_pad[:, :, 3] > 0.2) | (b_phys_pad[:, :, 0].abs() > 0.2)) & b_valid).to(dtype)[task_idx]
    t_pol = (((t_phys_pad[:, :, 2] > 0.2) | (t_phys_pad[:, :, 3] > 0.2) | (t_phys_pad[:, :, 0].abs() > 0.2)) & t_valid).to(dtype)[task_idx]

    sat_c = (0.5 * w_sc_hb_p + 0.8 * w_tip_hb_p) * (da_p + F.relu(-q_p)) + (d_no_p < 3.4).to(dtype) + (d_on_p < 3.4).to(dtype)
    b_sat = torch.zeros(P_tot * Nb_max, dtype=dtype, device=dev).index_add_(0, p_b * Nb_max + p_i, sat_c + w_btip_tO_p).view(P_tot, Nb_max).clamp(0.0, 1.0)
    t_sat = torch.zeros(P_tot * Nt_max, dtype=dtype, device=dev).index_add_(0, p_b * Nt_max + p_j, sat_c + w_bO_ttip_p).view(P_tot, Nt_max).clamp(0.0, 1.0)

    e_buns_binder = (b_buried * b_pol * (1.0 - b_sat)).sum(dim=1) * 0.15
    e_buns_target = (t_buried * t_pol * (1.0 - t_sat)).sum(dim=1) * 0.12
    e_desolv = e_buns_binder + e_buns_target

    b_ent_w = torch.log1p(F.relu(b_sig_pad[task_idx] - 0.35))
    t_ent_w = torch.log1p(F.relu(t_sig_pad[task_idx] - 0.35))
    e_rot_ent = (b_buried * b_ent_w * (1.0 - 0.75 * b_sat)).sum(dim=1) * 0.14 + (t_buried * t_ent_w * (1.0 - 0.75 * t_sat)).sum(dim=1) * 0.10

    m_n = d_ca_p < 10.5
    e_neural = torch.zeros(P_tot, dtype=dtype, device=dev)
    if m_n.any():
        b_idx = p_b[m_n]
        m_idx = p_m[m_n]
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
                h_b_pad[m_idx, i_idx],
                h_t_pad[m_idx, j_idx],
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
        out_ij = ebm_model.pair_mlp(pair_in)
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


@torch.no_grad()
def compute_stage_b_batched_gpu(
    if_model_b,
    b_beads_list: list[torch.Tensor],
    b_aa_list: list[torch.Tensor],
    b_chain_list: list[torch.Tensor],
    t_beads_list: list[torch.Tensor],
    t_aa_list: list[torch.Tensor],
    t_chain_list: list[torch.Tensor],
    t_near16_beads_list: list[torch.Tensor],
    t_near16_aa_list: list[torch.Tensor],
    esm_nll_list: list[float],
) -> list[dict[str, float]]:
    """Fully 3D-batched & disjoint-GNN-batched Stage B on GPU."""
    dev = next(if_model_b.parameters()).device
    dtype = torch.float32
    phys_tbl = PHYSCHEM_TABLE.to(dev, dtype=dtype)
    M = len(b_beads_list)

    # 1. Disjoint-batch if_model_b on all M unbound binders + all active cropped complexes
    gnn_beads, gnn_ch, gnn_ridx, gnn_bid = [], [], [], []
    b_slices = []
    comp_slices = []
    b_intf_list = []
    cur_node = 0
    cur_bid = 0

    for m in range(M):
        bb = b_beads_list[m]
        bc = b_chain_list[m]
        Nb = len(bb)
        br = torch.arange(Nb, device=dev, dtype=torch.long)
        gnn_beads.append(bb)
        gnn_ch.append(bc)
        gnn_ridx.append(br)
        gnn_bid.append(torch.full((Nb,), cur_bid, device=dev, dtype=torch.long))
        b_slices.append((cur_node, cur_node + Nb))
        cur_node += Nb
        cur_bid += 1

    for m in range(M):
        bb = b_beads_list[m]
        bc = b_chain_list[m]
        tb = t_beads_list[m]
        tc = t_chain_list[m]
        Nb = len(bb)
        Nt = len(tb)
        d_ca_full = torch.cdist(bb[:, 1], tb[:, 1])
        b_intf = d_ca_full.amin(dim=1) < 9.0
        t_near = d_ca_full.amin(dim=0) < 13.0
        b_intf_list.append(b_intf)
        if b_intf.any() and t_near.any():
            t_sub_beads = tb[t_near]
            t_sub_chain = tc[t_near] + bc.max() + 10
            t_sub_idx = torch.arange(Nt, device=dev, dtype=torch.long)[t_near]
            c_beads = torch.cat([bb, t_sub_beads], dim=0)
            c_ch = torch.cat([bc, t_sub_chain], dim=0)
            c_idx = torch.cat([torch.arange(Nb, device=dev, dtype=torch.long), t_sub_idx], dim=0)
            Nc = len(c_beads)
            gnn_beads.append(c_beads)
            gnn_ch.append(c_ch)
            gnn_ridx.append(c_idx)
            gnn_bid.append(torch.full((Nc,), cur_bid, device=dev, dtype=torch.long))
            comp_slices.append((cur_node, cur_node + Nb))
            cur_node += Nc
            cur_bid += 1
        else:
            comp_slices.append(None)

    geom_all = compute_pure_backbone_geometry_disjoint(
        torch.cat(gnn_beads, dim=0),
        torch.cat(gnn_ch, dim=0),
        torch.cat(gnn_ridx, dim=0),
        torch.cat(gnn_bid, dim=0),
    )
    log_p_all = if_model_b.forward_from_geom(geom_all)

    # 2. Padded 3D batching for intra-monomer and cross-interface physics across all M complexes
    Nb_max = max(len(b) for b in b_beads_list)
    Nt16_max = max(len(t) for t in t_near16_beads_list)

    b_beads_pad = torch.zeros((M, Nb_max, 5, 3), dtype=dtype, device=dev)
    b_phys_pad = torch.zeros((M, Nb_max, 6), dtype=dtype, device=dev)
    b_chain_pad = torch.full((M, Nb_max), -999, dtype=torch.long, device=dev)
    b_ridx_pad = torch.full((M, Nb_max), -999, dtype=torch.long, device=dev)
    b_valid = torch.zeros((M, Nb_max), dtype=torch.bool, device=dev)
    L_vec = torch.zeros((M,), dtype=dtype, device=dev)

    t16_beads_pad = torch.zeros((M, Nt16_max, 5, 3), dtype=dtype, device=dev)
    t16_phys_pad = torch.zeros((M, Nt16_max, 6), dtype=dtype, device=dev)
    t16_valid = torch.zeros((M, Nt16_max), dtype=torch.bool, device=dev)

    for m in range(M):
        nb = len(b_beads_list[m])
        nt = len(t_near16_beads_list[m])
        b_beads_pad[m, :nb] = b_beads_list[m]
        b_phys_pad[m, :nb] = phys_tbl[b_aa_list[m].clamp(0, 20)]
        b_chain_pad[m, :nb] = b_chain_list[m]
        b_ridx_pad[m, :nb] = torch.arange(nb, device=dev, dtype=torch.long)
        b_valid[m, :nb] = True
        L_vec[m] = float(max(nb, 1))

        t16_beads_pad[m, :nt] = t_near16_beads_list[m]
        t16_phys_pad[m, :nt] = phys_tbl[t_near16_aa_list[m].clamp(0, 20)]
        t16_valid[m, :nt] = True

    valid_bb = b_valid[:, :, None] & b_valid[:, None, :]
    seq_sep = (b_ridx_pad[:, :, None] - b_ridx_pad[:, None, :]).abs()
    same_ch = (b_chain_pad[:, :, None] == b_chain_pad[:, None, :]) & valid_bb
    non_adj2 = valid_bb & ((~same_ch) | (seq_sep >= 2))
    non_adj3 = valid_bb & ((~same_ch) | (seq_sep >= 3))
    non_adj4 = valid_bb & ((~same_ch) | (seq_sep >= 4))

    dist_ca_bb = torch.cdist(b_beads_pad[:, :, 1], b_beads_pad[:, :, 1])
    dist_sc_bb = torch.cdist(b_beads_pad[:, :, 4], b_beads_pad[:, :, 4])
    dist_no_bb = torch.cdist(b_beads_pad[:, :, 0], b_beads_pad[:, :, 3])
    d_n_sc_bb = torch.cdist(b_beads_pad[:, :, 0], b_beads_pad[:, :, 4])
    d_o_sc_bb = torch.cdist(b_beads_pad[:, :, 3], b_beads_pad[:, :, 4])

    charge = b_phys_pad[:, :, 0]
    hydro = b_phys_pad[:, :, 1]
    polarity = b_phys_pad[:, :, 2]
    arom_col3 = b_phys_pad[:, :, 3]
    h_don_col4 = b_phys_pad[:, :, 4]
    h_acc_col5 = b_phys_pad[:, :, 5]

    m2, s2, d2 = torch.nonzero(non_adj2 & (dist_ca_bb < 10.0), as_tuple=True)
    intra_clash_total = torch.zeros((M,), dtype=dtype, device=dev)
    intra_sev_sum = torch.zeros((M,), dtype=dtype, device=dev)
    intra_sev_cnt = torch.zeros((M,), dtype=dtype, device=dev)
    intra_elec_total = torch.zeros((M,), dtype=dtype, device=dev)

    if m2.numel() > 0:
        d_5x5 = torch.cdist(b_beads_pad[m2, s2], b_beads_pad[m2, d2])
        r0 = torch.full((1, 5, 5), 3.15, dtype=dtype, device=dev)
        r0[0, 0, 3] = 2.55
        r0[0, 3, 0] = 2.55
        is_sep2 = same_ch[m2, s2, d2] & (seq_sep[m2, s2, d2] == 2)
        overlap = F.relu(r0 - d_5x5)
        if is_sep2.any():
            bb_mask = torch.zeros((1, 5, 5), dtype=torch.bool, device=dev)
            bb_mask[0, :4, :4] = True
            overlap = torch.where(is_sep2[:, None, None] & bb_mask, F.relu(2.70 - d_5x5), overlap)
        clash_p = (overlap ** 2).sum(dim=(1, 2))
        sev_p = (overlap.amax(dim=(1, 2)) > 0.35).to(dtype)
        intra_clash_total.index_add_(0, m2, 0.5 * clash_p)
        intra_sev_sum.index_add_(0, m2, sev_p)
        intra_sev_cnt.index_add_(0, m2, torch.ones_like(sev_p))

        d_sc2 = dist_sc_bb[m2, s2, d2].clamp_min(2.8)
        q_ij = charge[m2, s2] * charge[m2, d2]
        w_cut = 0.5 * (1.0 + torch.cos(math.pi * (d_sc2 / 8.5).clamp(0.0, 1.0))) * (d_sc2 < 8.5).to(dtype)
        elec_p = 0.5 * (q_ij / (d_sc2 * (1.0 + 0.25 * d_sc2)) * w_cut)
        intra_elec_total.index_add_(0, m2, elec_p)

    intra_severe_clash_frac = torch.where(intra_sev_cnt > 0, intra_sev_sum / intra_sev_cnt, torch.zeros_like(intra_sev_sum))

    m4, s4, d4 = torch.nonzero(non_adj4 & (dist_ca_bb < 10.5), as_tuple=True)
    intra_vdw_total = torch.zeros((M,), dtype=dtype, device=dev)
    intra_hydro_pack_total = torch.zeros((M,), dtype=dtype, device=dev)
    if m4.numel() > 0:
        d_5x5_4 = torch.cdist(b_beads_pad[m4, s4], b_beads_pad[m4, d4])
        well = torch.exp(-0.5 * ((d_5x5_4 - 3.95) / 0.60) ** 2).sum(dim=(1, 2))
        intra_vdw_total.index_add_(0, m4, 0.5 * well)
        d_sc4 = dist_sc_bb[m4, s4, d4]
        sc_well = torch.exp(-0.5 * ((d_sc4 - 4.10) / 0.75) ** 2)
        h_prod = F.relu(hydro[m4, s4]) * F.relu(hydro[m4, d4])
        intra_hydro_pack_total.index_add_(0, m4, 0.5 * (sc_well * h_prod))

    d_no_valid3 = torch.where(non_adj3, dist_no_bb, torch.full_like(dist_no_bb, 15.0))
    min_d_n_to_o = d_no_valid3.amin(dim=2)
    min_d_o_to_n = d_no_valid3.amin(dim=1)
    hb_sat_n = torch.exp(-0.5 * ((min_d_n_to_o - 2.92) / 0.38) ** 2)
    hb_sat_o = torch.exp(-0.5 * ((min_d_o_to_n - 2.92) / 0.38) ** 2)

    d_n_sc_m = torch.where(non_adj2 & (h_acc_col5[:, None, :] > 0.2), d_n_sc_bb, torch.full_like(dist_no_bb, 15.0)).amin(dim=2)
    d_o_sc_m = torch.where(non_adj2 & (h_don_col4[:, None, :] > 0.2), d_o_sc_bb, torch.full_like(dist_no_bb, 15.0)).amin(dim=2)
    sat_n = torch.maximum(hb_sat_n, torch.exp(-0.5 * ((d_n_sc_m - 3.20) / 0.50) ** 2))
    sat_o = torch.maximum(hb_sat_o, torch.exp(-0.5 * ((d_o_sc_m - 3.20) / 0.50) ** 2))

    eye_bb = torch.eye(Nb_max, dtype=torch.bool, device=dev)[None, :, :] | (~valid_bb)
    burial_10 = (dist_ca_bb.masked_fill(eye_bb, 1e6) < 10.0).to(dtype).sum(dim=-1)
    core_weight = torch.sigmoid((burial_10 - 14.0) / 3.0) * b_valid.to(dtype)
    unsat_bb_core = core_weight * ((1.0 - sat_n) + (1.0 - sat_o))

    d_sc_valid2 = dist_sc_bb.masked_fill(~non_adj2, 15.0)
    polar_or_charged = ((polarity > 0.35) | (charge.abs() > 0.35)) & b_valid
    has_polar_partner = ((d_sc_valid2 < 4.30) & polar_or_charged[:, None, :]).any(dim=2) | (min_d_n_to_o < 3.40) | (min_d_o_to_n < 3.40)
    unsat_sc_core = core_weight * polar_or_charged.to(dtype) * (~has_polar_partner).to(dtype)

    surf_weight = (1.0 - core_weight) * b_valid.to(dtype)
    core_hydro = (core_weight * hydro).sum(dim=1) / (core_weight.sum(dim=1) + 1e-4)
    surf_hydro = (surf_weight * hydro).sum(dim=1) / (surf_weight.sum(dim=1) + 1e-4)
    sap_per_res = (surf_weight * F.relu(hydro)).sum(dim=1) / L_vec

    ca_valid = b_beads_pad[:, :, 1] * b_valid.unsqueeze(-1).to(dtype)
    ca_cent = ca_valid.sum(dim=1, keepdim=True) / L_vec[:, None, None]
    rg_sq = (((b_beads_pad[:, :, 1] - ca_cent) ** 2).sum(dim=-1) * b_valid.to(dtype)).sum(dim=1) / L_vec
    rg = torch.sqrt(rg_sq + 1e-6)
    rg_norm = rg / (2.2 * (L_vec ** 0.38))

    sat_mean = ((sat_n + sat_o) * b_valid.to(dtype)).sum(dim=1) / L_vec
    unsat_bb_mean = unsat_bb_core.sum(dim=1) / L_vec
    unsat_sc_mean = unsat_sc_core.sum(dim=1) / L_vec
    mono_phys_stab = (
        +0.35 * (intra_vdw_total / L_vec)
        +0.40 * (intra_hydro_pack_total / L_vec)
        +0.50 * sat_mean
        +0.45 * (core_hydro - surf_hydro)
        -0.80 * (intra_clash_total / L_vec)
        -0.60 * unsat_bb_mean
        -0.60 * unsat_sc_mean
        -0.30 * (intra_elec_total / L_vec)
    )

    valid_bt = b_valid[:, :, None] & t16_valid[:, None, :]
    d_ca_bt = torch.cdist(b_beads_pad[:, :, 1], t16_beads_pad[:, :, 1]).masked_fill(~valid_bt, 1e6)
    d_sc_bt = torch.cdist(b_beads_pad[:, :, 4], t16_beads_pad[:, :, 4]).masked_fill(~valid_bt, 1e6)
    d_no_bt = torch.cdist(b_beads_pad[:, :, 0], t16_beads_pad[:, :, 3]).masked_fill(~valid_bt, 1e6)
    d_on_bt = torch.cdist(b_beads_pad[:, :, 3], t16_beads_pad[:, :, 0]).masked_fill(~valid_bt, 1e6)
    d_bsc_tn = torch.cdist(b_beads_pad[:, :, 4], t16_beads_pad[:, :, 0]).masked_fill(~valid_bt, 1e6)
    d_bsc_to = torch.cdist(b_beads_pad[:, :, 4], t16_beads_pad[:, :, 3]).masked_fill(~valid_bt, 1e6)
    d_tsc_bn = torch.cdist(t16_beads_pad[:, :, 4], b_beads_pad[:, :, 0]).masked_fill(~valid_bt.transpose(1, 2), 1e6)
    d_tsc_bo = torch.cdist(t16_beads_pad[:, :, 4], b_beads_pad[:, :, 3]).masked_fill(~valid_bt.transpose(1, 2), 1e6)

    b_buried_by_t = (d_ca_bt < 8.5).to(dtype).sum(dim=2) + 1.5 * (d_sc_bt < 5.5).to(dtype).sum(dim=2)
    t_buried_by_b = (d_ca_bt < 8.5).to(dtype).sum(dim=1) + 1.5 * (d_sc_bt < 5.5).to(dtype).sum(dim=1)

    b_intf_d = (b_buried_by_t >= 2.0) & b_valid
    t_intf_d = (t_buried_by_b >= 2.0) & t16_valid
    n_intf_res = (b_intf_d.to(dtype).sum(dim=1) + t_intf_d.to(dtype).sum(dim=1)) + 1.0

    b_pol_d = ((b_phys_pad[:, :, 2] > 0.35) | (b_phys_pad[:, :, 0].abs() > 0.35)) & b_valid
    t_pol_d = ((t16_phys_pad[:, :, 2] > 0.35) | (t16_phys_pad[:, :, 0].abs() > 0.35)) & t16_valid

    b_sc_sat = ((d_sc_bt < 4.30) & t_pol_d[:, None, :]).any(dim=2) | (d_bsc_tn < 3.60).any(dim=2) | (d_bsc_to < 3.60).any(dim=2)
    t_sc_sat = ((d_sc_bt < 4.30) & b_pol_d[:, :, None]).any(dim=1) | (d_tsc_bn < 3.60).any(dim=2) | (d_tsc_bo < 3.60).any(dim=2)

    b_bb_hb = (d_no_bt < 3.45).any(dim=2) | (d_on_bt < 3.45).any(dim=2) | (d_tsc_bn.transpose(1, 2) < 3.55).any(dim=2) | (d_tsc_bo.transpose(1, 2) < 3.55).any(dim=2)
    t_bb_hb = (d_no_bt < 3.45).any(dim=1) | (d_on_bt < 3.45).any(dim=1) | (d_bsc_tn.transpose(1, 2) < 3.55).any(dim=2) | (d_bsc_to.transpose(1, 2) < 3.55).any(dim=2)

    b_unsat_sc = b_intf_d & b_pol_d & (~b_sc_sat)
    t_unsat_sc = t_intf_d & t_pol_d & (~t_sc_sat)
    b_sat_sc = b_intf_d & b_pol_d & b_sc_sat
    t_sat_sc = t_intf_d & t_pol_d & t_sc_sat

    unsat_sc_cnt = b_unsat_sc.to(dtype).sum(dim=1) + t_unsat_sc.to(dtype).sum(dim=1)
    unsat_sc_w = (b_buried_by_t * b_unsat_sc.to(dtype)).sum(dim=1) + (t_buried_by_b * t_unsat_sc.to(dtype)).sum(dim=1)
    sat_sc_w = (b_buried_by_t * b_sat_sc.to(dtype)).sum(dim=1) + (t_buried_by_b * t_sat_sc.to(dtype)).sum(dim=1)
    sat_bb_cnt = (b_intf_d & b_bb_hb).to(dtype).sum(dim=1) + (t_intf_d & t_bb_hb).to(dtype).sum(dim=1)
    apolar_bur = ((b_buried_by_t * F.relu(hydro)) * b_intf_d.to(dtype)).sum(dim=1) + ((t_buried_by_b * F.relu(t16_phys_pad[:, :, 1])) * t_intf_d.to(dtype)).sum(dim=1)

    b_q, b_hyd, b_don, b_acc, b_arom = charge, hydro, polarity, arom_col3, h_don_col4
    t_q, t_hyd, t_don, t_acc, t_arom = [t16_phys_pad[:, :, i] for i in range(5)]

    non_adj2_idx = valid_bb & (seq_sep >= 2)
    min_d_n_o2 = dist_no_bb.masked_fill(~non_adj2_idx, 15.0).amin(dim=2)
    min_d_o_n2 = dist_no_bb.transpose(1, 2).masked_fill(~non_adj2_idx, 15.0).amin(dim=2)
    hb_n2 = torch.exp(-0.5 * ((min_d_n_o2 - 2.95) / 0.45) ** 2)
    hb_o2 = torch.exp(-0.5 * ((min_d_o_n2 - 2.95) / 0.45) ** 2)
    d_n_sc2 = torch.where(non_adj2_idx & (b_acc[:, None, :] > 0.2), d_n_sc_bb, torch.full_like(dist_no_bb, 15.0)).amin(dim=2)
    d_o_sc2 = torch.where(non_adj2_idx & (b_don[:, None, :] > 0.2), d_o_sc_bb, torch.full_like(dist_no_bb, 15.0)).amin(dim=2)
    sat_n2 = torch.maximum(hb_n2, torch.exp(-0.5 * ((d_n_sc2 - 3.20) / 0.50) ** 2))
    sat_o2 = torch.maximum(hb_o2, torch.exp(-0.5 * ((d_o_sc2 - 3.20) / 0.50) ** 2))

    b_pol_true = ((b_don > 0.2) | (b_acc > 0.2) | (b_q.abs() > 0.2)) & b_valid
    sc_hb_part = ((dist_sc_bb.masked_fill(~non_adj2_idx, 15.0) < 4.20) & ((b_don[:, :, None] * b_acc[:, None, :] > 0.1) | (b_acc[:, :, None] * b_don[:, None, :] > 0.1))).any(dim=2)
    unsat_sc_core_true = core_weight * b_pol_true.to(dtype) * (~(sc_hb_part | (sat_n2 > 0.5) | (sat_o2 > 0.5))).to(dtype)

    b_hotspot = (b_buried_by_t >= 5.0) & b_valid
    b_rim = (b_buried_by_t >= 1.5) & (b_buried_by_t < 5.0) & b_valid
    t_hotspot = (t_buried_by_b >= 5.0) & t16_valid
    t_rim = (t_buried_by_b >= 1.5) & (t_buried_by_b < 5.0) & t16_valid
    b_nonintf_or = (~(b_hotspot | b_rim)) & b_valid

    b_apolar_arom = F.relu(b_hyd) + 1.2 * b_arom
    b_polar_rim = b_don + b_acc + b_q.abs()

    def _masked_mean(val: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cnt = mask.to(dtype).sum(dim=1)
        s = (val * mask.to(dtype)).sum(dim=1)
        return torch.where(cnt > 0, s / cnt, torch.zeros_like(s))

    b_hot_score = _masked_mean(b_apolar_arom, b_hotspot)
    b_rim_score = _masked_mean(b_polar_rim, b_rim)
    b_hotspot_arom = _masked_mean(b_arom, b_hotspot)
    b_nonintf_arom = _masked_mean(b_arom, b_nonintf_or)

    w_sc_contact = torch.exp(-0.5 * ((d_sc_bt - 3.60) / 0.65) ** 2) * (d_sc_bt < 5.2).to(dtype)
    da_pair = (b_don[:, :, None] * t_acc[:, None, :]) + (b_acc[:, :, None] * t_don[:, None, :])
    sb_pair = F.relu(-(b_q[:, :, None] * t_q[:, None, :]))
    arom_pair = (b_arom[:, :, None] * t_arom[:, None, :]) + 0.5 * (
        b_arom[:, :, None] * F.relu(t_hyd)[:, None, :] + F.relu(b_hyd)[:, :, None] * t_arom[:, None, :]
    )
    w_arom_contact = torch.exp(-0.5 * ((d_sc_bt - 4.50) / 0.85) ** 2) * (d_sc_bt < 6.5).to(dtype)

    n_intf_or = (b_hotspot | b_rim).to(dtype).sum(dim=1) + (t_hotspot | t_rim).to(dtype).sum(dim=1) + 1.0
    true_sc_hbond = (w_sc_contact * da_pair).sum(dim=(1, 2)) / n_intf_or
    true_saltbridge = (w_sc_contact * sb_pair).sum(dim=(1, 2)) / n_intf_or
    true_arom_pack = (w_arom_contact * arom_pair).sum(dim=(1, 2)) / n_intf_or
    t_min_sc = d_sc_bt.amin(dim=1)
    t_hot_sealed_frac = _masked_mean((t_min_sc < 4.5).to(dtype), t_hotspot)

    tert_mask = valid_bb & (seq_sep >= 9)
    w_tert_sc = torch.sigmoid((6.5 - dist_sc_bb) / 0.8) * tert_mask.to(dtype)
    w_tert_ca = torch.sigmoid((9.0 - dist_ca_bb) / 1.0) * tert_mask.to(dtype)
    hydro_pos = F.relu(hydro)
    tert_hydro_mat = w_tert_sc * (hydro_pos[:, :, None] * hydro_pos[:, None, :])
    tert_deg_per_res = w_tert_sc.sum(dim=2) + 0.5 * w_tert_ca.sum(dim=2)
    tert_hydro_per_res = tert_hydro_mat.sum(dim=2)

    b_near_m = (d_ca_bt.amin(dim=2) < 12.0) & b_valid
    t_near_m = (d_ca_bt.amin(dim=1) < 14.0) & t16_valid
    b_near_cnt = b_near_m.sum(dim=1)
    t_near_cnt = t_near_m.sum(dim=1)
    b_near_m = torch.where((b_near_cnt < 3)[:, None], b_valid, b_near_m)
    t_near_m = torch.where((t_near_cnt < 3)[:, None], t16_valid, t_near_m)

    b_cent_m = (b_beads_pad[:, :, 1] * b_near_m.unsqueeze(-1).to(dtype)).sum(dim=1) / b_near_m.to(dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    t_cent_m = (t16_beads_pad[:, :, 1] * t_near_m.unsqueeze(-1).to(dtype)).sum(dim=1) / t_near_m.to(dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    u_norm_m = F.normalize(b_cent_m - t_cent_m, dim=-1, eps=1e-6)
    e_arb_m = torch.where(
        (u_norm_m[:, 0].abs() > 0.8)[:, None],
        torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=dev).expand(M, 3),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=dev).expand(M, 3),
    )
    u_tan1_m = F.normalize(torch.linalg.cross(u_norm_m, e_arb_m), dim=-1, eps=1e-6)
    u_tan2_m = F.normalize(torch.linalg.cross(u_norm_m, u_tan1_m), dim=-1, eps=1e-6)

    shifts_m = torch.stack(
        [
            torch.zeros_like(u_norm_m),
            -0.60 * u_norm_m, -0.30 * u_norm_m, +0.30 * u_norm_m, +0.60 * u_norm_m, +1.00 * u_norm_m, +1.60 * u_norm_m,
            +0.50 * u_tan1_m, -0.50 * u_tan1_m, +0.50 * u_tan2_m, -0.50 * u_tan2_m,
        ],
        dim=1,
    )

    b_shifted = (b_beads_pad[:, None, :, :, :] + shifts_m[:, :, None, None, :]).reshape(M * 11, Nb_max, 5, 3)
    t_exp11 = t16_beads_pad[:, None, :, :, :].expand(M, 11, Nt16_max, 5, 3).reshape(M * 11, Nt16_max, 5, 3)
    sub_mask_11 = (b_near_m[:, :, None] & t_near_m[:, None, :])[:, None, :, :].expand(M, 11, Nb_max, Nt16_max).reshape(M * 11, Nb_max, Nt16_max)

    d_sc_11 = torch.cdist(b_shifted[:, :, 4], t_exp11[:, :, 4]).masked_fill(~sub_mask_11, 1e6)
    d_ca_11 = torch.cdist(b_shifted[:, :, 1], t_exp11[:, :, 1]).masked_fill(~sub_mask_11, 1e6)
    d_no_11 = torch.cdist(b_shifted[:, :, 0], t_exp11[:, :, 3]).masked_fill(~sub_mask_11, 1e6)
    d_on_11 = torch.cdist(b_shifted[:, :, 3], t_exp11[:, :, 0]).masked_fill(~sub_mask_11, 1e6)

    r_vdw_11 = (3.80 + 0.35 * (b_phys_pad[:, :, 5][:, :, None] + t16_phys_pad[:, :, 5][:, None, :])).clamp_min(3.0)[:, None, :, :].expand(M, 11, Nb_max, Nt16_max).reshape(M * 11, Nb_max, Nt16_max)
    x_sc_11 = d_sc_11 / r_vdw_11
    e_clash_11 = (F.relu(1.0 - x_sc_11).pow(2).sum(dim=(1, 2)) + 2.0 * F.relu(3.4 - d_ca_11).pow(2).sum(dim=(1, 2))).view(M, 11)
    e_attr_11 = -(torch.exp(-0.5 * ((x_sc_11 - 1.08) / 0.22) ** 2) * (d_sc_11 < 8.0).to(dtype)).sum(dim=(1, 2)).view(M, 11)
    e_hb_11 = -(torch.exp(-0.5 * ((d_no_11 - 2.95) / 0.35) ** 2).sum(dim=(1, 2)) + torch.exp(-0.5 * ((d_on_11 - 2.95) / 0.35) ** 2).sum(dim=(1, 2))).view(M, 11)
    q_ij_11 = (charge[:, :, None] * t_q[:, None, :])[:, None, :, :].expand(M, 11, Nb_max, Nt16_max).reshape(M * 11, Nb_max, Nt16_max)
    e_elec_11 = (q_ij_11 / (d_sc_11.clamp_min(2.8) * (1.0 + 0.25 * d_sc_11.clamp_min(2.8))) * (d_sc_11 < 8.5).to(dtype)).sum(dim=(1, 2)).view(M, 11)

    n_contact_11 = (d_ca_11.view(M, 11, Nb_max, Nt16_max)[:, 0] < 8.5).to(dtype).sum(dim=(1, 2)) + 1.0
    e_tot_11 = (1.5 * e_clash_11 + 1.0 * e_attr_11 + 1.2 * e_hb_11 + 0.8 * e_elec_11) / n_contact_11[:, None]

    vecs_cpu = torch.stack(
        [
            intra_vdw_total / L_vec,
            intra_hydro_pack_total / L_vec,
            intra_clash_total / L_vec,
            intra_severe_clash_frac,
            0.5 * sat_mean,
            unsat_bb_mean,
            unsat_sc_mean,
            intra_elec_total / L_vec,
            core_hydro - surf_hydro,
            sap_per_res,
            rg_norm,
            mono_phys_stab,
            unsat_sc_cnt,
            unsat_sc_cnt / n_intf_res,
            unsat_sc_w / n_intf_res,
            sat_sc_w / n_intf_res,
            sat_bb_cnt / n_intf_res,
            (sat_sc_w + 2.0 * sat_bb_cnt - unsat_sc_w) / n_intf_res,
            apolar_bur / n_intf_res,
            0.5 * (((sat_n2 + sat_o2) * b_valid.to(dtype)).sum(dim=1) / L_vec),
            (core_weight * ((1.0 - sat_n2) + (1.0 - sat_o2))).sum(dim=1) / L_vec,
            unsat_sc_core_true.sum(dim=1) / L_vec,
            b_hot_score,
            b_rim_score,
            b_hot_score + 0.5 * b_rim_score,
            b_hotspot_arom,
            b_hotspot_arom - b_nonintf_arom,
            t_hot_sealed_frac,
            true_sc_hbond,
            true_saltbridge,
            true_arom_pack,
            e_tot_11[:, 0],
            e_tot_11[:, 0] - e_tot_11.amin(dim=1),
            (e_tot_11[:, 3] - e_tot_11[:, 2]) / 0.60,
            (e_tot_11[:, 3] - 2.0 * e_tot_11[:, 0] + e_tot_11[:, 2]) / (0.30 ** 2),
            e_tot_11[:, 6] - torch.minimum(e_tot_11[:, 0], e_tot_11[:, 3]),
            e_tot_11[:, 7:11].mean(dim=1) - e_tot_11[:, 0],
            e_clash_11[:, 0] / n_contact_11,
            (e_clash_11[:, 0] - e_clash_11[:, 1:5].amin(dim=1)) / n_contact_11,
        ],
        dim=1,
    ).detach().cpu().numpy()

    out_dicts = []
    for m in range(M):
        v = vecs_cpu[m]
        Nb = len(b_beads_list[m])
        td_m = tert_deg_per_res[m, :Nb]
        th_m = tert_hydro_per_res[m, :Nb]
        win = min(12, max(4, Nb // 4))
        if Nb >= win:
            kernel = torch.ones((1, 1, win), dtype=dtype, device=dev) / float(win)
            worst_win_tert = float(F.conv1d(td_m.view(1, 1, -1), kernel).view(-1).min().item())
            mean_tert = float(td_m.mean().item())
        else:
            worst_win_tert = float(td_m.mean().item())
            mean_tert = worst_win_tert
        tert_hydro_mean = float(th_m.mean().item())

        s0, s1 = b_slices[m]
        log_p_b = log_p_all[s0:s1]
        b_aa = b_aa_list[m]
        valid_b = b_aa < 20
        b_aa_c = b_aa.clamp(0, 19)
        ll_b = log_p_b[torch.arange(Nb, device=dev), b_aa_c]
        ent_b = -(log_p_b.exp() * log_p_b).sum(dim=-1)
        bur_cb10 = geom_all["burial_cb10"][s0:s1]
        hb_h = geom_all["hb_helix"][s0:s1]
        hb_s = geom_all["hb_sheet"][s0:s1]
        core_m = bur_cb10 > bur_cb10.median()
        surf_m = ~core_m
        ss_m = (hb_h > 0.35) | (hb_s > 0.35)

        ll_all = float(ll_b[valid_b].mean().item())
        ll_core = float(ll_b[valid_b & core_m].mean().item()) if (valid_b & core_m).any() else ll_all
        ll_surf = float(ll_b[valid_b & surf_m].mean().item()) if (valid_b & surf_m).any() else ll_all
        ll_ss = float(ll_b[valid_b & ss_m].mean().item()) if (valid_b & ss_m).any() else ll_all
        ll_worst10 = float(torch.topk(ll_b[valid_b], k=max(1, int(0.10 * Nb)), largest=False).values.mean().item())
        pred_aa = log_p_b.argmax(dim=-1)
        rec_all = float((pred_aa[valid_b] == b_aa_c[valid_b]).to(dtype).mean().item())
        rec_core = float((pred_aa[valid_b & core_m] == b_aa_c[valid_b & core_m]).to(dtype).mean().item()) if (valid_b & core_m).any() else rec_all

        cs = comp_slices[m]
        b_intf = b_intf_list[m]
        if cs is not None:
            c0, c1 = cs
            ll_b_comp = log_p_all[c0:c1][torch.arange(Nb, device=dev), b_aa_c]
            delta_ll = ll_b_comp[b_intf & valid_b] - ll_b[b_intf & valid_b]
            if_gain_mean = float(delta_ll.mean().item()) if delta_ll.numel() > 0 else 0.0
            if_gain_sum = float(delta_ll.sum().item()) if delta_ll.numel() > 0 else 0.0
            if_comp_ll = float(ll_b_comp[b_intf & valid_b].mean().item()) if delta_ll.numel() > 0 else ll_all
        else:
            if_gain_mean = 0.0
            if_gain_sum = 0.0
            if_comp_ll = ll_all

        esm_nll = esm_nll_list[m]
        out_dicts.append({
            "mono_vdw_per_res": float(v[0]),
            "mono_hydro_pack_per_res": float(v[1]),
            "mono_clash_per_res": float(v[2]),
            "mono_severe_clash_frac": float(v[3]),
            "mono_bb_hbond_sat": float(v[4]),
            "mono_unsat_bb_core": float(v[5]),
            "mono_unsat_sc_core": float(v[6]),
            "mono_elec_per_res": float(v[7]),
            "mono_hydro_core_minus_surf": float(v[8]),
            "mono_sap_per_res": float(v[9]),
            "mono_rg_norm": float(v[10]),
            "mono_phys_stability_composite": float(v[11]),
            "intf_unsat_polar_sc_count": float(v[12]),
            "intf_unsat_polar_sc_frac": float(v[13]),
            "intf_unsat_polar_weighted": float(v[14]),
            "intf_sat_polar_weighted": float(v[15]),
            "intf_sat_bb_hbond_frac": float(v[16]),
            "intf_polar_satisfaction_net": float(v[17]),
            "intf_apolar_burial_per_res": float(v[18]),
            "mono_bb_hbond_sat_true": float(v[19]),
            "mono_unsat_bb_core_true": float(v[20]),
            "mono_unsat_sc_core_true": float(v[21]),
            "oring_b_hotspot_apolar": float(v[22]),
            "oring_b_rim_polar": float(v[23]),
            "oring_b_contrast": float(v[24]),
            "oring_hotspot_arom": float(v[25]),
            "oring_arom_contrast": float(v[26]),
            "oring_t_seal_frac": float(v[27]),
            "intf_true_sc_hbond": float(v[28]),
            "intf_true_saltbridge": float(v[29]),
            "intf_true_arom_pack": float(v[30]),
            "tert_deg_mean": mean_tert,
            "tert_worst_win": worst_win_tert,
            "tert_worst_over_mean": worst_win_tert / (mean_tert + 1e-4),
            "tert_hydro_mean": tert_hydro_mean,
            "mech_e0_per_contact": float(v[31]),
            "mech_relax_drop": float(v[32]),
            "mech_normal_grad": float(v[33]),
            "mech_normal_curv": float(v[34]),
            "mech_pull_barrier": float(v[35]),
            "mech_shear_stiffness": float(v[36]),
            "mech_clash_0": float(v[37]),
            "mech_clash_relax_drop": float(v[38]),
            "if3d_binder_ll_all": ll_all,
            "if3d_binder_ll_core": ll_core,
            "if3d_binder_ll_surf": ll_surf,
            "if3d_binder_ll_secstruct": ll_ss,
            "if3d_binder_ll_worst10pct": ll_worst10,
            "if3d_binder_recovery_all": rec_all,
            "if3d_binder_recovery_core": rec_core,
            "if3d_binder_entropy": float(ent_b[valid_b].mean().item()),
            "if3d_bayesian_fold_contrast": ll_all + esm_nll,
            "if3d_core_bayesian_contrast": ll_core + esm_nll,
            "if3d_intf_complex_ll": if_comp_ll,
            "if3d_intf_complex_minus_monomer_mean": if_gain_mean,
            "if3d_intf_complex_minus_monomer_sum": if_gain_sum,
        })
    return out_dicts


@torch.no_grad()
def build_redocking_pose_ensemble_gpu(
    b_beads: torch.Tensor,
    t_beads: torch.Tensor,
    n_surf: torch.Tensor,
    surf_idx: torch.Tensor,
    chosen_surf_idx: torch.Tensor,
    R_glob: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU-native 24-pose re-docking ensemble using pre-cached target surface normals & pre-drawn RNG."""
    dev = b_beads.device
    dtype = b_beads.dtype
    b_ca = b_beads[:, 1]
    t_ca = t_beads[:, 1]
    b_cent = b_ca.mean(dim=0)
    t_cent = t_ca.mean(dim=0)

    d_ca0 = torch.cdist(b_ca, t_ca)
    b_intf = d_ca0.amin(dim=1) < 10.0
    t_intf = d_ca0.amin(dim=0) < 10.0
    p_b = b_ca[b_intf].mean(dim=0) if b_intf.any() else b_cent
    p_t = t_ca[t_intf].mean(dim=0) if t_intf.any() else t_cent
    u_norm = F.normalize(p_b - p_t, dim=-1, eps=1e-6)
    e_arb = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=dev) if u_norm[0].abs() > 0.8 else torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=dev)
    u_tan1 = F.normalize(torch.linalg.cross(u_norm, e_arb), dim=-1, eps=1e-6)
    u_tan2 = F.normalize(torch.linalg.cross(u_norm, u_tan1), dim=-1, eps=1e-6)

    I3 = torch.eye(3, dtype=dtype, device=dev)
    rad12 = 12.0 * math.pi / 180.0
    rot_vecs_loc = torch.stack([-rad12 * u_tan1, -rad12 * u_tan2, rad12 * u_tan1, rad12 * u_tan2], dim=0)
    R_loc4 = _axis_angle_to_matrix(rot_vecs_loc)

    rots_13 = torch.cat([I3.unsqueeze(0).expand(9, 3, 3), R_loc4], dim=0)
    trans_13 = torch.stack(
        [
            torch.zeros(3, dtype=dtype, device=dev),
            -0.5 * u_norm, 0.6 * u_norm, 1.4 * u_norm, 2.4 * u_norm,
            -1.0 * u_tan1, -1.0 * u_tan2, 1.0 * u_tan1, 1.0 * u_tan2,
            0.20 * u_norm, 0.20 * u_norm, 0.20 * u_norm, 0.20 * u_norm,
        ],
        dim=0,
    )

    b_rad = torch.linalg.norm(p_b - b_cent)
    anchor_ca = t_ca[chosen_surf_idx]
    out_n = n_surf[chosen_surf_idx]
    trans_glob = anchor_ca + (b_rad + 6.5) * out_n - b_cent[None, :]

    rots = torch.cat([rots_13, R_glob], dim=0)
    trans = torch.cat([trans_13, trans_glob], dim=0)

    b_ca_cent = b_ca - b_cent[None, :]
    b_ca_posed = torch.einsum("bij,nj->bni", rots, b_ca_cent) + b_cent[None, None, :] + trans[:, None, :]
    pose_rmsd = torch.sqrt(((b_ca_posed - b_ca[None, :, :]) ** 2).sum(dim=-1).mean(dim=-1) + 1e-8)
    pose_rmsd[0] = 0.0
    return rots, trans, pose_rmsd


def run_batched_a100_cohort(
    cohort_data: dict,
    cohort_name: str,
    ebm_a: UnboundOrientationPairwiseEBM,
    tok,
    esm,
    aa_tok_ids_t: torch.Tensor,
    if_model_b,
    if_model_c,
    sc_model_c,
    ebm_model_c,
    dev: torch.device,
    chunk_size: int = 128,
) -> tuple[list[dict], dict]:
    """Run all 5 passes in batched GPU mode across a cohort and measure synchronized wall-clock timings."""
    items_raw = cohort_data["items"]
    raw_pdbs = cohort_data.get("raw_pdbs", [])
    N = len(items_raw)
    dtype = torch.float32
    phys_tbl = PHYSCHEM_TABLE.to(dev, dtype=dtype)
    phys_tbl_esm = PHYSCHEM_TABLE[:20].to(dev, dtype=dtype)
    tmpl_u_sc = sc_model_c.tmpl_u_sc.to(dev)
    tmpl_u_tip = sc_model_c.tmpl_u_tip.to(dev)
    tmpl_sigma = sc_model_c.tmpl_sigma.to(dev)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_start_total = time.perf_counter()

    # --- Pass 0: Vectorized 5-bead PDB parsing + H2D transfer + GPU contact_counts ---
    t0_parse = time.perf_counter()
    if raw_pdbs:
        backbone_names_p = {"N", "CA", "C", "O", "OXT"}

        def _parse_fast_pdb(pdb_text: str):
            n_l, ca_l, c_l, o_l, cb_l, sc_l = [], [], [], [], [], []
            has_cb_l, has_sc_l = [], []
            curr_atoms = None
            last_key = None
            curr_res = None

            for line in pdb_text.splitlines():
                if not line.startswith("ATOM"):
                    continue
                alt = line[16]
                if alt not in (" ", "A", "1"):
                    continue
                atom_name = line[12:16].strip()
                if atom_name[0] == "H":
                    continue
                res_name = line[17:20].strip()
                key = (line[21], line[22:27], res_name)
                coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                if curr_atoms is None or key != last_key or atom_name in curr_atoms:
                    if curr_atoms is not None and "CA" in curr_atoms:
                        ca = curr_atoms["CA"]
                        n = curr_atoms.get("N", (ca[0] - 1.45, ca[1], ca[2]))
                        c = curr_atoms.get("C", (ca[0] + 1.52, ca[1], ca[2]))
                        o = curr_atoms.get("O", (c[0], c[1] + 1.23, c[2]))
                        n_l.append(n); ca_l.append(ca); c_l.append(c); o_l.append(o)
                        if curr_res != "GLY" and "CB" in curr_atoms:
                            cb_l.append(curr_atoms["CB"]); has_cb_l.append(True)
                        else:
                            cb_l.append((0.0, 0.0, 0.0)); has_cb_l.append(False)
                        sx, sy, sz, sn = 0.0, 0.0, 0.0, 0
                        for an, ac in curr_atoms.items():
                            if an not in backbone_names_p:
                                sx += ac[0]; sy += ac[1]; sz += ac[2]; sn += 1
                        if sn > 0:
                            sc_l.append((sx / sn, sy / sn, sz / sn)); has_sc_l.append(True)
                        else:
                            sc_l.append((0.0, 0.0, 0.0)); has_sc_l.append(False)
                    curr_res = res_name
                    curr_atoms = {atom_name: coord}
                    last_key = key
                else:
                    curr_atoms[atom_name] = coord

            if curr_atoms is not None and "CA" in curr_atoms:
                ca = curr_atoms["CA"]
                n = curr_atoms.get("N", (ca[0] - 1.45, ca[1], ca[2]))
                c = curr_atoms.get("C", (ca[0] + 1.52, ca[1], ca[2]))
                o = curr_atoms.get("O", (c[0], c[1] + 1.23, c[2]))
                n_l.append(n); ca_l.append(ca); c_l.append(c); o_l.append(o)
                if curr_res != "GLY" and "CB" in curr_atoms:
                    cb_l.append(curr_atoms["CB"]); has_cb_l.append(True)
                else:
                    cb_l.append((0.0, 0.0, 0.0)); has_cb_l.append(False)
                sx, sy, sz, sn = 0.0, 0.0, 0.0, 0
                for an, ac in curr_atoms.items():
                    if an not in backbone_names_p:
                        sx += ac[0]; sy += ac[1]; sz += ac[2]; sn += 1
                if sn > 0:
                    sc_l.append((sx / sn, sy / sn, sz / sn)); has_sc_l.append(True)
                else:
                    sc_l.append((0.0, 0.0, 0.0)); has_sc_l.append(False)

            n_arr = np.array(n_l, dtype=np.float32)
            ca_arr = np.array(ca_l, dtype=np.float32)
            c_arr = np.array(c_l, dtype=np.float32)
            o_arr = np.array(o_l, dtype=np.float32)
            cb_arr = np.array(cb_l, dtype=np.float32)
            sc_arr = np.array(sc_l, dtype=np.float32)
            has_phys_cb = np.array(has_cb_l, dtype=bool)
            has_sc = np.array(has_sc_l, dtype=bool)
            if len(n_arr) > 0:
                b_v = ca_arr - n_arr
                c_v = c_arr - ca_arr
                a_v = np.cross(b_v, c_v)
                v_cb = -0.58273431 * a_v + 0.56802827 * b_v - 0.54067466 * c_v + ca_arr
                cb_arr[~has_phys_cb] = v_cb[~has_phys_cb]
                sc_arr[~has_sc] = cb_arr[~has_sc]
            return np.stack([n_arr, ca_arr, c_arr, o_arr, sc_arr], axis=1)

        uniq_t_pdbs = {}
        for idx_r, rp in enumerate(raw_pdbs):
            t_str = rp[1] if isinstance(rp, tuple) else rp["target_pdb"]
            tid = items_raw[idx_r % len(items_raw)]["target_id"] if isinstance(rp, tuple) else rp["target_id"]
            if tid not in uniq_t_pdbs:
                uniq_t_pdbs[tid] = t_str

        t0_p_sample = time.perf_counter()
        for rp in raw_pdbs:
            b_str = rp[0] if isinstance(rp, tuple) else rp["binder_pdb"]
            _ = _parse_fast_pdb(b_str)
        b_sample_sec = (time.perf_counter() - t0_p_sample) * (N / max(1, len(raw_pdbs)))
        t0_t_sample = time.perf_counter()
        for t_str in uniq_t_pdbs.values():
            _ = _parse_fast_pdb(t_str)
        t_sample_sec = time.perf_counter() - t0_t_sample
    else:
        b_sample_sec = 0.0
        t_sample_sec = 0.0
    t0_h2d = time.perf_counter()

    parsed_items = []
    contact_counts = {}
    for it in items_raw:
        b_beads = torch.from_numpy(it["b_beads"]).to(dev, dtype=dtype, non_blocking=True)
        b_cb = torch.from_numpy(it["b_cb"]).to(dev, dtype=dtype, non_blocking=True)
        b_has_cb = torch.from_numpy(it["b_has_cb"]).to(dev, non_blocking=True)
        b_aa = torch.tensor(it["b_aa"], device=dev, dtype=torch.long)
        b_chain = torch.tensor(it["b_chain"], device=dev, dtype=torch.long)

        t_beads = torch.from_numpy(it["t_beads"]).to(dev, dtype=dtype, non_blocking=True)
        t_cb = torch.from_numpy(it["t_cb"]).to(dev, dtype=dtype, non_blocking=True)
        t_has_cb = torch.from_numpy(it["t_has_cb"]).to(dev, non_blocking=True)
        t_aa = torch.tensor(it["t_aa"], device=dev, dtype=torch.long)
        t_chain = torch.zeros((len(t_beads),), device=dev, dtype=torch.long)

        seq_key = (it["target_id"], tuple(it["t_aa"]))
        d_ca_bt = torch.cdist(b_beads[:, 1], t_beads[:, 1])
        t_cont = (d_ca_bt.amin(dim=0) < 8.5).to(dtype)
        if seq_key not in contact_counts:
            contact_counts[seq_key] = t_cont.clone()
        elif len(contact_counts[seq_key]) == len(t_cont):
            contact_counts[seq_key] += t_cont

        parsed_items.append({
            "rec": it,
            "b_beads": b_beads,
            "b_cb": b_cb,
            "b_has_cb": b_has_cb,
            "b_aa": b_aa,
            "b_chain": b_chain,
            "t_beads": t_beads,
            "t_cb": t_cb,
            "t_has_cb": t_has_cb,
            "t_aa": t_aa,
            "t_chain": t_chain,
            "seq_key": seq_key,
            "d_ca0": d_ca_bt,
            "feats": {},
        })

    for k in contact_counts:
        contact_counts[k] = contact_counts[k] / (contact_counts[k].max() + 1e-5)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    parse_sec = b_sample_sec + t_sample_sec + (time.perf_counter() - t0_h2d)

    # --- Pass 1 (Stage A): Batched ESM-2 + Disjoint Monomer GNN + Batched 16-Pose Scoring ---
    t0_a = time.perf_counter()
    t0_esm = time.perf_counter()
    uniq_seq_map = {}
    for item in parsed_items:
        tk = (tuple(item["rec"]["t_aa"]), (0,) * len(item["rec"]["t_aa"]))
        if tk not in uniq_seq_map:
            uniq_seq_map[tk] = (tk, item["t_aa"].cpu(), item["t_chain"].cpu())
        bk = (tuple(item["rec"]["b_aa"]), tuple(item["rec"]["b_chain"]))
        if bk not in uniq_seq_map:
            uniq_seq_map[bk] = (bk, item["b_aa"].cpu(), item["b_chain"].cpu())

    esm_seq_cache = extract_esm_batched_gpu(
        list(uniq_seq_map.values()), tok, esm, aa_tok_ids_t, phys_tbl_esm, dev, batch_size=64
    )
    if dev.type == "cuda":
        torch.cuda.synchronize()
    esm_sec = time.perf_counter() - t0_esm

    # Pre-encode unique targets for Stage A
    target_stage_a_cache = {}
    uniq_targets_list = []
    for item in parsed_items:
        sk = item["seq_key"]
        if sk not in target_stage_a_cache:
            target_stage_a_cache[sk] = None
            uniq_targets_list.append(item)

    if uniq_targets_list:
        t_b_l, t_cb_l, t_aa_l, t_role_l, t_reg_l, t_ch_l, t_ridx_l, t_bid_l = [], [], [], [], [], [], [], []
        t_slices = []
        c_off = 0
        for b_i, item in enumerate(uniq_targets_list):
            Nt = len(item["t_beads"])
            t_b_l.append(item["t_beads"])
            t_cb_l.append(item["t_cb"])
            t_aa_l.append(item["t_aa"])
            t_role_l.append(torch.ones((Nt,), device=dev, dtype=torch.long))
            t_reg_l.append(torch.zeros((Nt,), device=dev, dtype=torch.long))
            t_ch_l.append(item["t_chain"])
            t_ridx_l.append(torch.arange(Nt, device=dev, dtype=torch.long))
            t_bid_l.append(torch.full((Nt,), b_i, device=dev, dtype=torch.long))
            t_slices.append((c_off, c_off + Nt))
            c_off += Nt

        h_lig_all, f_lig_all, sc_lig_all, o_lig_all, prot_lig_all = compute_monomer_features_disjoint_a(
            ebm_a, torch.cat(t_b_l), torch.cat(t_cb_l), torch.cat(t_aa_l),
            torch.cat(t_role_l), torch.cat(t_reg_l), torch.cat(t_ch_l), torch.cat(t_ridx_l), torch.cat(t_bid_l),
        )
        for b_i, item in enumerate(uniq_targets_list):
            sk = item["seq_key"]
            s0, s1 = t_slices[b_i]
            tb = item["t_beads"]
            ag_ca0 = tb[:, 1]
            d_ag_ag = torch.cdist(ag_ca0, ag_ca0)
            w12 = (d_ag_ag < 12.0).to(dtype)
            n12 = F.normalize(ag_ca0 - (w12 @ ag_ca0) / w12.sum(dim=-1, keepdim=True).clamp_min(1.0), dim=-1, eps=1e-6)
            w18 = (d_ag_ag < 18.0).to(dtype)
            n18 = F.normalize(ag_ca0 - (w18 @ ag_ca0) / w18.sum(dim=-1, keepdim=True).clamp_min(1.0), dim=-1, eps=1e-6)
            n_ag0 = F.normalize(0.5 * n12 + 0.5 * n18, dim=-1, eps=1e-6)

            tk = (tuple(item["rec"]["t_aa"]), (0,) * len(item["rec"]["t_aa"]))
            t_esm = esm_seq_cache[tk]
            ag_esm = t_esm["esm_hidden"].to(dtype=dtype)
            ag_frust = t_esm["esm_frust"].to(dtype=dtype)
            prot_lig = prot_lig_all[s0:s1]
            surf_m = prot_lig > -0.20
            if int(surf_m.sum().item()) < 10:
                surf_m = torch.ones_like(prot_lig, dtype=torch.bool)
            f_mean = ag_frust[surf_m].mean(dim=0, keepdim=True)
            f_std = ag_frust[surf_m].std(dim=0, keepdim=True) + 1e-3
            ag_frust_z = ((ag_frust - f_mean) / f_std).clamp(-3.5, 3.5)
            e_mean = ag_esm[surf_m].mean(dim=0, keepdim=True)
            ag_esm_c = F.normalize(ag_esm - e_mean, dim=-1, eps=1e-6)

            target_stage_a_cache[sk] = {
                "lig_beads": tb,
                "lig_cb": item["t_cb"],
                "lig_has_phys_cb": item["t_has_cb"],
                "lig_aa": item["t_aa"],
                "lig_region": torch.zeros((len(tb),), device=dev, dtype=torch.long),
                "h_lig": h_lig_all[s0:s1],
                "frames_lig": f_lig_all[s0:s1],
                "sc_vec_lig": sc_lig_all[s0:s1],
                "o_vec_lig": o_lig_all[s0:s1],
                "prot_lig": prot_lig,
                "lig_phys": phys_tbl[item["t_aa"].clamp(0, 20)],
                "n_ag0": n_ag0,
                "ag_frust_z": ag_frust_z,
                "ag_esm_c": ag_esm_c,
                "t_esm_frust": ag_frust,
            }

    rng_a = np.random.default_rng(2026)
    frust_names = ["nll", "entropy", "hydro_frust", "arom_frust", "charge_frust", "exp_hydro"]
    decoy_slice = slice(7, 16)

    for c_start in range(0, N, chunk_size):
        chunk = parsed_items[c_start : c_start + chunk_size]
        M_c = len(chunk)
        b_b_l, b_cb_l, b_aa_l, b_role_l, b_reg_l, b_ch_l, b_ridx_l, b_bid_l = [], [], [], [], [], [], [], []
        b_slices = []
        c_off = 0
        for m, item in enumerate(chunk):
            bb = item["b_beads"]
            Nb = len(bb)
            min_d_b = item["d_ca0"].amin(dim=1)
            rec_region = torch.full((Nb,), REGION_AB_VFW, dtype=torch.long, device=dev)
            rim_mask = (min_d_b >= 8.0) & (min_d_b < 11.5)
            core_mask = min_d_b < 8.0
            if int(core_mask.sum().item()) < 4:
                _, top_idx = torch.topk(min_d_b, k=min(8, Nb), largest=False)
                core_mask = torch.zeros_like(core_mask)
                core_mask[top_idx] = True
            rec_region[rim_mask] = REGION_AB_CDR1
            rec_region[core_mask] = REGION_AB_CDRH3
            item["rec_region"] = rec_region

            b_b_l.append(bb)
            b_cb_l.append(item["b_cb"])
            b_aa_l.append(item["b_aa"])
            b_role_l.append(torch.zeros((Nb,), device=dev, dtype=torch.long))
            b_reg_l.append(rec_region)
            b_ch_l.append(item["b_chain"])
            b_ridx_l.append(torch.arange(Nb, device=dev, dtype=torch.long))
            b_bid_l.append(torch.full((Nb,), m, device=dev, dtype=torch.long))
            b_slices.append((c_off, c_off + Nb))
            c_off += Nb

        h_rec_all, f_rec_all, sc_rec_all, o_rec_all, prot_rec_all = compute_monomer_features_disjoint_a(
            ebm_a, torch.cat(b_b_l), torch.cat(b_cb_l), torch.cat(b_aa_l),
            torch.cat(b_role_l), torch.cat(b_reg_l), torch.cat(b_ch_l), torch.cat(b_ridx_l), torch.cat(b_bid_l),
        )

        rots_a_list, trans_a_list = generate_surface_packed_sabdab2_pool_batched_gpu(
            b_b_l, b_reg_l, [it["t_beads"] for it in chunk], rng_a, n_poses=16
        )

        enc_list = []
        for m, item in enumerate(chunk):
            s0, s1 = b_slices[m]
            bb = item["b_beads"]
            rec_region = item["rec_region"]
            t_cached = target_stage_a_cache[item["seq_key"]]
            cdr_centroid0, u_para0 = compute_antibody_paratope_geometry(bb[:, 1], rec_region)
            item["u_para0"] = u_para0

            bk = (tuple(item["rec"]["b_aa"]), tuple(item["rec"]["b_chain"]))
            b_esm = esm_seq_cache[bk]
            ab_esm = b_esm["esm_hidden"].to(dtype=dtype)
            item["b_frust"] = b_esm["esm_frust"]

            cdr_mask = rec_region >= REGION_AB_CDR1
            if cdr_mask.any():
                w_cdr_ab = (1.0 + 0.5 * (rec_region[cdr_mask] >= REGION_AB_CDRL3).to(dtype)).unsqueeze(-1)
                w_cdr_ab = w_cdr_ab / w_cdr_ab.sum().clamp_min(1e-6)
                ab_esm_pool = F.normalize((w_cdr_ab * ab_esm[cdr_mask]).sum(dim=0), dim=-1, eps=1e-6)
            else:
                ab_esm_pool = F.normalize(ab_esm.mean(dim=0), dim=-1, eps=1e-6)

            enc_list.append({
                "h_rec": h_rec_all[s0:s1],
                "frames_rec": f_rec_all[s0:s1],
                "sc_vec_rec": sc_rec_all[s0:s1],
                "o_vec_rec": o_rec_all[s0:s1],
                "prot_rec": prot_rec_all[s0:s1],
                "rec_beads": bb,
                "rec_cb": item["b_cb"],
                "rec_region": rec_region,
                "rec_has_phys_cb": item["b_has_cb"],
                "rec_phys": phys_tbl[item["b_aa"].clamp(0, 20)],
                "h_lig": t_cached["h_lig"],
                "frames_lig": t_cached["frames_lig"],
                "sc_vec_lig": t_cached["sc_vec_lig"],
                "o_vec_lig": t_cached["o_vec_lig"],
                "prot_lig": t_cached["prot_lig"],
                "lig_beads": item["t_beads"],
                "lig_cb": item["t_cb"],
                "lig_region": t_cached["lig_region"],
                "lig_has_phys_cb": item["t_has_cb"],
                "lig_phys": t_cached["lig_phys"],
                "receptor_is_antibody": True,
                "cdr_centroid0": cdr_centroid0,
                "u_para0": u_para0,
                "n_ag0": t_cached["n_ag0"],
                "has_esm": True,
                "ag_frust_z": t_cached["ag_frust_z"],
                "ag_esm_c": t_cached["ag_esm_c"],
                "ab_esm_pool": ab_esm_pool,
            })

        comp_out_batch = batched_score_stage_a_multi_complex(ebm_a, enc_list, rots_a_list, trans_a_list)
        comp_out_np = {k: v.view(M_c, 16).detach().cpu().numpy().astype(np.float64) for k, v in comp_out_batch.items()}

        for m, item in enumerate(chunk):
            feats = item["feats"]
            for k_name, mat in comp_out_np.items():
                arr = mat[m]
                v0 = float(arr[0])
                v_near = float(np.mean(arr[1:7]))
                v_dec_mean = float(np.mean(arr[decoy_slice]))
                v_dec_min = float(np.min(arr[decoy_slice]))
                v_dec_std = float(np.std(arr[decoy_slice])) + 1e-4
                feats[f"ebm_{k_name}_design"] = v0
                feats[f"ebm_{k_name}_near"] = v_near
                feats[f"ebm_{k_name}_decoy_mean"] = v_dec_mean
                feats[f"ebm_{k_name}_funnel_gap"] = v0 - v_dec_mean
                feats[f"ebm_{k_name}_hardneg_margin"] = v0 - v_dec_min
                feats[f"ebm_{k_name}_zscore"] = (v0 - v_dec_mean) / v_dec_std

            bb = item["b_beads"]
            tb = item["t_beads"]
            d_ca = item["d_ca0"]
            min_d_b = d_ca.amin(dim=1)
            min_d_t = d_ca.amin(dim=0)
            t_near16 = min_d_t < 16.0
            if not bool(t_near16.any()):
                t_near16 = torch.ones((len(tb),), dtype=torch.bool, device=dev)
            t_beads_near16 = tb[t_near16]
            t_aa_near16 = item["t_aa"][t_near16]
            item["t_beads_near16"] = t_beads_near16
            item["t_aa_near16"] = t_aa_near16

            d_sc = torch.cdist(bb[:, 4], t_beads_near16[:, 4])
            all_bead_d = torch.cdist(bb.reshape(-1, 3), t_beads_near16.reshape(-1, 3))
            intf_b_mask = min_d_b < 8.5
            intf_t_mask = min_d_t < 8.5
            if not bool(intf_b_mask.any()):
                intf_b_mask = torch.ones((len(bb),), dtype=torch.bool, device=dev)
            if not bool(intf_t_mask.any()):
                intf_t_mask = torch.ones((len(tb),), dtype=torch.bool, device=dev)

            t_cached = target_stage_a_cache[item["seq_key"]]
            b_frust = item["b_frust"]
            t_frust = t_cached["t_esm_frust"]
            b_f_all = b_frust.mean(dim=0).detach().cpu().numpy()
            b_f_intf = b_frust[intf_b_mask].mean(dim=0).detach().cpu().numpy()
            t_f_intf = t_frust[intf_t_mask].mean(dim=0).detach().cpu().numpy()

            feats["binder_length"] = float(len(bb))
            feats["intf_n_ca8"] = float((d_ca < 8.0).sum().item())
            feats["intf_n_sc5"] = float((d_sc < 5.0).sum().item())
            feats["intf_min_bead_dist"] = float(all_bead_d.min().item())
            feats["intf_binder_frac"] = float(intf_b_mask.to(dtype).mean().item())
            feats["intf_head_on_cos"] = float(-(item["u_para0"] * t_cached["n_ag0"][intf_t_mask].mean(dim=0)).sum().item())

            for f_i, f_n in enumerate(frust_names):
                feats[f"esm_binder_all_{f_n}"] = float(b_f_all[f_i])
                feats[f"esm_binder_intf_{f_n}"] = float(b_f_intf[f_i])
                feats[f"esm_target_intf_{f_n}"] = float(t_f_intf[f_i])
                feats[f"esm_binder_intf_minus_all_{f_n}"] = float(b_f_intf[f_i] - b_f_all[f_i])

    if dev.type == "cuda":
        torch.cuda.synchronize()
    stage_a_sec = time.perf_counter() - t0_a

    # --- Pass 2 (Stage B): Disjoint 3D IF GNN + 3D-Batched Monomer & Interface Physics ---
    t0_b = time.perf_counter()
    for c_start in range(0, N, chunk_size):
        chunk = parsed_items[c_start : c_start + chunk_size]
        sb_dicts = compute_stage_b_batched_gpu(
            if_model_b,
            [it["b_beads"] for it in chunk],
            [it["b_aa"] for it in chunk],
            [it["b_chain"] for it in chunk],
            [it["t_beads"] for it in chunk],
            [it["t_aa"] for it in chunk],
            [it["t_chain"] for it in chunk],
            [it["t_beads_near16"] for it in chunk],
            [it["t_aa_near16"] for it in chunk],
            [it["feats"]["esm_binder_all_nll"] for it in chunk],
        )
        for it, d_b in zip(chunk, sb_dicts):
            it["feats"].update(d_b)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    stage_b_sec = time.perf_counter() - t0_b

    # --- Pass 3, 4, 5 (Stage C): Target Pre-Encoding + Disjoint Autograd Forces + Sidechain/Re-Dock/Alchem + 6-DoF Relax ---
    t0_c_total = time.perf_counter()
    stage_c_autograd_sec = 0.0
    stage_c_sc_redock_sec = 0.0
    stage_c_relax_sec = 0.0

    # Target pre-encoding for Stage C (once per unique receptor)
    target_seq_cache_c = {}
    for item in parsed_items:
        sk = item["seq_key"]
        if sk not in target_seq_cache_c:
            tb = item["t_beads"]
            ta = item["t_aa"]
            tc = item["t_chain"]
            Nt = len(tb)
            t0_g = time.perf_counter()
            with torch.enable_grad():
                tb_g = tb.clone().requires_grad_(True)
                t_geom0 = compute_pure_backbone_geometry_disjoint(
                    tb_g, tc, torch.arange(Nt, device=dev, dtype=torch.long), torch.zeros(Nt, device=dev, dtype=torch.long)
                )
                lp_t0 = if_model_c.forward_from_geom(t_geom0)
                ll_t0_sum = lp_t0[torch.arange(Nt, device=dev), ta.clamp(0, 19)].sum()
                grad_t0 = torch.autograd.grad(ll_t0_sum, tb_g)[0].detach()
                f_t_apo = torch.norm(grad_t0[:, :4].reshape(Nt, -1), dim=-1)
                lp_t0 = lp_t0.detach()
                t_geom0 = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in t_geom0.items()}
            if dev.type == "cuda":
                torch.cuda.synchronize()
            stage_c_autograd_sec += time.perf_counter() - t0_g

            t0_sc = time.perf_counter()
            with torch.no_grad():
                t_sc_hat, u_local_t, t_sigma, t_tip_hat, u_tip_local_t, t_sigma_tip = sc_model_c.forward_from_geom(
                    t_geom0, ta, return_tip=True
                )
                h_t = ebm_model_c.encode_partner(t_geom0, ta)
                t_erased = tb.clone()
                t_erased[:, 4] = t_sc_hat
                t_sig_comb = 0.5 * (t_sigma + t_sigma_tip)

                t_ca = tb[:, 1]
                d_tt = torch.cdist(t_ca, t_ca)
                w12 = (d_tt < 12.0).to(dtype)
                n_surf = F.normalize(t_ca - (w12 @ t_ca) / w12.sum(dim=-1, keepdim=True).clamp_min(1.0), dim=-1, eps=1e-6)
                burial_t = (d_tt < 10.0).to(dtype).sum(dim=-1)
                surf_idx = torch.where(burial_t <= torch.quantile(burial_t, 0.65))[0]
                if surf_idx.numel() == 0:
                    surf_idx = torch.arange(Nt, device=dev)

                d_tt_m = d_tt.masked_fill(torch.eye(Nt, dtype=torch.bool, device=dev), 1e6)
                bur6 = (d_tt_m < 6.0).to(dtype).sum(dim=-1)
                bur14 = (d_tt_m < 14.0).to(dtype).sum(dim=-1)
                target_seq_cache_c[sk] = {
                    "t_sc_hat": t_sc_hat,
                    "t_tip_hat": t_tip_hat,
                    "t_erased": t_erased,
                    "t_sig_comb": t_sig_comb,
                    "h_t": h_t,
                    "lp_t_apo": lp_t0,
                    "f_t_apo": f_t_apo,
                    "concavity": bur14 / (bur6 + 2.0),
                    "n_surf": n_surf,
                    "surf_idx": surf_idx,
                }
            if dev.type == "cuda":
                torch.cuda.synchronize()
            stage_c_sc_redock_sec += time.perf_counter() - t0_sc

    rng_c = np.random.default_rng(2026)
    I_rot = torch.eye(3, dtype=dtype, device=dev).unsqueeze(0)
    Z_tr = torch.zeros((1, 3), dtype=dtype, device=dev)
    polar_pool = torch.tensor([3, 6, 11, 1, 2, 5], dtype=torch.long, device=dev)
    alphas_relax = torch.tensor([0.20, 0.45, 0.85], dtype=dtype, device=dev)

    for c_start in range(0, N, chunk_size):
        chunk = parsed_items[c_start : c_start + chunk_size]
        M_c = len(chunk)

        # --- Pass 4: Single Batched Disjoint Autograd Pass for Apo Binders + Holo Complexes ---
        t0_g = time.perf_counter()
        apo_beads_l, apo_ch_l, apo_ridx_l, apo_bid_l, apo_aa_l = [], [], [], [], []
        holo_beads_l, holo_ch_l, holo_ridx_l, holo_bid_l, holo_aa_l = [], [], [], [], []
        apo_slices, holo_slices = [], []
        apo_off, holo_off = 0, 0

        for m, item in enumerate(chunk):
            bb = item["b_beads"]
            tb = item["t_beads"]
            ba = item["b_aa"]
            ta = item["t_aa"]
            Nb = len(bb)
            d_ca0 = item["d_ca0"]
            min_d_b = d_ca0.amin(dim=1)
            min_d_t = d_ca0.amin(dim=0)
            b_intf = min_d_b < 9.0
            t_near_idx = torch.where(min_d_t < 15.0)[0]
            if t_near_idx.numel() < 5:
                t_near_idx = torch.argsort(min_d_t)[: min(25, len(tb))]
            Nt_n = len(t_near_idx)
            item["b_intf"] = b_intf
            item["t_near_idx"] = t_near_idx

            apo_beads_l.append(bb)
            apo_ch_l.append(item["b_chain"])
            apo_ridx_l.append(torch.arange(Nb, device=dev, dtype=torch.long))
            apo_bid_l.append(torch.full((Nb,), m, device=dev, dtype=torch.long))
            apo_aa_l.append(ba)
            apo_slices.append((apo_off, apo_off + Nb))
            apo_off += Nb

            comp_aa = torch.cat([ba, ta[t_near_idx]], dim=0)
            comp_ch = torch.cat([torch.zeros(Nb, dtype=torch.long, device=dev), torch.ones(Nt_n, dtype=torch.long, device=dev)], dim=0)
            comp_ridx = torch.cat([torch.arange(Nb, dtype=torch.long, device=dev), t_near_idx + 1000], dim=0)
            item["comp_aa"] = comp_aa

            holo_beads_l.append(torch.cat([bb, tb[t_near_idx]], dim=0))
            holo_ch_l.append(comp_ch)
            holo_ridx_l.append(comp_ridx)
            holo_bid_l.append(torch.full((Nb + Nt_n,), M_c + m, device=dev, dtype=torch.long))
            holo_aa_l.append(comp_aa)
            holo_slices.append((holo_off, holo_off + Nb + Nt_n))
            holo_off += Nb + Nt_n

        with torch.enable_grad():
            all_beads_g = torch.cat(apo_beads_l + holo_beads_l, dim=0).clone().requires_grad_(True)
            all_ch = torch.cat(apo_ch_l + holo_ch_l, dim=0)
            all_ridx = torch.cat(apo_ridx_l + holo_ridx_l, dim=0)
            all_bid = torch.cat(apo_bid_l + holo_bid_l, dim=0)
            all_aa_c = torch.cat(apo_aa_l + holo_aa_l, dim=0).clamp(0, 19)

            geom_both_g = compute_pure_backbone_geometry_disjoint(all_beads_g, all_ch, all_ridx, all_bid)
            lp_both_g = if_model_c.forward_from_geom(geom_both_g)
            ll_both_vec = lp_both_g[torch.arange(len(all_beads_g), device=dev), all_aa_c]
            grad_both = torch.autograd.grad(ll_both_vec.sum(), all_beads_g)[0].detach()
            f_both = torch.norm(grad_both[:, :4].reshape(len(all_beads_g), -1), dim=-1)
            ll_both = ll_both_vec.detach()

            N_apo = apo_off
            edge_apo_m = geom_both_g["src"] < N_apo
            b_geom_cat = {
                "node_feats": geom_both_g["node_feats"][:N_apo].detach(),
                "src": geom_both_g["src"][edge_apo_m].detach(),
                "dst": geom_both_g["dst"][edge_apo_m].detach(),
                "edge_attr": geom_both_g["edge_attr"][edge_apo_m].detach(),
                "burial_cb10": geom_both_g["burial_cb10"][:N_apo].detach(),
                "frames": geom_both_g["frames"][:N_apo].detach(),
                "ca_pos": geom_both_g["ca_pos"][:N_apo].detach(),
            }
            edge_holo_m = geom_both_g["src"] >= N_apo
            cg_cat = {
                "node_feats": geom_both_g["node_feats"][N_apo:].detach(),
                "src": (geom_both_g["src"][edge_holo_m] - N_apo).detach(),
                "dst": (geom_both_g["dst"][edge_holo_m] - N_apo).detach(),
                "edge_attr": geom_both_g["edge_attr"][edge_holo_m].detach(),
                "frames": geom_both_g["frames"][N_apo:].detach(),
                "ca_pos": geom_both_g["ca_pos"][N_apo:].detach(),
            }
            del geom_both_g, lp_both_g, ll_both_vec, all_beads_g, grad_both

        if dev.type == "cuda":
            torch.cuda.synchronize()
        stage_c_autograd_sec += time.perf_counter() - t0_g

        # --- Pass 3: Disjoint Sidechain Erasure + Re-Docking + Alchemical Counterfactuals ---
        t0_sc = time.perf_counter()

        apo_aa_cat = torch.cat(apo_aa_l, dim=0)
        holo_aa_cat = torch.cat(holo_aa_l, dim=0)

        # Build alchemical counterfactual sequences m_idx=2 (Buns) and m_idx=3 (Ala) & advance rng_c identically
        m2_aa_l, m3_aa_l = [], []
        chosen_surf_l, R_glob_l = [], []
        for m, item in enumerate(chunk):
            t_info = target_seq_cache_c[item["seq_key"]]
            surf_idx = t_info["surf_idx"]
            # 1) RNG calls for build_redocking_pose_ensemble(n_global=11)
            c_idx_np = rng_c.choice(len(surf_idx), size=11, replace=True)
            r_rv_np = rng_c.normal(0.0, 1.2, size=(11, 3)).astype(np.float32)
            chosen_surf_l.append(surf_idx[torch.from_numpy(c_idx_np).to(dev)])
            R_glob_l.append(_axis_angle_to_matrix(torch.from_numpy(r_rv_np).to(dev)))

            # 2) RNG calls & sequence construction for build_inplace_mirage_sequences
            ba = item["b_aa"]
            min_d = item["d_ca0"].amin(dim=1)
            intf_idx = torch.where(min_d < 9.0)[0]
            core_idx = torch.where(min_d < 6.8)[0]
            if intf_idx.numel() < 3:
                intf_idx = torch.argsort(min_d)[: min(8, len(ba))]
            if core_idx.numel() < 2:
                core_idx = intf_idx[: max(2, len(intf_idx) // 2)]
            _ = rng_c.permutation(len(intf_idx))
            _ = rng_c.permutation(len(intf_idx))

            aa_buns = ba.clone()
            aa_buns[core_idx] = polar_pool[torch.arange(len(core_idx), device=dev) % len(polar_pool)]
            aa_ala = ba.clone()
            aa_ala[intf_idx] = 0
            m2_aa_l.append(aa_buns)
            m3_aa_l.append(aa_ala)

        m2_aa_cat = torch.cat(m2_aa_l, dim=0)
        m3_aa_cat = torch.cat(m3_aa_l, dim=0)

        # Batched sc_model_c forward passes
        b_sc_all, _, b_sig_all, b_tip_all, _, b_sigt_all, mdn_b_all = sc_model_c.forward_from_geom(
            b_geom_cat, apo_aa_cat, return_tip=True, return_mdn=True
        )
        # Concatenate holo + m2 + m3 into a single disjoint graph for sc_model_c (return_tip=True)
        comb_sc_geom = {
            "node_feats": torch.cat([cg_cat["node_feats"], b_geom_cat["node_feats"], b_geom_cat["node_feats"]], dim=0),
            "src": torch.cat([cg_cat["src"], b_geom_cat["src"] + holo_off, b_geom_cat["src"] + holo_off + N_apo], dim=0),
            "dst": torch.cat([cg_cat["dst"], b_geom_cat["dst"] + holo_off, b_geom_cat["dst"] + holo_off + N_apo], dim=0),
            "edge_attr": torch.cat([cg_cat["edge_attr"], b_geom_cat["edge_attr"], b_geom_cat["edge_attr"]], dim=0),
            "frames": torch.cat([cg_cat["frames"], b_geom_cat["frames"], b_geom_cat["frames"]], dim=0),
            "ca_pos": torch.cat([cg_cat["ca_pos"], b_geom_cat["ca_pos"], b_geom_cat["ca_pos"]], dim=0),
        }
        comb_sc_aa = torch.cat([holo_aa_cat, m2_aa_cat, m3_aa_cat], dim=0)
        sc_hm_all, _, sig_hm_all, tip_hm_all, _, sigt_hm_all = sc_model_c.forward_from_geom(
            comb_sc_geom, comb_sc_aa, return_tip=True
        )
        sc_comp_all = sc_hm_all[:holo_off]
        sig_comp_comb_all = 0.5 * (sig_hm_all[:holo_off] + sigt_hm_all[:holo_off])
        tip_comp_all = tip_hm_all[:holo_off]

        m2_sc_all = sc_hm_all[holo_off : holo_off + N_apo]
        m2_tip_all = tip_hm_all[holo_off : holo_off + N_apo]
        m2_sig_all = 0.5 * (sig_hm_all[holo_off : holo_off + N_apo] + sigt_hm_all[holo_off : holo_off + N_apo])

        m3_sc_all = sc_hm_all[holo_off + N_apo :]
        m3_tip_all = tip_hm_all[holo_off + N_apo :]
        m3_sig_all = 0.5 * (sig_hm_all[holo_off + N_apo :] + sigt_hm_all[holo_off + N_apo :])

        # Single disjoint-batch ebm_model_c.encode_partner for (apo, m2, m3)
        enc_3x_geom = {
            "node_feats": torch.cat([b_geom_cat["node_feats"]] * 3, dim=0),
            "src": torch.cat([b_geom_cat["src"], b_geom_cat["src"] + N_apo, b_geom_cat["src"] + 2 * N_apo], dim=0),
            "dst": torch.cat([b_geom_cat["dst"], b_geom_cat["dst"] + N_apo, b_geom_cat["dst"] + 2 * N_apo], dim=0),
            "edge_attr": torch.cat([b_geom_cat["edge_attr"]] * 3, dim=0),
        }
        h_b_3x = ebm_model_c.encode_partner(enc_3x_geom, torch.cat([apo_aa_cat, m2_aa_cat, m3_aa_cat], dim=0))
        h_b_all = h_b_3x[:N_apo]
        h_bm2_all = h_b_3x[N_apo : 2 * N_apo]
        h_bm3_all = h_b_3x[2 * N_apo :]

        # Assemble multi-complex pose tasks for batched_score_poses_multi_complex:
        # Per complex m: Task 4m = input_sc (1 pose), Task 4m+1 = erased 24-pose, Task 4m+2 = m2 (1 pose), Task 4m+3 = m3 (1 pose)
        sc_b_list, sc_btip_list, sc_baa_list, sc_hb_list, sc_bsig_list = [], [], [], [], []
        sc_t_list, sc_ttip_list, sc_taa_list, sc_ht_list, sc_tsig_list = [], [], [], [], []
        sc_rots_list, sc_trans_list = [], []
        pose_rmsd_list = []

        for m, item in enumerate(chunk):
            s0, s1 = apo_slices[m]
            bb = item["b_beads"]
            ba = item["b_aa"]
            tb = item["t_beads"]
            ta = item["t_aa"]
            t_info = target_seq_cache_c[item["seq_key"]]

            b_sc_hat = b_sc_all[s0:s1]
            b_tip_hat = b_tip_all[s0:s1]
            b_sig_comb = 0.5 * (b_sig_all[s0:s1] + b_sigt_all[s0:s1])
            b_erased = bb.clone()
            b_erased[:, 4] = b_sc_hat
            h_b = h_b_all[s0:s1]

            item["b_sc_hat"] = b_sc_hat
            item["b_tip_hat"] = b_tip_hat
            item["b_sigma"] = b_sig_all[s0:s1]
            item["b_sig_comb"] = b_sig_comb
            item["b_erased"] = b_erased
            item["h_b"] = h_b
            item["x_sc_modes"] = mdn_b_all["x_sc_modes"][s0:s1]

            t_erased = t_info["t_erased"]
            t_tip_hat = t_info["t_tip_hat"]
            t_sig_comb = t_info["t_sig_comb"]
            h_t = t_info["h_t"]

            rots_c, trans_c, pose_rmsd = build_redocking_pose_ensemble_gpu(
                b_erased, t_erased, t_info["n_surf"], t_info["surf_idx"], chosen_surf_l[m], R_glob_l[m]
            )
            pose_rmsd_list.append(pose_rmsd)

            # Crop target to <22A for single-pose tasks (input_sc, m2, m3) and 24-pose task
            tk_0 = item["d_ca0"].amin(dim=0) < 22.0
            if not bool(tk_0.any()):
                tk_0[0] = True

            b_cent = b_erased[:, 1].mean(dim=0, keepdim=True)
            b_posed_ca = torch.einsum("bij,nj->bni", rots_c, b_erased[:, 1] - b_cent) + b_cent[None, :, :] + trans_c[:, None, :]
            d_ca_24 = torch.cdist(b_posed_ca, t_erased[None, :, 1].expand(24, -1, -1))
            tk_24 = d_ca_24.amin(dim=(0, 1)) < 22.0
            if not bool(tk_24.any()):
                tk_24[0] = True

            # Task 4m: input_sc (1 pose)
            sc_b_list.append(bb)
            sc_btip_list.append(bb[:, 4])
            sc_baa_list.append(ba)
            sc_hb_list.append(h_b)
            sc_bsig_list.append(b_sig_comb)
            sc_t_list.append(tb[tk_0])
            sc_ttip_list.append(tb[tk_0, 4])
            sc_taa_list.append(ta[tk_0])
            sc_ht_list.append(h_t[tk_0])
            sc_tsig_list.append(t_sig_comb[tk_0])
            sc_rots_list.append(I_rot)
            sc_trans_list.append(Z_tr)

            # Task 4m+1: erased 24-pose
            sc_b_list.append(b_erased)
            sc_btip_list.append(b_tip_hat)
            sc_baa_list.append(ba)
            sc_hb_list.append(h_b)
            sc_bsig_list.append(b_sig_comb)
            sc_t_list.append(t_erased[tk_24])
            sc_ttip_list.append(t_tip_hat[tk_24])
            sc_taa_list.append(ta[tk_24])
            sc_ht_list.append(h_t[tk_24])
            sc_tsig_list.append(t_sig_comb[tk_24])
            sc_rots_list.append(rots_c)
            sc_trans_list.append(trans_c)

            # Task 4m+2: mirage m2 (1 pose)
            m2_erased = bb.clone()
            m2_erased[:, 4] = m2_sc_all[s0:s1]
            sc_b_list.append(m2_erased)
            sc_btip_list.append(m2_tip_all[s0:s1])
            sc_baa_list.append(m2_aa_l[m])
            sc_hb_list.append(h_bm2_all[s0:s1])
            sc_bsig_list.append(m2_sig_all[s0:s1])
            sc_t_list.append(t_erased[tk_0])
            sc_ttip_list.append(t_tip_hat[tk_0])
            sc_taa_list.append(ta[tk_0])
            sc_ht_list.append(h_t[tk_0])
            sc_tsig_list.append(t_sig_comb[tk_0])
            sc_rots_list.append(I_rot)
            sc_trans_list.append(Z_tr)

            # Task 4m+3: mirage m3 (1 pose)
            m3_erased = bb.clone()
            m3_erased[:, 4] = m3_sc_all[s0:s1]
            sc_b_list.append(m3_erased)
            sc_btip_list.append(m3_tip_all[s0:s1])
            sc_baa_list.append(m3_aa_l[m])
            sc_hb_list.append(h_bm3_all[s0:s1])
            sc_bsig_list.append(m3_sig_all[s0:s1])
            sc_t_list.append(t_erased[tk_0])
            sc_ttip_list.append(t_tip_hat[tk_0])
            sc_taa_list.append(ta[tk_0])
            sc_ht_list.append(h_t[tk_0])
            sc_tsig_list.append(t_sig_comb[tk_0])
            sc_rots_list.append(I_rot)
            sc_trans_list.append(Z_tr)

        out_sc_batch = batched_score_poses_multi_complex(
            ebm_model_c,
            sc_b_list, sc_btip_list, sc_baa_list, sc_hb_list, sc_bsig_list,
            sc_t_list, sc_ttip_list, sc_taa_list, sc_ht_list, sc_tsig_list,
            sc_rots_list, sc_trans_list,
            return_components=True,
        )

        # Unpack Pass 3 & Pass 4 features per complex
        for m, item in enumerate(chunk):
            s0, s1 = apo_slices[m]
            h0, h1 = holo_slices[m]
            bb = item["b_beads"]
            ba = item["b_aa"]
            tb = item["t_beads"]
            ta = item["t_aa"]
            Nb = len(bb)
            valid_b = ba < 20
            b_intf = item["b_intf"]
            t_near_idx = item["t_near_idx"]
            t_info = target_seq_cache_c[item["seq_key"]]

            ll_b = ll_both[s0:s1]
            f_b_apo = f_both[s0:s1]
            ll_comp_m = ll_both[N_apo + h0 : N_apo + h1]
            f_comp_m = f_both[N_apo + h0 : N_apo + h1]
            ll_b_holo = ll_comp_m[:Nb]
            ll_t_holo = ll_comp_m[Nb:]

            bur_cb10 = b_geom_cat["burial_cb10"][s0:s1]
            core_m = bur_cb10 > bur_cb10.median()
            gen_if_ll_all = float(ll_b[valid_b].mean().item())
            if Nb >= 5:
                win5_ll = F.avg_pool1d(ll_b.view(1, 1, -1), kernel_size=5, stride=1).view(-1)
                gen_if_worst_win5 = float(win5_ll.min().item())
            else:
                gen_if_worst_win5 = gen_if_ll_all

            ta_n = ta[t_near_idx].clamp(0, 19)
            ll_t_apo = t_info["lp_t_apo"][t_near_idx, ta_n]
            d_ca0 = item["d_ca0"]
            t_intf_n = (d_ca0[:, t_near_idx].amin(dim=0) < 9.0) & (ta[t_near_idx] < 20)
            b_intf_v = b_intf & valid_b
            holo_if_b_intf = float(ll_b_holo[b_intf_v].mean().item()) if b_intf_v.any() else float(ll_b_holo.mean().item())
            holo_minus_apo_if_t = float((ll_t_holo[t_intf_n] - ll_t_apo[t_intf_n]).mean().item()) if t_intf_n.any() else 0.0

            bb_force_norm_all = float(f_b_apo.mean().item())
            bb_force_norm_intf = float(f_b_apo[b_intf_v].mean().item()) if b_intf_v.any() else bb_force_norm_all
            bb_force_core = float(f_b_apo[core_m].mean().item()) if core_m.any() else bb_force_norm_all
            bb_force_intf_minus_core = bb_force_norm_intf - bb_force_core
            bb_force_holo_b_intf = float(f_comp_m[:Nb][b_intf_v].mean().item()) if b_intf_v.any() else bb_force_norm_intf
            bb_force_holo_minus_apo_b = bb_force_holo_b_intf - bb_force_norm_intf
            bb_force_holo_minus_apo_t = float((f_comp_m[Nb:][t_intf_n] - t_info["f_t_apo"][t_near_idx][t_intf_n]).mean().item()) if t_intf_n.any() else 0.0

            R_b = b_geom_cat["frames"][s0:s1]
            b_aa_c20 = ba.clamp(0, 20)
            b_sc_tmpl = bb[:, 1] + torch.einsum("nij,nj->ni", R_b, tmpl_u_sc[b_aa_c20])
            b_sc_hat = item["b_sc_hat"]
            b_sigma = item["b_sigma"]

            non_gly = valid_b & (ba != 5)
            sc_dist = torch.sqrt(((bb[:, 4] - b_sc_hat) ** 2).sum(dim=-1) + 1e-8)
            sc_z = sc_dist / b_sigma.clamp_min(0.15)
            sc_dist_tmpl_vec = torch.sqrt(((bb[:, 4] - b_sc_tmpl) ** 2).sum(dim=-1) + 1e-8)
            intf_ng = b_intf & non_gly
            nonintf_ng = (~b_intf) & non_gly

            x_sc_modes = item["x_sc_modes"]
            d_modes = torch.sqrt(((bb[:, 4].unsqueeze(1) - x_sc_modes) ** 2).sum(dim=-1) + 1e-8)
            sc_min_mode_dist = d_modes.amin(dim=-1)

            v_sup = bb[:, 4] - bb[:, 1]
            v_rec = b_sc_hat - bb[:, 1]
            r_sup = torch.linalg.norm(v_sup, dim=-1).clamp_min(1e-5)
            r_rec = torch.linalg.norm(v_rec, dim=-1).clamp_min(1e-5)
            cos_ang = ((v_sup * v_rec).sum(dim=-1) / (r_sup * r_rec)).clamp(-1.0, 1.0)
            sc_angular_strain = 1.0 - cos_ang

            if intf_ng.any():
                sc_intf_min_mode_dist = float(sc_min_mode_dist[intf_ng].mean().item())
                sc_intf_angular_strain = float(sc_angular_strain[intf_ng].mean().item())
                sc_intf_dist_tmpl = float(sc_dist_tmpl_vec[intf_ng].mean().item())
            else:
                sc_intf_min_mode_dist = float(sc_min_mode_dist[non_gly].mean().item()) if non_gly.any() else 0.0
                sc_intf_angular_strain = 0.0
                sc_intf_dist_tmpl = float(sc_dist_tmpl_vec[non_gly].mean().item()) if non_gly.any() else 0.0

            sc_intf_dist = float(sc_dist[intf_ng].mean().item()) if intf_ng.any() else float(sc_dist[non_gly].mean().item())
            sc_nonintf_dist = float(sc_dist[nonintf_ng].mean().item()) if nonintf_ng.any() else sc_intf_dist
            sc_intf_z_strain = float(sc_z[intf_ng].mean().item()) if intf_ng.any() else 0.0

            if intf_ng.any():
                k_hot = min(3, int(intf_ng.sum().item()))
                sc_intf_top3_dist_tmpl = float(torch.topk(sc_dist_tmpl_vec[intf_ng], k=k_hot).values.mean().item())
                d_min_b = d_ca0.amin(dim=1)
                w_core = torch.sigmoid((7.5 - d_min_b[intf_ng]) / 0.8) + 1e-4
                sc_core_weighted_z = float(((sc_z[intf_ng] * w_core).sum() / w_core.sum()).item())
            else:
                sc_intf_top3_dist_tmpl = sc_intf_dist_tmpl
                sc_core_weighted_z = sc_intf_z_strain

            # Slice pose energies from out_sc_batch (27 poses per complex: 0=in, 1..24=erased, 25=m2, 26=m3)
            p_base = m * 27
            e0_input = float(out_sc_batch["total"][p_base].item())
            e_tot = out_sc_batch["total"][p_base + 1 : p_base + 25]
            e0_erased = float(e_tot[0].item())
            mirage_clash_jump = float((out_sc_batch["clash"][p_base + 1] - out_sc_batch["clash"][p_base]).item())
            e_m2 = float(out_sc_batch["total"][p_base + 25].item())
            e_m3 = float(out_sc_batch["total"][p_base + 26].item())

            alchem_buns_mirage_gap = e0_erased - e_m2
            alchem_ala_scan_ddg = e0_erased - e_m3

            pose_rmsd = pose_rmsd_list[m]
            e_local = e_tot[:13]
            w_boltz = F.softmax(-e_tot / 0.50, dim=0)
            redock_boltz_rmsd_t05 = float((w_boltz * pose_rmsd).sum().item())

            far_m = pose_rmsd > 5.0
            e_far_mean = float(e_tot[far_m].mean().item()) if far_m.any() else float(e_tot[1:].mean().item())
            e_far_std = float(e_tot[far_m].std().item()) + 1e-4
            redock_global_zscore = (float(e_local.min().item()) - e_far_mean) / e_far_std

            n_intf_c = float((d_ca0 < 8.5).sum().item()) + 1.0

            b_ca = bb[:, 1]
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

            ss_helix_all = float(is_helix.to(dtype).mean().item())
            ss_sheet_all = float(is_sheet.to(dtype).mean().item())
            if b_intf.any():
                ss_helix_intf = float(is_helix[b_intf].to(dtype).mean().item())
                ss_sheet_intf = float(is_sheet[b_intf].to(dtype).mean().item())
                ss_loop_intf = float(is_loop[b_intf].to(dtype).mean().item())
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
            t_cons = contact_counts[item["seq_key"]]
            if t_intf.any() and len(t_cons) == len(t_intf):
                epi_concavity = float(t_info["concavity"][t_intf].mean().item())
                epi_consensus = float(t_cons[t_intf].mean().item())
            else:
                epi_concavity = float(t_info["concavity"].mean().item())
                epi_consensus = 0.0

            buns_b0 = float(out_sc_batch["buns_binder"][p_base + 1].item())
            buns_t0 = float(out_sc_batch["buns_target"][p_base + 1].item())

            item["feats"].update({
                "gen_if_worst_win5": gen_if_worst_win5,
                "sc_intf_dist": sc_intf_dist,
                "sc_intf_minus_nonintf_dist": sc_intf_dist - sc_nonintf_dist,
                "sc_intf_z_strain": sc_intf_z_strain,
                "sc_intf_min_mode_dist": sc_intf_min_mode_dist,
                "sc_intf_angular_strain": sc_intf_angular_strain,
                "sc_core_weighted_z": sc_core_weighted_z,
                "sc_intf_top3_dist_tmpl": sc_intf_top3_dist_tmpl,
                "mirage_clash_jump": mirage_clash_jump / n_intf_c,
                "gen_ebm_e0_erased": e0_erased,
                "gen_ebm_e0_input": e0_input,
                "gen_ebm_e0_per_contact": e0_erased / n_intf_c,
                "gen_ebm_vdw_e0": float(out_sc_batch["vdw"][p_base + 1].item()) / n_intf_c,
                "gen_ebm_rot_ent_e0": float(out_sc_batch["rot_ent"][p_base + 1].item()) / n_intf_c,
                "gen_ebm_buns_total_e0": (buns_b0 + buns_t0) / n_intf_c,
                "alchem_buns_mirage_gap": alchem_buns_mirage_gap,
                "alchem_ala_scan_ddg": alchem_ala_scan_ddg,
                "redock_boltz_rmsd_t05": redock_boltz_rmsd_t05,
                "redock_global_zscore": redock_global_zscore,
                "ss_sheet_all": ss_sheet_all,
                "ss_helix_intf": ss_helix_intf,
                "ss_sheet_intf": ss_sheet_intf,
                "ss_loop_intf": ss_loop_intf,
                "ss_max_helix_intf_run": float(max_h_run),
                "epi_concavity": epi_concavity,
                "epi_consensus_density": epi_consensus,
                "holo_if_b_intf": holo_if_b_intf,
                "holo_minus_apo_if_t": holo_minus_apo_if_t,
                "bb_force_intf_minus_core": bb_force_intf_minus_core,
                "bb_force_holo_minus_apo_b": bb_force_holo_minus_apo_b,
                "bb_force_holo_minus_apo_t": bb_force_holo_minus_apo_t,
            })
            item["sc_comp_m"] = sc_comp_all[h0:h1]
            item["tip_comp_m"] = tip_comp_all[h0:h1]
            item["sig_comp_comb_m"] = sig_comp_comb_all[h0:h1]

        if dev.type == "cuda":
            torch.cuda.synchronize()
        stage_c_sc_redock_sec += time.perf_counter() - t0_sc

        # --- Pass 5: Batched Continuous 6-DoF Autograd Relaxation ---
        t0_rel = time.perf_counter()
        rel_b_l, rel_btip_l, rel_baa_l, rel_hb_l, rel_bsig_l = [], [], [], [], []
        rel_t_l, rel_ttip_l, rel_taa_l, rel_ht_l, rel_tsig_l = [], [], [], [], []
        holo_b_l, holo_btip_l, holo_bsig_l = [], [], []
        holo_t_l, holo_ttip_l, holo_tsig_l = [], [], []
        u_norm_rel_l = []

        for m, item in enumerate(chunk):
            bb = item["b_beads"]
            Nb = len(bb)
            t_near_idx = item["t_near_idx"]
            t_info = target_seq_cache_c[item["seq_key"]]
            t_b_n = item["t_beads"][t_near_idx]
            t_a_n = item["t_aa"][t_near_idx]
            h_t_n = t_info["h_t"][t_near_idx]
            t_erased_n = t_info["t_erased"][t_near_idx]
            t_tip_n = t_info["t_tip_hat"][t_near_idx]
            t_sig_n = t_info["t_sig_comb"][t_near_idx]

            rel_b_l.append(item["b_erased"])
            rel_btip_l.append(item["b_tip_hat"])
            rel_baa_l.append(item["b_aa"])
            rel_hb_l.append(item["h_b"])
            rel_bsig_l.append(item["b_sig_comb"])
            rel_t_l.append(t_erased_n)
            rel_ttip_l.append(t_tip_n)
            rel_taa_l.append(t_a_n)
            rel_ht_l.append(h_t_n)
            rel_tsig_l.append(t_sig_n)

            sc_comp_m = item["sc_comp_m"]
            tip_comp_m = item["tip_comp_m"]
            sig_comp_m = item["sig_comp_comb_m"]
            b_holo = bb.clone()
            b_holo[:, 4] = sc_comp_m[:Nb]
            t_holo_n = t_b_n.clone()
            t_holo_n[:, 4] = sc_comp_m[Nb:]
            holo_b_l.append(b_holo)
            holo_btip_l.append(tip_comp_m[:Nb])
            holo_bsig_l.append(sig_comp_m[:Nb])
            holo_t_l.append(t_holo_n)
            holo_ttip_l.append(tip_comp_m[Nb:])
            holo_tsig_l.append(sig_comp_m[Nb:])

            b_intf10 = item["d_ca0"].amin(dim=1) < 10.0
            p_b = bb[b_intf10, 1].mean(dim=0) if b_intf10.any() else bb[:, 1].mean(dim=0)
            p_t = t_b_n[:, 1].mean(dim=0)
            u_norm_rel_l.append(F.normalize(p_b - p_t, dim=-1, eps=1e-6))

        with torch.enable_grad():
            w_vec_all = torch.zeros((M_c, 3), dtype=dtype, device=dev, requires_grad=True)
            t_vec_all = torch.zeros((M_c, 3), dtype=dtype, device=dev, requires_grad=True)
            R_poses_all = _axis_angle_to_matrix(w_vec_all)
            c_apo_g = batched_score_poses_multi_complex(
                ebm_model_c,
                rel_b_l, rel_btip_l, rel_baa_l, rel_hb_l, rel_bsig_l,
                rel_t_l, rel_ttip_l, rel_taa_l, rel_ht_l, rel_tsig_l,
                [R_poses_all[m : m + 1] for m in range(M_c)],
                [t_vec_all[m : m + 1] for m in range(M_c)],
                return_components=True,
            )
            g_w_all, g_t_all = torch.autograd.grad(c_apo_g["total"].sum(), [w_vec_all, t_vec_all])
            g_w_all = g_w_all.detach()
            g_t_all = g_t_all.detach()
            e0_apo_all = c_apo_g["total"].detach()
            elec0_apo_all = c_apo_g["elec"].detach()
            del c_apo_g, R_poses_all, w_vec_all, t_vec_all

        with torch.no_grad():
            u_t_steps = -g_t_all / (torch.norm(g_t_all, dim=-1, keepdim=True) + 1e-6)
            u_w_steps = -g_w_all / (torch.norm(g_w_all, dim=-1, keepdim=True) + 1e-6)
            holo_rots_l = []
            holo_trans_l = []
            for m in range(M_c):
                t_steps = alphas_relax[:, None] * u_t_steps[m : m + 1, :]
                w_steps = (0.35 * alphas_relax)[:, None] * u_w_steps[m : m + 1, :]
                R_steps = _axis_angle_to_matrix(w_steps)
                holo_rots_l.append(torch.cat([I_rot, R_steps], dim=0))
                holo_trans_l.append(torch.cat([Z_tr, t_steps], dim=0))

            e_holo_4_all = batched_score_poses_multi_complex(
                ebm_model_c,
                holo_b_l, holo_btip_l, rel_baa_l, rel_hb_l, holo_bsig_l,
                holo_t_l, holo_ttip_l, rel_taa_l, rel_ht_l, holo_tsig_l,
                holo_rots_l, holo_trans_l,
                return_components=False,
            ).view(M_c, 4)

            for m, item in enumerate(chunk):
                e0_val = float(e0_apo_all[m].item())
                e_holo_val = float(e_holo_4_all[m, 0].item())
                e_holo_min = float(e_holo_4_all[m].min().item())
                norm_repul = float((-torch.dot(g_t_all[m], u_norm_rel_l[m])).item())
                item["feats"].update({
                    "gen_ebm_elec_e0": float(elec0_apo_all[m].item()),
                    "ebm_holo_sc_gain_raw": e0_val - e_holo_val,
                    "ebm_normal_repulsion_force": norm_repul,
                    "ebm_holo_cont_relax_emin": e_holo_min,
                })
                for k_tmp in list(item.keys()):
                    if k_tmp not in ("rec", "feats"):
                        del item[k_tmp]

        if dev.type == "cuda":
            torch.cuda.synchronize()
        stage_c_relax_sec += time.perf_counter() - t0_rel

    if dev.type == "cuda":
        torch.cuda.synchronize()
    stage_c_total_sec = time.perf_counter() - t0_c_total
    total_wall_sec = time.perf_counter() - t_start_total
    cached_esm_wall_sec = max(1e-6, total_wall_sec - esm_sec)

    output_rows = [
        {
            "candidate_id": it["rec"]["candidate_id"],
            "target_id": it["rec"]["target_id"],
            "label": int(it["rec"]["label"]),
            "features": it["feats"],
        }
        for it in parsed_items
    ]
    timing_summary = {
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else str(dev),
        "n_complexes": N,
        "total_wall_sec": total_wall_sec,
        "cached_esm_wall_sec": cached_esm_wall_sec,
        "throughput_complexes_per_sec": N / total_wall_sec,
        "cached_esm_throughput_complexes_per_sec": N / cached_esm_wall_sec,
        "ms_per_complex_total": 1000.0 * total_wall_sec / N,
        "ms_per_complex_cached_esm_total": 1000.0 * cached_esm_wall_sec / N,
        "ms_per_complex_esm2_actual": 1000.0 * esm_sec / N,
        "ms_per_complex_parse_5bead": 1000.0 * parse_sec / N,
        "ms_per_complex_stage_a_orientation_ebm_esm2": 1000.0 * stage_a_sec / N,
        "ms_per_complex_stage_b_monomer_oring_if": 1000.0 * stage_b_sec / N,
        "ms_per_complex_stage_c_total": 1000.0 * stage_c_total_sec / N,
        "ms_per_complex_stage_c_autograd_bb_force": 1000.0 * stage_c_autograd_sec / N,
        "ms_per_complex_stage_c_cont_6dof_relax": 1000.0 * stage_c_relax_sec / N,
        "ms_per_complex_stage_c_sidechain_redock_alchem": 1000.0 * stage_c_sc_redock_sec / N,
    }
    print(
        f"[{cohort_name} | N={N}] Total={timing_summary['ms_per_complex_total']:.2f} ms/complex "
        f"({timing_summary['throughput_complexes_per_sec']:.1f} complexes/s) | "
        f"Parse={timing_summary['ms_per_complex_parse_5bead']:.2f} ms | "
        f"Pass1={timing_summary['ms_per_complex_stage_a_orientation_ebm_esm2']:.2f} ms (ESM2={timing_summary['ms_per_complex_esm2_actual']:.2f} ms) | "
        f"Pass2={timing_summary['ms_per_complex_stage_b_monomer_oring_if']:.2f} ms | "
        f"Pass3={timing_summary['ms_per_complex_stage_c_sidechain_redock_alchem']:.2f} ms | "
        f"Pass4={timing_summary['ms_per_complex_stage_c_autograd_bb_force']:.2f} ms | "
        f"Pass5={timing_summary['ms_per_complex_stage_c_cont_6dof_relax']:.2f} ms",
        flush=True,
    )
    return output_rows, timing_summary


def main():
    import gc, traceback
    for attr in ("last_traceback", "last_value", "last_type", "last_exc"):
        if hasattr(sys, attr):
            obj = getattr(sys, attr)
            if attr == "last_traceback" and obj is not None:
                try:
                    traceback.clear_frames(obj)
                except Exception:
                    pass
            setattr(sys, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Batched BC-EBM Benchmark on device: {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'})", flush=True)

    payload_path = Path("/content/bcebm_compact.pkl.gz")
    if not payload_path.exists():
        payload_path = Path("/tmp/bcebm_compact_small.pkl.gz")
    with gzip.open(payload_path, "rb") as f:
        payload = pickle.load(f)

    ckpt_dir = Path("/content/results") if Path("/content/results").exists() else PROJECT_DIR / "results"
    ckpt_a = torch.load(ckpt_dir / "singlepose_hardneg_orientation_ebm.pt", map_location=dev, weights_only=False)
    state_a = ckpt_a.get("model_state", ckpt_a.get("model_state_dict", ckpt_a))
    ebm_a = UnboundOrientationPairwiseEBM(
        hidden=64,
        monomer_layers=3,
        neural_interface_weight=ckpt_a.get("neural_interface_weight", 0.11),
        neural_veto_weight=ckpt_a.get("neural_veto_weight", 0.008),
        neural_veto_thresh=ckpt_a.get("neural_veto_thresh", -6.00),
        neural_veto_temp=ckpt_a.get("neural_veto_temp", 1.00),
        esm_rank=16,
        esm_bilin_weight=ckpt_a.get("esm_bilin_weight", 0.020),
        enable_head_on_esm=True,
    ).to(dev).eval()
    ebm_a.load_state_dict(state_a)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    esm = AutoModelForMaskedLM.from_pretrained(MODEL_ID).to(dev).eval()
    aa_tok_ids_t = torch.tensor([tok.convert_tokens_to_ids(a) for a in _AA1_LIST], device=dev, dtype=torch.long)

    if_model_b = train_or_load_backbone_inverse_folding_gnn(force_retrain=False).to(dev).eval()
    if_model_c, sc_model_c, ebm_model_c = train_or_load_general_models(force_retrain=False)
    if_model_c.to(dev).eval()
    sc_model_c.to(dev).eval()
    ebm_model_c.to(dev).eval()

    # Warmup on 10 complexes
    warmup_cohort = {"items": payload["ov100"]["items"][:10], "raw_pdbs": []}
    _, _ = run_batched_a100_cohort(
        warmup_cohort, "Warmup-10", ebm_a, tok, esm, aa_tok_ids_t,
        if_model_b, if_model_c, sc_model_c, ebm_model_c, dev, chunk_size=10
    )

    if os.environ.get("VERIFY_ONLY") == "1":
        # Verify feature parity against ref_ov100_rows
        ov_rows, _ = run_batched_a100_cohort(
            {"items": payload["ov100"]["items"], "raw_pdbs": []},
            "Overath-100-Verify", ebm_a, tok, esm, aa_tok_ids_t,
            if_model_b, if_model_c, sc_model_c, ebm_model_c, dev, chunk_size=100
        )
        ref_rows = payload["ref_ov100_rows"]
        max_rel_diff = 0.0
        worst_feat = ""
        for r_ref, r_bat in zip(ref_rows, ov_rows[: len(ref_rows)]):
            for k, v_ref in r_ref["features"].items():
                if k.startswith("ext_") or k not in r_bat["features"]:
                    continue
                v_bat = r_bat["features"][k]
                rel = abs(v_ref - v_bat) / max(abs(v_ref), 1.0)
                if rel > max_rel_diff:
                    max_rel_diff = rel
                    worst_feat = f"{k} (ref={v_ref:.6f}, bat={v_bat:.6f})"
        print(f"Max relative feature diff across Overath-100 reference rows: {max_rel_diff:.2e} [{worst_feat}]", flush=True)
        return

    # Full timed runs on A100 across all 3 cohorts (best of 3 warm runs)
    ov_rows, ov_timing = None, None
    for r_i in range(3):
        gc.collect()
        torch.cuda.empty_cache()
        rows_i, tim_i = run_batched_a100_cohort(
            payload["ov100"], f"Overath-100 (~200-aa) [run {r_i+1}]", ebm_a, tok, esm, aa_tok_ids_t,
            if_model_b, if_model_c, sc_model_c, ebm_model_c, dev, chunk_size=25
        )
        if ov_timing is None or tim_i["ms_per_complex_total"] < ov_timing["ms_per_complex_total"]:
            ov_rows, ov_timing = rows_i, tim_i

    sc_rows, sc_timing = None, None
    for r_i in range(3):
        gc.collect()
        torch.cuda.empty_cache()
        rows_i, tim_i = run_batched_a100_cohort(
            payload["scorp"], f"Scorpion-243 (~65-aa) [run {r_i+1}]", ebm_a, tok, esm, aa_tok_ids_t,
            if_model_b, if_model_c, sc_model_c, ebm_model_c, dev, chunk_size=32
        )
        if sc_timing is None or tim_i["ms_per_complex_total"] < sc_timing["ms_per_complex_total"]:
            sc_rows, sc_timing = rows_i, tim_i

    ni_rows, ni_timing = None, None
    for r_i in range(2):
        gc.collect()
        torch.cuda.empty_cache()
        rows_i, tim_i = run_batched_a100_cohort(
            payload["nipah"], f"Nipah-881 (532-aa) [run {r_i+1}]", ebm_a, tok, esm, aa_tok_ids_t,
            if_model_b, if_model_c, sc_model_c, ebm_model_c, dev, chunk_size=32
        )
        if ni_timing is None or tim_i["ms_per_complex_total"] < ni_timing["ms_per_complex_total"]:
            ni_rows, ni_timing = rows_i, tim_i

    # Check feature parity against ref_ov100_rows
    ref_rows = payload["ref_ov100_rows"]
    max_rel_diff = 0.0
    for r_ref, r_bat in zip(ref_rows, ov_rows[: len(ref_rows)]):
        for k, v_ref in r_ref["features"].items():
            if k.startswith("ext_") or k not in r_bat["features"]:
                continue
            v_bat = r_bat["features"][k]
            rel = abs(v_ref - v_bat) / max(abs(v_ref), 1.0)
            if rel > max_rel_diff:
                max_rel_diff = rel

    summary = {
        "gpu_device": torch.cuda.get_device_name(0) if dev.type == "cuda" else str(dev),
        "max_rel_feature_diff_vs_cpu_ref": max_rel_diff,
        "overath_100_timing": ov_timing,
        "scorpion_243_timing": sc_timing,
        "nipah_881_timing": ni_timing,
        "speedup_vs_af3_180s_overath_uncached": 180000.0 / ov_timing["ms_per_complex_total"],
        "speedup_vs_af3_180s_overath_cached": 180000.0 / ov_timing["ms_per_complex_cached_esm_total"],
        "speedup_vs_af3_180s_scorpion_uncached": 180000.0 / sc_timing["ms_per_complex_total"],
        "speedup_vs_af3_180s_scorpion_cached": 180000.0 / sc_timing["ms_per_complex_cached_esm_total"],
        "speedup_vs_af3_180s_nipah_uncached": 180000.0 / ni_timing["ms_per_complex_total"],
        "speedup_vs_af3_180s_nipah_cached": 180000.0 / ni_timing["ms_per_complex_cached_esm_total"],
        "speedup_vs_boltz1_35s_overath_uncached": 35000.0 / ov_timing["ms_per_complex_total"],
        "speedup_vs_boltz1_35s_overath_cached": 35000.0 / ov_timing["ms_per_complex_cached_esm_total"],
        "speedup_vs_boltz1_35s_scorpion_uncached": 35000.0 / sc_timing["ms_per_complex_total"],
        "speedup_vs_boltz1_35s_scorpion_cached": 35000.0 / sc_timing["ms_per_complex_cached_esm_total"],
        "speedup_vs_boltz1_35s_nipah_uncached": 35000.0 / ni_timing["ms_per_complex_total"],
        "speedup_vs_boltz1_35s_nipah_cached": 35000.0 / ni_timing["ms_per_complex_cached_esm_total"],
        "overath_3532_campaign_a100_gpu_sec": 3532.0 * ov_timing["ms_per_complex_total"] / 1000.0,
        "overath_3532_campaign_a100_gpu_hours": (3532.0 * ov_timing["ms_per_complex_total"] / 1000.0) / 3600.0,
    }
    print("A100_JSON_START")
    print(json.dumps(summary, indent=2))
    print("A100_JSON_END")


if __name__ == "__main__":
    main()
