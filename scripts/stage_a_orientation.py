"""Unbound-Anchored Orientation- and Paratope-Aware Pairwise EBM for biomolecular complexes.

Key properties:
1. Full backbone (N, CA, C, O) + physical CB + heavy side-chain centroid representation.
2. SE(3)-invariant local residue frames F_i in SO(3) and 6D relative frame orientations.
3. Unbound monomer contextual GNN runs ONCE per complex (independent of rigid docking pose)
   with Half-Sphere Exposure (HSE-up/down), surface protrusion index, backbone loop curvature,
   and sequence-derived IMGT region channels (Antigen, Constant, V-Framework, CDR1, CDR2, CDR-L3, CDR-H3).
4. Fully vectorized cross-interface pairwise energy evaluation across pose batches with:
   - 3D Antibody Paratope Axis alignment (u_paratope) and flat-bottom constant-domain exclusion
   - Soft-core short-range steric potential (tolerant to unrelaxed unbound CDR loop overlaps)
   - Size-controlled interface aggregation guaranteeing E(infinity) = 0 identically.
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

AA3 = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, "GLN": 5, "GLU": 6,
    "GLY": 7, "HIS": 8, "ILE": 9, "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13,
    "PRO": 14, "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}

# Physicochemical lookup tables indexed by AA3 integer (0..19 standard AAs, 20 unknown)
# Order in AA3: ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL, UNK
_AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
]
_CHARGE = {
    "ARG": 1.0, "LYS": 1.0, "HIS": 0.2, "ASP": -1.0, "GLU": -1.0,
}
_HYDROPATHY = {
    "ILE": 1.00, "VAL": 0.93, "LEU": 0.84, "PHE": 0.62, "CYS": 0.56,
    "MET": 0.42, "ALA": 0.40, "GLY": -0.09, "THR": -0.16, "SER": -0.18,
    "TRP": -0.20, "TYR": -0.29, "PRO": -0.36, "HIS": -0.71, "GLU": -0.78,
    "GLN": -0.78, "ASP": -0.78, "ASN": -0.78, "LYS": -0.87, "ARG": -1.00,
}
_DONOR = {
    "ARG": 1.0, "LYS": 1.0, "TRP": 0.7, "ASN": 0.7, "GLN": 0.7,
    "HIS": 0.5, "SER": 0.5, "THR": 0.5, "TYR": 0.5, "CYS": 0.2,
}
_ACCEPTOR = {
    "ASP": 1.0, "GLU": 1.0, "ASN": 0.7, "GLN": 0.7, "HIS": 0.5,
    "SER": 0.5, "THR": 0.5, "TYR": 0.5, "MET": 0.2,
}
_AROMATIC = {"PHE": 1.0, "TYR": 1.0, "TRP": 1.0, "HIS": 0.6}
_VOLUME = {
    "GLY": 0.00, "ALA": 0.16, "SER": 0.18, "CYS": 0.27, "ASP": 0.30,
    "PRO": 0.32, "THR": 0.34, "ASN": 0.35, "VAL": 0.48, "GLU": 0.46,
    "GLN": 0.51, "HIS": 0.55, "MET": 0.62, "ILE": 0.65, "LEU": 0.65,
    "LYS": 0.66, "ARG": 0.73, "PHE": 0.77, "TYR": 0.79, "TRP": 1.00,
}

# Region IDs (7 classes: 0..6):
# 0 = Antigen residue
# 1 = Antibody Constant domain (C_H1 / C_L)
# 2 = Antibody V-domain Framework (FR1..FR4)
# 3 = Antibody CDR1 (H1 / L1, IMGT 27..38)
# 4 = Antibody CDR2 (H2 / L2, IMGT 56..65)
# 5 = Antibody CDR-L3 (Light chain IMGT 105..117)
# 6 = Antibody CDR-H3 (Heavy chain IMGT 105..117)
REGION_ANTIGEN = 0
REGION_AB_CONST = 1
REGION_AB_VFW = 2
REGION_AB_CDR1 = 3
REGION_AB_CDR2 = 4
REGION_AB_CDRL3 = 5
REGION_AB_CDRH3 = 6
NUM_REGIONS = 7

_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

_CDR_ANNOTATIONS_CACHE: dict | None = None
_ESM2_MONOMER_CACHE: dict | None = None


def _get_coverage_cdr_annotations() -> dict:
    global _CDR_ANNOTATIONS_CACHE
    if _CDR_ANNOTATIONS_CACHE is not None:
        return _CDR_ANNOTATIONS_CACHE
    zip_path = Path(__file__).resolve().parent.parent / "results" / "antibody_roles.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            raw = json.loads(zf.read("antibody_roles.json"))
            _CDR_ANNOTATIONS_CACHE = raw.get("annotations", raw)
    else:
        _CDR_ANNOTATIONS_CACHE = {}
    return _CDR_ANNOTATIONS_CACHE


def _get_esm2_monomer_cache() -> dict:
    global _ESM2_MONOMER_CACHE
    if _ESM2_MONOMER_CACHE is not None:
        return _ESM2_MONOMER_CACHE
    cache_path = Path(__file__).resolve().parent.parent / "results" / "esm2_frustration_monomer_cache.pt"
    if cache_path.exists():
        _ESM2_MONOMER_CACHE = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        _ESM2_MONOMER_CACHE = {}
    return _ESM2_MONOMER_CACHE


def _build_physchem_table() -> torch.Tensor:
    rows = []
    for name in _AA_ORDER:
        rows.append([
            _CHARGE.get(name, 0.0),
            _HYDROPATHY.get(name, 0.0),
            _DONOR.get(name, 0.0),
            _ACCEPTOR.get(name, 0.0),
            _AROMATIC.get(name, 0.0),
            _VOLUME.get(name, 0.35),
        ])
    return torch.tensor(rows, dtype=torch.float32)


PHYSCHEM_TABLE = _build_physchem_table()  # [21, 6]


def parse_residues_sequential(pdb_text: str) -> list[dict]:
    """Parse ATOM records into sequential residues matching coverage_pipeline.py CA extraction 1-to-1.

    Handles stripped insertion codes in DOCKGROUND monomer PDBs by starting a new residue
    whenever (chain, resseq, resname) changes OR an atom name repeats within the current block.
    """
    residues = []
    curr = None
    last_key = None
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        alt = line[16]
        if alt not in (" ", "A", "1"):
            continue
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain_id = line[21]
        res_seq = line[22:27]
        coord = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
        key = (chain_id, res_seq, res_name)
        if curr is None or key != last_key or atom_name in curr["atoms"]:
            if curr is not None and "CA" in curr["atoms"]:
                residues.append(curr)
            curr = {
                "chain": chain_id,
                "res_seq": res_seq.strip(),
                "res_name": res_name,
                "atoms": {atom_name: coord},
            }
            last_key = key
        else:
            curr["atoms"][atom_name] = coord
    if curr is not None and "CA" in curr["atoms"]:
        residues.append(curr)
    return residues


def assign_anarci_regions_from_residues(residues: list[dict], chain_annot: dict) -> np.ndarray:
    """Assign sequence-derived ANARCI/IMGT region IDs (1..6) to antibody monomer residues."""
    n_res = len(residues)
    regions = np.full((n_res,), REGION_AB_VFW, dtype=np.int64)
    if not chain_annot:
        # Fallback if no annotation dict: first ~112 residues per chain are V-domain
        by_chain: dict[str, list[int]] = {}
        for i, r in enumerate(residues):
            by_chain.setdefault(r["chain"], []).append(i)
        for indices in by_chain.values():
            for pos, idx in enumerate(indices):
                if pos >= 115:
                    regions[idx] = REGION_AB_CONST
        return regions

    # Normalize chain_annot whether keyed by chain_type ("H"/"L"/"K") or by PDB chain_id
    chain_map: dict[str, dict] = {}
    for k, v in chain_annot.items():
        if isinstance(v, dict) and "chain" in v:
            chain_map[v["chain"]] = {
                "is_heavy": (k == "H"),
                "cdr1": v.get("cdr1", "").replace("-", ""),
                "cdr2": v.get("cdr2", "").replace("-", ""),
                "cdr3": v.get("cdr3", "").replace("-", ""),
            }
        elif isinstance(v, dict):
            cdrs = v.get("cdrs", v)
            chain_map[k] = {
                "is_heavy": (v.get("chain_type", "H") == "H"),
                "cdr1": cdrs.get("cdr1", "").replace("-", ""),
                "cdr2": cdrs.get("cdr2", "").replace("-", ""),
                "cdr3": cdrs.get("cdr3", "").replace("-", ""),
            }

    by_chain_indices: dict[str, list[int]] = {}
    for i, r in enumerate(residues):
        by_chain_indices.setdefault(r["chain"], []).append(i)

    for ch_id, indices in by_chain_indices.items():
        if ch_id not in chain_map:
            continue
        info = chain_map[ch_id]
        is_heavy = info["is_heavy"]
        cdrs = info

        # Deduplicate by res_seq to match annotate_antibody_roles.py sequence construction
        dedup_seq_chars = []
        dedup_to_all: list[list[int]] = []
        seen_res_seq: dict[str, int] = {}
        for idx in indices:
            rseq = residues[idx]["res_seq"]
            aa1 = _AA3_TO_1.get(residues[idx]["res_name"], "X")
            if rseq not in seen_res_seq:
                seen_res_seq[rseq] = len(dedup_seq_chars)
                dedup_seq_chars.append(aa1)
                dedup_to_all.append([idx])
            else:
                dedup_to_all[seen_res_seq[rseq]].append(idx)
        dedup_seq = "".join(dedup_seq_chars)

        # V-domain ends around dedup position 115..122; beyond is constant domain (C_H1/C_L)
        v_end_dedup = min(len(dedup_seq), 118)
        for d_i in range(v_end_dedup, len(dedup_seq)):
            for full_idx in dedup_to_all[d_i]:
                regions[full_idx] = REGION_AB_CONST

        # Match CDR1, CDR2, CDR3 substrings inside V-domain
        search_Ranges = [
            ("cdr1", REGION_AB_CDR1, 15, 50),
            ("cdr2", REGION_AB_CDR2, 40, 78),
            ("cdr3", REGION_AB_CDRH3 if is_heavy else REGION_AB_CDRL3, 80, 128),
        ]
        for cdr_key, reg_code, lo, hi in search_Ranges:
            cdr_seq = cdrs.get(cdr_key, "")
            if not cdr_seq:
                continue
            window = dedup_seq[lo:min(len(dedup_seq), hi)]
            pos = window.find(cdr_seq)
            if pos != -1:
                start_d = lo + pos
                end_d = start_d + len(cdr_seq)
            else:
                pos_full = dedup_seq[:128].find(cdr_seq)
                if pos_full == -1:
                    continue
                start_d = pos_full
                end_d = start_d + len(cdr_seq)
            for d_i in range(start_d, min(end_d, len(dedup_to_all))):
                for full_idx in dedup_to_all[d_i]:
                    regions[full_idx] = reg_code
                # Ensure any residue up to end_d + 10 is at least V-framework, not constant
                for v_i in range(start_d, min(end_d + 11, len(dedup_to_all))):
                    for full_idx in dedup_to_all[v_i]:
                        if regions[full_idx] == REGION_AB_CONST:
                            regions[full_idx] = REGION_AB_VFW

    return regions


def assign_imgt_regions_from_numbers(
    imgt_numbers: np.ndarray,
    is_antibody: np.ndarray,
    is_heavy_chain: np.ndarray,
) -> np.ndarray:
    """Assign region IDs (0..6) directly from IMGT residue numbers (e.g., SAbDab2 auth_seq_id)."""
    n = len(imgt_numbers)
    regions = np.zeros((n,), dtype=np.int64)
    ab_mask = is_antibody.astype(bool)
    if not np.any(ab_mask):
        return regions

    # Default antibody residues to V-framework (2) or Constant (1)
    regions[ab_mask & (imgt_numbers <= 128)] = REGION_AB_VFW
    regions[ab_mask & (imgt_numbers > 128)] = REGION_AB_CONST

    cdr1 = ab_mask & (imgt_numbers >= 27) & (imgt_numbers <= 38)
    cdr2 = ab_mask & (imgt_numbers >= 56) & (imgt_numbers <= 65)
    cdr3 = ab_mask & (imgt_numbers >= 105) & (imgt_numbers <= 117)

    regions[cdr1] = REGION_AB_CDR1
    regions[cdr2] = REGION_AB_CDR2
    regions[cdr3 & (~is_heavy_chain.astype(bool))] = REGION_AB_CDRL3
    regions[cdr3 & is_heavy_chain.astype(bool)] = REGION_AB_CDRH3
    return regions


def virtual_cb_from_backbone(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ideal tetrahedral virtual CB position from N, CA, C coordinates (shape [..., 3])."""
    b = ca - n
    c_vec = c - ca
    a = np.cross(b, c_vec)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca


def extract_monomer_geometry(pdb_text: str) -> dict[str, np.ndarray | list[dict]]:
    """Extract backbone N, CA, C, O, physical/virtual CB, and side-chain centroid arrays."""
    residues = parse_residues_sequential(pdb_text)
    n_res = len(residues)
    n_arr = np.zeros((n_res, 3), dtype=np.float32)
    ca_arr = np.zeros((n_res, 3), dtype=np.float32)
    c_arr = np.zeros((n_res, 3), dtype=np.float32)
    o_arr = np.zeros((n_res, 3), dtype=np.float32)
    cb_arr = np.zeros((n_res, 3), dtype=np.float32)
    sc_arr = np.zeros((n_res, 3), dtype=np.float32)
    has_phys_cb = np.zeros((n_res,), dtype=bool)
    has_sc = np.zeros((n_res,), dtype=bool)

    backbone_names = {"N", "CA", "C", "O", "OXT"}
    for i, res in enumerate(residues):
        atoms = res["atoms"]
        ca = atoms["CA"]
        n = atoms.get("N", ca + np.array([-1.45, 0.0, 0.0]))
        c = atoms.get("C", ca + np.array([1.52, 0.0, 0.0]))
        o = atoms.get("O", c + np.array([0.0, 1.23, 0.0]))
        v_cb = virtual_cb_from_backbone(n, ca, c)
        if res["res_name"] != "GLY" and "CB" in atoms:
            cb = atoms["CB"]
            has_phys_cb[i] = True
        else:
            cb = v_cb
            has_phys_cb[i] = False

        sc_coords = [
            coord for name, coord in atoms.items()
            if name not in backbone_names and not name.startswith("H")
        ]
        if sc_coords:
            sc_centroid = np.mean(sc_coords, axis=0)
            has_sc[i] = True
        else:
            sc_centroid = cb
            has_sc[i] = False

        n_arr[i] = n
        ca_arr[i] = ca
        c_arr[i] = c
        o_arr[i] = o
        cb_arr[i] = cb
        sc_arr[i] = sc_centroid

    beads = np.stack([n_arr, ca_arr, c_arr, o_arr, sc_arr], axis=1)
    return {
        "beads": beads,              # [N_res, 5, 3] float32: 0=N, 1=CA, 2=C, 3=O, 4=SC
        "cb": cb_arr,                # [N_res, 3] float32 (physical CB or virtual CB for Gly)
        "has_phys_cb": has_phys_cb,  # [N_res] bool
        "has_sc": has_sc,            # [N_res] bool
        "residues": residues,        # parsed sequential residue dicts
    }


def kabsch_rigid_transforms(ref_ca: np.ndarray, pose_ca: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute rigid transforms (R, t) mapping ref_ca [M, 3] -> pose_ca [P, M, 3]."""
    ref64 = ref_ca.astype(np.float64)
    pose64 = pose_ca.astype(np.float64)
    ref_mean = ref64.mean(axis=0, keepdims=True)  # [1, 3]
    ref_centered = ref64 - ref_mean              # [M, 3]

    pose_mean = pose64.mean(axis=1, keepdims=True)  # [P, 1, 3]
    pose_centered = pose64 - pose_mean              # [P, M, 3]

    cov = np.einsum("pmi,mj->pij", pose_centered, ref_centered)
    u, _, vt = np.linalg.svd(cov)
    det = np.linalg.det(u @ vt)
    d = np.ones((len(pose_ca), 3), dtype=np.float64)
    d[:, 2] = np.where(det < 0, -1.0, 1.0)
    rot = (u * d[:, None, :]) @ vt  # [P, 3, 3]
    trans = pose_mean[:, 0, :] - np.einsum("mi,pji->pj", ref_mean, rot)  # [P, 3]

    fitted = np.einsum("mi,pji->pmj", ref64, rot) + trans[:, None, :]
    rmsd = np.sqrt(((fitted - pose64) ** 2).sum(axis=-1).mean(axis=-1))
    return rot.astype(np.float32), trans.astype(np.float32), float(rmsd.max())


def build_local_frames(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Construct right-handed orthonormal local frames F_i in SO(3) from N, CA, C."""
    v1 = c - ca
    e1 = F.normalize(v1, dim=-1, eps=1e-6)
    v2 = n - ca
    u2 = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(u2, dim=-1, eps=1e-6)
    e3 = torch.linalg.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)


def rbf_encode(distances: torch.Tensor, d_min: float, d_max: float, num_bins: int) -> torch.Tensor:
    """Gaussian radial basis encoding of distances."""
    centers = torch.linspace(d_min, d_max, num_bins, device=distances.device, dtype=distances.dtype)
    gamma = (num_bins / max(d_max - d_min, 1e-3)) ** 2
    return torch.exp(-gamma * (distances.unsqueeze(-1) - centers) ** 2)


def compute_antibody_paratope_geometry(
    ca_pos: torch.Tensor,
    region: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SE(3)-covariant antibody CDR centroid and unit paratope axis u_paratope.

    Args:
        ca_pos: [N_ab, 3] antibody CA coordinates
        region: [N_ab] integer region labels (1..6)
    Returns:
        cdr_centroid: [3] weighted centroid of CDR loops
        u_paratope: [3] unit vector pointing from V-domain framework centroid through CDR centroid
    """
    dtype = ca_pos.dtype
    device = ca_pos.device

    # Weights: CDR-H3=3.0, CDR-L3=2.0, CDR1/2=1.0
    w_cdr = torch.zeros((len(ca_pos),), dtype=dtype, device=device)
    w_cdr = torch.where(region == REGION_AB_CDR1, torch.full_like(w_cdr, 1.0), w_cdr)
    w_cdr = torch.where(region == REGION_AB_CDR2, torch.full_like(w_cdr, 1.0), w_cdr)
    w_cdr = torch.where(region == REGION_AB_CDRL3, torch.full_like(w_cdr, 2.0), w_cdr)
    w_cdr = torch.where(region == REGION_AB_CDRH3, torch.full_like(w_cdr, 3.0), w_cdr)

    if w_cdr.sum() > 0:
        cdr_centroid = (w_cdr[:, None] * ca_pos).sum(dim=0) / w_cdr.sum().clamp_min(1e-6)
    else:
        cdr_centroid = ca_pos.mean(dim=0)

    vfw_mask = (region == REGION_AB_VFW)
    if vfw_mask.any():
        vfw_centroid = ca_pos[vfw_mask].mean(dim=0)
    else:
        vfw_centroid = ca_pos.mean(dim=0)

    axis = cdr_centroid - vfw_centroid
    norm = torch.linalg.norm(axis)
    if norm > 1e-4:
        u_paratope = axis / norm
    else:
        u_paratope = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    return cdr_centroid, u_paratope


class MonomerFrameGNNLayer(nn.Module):
    """SE(3)-invariant orientation-aware message-passing layer on unbound monomers."""

    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.gate = nn.Linear(2 * hidden + edge_dim, hidden)
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        h: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if edge_src.numel() == 0:
            return h
        cat = torch.cat([h[edge_src], h[edge_dst], edge_attr], dim=-1)
        msg = self.msg_mlp(cat) * torch.sigmoid(self.gate(cat))
        agg = torch.zeros_like(h)
        agg.index_add_(0, edge_dst, msg)
        deg = torch.zeros((h.shape[0], 1), device=h.device, dtype=h.dtype)
        deg.index_add_(0, edge_dst, torch.ones((edge_dst.numel(), 1), device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0)
        return self.norm(h + self.upd_mlp(torch.cat([h, agg], dim=-1)))


class UnboundOrientationPairwiseEBM(nn.Module):
    """Unbound-Anchored Orientation- and Paratope-Aware Pairwise Energy-Based Model.

    1. Encodes unbound antibody and antigen monomers ONCE per complex using local
       backbone frames (N, CA, C, O, CB, side-chain centroids), Half-Sphere Exposure
       (HSE-up/down), protrusion index, backbone loop curvature, physicochemical
       features, and 7-way IMGT region identity.
    2. Evaluates any batch of B rigid docking poses simultaneously in a single
       vectorized pass using orientation-aware cross-interface pair features,
       3D paratope-axis alignment, soft-core short-range sterics, and flat-bottom
       constant-domain exclusion.
    3. Satisfies E(infinity) = 0 identically and SE(3) invariance up to floating-point precision.
    """

    def __init__(
        self,
        hidden: int = 64,
        monomer_layers: int = 3,
        monomer_cutoff: float = 12.0,
        interface_cutoff: float = 12.0,
        neural_interface_weight: float = 0.11,
        neural_veto_weight: float = 0.008,
        neural_veto_thresh: float = -6.00,
        neural_veto_temp: float = 1.00,
        esm_rank: int = 16,
        esm_bilin_weight: float = 0.025,
        enable_head_on_esm: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        self.monomer_cutoff = monomer_cutoff
        self.interface_cutoff = interface_cutoff
        self.neural_interface_weight = neural_interface_weight
        self.neural_veto_weight = neural_veto_weight
        self.neural_veto_thresh = neural_veto_thresh
        self.neural_veto_temp = neural_veto_temp
        self.esm_rank = esm_rank
        self.esm_bilin_weight = esm_bilin_weight
        self.enable_head_on_esm = enable_head_on_esm

        # Node input (46 dims):
        # 21 one-hot AA + 2 role + 7 IMGT region + 6 physchem + 3 sc_local
        # + 3 burial(6,8,12) + 3 HSE(up, down, protrusion) + 1 backbone loop curvature = 46
        self.node_in_dim = 46
        self.node_in = nn.Sequential(
            nn.Linear(self.node_in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        # Monomer edge attr: 16 RBF(d_CA) + 8 RBF(d_SC) + 3 dir_i + 3 dir_j + 6 rot_6d + 5 seq_sep = 41
        self.monomer_layers = nn.ModuleList(
            [MonomerFrameGNNLayer(hidden, edge_dim=41) for _ in range(monomer_layers)]
        )

        # Cross-interface pair feature dim (84 dims):
        # 12 RBF(d_CA) + 12 RBF(d_CB) + 12 RBF(d_SC) + 8 RBF(d_NO) + 8 RBF(d_ON)
        # + 3 u_ab + 3 u_ag + 6 Q_ab_ag(6D) + 5 sc/o orientation cosines
        # + 6 physchem pair products + 6 Ab region one-hot (1..6)
        # + 3 paratope directional features (cos_para, cdr_prox, ag_protrusion) = 84
        self.pair_geom_dim = 84
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * hidden + self.pair_geom_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2),  # [pair_energy, pair_attention_logit]
        )
        nn.init.normal_(self.pair_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.pair_mlp[-1].bias)

        # Zero-Shortcut Surface-Centered Bilinear ESM-2 & Evolutionary Frustration Head
        self.esm_proj_ab = nn.Linear(320, esm_rank)
        self.esm_proj_ag = nn.Linear(320, esm_rank, bias=False)
        nn.init.normal_(self.esm_proj_ag.weight, mean=0.0, std=1e-2)
        self.esm_frust_weights = nn.Parameter(
            torch.tensor([-0.002, -0.004, 0.0, 0.0, 0.0, 0.008])
        )
        # Multi-Scale Head-On Antigen Surface Normal Grazing Veto & Alignment weights
        self.head_on_veto_weight = nn.Parameter(torch.tensor(0.080))
        self.head_on_lin_weight = nn.Parameter(torch.tensor(0.055))

        # Calibrated physical & structural prior parameters (softplus-constrained where positive)
        self.soft_clash_weight = nn.Parameter(torch.tensor(0.50))
        self.vdw_attr_weight = nn.Parameter(torch.tensor(0.25))
        self.electrostatic_weight = nn.Parameter(torch.tensor(0.15))
        self.hydrophobic_weight = nn.Parameter(torch.tensor(0.15))
        # softplus(-1.82) = 0.150 (balances macro paratope steering with 5-bead micro packing)
        self.paratope_weight = nn.Parameter(torch.tensor(-1.82))
        self.const_repulse_weight = nn.Parameter(torch.tensor(1.20))

    def reset_and_freeze_calibrated_physics(self, freeze: bool = True) -> None:
        """Restore calibrated physical scalar weights and optionally freeze them during neural residual training."""
        with torch.no_grad():
            self.soft_clash_weight.fill_(0.50)
            self.vdw_attr_weight.fill_(0.25)
            self.electrostatic_weight.fill_(0.15)
            self.hydrophobic_weight.fill_(0.15)
            self.paratope_weight.fill_(-1.82)
            self.const_repulse_weight.fill_(1.20)
        for p in (
            self.soft_clash_weight,
            self.vdw_attr_weight,
            self.electrostatic_weight,
            self.hydrophobic_weight,
            self.paratope_weight,
            self.const_repulse_weight,
        ):
            p.requires_grad_(not freeze)

    def _compute_monomer_features(
        self,
        beads: torch.Tensor,
        cb: torch.Tensor,
        aa: torch.Tensor,
        role: torch.Tensor,
        region: torch.Tensor,
        chain: torch.Tensor,
        residue_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute local frames, unit vectors, protrusion, and contextual monomer embeddings h."""
        n_pos = beads[:, 0]
        ca_pos = beads[:, 1]
        c_pos = beads[:, 2]
        o_pos = beads[:, 3]
        sc_pos = beads[:, 4]

        frames = build_local_frames(n_pos, ca_pos, c_pos)  # [N, 3, 3]
        sc_vec = F.normalize(sc_pos - ca_pos, dim=-1, eps=1e-6)
        cb_vec = F.normalize(cb - ca_pos, dim=-1, eps=1e-6)
        o_vec = F.normalize(o_pos - c_pos, dim=-1, eps=1e-6)

        # Local frame coordinates of SC centroid (SE(3)-invariant internal geometry)
        sc_local = torch.einsum("nij,ni->nj", frames, sc_pos - ca_pos) / 5.0  # [N, 3]

        # Intra-partner distance matrix for surface exposure, HSE, & monomer graph
        dist_ca = torch.cdist(ca_pos, ca_pos)
        eye = torch.eye(len(ca_pos), device=ca_pos.device, dtype=torch.bool)
        dist_masked = dist_ca.masked_fill(eye, 1e6)

        # Isotropic sphere burial at 6, 8, 12 A
        burial_6 = (dist_masked < 6.0).sum(dim=-1, keepdim=True).to(ca_pos.dtype) / 10.0
        burial_8 = (dist_masked < 8.0).sum(dim=-1, keepdim=True).to(ca_pos.dtype) / 20.0
        burial_12 = (dist_masked < 12.0).sum(dim=-1, keepdim=True).to(ca_pos.dtype) / 50.0

        # Half-Sphere Exposure (HSE-up vs HSE-down) along CB/SC vector within 12 A
        # Tells the network whether residue sits on a convex protruding loop vs concave pocket
        nbr_12 = (dist_masked < 12.0)  # [N, N]
        rel_unit = F.normalize(ca_pos[None, :, :] - ca_pos[:, None, :], dim=-1, eps=1e-6)  # [N, N, 3]
        dot_cb = (rel_unit * cb_vec[:, None, :]).sum(dim=-1)  # [N, N]
        hse_up = (nbr_12 & (dot_cb > 0.0)).sum(dim=-1, keepdim=True).to(ca_pos.dtype) / 25.0
        hse_down = (nbr_12 & (dot_cb <= 0.0)).sum(dim=-1, keepdim=True).to(ca_pos.dtype) / 25.0
        protrusion = (hse_down - hse_up) / (hse_down + hse_up + 0.1)  # [-1, +1]

        # Backbone loop curvature ||CA_{i+1} - 2*CA_i + CA_{i-1}|| / 4.0 along same chain
        curvature = torch.zeros((len(ca_pos), 1), device=ca_pos.device, dtype=ca_pos.dtype)
        if len(ca_pos) >= 3:
            same_prev = (chain[1:-1] == chain[:-2])
            same_next = (chain[1:-1] == chain[2:])
            valid_turn = (same_prev & same_next).to(ca_pos.dtype).unsqueeze(-1)
            sec_diff = ca_pos[2:] - 2.0 * ca_pos[1:-1] + ca_pos[:-2]
            curvature[1:-1] = (torch.linalg.norm(sec_diff, dim=-1, keepdim=True) / 4.0).clamp(0.0, 2.0) * valid_turn

        aa_clamped = aa.clamp(0, 20)
        aa_oh = F.one_hot(aa_clamped, 21).to(ca_pos.dtype)
        role_oh = F.one_hot(role.clamp(0, 1), 2).to(ca_pos.dtype)
        region_oh = F.one_hot(region.clamp(0, NUM_REGIONS - 1), NUM_REGIONS).to(ca_pos.dtype)
        physchem = PHYSCHEM_TABLE.to(ca_pos.device)[aa_clamped]  # [N, 6]

        node_feats = torch.cat(
            [
                aa_oh,
                role_oh,
                region_oh,
                physchem,
                sc_local,
                burial_6,
                burial_8,
                burial_12,
                hse_up,
                hse_down,
                protrusion,
                curvature,
            ],
            dim=-1,
        )  # [N, 46]
        h = self.node_in(node_feats)

        # Build intra-monomer graph edges
        src, dst = torch.nonzero((dist_masked < self.monomer_cutoff), as_tuple=True)
        if src.numel() > 0:
            d_ca = dist_ca[src, dst]
            d_sc = torch.linalg.norm(sc_pos[dst] - sc_pos[src], dim=-1)
            rbf_ca = rbf_encode(d_ca, 0.0, self.monomer_cutoff, 16)
            rbf_sc = rbf_encode(d_sc, 0.0, self.monomer_cutoff, 8)

            rel_vec = F.normalize(ca_pos[dst] - ca_pos[src], dim=-1, eps=1e-6)
            dir_i = torch.einsum("nij,ni->nj", frames[src], rel_vec)
            dir_j = torch.einsum("nij,ni->nj", frames[dst], -rel_vec)
            q_ij = torch.einsum("nij,nik->njk", frames[src], frames[dst])
            rot_6d = q_ij[:, :, :2].reshape(-1, 6)

            same_chain = (chain[src] == chain[dst])
            seq_diff = (residue_index[dst] - residue_index[src]).abs()
            seq_feats = torch.stack(
                [
                    same_chain.to(ca_pos.dtype),
                    (same_chain & (seq_diff == 1)).to(ca_pos.dtype),
                    (same_chain & (seq_diff == 2)).to(ca_pos.dtype),
                    (same_chain & (seq_diff <= 4)).to(ca_pos.dtype),
                    torch.clamp(seq_diff.to(ca_pos.dtype) / 32.0, max=1.0) * same_chain.to(ca_pos.dtype),
                ],
                dim=-1,
            )
            edge_attr = torch.cat([rbf_ca, rbf_sc, dir_i, dir_j, rot_6d, seq_feats], dim=-1)
            for layer in self.monomer_layers:
                h = layer(h, src, dst, edge_attr)

        return h, frames, sc_vec, o_vec, protrusion.squeeze(-1)

    def _default_region_if_missing(
        self,
        role: torch.Tensor,
        chain: torch.Tensor,
        n_res: int,
    ) -> torch.Tensor:
        """Fallback region assignment if complex_data does not supply rec_region/lig_region."""
        is_ab = bool((role == 0).any().item())
        if not is_ab:
            return torch.zeros((n_res,), dtype=torch.long, device=role.device)
        reg = torch.full((n_res,), REGION_AB_VFW, dtype=torch.long, device=role.device)
        for ch_val in torch.unique(chain):
            idx = torch.nonzero(chain == ch_val, as_tuple=True)[0]
            if len(idx) > 118:
                reg[idx[118:]] = REGION_AB_CONST
        return reg

    def encode_unbound_complex(self, complex_data: dict) -> dict[str, torch.Tensor]:
        """Run unbound monomer GNN once on receptor and ligand partners."""
        device = next(self.parameters()).device
        rec_beads = complex_data["rec_beads"].to(device)
        rec_cb = complex_data["rec_cb"].to(device)
        rec_aa = complex_data["rec_aa"].to(device)
        rec_role = complex_data["rec_role"].to(device)
        rec_chain = complex_data["rec_chain"].to(device)
        rec_res_idx = complex_data["rec_residue_index"].to(device)
        if "rec_region" in complex_data:
            rec_region = complex_data["rec_region"].to(device)
        else:
            rec_region = self._default_region_if_missing(rec_role, rec_chain, len(rec_aa))

        lig_beads = complex_data["lig_beads"].to(device)
        lig_cb = complex_data["lig_cb"].to(device)
        lig_aa = complex_data["lig_aa"].to(device)
        lig_role = complex_data["lig_role"].to(device)
        lig_chain = complex_data["lig_chain"].to(device)
        lig_res_idx = complex_data["lig_residue_index"].to(device)
        if "lig_region" in complex_data:
            lig_region = complex_data["lig_region"].to(device)
        else:
            lig_region = self._default_region_if_missing(lig_role, lig_chain, len(lig_aa))

        h_rec, frames_rec, sc_vec_rec, o_vec_rec, prot_rec = self._compute_monomer_features(
            rec_beads, rec_cb, rec_aa, rec_role, rec_region, rec_chain, rec_res_idx
        )
        h_lig, frames_lig, sc_vec_lig, o_vec_lig, prot_lig = self._compute_monomer_features(
            lig_beads, lig_cb, lig_aa, lig_role, lig_region, lig_chain, lig_res_idx
        )

        receptor_is_ab = bool(complex_data["receptor_is_antibody"])
        if receptor_is_ab:
            cdr_centroid0, u_para0 = compute_antibody_paratope_geometry(rec_beads[:, 1], rec_region)
            ag_ca0 = lig_beads[:, 1]
            ag_prot = prot_lig
            ab_reg = rec_region
        else:
            cdr_centroid0, u_para0 = compute_antibody_paratope_geometry(lig_beads[:, 1], lig_region)
            ag_ca0 = rec_beads[:, 1]
            ag_prot = prot_rec
            ab_reg = lig_region

        # Multi-scale outward antigen surface normal n_ag0 in R^{N_ag x 3} (SO(3)-equivariant)
        d_ag_ag = torch.cdist(ag_ca0, ag_ca0)
        w12 = (d_ag_ag < 12.0).to(ag_ca0.dtype)
        n12 = F.normalize(
            ag_ca0 - (w12 @ ag_ca0) / w12.sum(dim=-1, keepdim=True).clamp_min(1.0),
            dim=-1,
            eps=1e-6,
        )
        w18 = (d_ag_ag < 18.0).to(ag_ca0.dtype)
        n18 = F.normalize(
            ag_ca0 - (w18 @ ag_ca0) / w18.sum(dim=-1, keepdim=True).clamp_min(1.0),
            dim=-1,
            eps=1e-6,
        )
        n_ag0 = F.normalize(0.5 * n12 + 0.5 * n18, dim=-1, eps=1e-6)

        # Zero-Shortcut Surface-Standardized ESM-2 Frustration & Surface-Centered Embeddings
        esm_entry = None
        if "rec_esm_hidden" in complex_data and "lig_esm_hidden" in complex_data:
            esm_entry = complex_data
        elif "pair_id" in complex_data:
            esm_entry = _get_esm2_monomer_cache().get(complex_data["pair_id"])

        has_esm = False
        ag_frust_z = None
        ag_esm_c = None
        ab_esm_pool = None
        if (
            esm_entry is not None
            and len(esm_entry["rec_esm_hidden"]) == len(rec_aa)
            and len(esm_entry["lig_esm_hidden"]) == len(lig_aa)
        ):
            has_esm = True
            if receptor_is_ab:
                ab_esm = esm_entry["rec_esm_hidden"].to(device=device, dtype=ag_ca0.dtype)
                ag_esm = esm_entry["lig_esm_hidden"].to(device=device, dtype=ag_ca0.dtype)
                ag_frust = esm_entry["lig_esm_frust"].to(device=device, dtype=ag_ca0.dtype)
            else:
                ab_esm = esm_entry["lig_esm_hidden"].to(device=device, dtype=ag_ca0.dtype)
                ag_esm = esm_entry["rec_esm_hidden"].to(device=device, dtype=ag_ca0.dtype)
                ag_frust = esm_entry["rec_esm_frust"].to(device=device, dtype=ag_ca0.dtype)

            surf_m = ag_prot > -0.20
            if int(surf_m.sum().item()) < 10:
                surf_m = torch.ones_like(ag_prot, dtype=torch.bool)
            f_mean = ag_frust[surf_m].mean(dim=0, keepdim=True)
            f_std = ag_frust[surf_m].std(dim=0, keepdim=True) + 1e-3
            ag_frust_z = ((ag_frust - f_mean) / f_std).clamp(-3.5, 3.5)

            e_mean = ag_esm[surf_m].mean(dim=0, keepdim=True)
            ag_esm_c = F.normalize(ag_esm - e_mean, dim=-1, eps=1e-6)

            cdr_mask = ab_reg >= REGION_AB_CDR1
            if cdr_mask.any():
                w_cdr_ab = (1.0 + 0.5 * (ab_reg[cdr_mask] >= REGION_AB_CDRL3).to(ag_ca0.dtype)).unsqueeze(-1)
                w_cdr_ab = w_cdr_ab / w_cdr_ab.sum().clamp_min(1e-6)
                ab_esm_pool = F.normalize((w_cdr_ab * ab_esm[cdr_mask]).sum(dim=0), dim=-1, eps=1e-6)
            else:
                ab_esm_pool = F.normalize(ab_esm.mean(dim=0), dim=-1, eps=1e-6)

        return {
            "h_rec": h_rec,
            "frames_rec": frames_rec,
            "sc_vec_rec": sc_vec_rec,
            "o_vec_rec": o_vec_rec,
            "prot_rec": prot_rec,
            "rec_beads": rec_beads,
            "rec_cb": rec_cb,
            "rec_region": rec_region,
            "rec_has_phys_cb": complex_data["rec_has_phys_cb"].to(device),
            "rec_phys": PHYSCHEM_TABLE.to(device)[rec_aa.clamp(0, 20)],
            "h_lig": h_lig,
            "frames_lig": frames_lig,
            "sc_vec_lig": sc_vec_lig,
            "o_vec_lig": o_vec_lig,
            "prot_lig": prot_lig,
            "lig_beads": lig_beads,
            "lig_cb": lig_cb,
            "lig_region": lig_region,
            "lig_has_phys_cb": complex_data["lig_has_phys_cb"].to(device),
            "lig_phys": PHYSCHEM_TABLE.to(device)[lig_aa.clamp(0, 20)],
            "receptor_is_antibody": receptor_is_ab,
            "cdr_centroid0": cdr_centroid0,
            "u_para0": u_para0,
            "n_ag0": n_ag0,
            "has_esm": has_esm,
            "ag_frust_z": ag_frust_z,
            "ag_esm_c": ag_esm_c,
            "ab_esm_pool": ab_esm_pool,
        }

    def score_poses_from_encoded(
        self,
        encoded: dict[str, torch.Tensor],
        rotations: torch.Tensor,
        translations: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Vectorized scoring of a batch of B rigid ligand poses (R_p, t_p) against fixed receptor.

        Args:
            encoded: output of encode_unbound_complex
            rotations: [B, 3, 3]
            translations: [B, 3]
        Returns:
            energies: [B]
        """
        device = encoded["h_rec"].device
        rotations = rotations.to(device)
        translations = translations.to(device)
        bsz = len(rotations)

        rec_beads = encoded["rec_beads"]        # [N_rec, 5, 3]
        rec_cb = encoded["rec_cb"]              # [N_rec, 3]
        rec_has_cb = encoded["rec_has_phys_cb"] # [N_rec]
        frames_rec = encoded["frames_rec"]      # [N_rec, 3, 3]
        sc_vec_rec = encoded["sc_vec_rec"]      # [N_rec, 3]
        o_vec_rec = encoded["o_vec_rec"]        # [N_rec, 3]
        prot_rec = encoded["prot_rec"]          # [N_rec]
        h_rec = encoded["h_rec"]                # [N_rec, H]
        rec_phys = encoded["rec_phys"]          # [N_rec, 6]
        rec_region = encoded["rec_region"]      # [N_rec]

        lig_beads0 = encoded["lig_beads"]       # [N_lig, 5, 3]
        lig_cb0 = encoded["lig_cb"]             # [N_lig, 3]
        lig_has_cb = encoded["lig_has_phys_cb"] # [N_lig]
        frames_lig0 = encoded["frames_lig"]     # [N_lig, 3, 3]
        sc_vec_lig0 = encoded["sc_vec_lig"]     # [N_lig, 3]
        o_vec_lig0 = encoded["o_vec_lig"]       # [N_lig, 3]
        prot_lig = encoded["prot_lig"]          # [N_lig]
        h_lig = encoded["h_lig"]                # [N_lig, H]
        lig_phys = encoded["lig_phys"]          # [N_lig, 6]
        lig_region = encoded["lig_region"]      # [N_lig]

        # Transform only ligand CA across B poses first to extract sparse active interface pairs
        rec_ca = rec_beads[:, 1]         # [N_rec, 3]
        lig_ca_p = torch.einsum("ni,bji->bnj", lig_beads0[:, 1], rotations) + translations[:, None, :]  # [B, N_lig, 3]

        # Pairwise CA distances [B, N_rec, N_lig]
        dist_ca = torch.cdist(rec_ca.unsqueeze(0).expand(bsz, -1, -1), lig_ca_p)
        n_rec, n_lig = dist_ca.shape[1], dist_ca.shape[2]

        # Single vectorized sparse interface extraction across all B poses
        mask = dist_ca < self.interface_cutoff
        counts = mask.reshape(bsz, -1).sum(dim=1)
        if (counts == 0).any():
            empty_b = torch.nonzero(counts == 0, as_tuple=True)[0]
            flat_d = dist_ca[empty_b].reshape(len(empty_b), -1)
            _, nearest_idx = torch.topk(flat_d, k=min(8, n_rec * n_lig), dim=1, largest=False)
            r_near = nearest_idx // n_lig
            l_near = nearest_idx % n_lig
            b_near = empty_b[:, None].expand_as(r_near)
            mask = mask.clone()
            mask[b_near, r_near, l_near] = True

        b_idx, rec_idx, lig_idx = torch.nonzero(mask, as_tuple=True)

        # Rotate ligand coordinates, frames, and vectors strictly on active pairs (b_idx, lig_idx)
        rot_pair = rotations[b_idx]      # [P_total, 3, 3]
        tr_pair = translations[b_idx]    # [P_total, 3]
        lig_beads_pair = torch.einsum("pki,pji->pkj", lig_beads0[lig_idx], rot_pair) + tr_pair[:, None, :]
        lig_cb_pair = torch.einsum("pi,pji->pj", lig_cb0[lig_idx], rot_pair) + tr_pair
        frames_lig_pair = torch.einsum("pij,pjk->pik", rot_pair, frames_lig0[lig_idx])
        sc_vec_lig_pair = torch.einsum("pi,pji->pj", sc_vec_lig0[lig_idx], rot_pair)
        o_vec_lig_pair = torch.einsum("pi,pji->pj", o_vec_lig0[lig_idx], rot_pair)

        # Transform antibody paratope axis and CDR centroid per pose
        cdr_centroid0 = encoded["cdr_centroid0"]
        u_para0 = encoded["u_para0"]
        if encoded["receptor_is_antibody"]:
            cdr_centroid_p = cdr_centroid0.unsqueeze(0).expand(bsz, -1)  # [B, 3]
            u_para_p = u_para0.unsqueeze(0).expand(bsz, -1)              # [B, 3]

            ab_beads = rec_beads[rec_idx]
            ag_beads = lig_beads_pair
            ab_cb = rec_cb[rec_idx]
            ag_cb = lig_cb_pair
            ab_has_cb = rec_has_cb[rec_idx]
            ag_has_cb = lig_has_cb[lig_idx]
            f_ab = frames_rec[rec_idx]
            f_ag = frames_lig_pair
            sc_ab = sc_vec_rec[rec_idx]
            sc_ag = sc_vec_lig_pair
            o_ab = o_vec_rec[rec_idx]
            o_ag = o_vec_lig_pair
            h_ab = h_rec[rec_idx]
            h_ag = h_lig[lig_idx]
            phys_ab = rec_phys[rec_idx]
            phys_ag = lig_phys[lig_idx]
            reg_ab = rec_region[rec_idx]
            prot_ag = prot_lig[lig_idx]
        else:
            cdr_centroid_p = torch.einsum("i,bji->bj", cdr_centroid0, rotations) + translations
            u_para_p = torch.einsum("i,bji->bj", u_para0, rotations)

            ab_beads = lig_beads_pair
            ag_beads = rec_beads[rec_idx]
            ab_cb = lig_cb_pair
            ag_cb = rec_cb[rec_idx]
            ab_has_cb = lig_has_cb[lig_idx]
            ag_has_cb = rec_has_cb[rec_idx]
            f_ab = frames_lig_pair
            f_ag = frames_rec[rec_idx]
            sc_ab = sc_vec_lig_pair
            sc_ag = sc_vec_rec[rec_idx]
            o_ab = o_vec_lig_pair
            o_ag = o_vec_rec[rec_idx]
            h_ab = h_lig[lig_idx]
            h_ag = h_rec[rec_idx]
            phys_ab = lig_phys[lig_idx]
            phys_ag = rec_phys[rec_idx]
            reg_ab = lig_region[lig_idx]
            prot_ag = prot_rec[rec_idx]

        # Key inter-atomic distances [P_total]
        d_ca = torch.linalg.norm(ag_beads[:, 1] - ab_beads[:, 1], dim=-1)
        d_cb = torch.linalg.norm(ag_cb - ab_cb, dim=-1)
        d_sc = torch.linalg.norm(ag_beads[:, 4] - ab_beads[:, 4], dim=-1)
        d_no = torch.linalg.norm(ag_beads[:, 3] - ab_beads[:, 0], dim=-1)
        d_on = torch.linalg.norm(ag_beads[:, 0] - ab_beads[:, 3], dim=-1)

        # Smooth cosine cutoff weight w(d_ca) -> 0 at interface_cutoff (guarantees E(inf) = 0)
        ratio = (d_ca / self.interface_cutoff).clamp(0.0, 1.0)
        w_pair = 0.5 * (1.0 + torch.cos(math.pi * ratio))

        # Directional vectors in local frames
        r_hat = F.normalize(ag_beads[:, 1] - ab_beads[:, 1], dim=-1, eps=1e-6)
        u_ab = torch.einsum("pij,pi->pj", f_ab, r_hat)
        u_ag = torch.einsum("pij,pi->pj", f_ag, -r_hat)
        q_ab_ag = torch.einsum("pij,pik->pjk", f_ab, f_ag)[:, :, :2].reshape(-1, 6)

        # Orientation cosines (side-chain pointing & carbonyl alignment)
        cos_sc_ab = (sc_ab * r_hat).sum(dim=-1, keepdim=True)
        cos_sc_ag = (sc_ag * (-r_hat)).sum(dim=-1, keepdim=True)
        cos_sc_mutual = (sc_ab * sc_ag).sum(dim=-1, keepdim=True)
        cos_o_ab = (o_ab * r_hat).sum(dim=-1, keepdim=True)
        cos_o_ag = (o_ag * (-r_hat)).sum(dim=-1, keepdim=True)

        # Soft-core clamped distances for neural RBFs & physicochemical products
        # Prevents the network from using <3.5 A unbound side-chain overlaps as a shortcut
        d_ca_soft = d_ca.clamp_min(3.8)
        d_cb_soft = d_cb.clamp_min(3.6)
        d_sc_soft = d_sc.clamp_min(3.6)
        d_no_soft = d_no.clamp_min(2.9)
        d_on_soft = d_on.clamp_min(2.9)

        q_prod = (phys_ab[:, 0] * phys_ag[:, 0]) / (d_sc_soft + 2.0)
        hp_prod = (
            torch.clamp(phys_ab[:, 1], min=0.0)
            * torch.clamp(phys_ag[:, 1], min=0.0)
            * torch.exp(-((d_sc_soft - 4.2) ** 2) / 4.0)
        )
        hb_ab_don = phys_ab[:, 2] * phys_ag[:, 3] * torch.exp(-((d_sc_soft - 3.5) ** 2) / 2.5)
        hb_ag_don = phys_ag[:, 2] * phys_ab[:, 3] * torch.exp(-((d_sc_soft - 3.5) ** 2) / 2.5)
        arom_prod = phys_ab[:, 4] * phys_ag[:, 4] * torch.exp(-((d_sc_soft - 4.8) ** 2) / 3.0)
        vol_sum = 0.5 * (phys_ab[:, 5] + phys_ag[:, 5])
        phys_pair = torch.stack(
            [q_prod, hp_prod, hb_ab_don, hb_ag_don, arom_prod, vol_sum], dim=-1
        )

        # Antibody region 6-way one-hot (1=Const, 2=V-FW, 3=CDR1, 4=CDR2, 5=CDR-L3, 6=CDR-H3)
        ab_reg_oh = F.one_hot((reg_ab - 1).clamp(0, 5), 6).to(d_ca.dtype)

        # 3D Paratope directional alignment & antigen protrusion features
        u_para_pair = u_para_p[b_idx]          # [P_total, 3]
        cdr_cent_pair = cdr_centroid_p[b_idx]  # [P_total, 3]
        cos_para = (r_hat * u_para_pair).sum(dim=-1, keepdim=True)
        d_ag_cdr = torch.linalg.norm(ag_beads[:, 1] - cdr_cent_pair, dim=-1, keepdim=True)
        cdr_prox = torch.exp(-(d_ag_cdr ** 2) / (2.0 * (14.0 ** 2)))
        ag_prot_feat = prot_ag.unsqueeze(-1)

        pair_geom = torch.cat(
            [
                rbf_encode(d_ca_soft, 2.0, self.interface_cutoff, 12),
                rbf_encode(d_cb_soft, 2.0, self.interface_cutoff, 12),
                rbf_encode(d_sc_soft, 2.0, self.interface_cutoff, 12),
                rbf_encode(d_no_soft, 2.0, 8.0, 8),
                rbf_encode(d_on_soft, 2.0, 8.0, 8),
                u_ab,
                u_ag,
                q_ab_ag,
                cos_sc_ab,
                cos_sc_ag,
                cos_sc_mutual,
                cos_o_ab,
                cos_o_ag,
                phys_pair,
                ab_reg_oh,
                cos_para,
                cdr_prox,
                ag_prot_feat,
            ],
            dim=-1,
        )  # [P_total, 84]

        pair_in = torch.cat([h_ab, h_ag, pair_geom], dim=-1)
        pair_out = self.pair_mlp(pair_in)  # [P_total, 2]
        e_pair = pair_out[:, 0]
        attn_logit = pair_out[:, 1]

        # Segmented size-controlled aggregation across B poses
        w_sum = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        w_sum.index_add_(0, b_idx, w_pair)
        w_norm = w_pair / (w_sum[b_idx] + 4.0)
        interface_gate = w_sum / (w_sum + 4.0)

        mean_e = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        mean_e.index_add_(0, b_idx, w_norm * e_pair)

        # Numerically stable segmented softmax for attention readout
        logit_shifted = torch.tanh(attn_logit / 4.0) * 4.0 + torch.log(w_pair.clamp_min(1e-6))
        exp_logit = torch.exp(logit_shifted)
        denom_attn = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        denom_attn.index_add_(0, b_idx, exp_logit)
        attn_w = exp_logit / denom_attn[b_idx].clamp_min(1e-8)

        attn_e = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        attn_e.index_add_(0, b_idx, attn_w * e_pair)
        raw_learned = 0.5 * mean_e + 0.5 * attn_e

        # Paratope specificity & flat-bottom constant-domain exclusion (all gated by w_pair so E(inf)=0)
        cdr_weight = (
            1.00 * (reg_ab == REGION_AB_CDRH3).to(d_ca.dtype)
            + 0.70 * (reg_ab == REGION_AB_CDRL3).to(d_ca.dtype)
            + 0.45 * ((reg_ab == REGION_AB_CDR1) | (reg_ab == REGION_AB_CDR2)).to(d_ca.dtype)
        )
        para_align = torch.clamp(cos_para.squeeze(-1), min=0.0) * cdr_prox.squeeze(-1)
        para_pair = -cdr_weight * (0.6 + 0.4 * para_align) * (1.0 + 0.25 * prot_ag.clamp(min=0.0))
        para_term = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        para_term.index_add_(0, b_idx, w_norm * para_pair)

        is_const = (reg_ab == REGION_AB_CONST).to(d_ca.dtype)
        const_frac = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        const_frac.index_add_(0, b_idx, w_norm * is_const)
        const_penalty = interface_gate * F.relu(const_frac - 0.05)

        # Single-pose Paratope Funnel Gate: g_para(x) in [0, 1]
        # Only engages the attractive atomic packing well when the pose contacts the CDR paratope
        # (and is not docked onto the constant domain), using ZERO pool-level statistics.
        cdr_contact_frac = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        cdr_contact_frac.index_add_(0, b_idx, w_norm * (cdr_weight > 0).to(d_ca.dtype))
        paratope_funnel_gate = interface_gate * torch.sigmoid(8.0 * (cdr_contact_frac - const_frac - 0.15))

        # One-sided neural compatibility veto + interface contact term:
        # Preserves exact E(inf) = 0, keeps 5-bead physical packing in control of compatible poses,
        # and penalizes chemically/orientationally incompatible false-positive decoys.
        neural_veto = F.softplus((raw_learned - self.neural_veto_thresh) / self.neural_veto_temp)
        learned_energy = (
            -self.neural_interface_weight * interface_gate
            + self.neural_veto_weight * paratope_funnel_gate * neural_veto
        )

        # 5-Bead Physical Atomic Packing (N, CA, C, O, SC_centroid)
        bead_dists = torch.cdist(ab_beads, ag_beads)  # [P_total, 5, 5]
        valid_ab = torch.ones((len(ab_beads), 5), device=device, dtype=torch.bool)
        valid_ag = torch.ones((len(ag_beads), 5), device=device, dtype=torch.bool)
        valid_ab[:, 4] = ab_has_cb
        valid_ag[:, 4] = ag_has_cb
        valid_mask = valid_ab[:, :, None] & valid_ag[:, None, :]

        # CDR3-selective soft-core steric radius: 3.0 A on rigid backbone/framework/CDR1-2,
        # 2.75 A on flexible induced-fit CDR3 loops (capped at 4.0 per pair to prevent single-rotamer blowup)
        is_cdr3 = ((reg_ab == REGION_AB_CDRL3) | (reg_ab == REGION_AB_CDRH3)).to(bead_dists.dtype)
        clash_cutoff = (3.0 - 0.25 * is_cdr3)[:, None, None]
        overlap = F.relu(clash_cutoff - bead_dists) * valid_mask.to(bead_dists.dtype)
        clash_pair = (overlap ** 2).sum(dim=(1, 2)).clamp(max=4.0)

        pair_counts = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        pair_counts.index_add_(0, b_idx, torch.ones_like(d_ca))
        inv_count = 1.0 / pair_counts[b_idx].clamp_min(1.0)

        clash_term = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        clash_term.index_add_(0, b_idx, inv_count * clash_pair)
        clash_term = interface_gate * clash_term

        # 5-bead van der Waals contact well around 4.0 A (exact Step 2A multi-bead formulation)
        min_bead_d = bead_dists.masked_fill(~valid_mask, 20.0).amin(dim=(1, 2))
        vdw_pair = -torch.exp(-((min_bead_d - 4.0) ** 2) / 1.5)
        vdw_well = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        vdw_well.index_add_(0, b_idx, inv_count * vdw_pair)

        # Multi-scale Head-On Antigen Surface Normal Grazing Veto & Zero-Shortcut Bilinear ESM-2 Term
        head_on_term = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        esm_term = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        if self.enable_head_on_esm and "n_ag0" in encoded:
            if encoded["receptor_is_antibody"]:
                n_ag_pair = torch.einsum("pi,pji->pj", encoded["n_ag0"][lig_idx], rot_pair)
                ag_idx_pair = lig_idx
            else:
                n_ag_pair = encoded["n_ag0"][rec_idx]
                ag_idx_pair = rec_idx

            is_cdr_pair = (reg_ab >= REGION_AB_CDR1).to(d_ca.dtype) * (
                d_ca < self.interface_cutoff
            ).to(d_ca.dtype)
            w_cdr_c = is_cdr_pair * torch.sigmoid((6.5 - min_bead_d) / 0.7)
            w_cdr_sum = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
            w_cdr_sum.index_add_(0, b_idx, w_cdr_c)
            w_cdr_norm = w_cdr_c / (w_cdr_sum[b_idx] + 3.0)

            head_on_pair = -(u_para_p[b_idx] * n_ag_pair).sum(dim=-1)
            head_on = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
            head_on.index_add_(0, b_idx, w_cdr_norm * head_on_pair)
            grazing_veto = F.softplus((0.55 - head_on) / 0.20) * 0.20
            head_on_term = (
                self.head_on_veto_weight.clamp(0.0, 0.25) * grazing_veto
                - self.head_on_lin_weight.clamp(0.0, 0.12) * head_on
            )

            if encoded.get("has_esm", False) and encoded.get("ag_frust_z") is not None:
                ag_frust_z = encoded["ag_frust_z"]
                ag_esm_c = encoded["ag_esm_c"]
                ab_esm_pool = encoded["ab_esm_pool"]

                ag_frust_pose = torch.zeros((bsz, 6), device=device, dtype=d_ca.dtype)
                ag_frust_pose.index_add_(
                    0, b_idx, w_cdr_norm.unsqueeze(-1) * ag_frust_z[ag_idx_pair]
                )
                ag_esm_pose = torch.zeros((bsz, 320), device=device, dtype=d_ca.dtype)
                ag_esm_pose.index_add_(
                    0, b_idx, w_cdr_norm.unsqueeze(-1) * ag_esm_c[ag_idx_pair]
                )
                e_frust = ag_frust_pose @ self.esm_frust_weights
                z_ab = torch.tanh(self.esm_proj_ab(ab_esm_pool))
                z_ag = self.esm_proj_ag(ag_esm_pose)
                e_bilin = -self.esm_bilin_weight * torch.tanh((z_ag @ z_ab) / 2.0)
                esm_term = e_frust + e_bilin

        # Electrostatic & hydrophobic complementarity (using actual d_sc with 3.0 A soft floor)
        d_sc_pack = d_sc.clamp_min(3.0)
        q_pack = (phys_ab[:, 0] * phys_ag[:, 0]) / (d_sc_pack + 2.0)
        hp_pack = (
            torch.clamp(phys_ab[:, 1], min=0.0)
            * torch.clamp(phys_ag[:, 1], min=0.0)
            * torch.exp(-((d_sc_pack - 4.2) ** 2) / 4.0)
        )
        hb_ab_pack = phys_ab[:, 2] * phys_ag[:, 3] * torch.exp(-((d_sc_pack - 3.5) ** 2) / 2.5)
        hb_ag_pack = phys_ag[:, 2] * phys_ab[:, 3] * torch.exp(-((d_sc_pack - 3.5) ** 2) / 2.5)
        arom_pack = phys_ab[:, 4] * phys_ag[:, 4] * torch.exp(-((d_sc_pack - 4.8) ** 2) / 3.0)

        elec_term = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        elec_term.index_add_(0, b_idx, w_norm * q_pack)

        hydro_pair = -(hp_pack + 0.5 * (hb_ab_pack + hb_ag_pack) + 0.5 * arom_pack)
        hydro_term = torch.zeros((bsz,), device=device, dtype=d_ca.dtype)
        hydro_term.index_add_(0, b_idx, w_norm * hydro_pair)

        micro_packing = (
            F.softplus(self.vdw_attr_weight) * vdw_well
            + self.electrostatic_weight * elec_term
            + F.softplus(self.hydrophobic_weight) * hydro_term
        )

        total_e = (
            learned_energy
            + F.softplus(self.paratope_weight) * para_term
            + F.softplus(self.const_repulse_weight) * const_penalty
            + F.softplus(self.soft_clash_weight) * clash_term
            + paratope_funnel_gate * (micro_packing + head_on_term + esm_term)
        )
        if return_components:
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
        return total_e


def load_coverage_complex_with_frames(outer_zip: zipfile.ZipFile, pdb_id: str) -> dict:
    """Load a coverage15 complex with full backbone frames, side-chain centroids, ANARCI regions, and rigid pose transforms."""
    inner_bytes = outer_zip.read(f"coverage15_{pdb_id}.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as archive:
        data = np.load(archive.open("candidates.npz"))
        run = json.loads(archive.read("run.json"))
        rec_text = archive.read("receptor.pdb").decode()
        lig_text = archive.read("ligand.pdb").decode()

    rec_geom = extract_monomer_geometry(rec_text)
    lig_geom = extract_monomer_geometry(lig_text)
    n_rec = len(rec_geom["beads"])
    n_lig = len(lig_geom["beads"])

    coords = data["coords"]  # [1000, N_rec + N_lig, 3]
    if coords.shape[1] != n_rec + n_lig:
        raise ValueError(f"{pdb_id}: node count mismatch {coords.shape[1]} vs {n_rec + n_lig}")

    lig_ref_ca = lig_geom["beads"][:, 1, :]
    lig_pose_ca = coords[:, n_rec:, :]
    rotations, translations, max_rmsd = kabsch_rigid_transforms(lig_ref_ca, lig_pose_ca)
    if max_rmsd > 0.002:
        raise ValueError(f"{pdb_id}: unexpected ligand rigid fit RMSD {max_rmsd:.6f} A")

    aa_all = np.array([AA3.get(str(value), 20) for value in data["aa"]], dtype=np.int64)
    chain_all = data["chain"].astype(np.int64)
    raw_role = data["role"].astype(np.int64)
    receptor_is_ab = bool(run["antibody_receptor"])
    if not receptor_is_ab:
        raw_role = 1 - raw_role
    res_idx_all = data["residue_index"].astype(np.int64)

    # Sequence-derived ANARCI CDR/Framework/Constant region labels (0..6)
    cdr_db = _get_coverage_cdr_annotations()
    pdb_annot = cdr_db.get(pdb_id, {})
    if receptor_is_ab:
        rec_region = assign_anarci_regions_from_residues(rec_geom["residues"], pdb_annot)
        lig_region = np.zeros((n_lig,), dtype=np.int64)
    else:
        rec_region = np.zeros((n_rec,), dtype=np.int64)
        lig_region = assign_anarci_regions_from_residues(lig_geom["residues"], pdb_annot)

    return {
        "pair_id": pdb_id,
        "cluster": pdb_id,
        "receptor_is_antibody": receptor_is_ab,
        "rec_beads": torch.from_numpy(rec_geom["beads"]),
        "rec_cb": torch.from_numpy(rec_geom["cb"]),
        "rec_has_phys_cb": torch.from_numpy(rec_geom["has_phys_cb"]),
        "rec_aa": torch.from_numpy(aa_all[:n_rec]),
        "rec_role": torch.from_numpy(raw_role[:n_rec]),
        "rec_region": torch.from_numpy(rec_region),
        "rec_chain": torch.from_numpy(chain_all[:n_rec]),
        "rec_residue_index": torch.from_numpy(res_idx_all[:n_rec]),
        "lig_beads": torch.from_numpy(lig_geom["beads"]),
        "lig_cb": torch.from_numpy(lig_geom["cb"]),
        "lig_has_phys_cb": torch.from_numpy(lig_geom["has_phys_cb"]),
        "lig_aa": torch.from_numpy(aa_all[n_rec:]),
        "lig_role": torch.from_numpy(raw_role[n_rec:]),
        "lig_region": torch.from_numpy(lig_region),
        "lig_chain": torch.from_numpy(chain_all[n_rec:]),
        "lig_residue_index": torch.from_numpy(res_idx_all[n_rec:]),
        "rotations": torch.from_numpy(rotations),
        "translations": torch.from_numpy(translations),
        "quality": data["dockq"].astype(np.float64).tolist(),
        "zdock_score": data["zdock_score"].astype(np.float64).tolist(),
        "max_fit_rmsd": max_rmsd,
    }


# --- ESM-2 Frustration & Surface-Packed Decoy Pool Helpers ---
MODEL_ID = "facebook/esm2_t6_8M_UR50D"

_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
_AA1_LIST = [_AA3_TO_1[a] for a in _AA_ORDER[:20]]


@torch.no_grad()
def extract_partner_esm(
    aa: torch.Tensor,
    chain: torch.Tensor,
    tok: AutoTokenizer,
    esm: AutoModelForMaskedLM,
    aa_tok_ids: list[int],
    phys_tbl_dev: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    n_res = len(aa)
    seq_chars = [_AA1_LIST[i] if i < 20 else "X" for i in aa.tolist()]
    frust = torch.zeros((n_res, 6), dtype=torch.float32)
    hidden = torch.zeros((n_res, 320), dtype=torch.float16)
    phys_cpu = PHYSCHEM_TABLE[:20]

    start = 0
    while start < n_res:
        end = start + 1
        while end < n_res and chain[end] == chain[start]:
            end += 1
        sub_seq = "".join(seq_chars[start:end])
        inputs = tok(sub_seq, return_tensors="pt", add_special_tokens=True).to(device)
        out = esm(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1][0, 1:-1].to(torch.float16).cpu()
        logits = out.logits[0, 1:-1][:, aa_tok_ids]
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        ent = -(probs * log_probs).sum(dim=-1).cpu()
        exp_phys = (probs @ phys_tbl_dev).cpu()
        log_probs_cpu = log_probs.cpu()
        probs_cpu = probs.cpu()

        for k in range(end - start):
            c_idx = int(aa[start + k].item())
            if c_idx < 20:
                nll = -log_probs_cpu[k, c_idx].item()
                act_p = phys_cpu[c_idx]
            else:
                nll = 3.0
                act_p = torch.zeros(6)
            ep = exp_phys[k]
            frust[start + k] = torch.tensor(
                [
                    nll,
                    ent[k].item(),
                    (act_p[1] - ep[1]).item(),
                    (act_p[4] - ep[4]).item(),
                    (act_p[0].abs() - probs_cpu[k] @ phys_cpu[:, 0].abs()).item(),
                    ep[1].item(),
                ],
                dtype=torch.float32,
            )
        hidden[start:end] = h
        start = end

    return {"esm_hidden": hidden, "esm_frust": frust}


def generate_surface_packed_sabdab2_pool(
    complex_data: dict,
    rng: np.random.Generator,
    n_poses: int = 48,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate surface-packed rigid antigen poses via analytical 5-bead ray-casting.

    Layout of B = n_poses (default 48):
      - Pose 0: exact native pose (quality = 1.00)
      - Poses 1..6 (6 near-native acceptable poses): small rotation (2.5..8.0 deg) +
        small translation, with any <3.15 A steric overlap gently relieved along u_axis.
      - Poses 7..41 (35 CDR-docked non-native surface-packed decoys): non-native rotations
        (30..180 deg) and lateral epitope shifts analytically ray-cast along u_axis so
        their minimum 5-bead inter-partner distance equals d_target ~ U(3.42, 4.12) A.
      - Poses 42..47 (6 framework/side surface-packed decoys): packed against non-CDR
        antibody surface with d_target ~ U(3.45, 4.15) A.
    """
    rec_beads = complex_data["rec_beads"].float()
    lig_beads0 = complex_data["lig_beads"].float()
    rec_ca = rec_beads[:, 1].numpy()
    lig_ca = lig_beads0[:, 1].numpy()
    rec_reg = complex_data["rec_region"].numpy()

    ag_center = lig_ca.mean(axis=0)
    cdr_mask = np.isin(
        rec_reg, [REGION_AB_CDR1, REGION_AB_CDR2, REGION_AB_CDRL3, REGION_AB_CDRH3]
    )
    cdr_center = rec_ca[cdr_mask].mean(axis=0) if cdr_mask.any() else rec_ca.mean(axis=0)
    fw_mask = ~cdr_mask
    fw_center = rec_ca[fw_mask].mean(axis=0) if fw_mask.any() else rec_ca.mean(axis=0)

    u_axis_np = ag_center - cdr_center
    u_axis_np = u_axis_np / max(np.linalg.norm(u_axis_np), 1e-6)
    u_axis = torch.from_numpy(u_axis_np.astype(np.float32))

    rots = np.zeros((n_poses, 3, 3), dtype=np.float32)
    trans = np.zeros((n_poses, 3), dtype=np.float32)
    qualities = np.zeros((n_poses,), dtype=np.float32)

    rots[0] = np.eye(3, dtype=np.float32)
    trans[0] = np.zeros((3,), dtype=np.float32)
    qualities[0] = 1.00

    n_near = 6
    n_cdr_decoy = n_poses - 1 - n_near - 6

    for p in range(1, n_poses):
        axis = rng.normal(size=3).astype(np.float32)
        axis /= max(np.linalg.norm(axis), 1e-6)
        if p <= n_near:
            ang = math.radians(float(rng.uniform(2.5, 7.8)))
            shift = rng.normal(0.0, 0.50, size=3).astype(np.float32)
            r_mat = axis_angle_to_matrix(axis, ang)
            t_vec = ag_center - (ag_center @ r_mat.T) + shift
        elif p <= n_near + n_cdr_decoy:
            ang = math.radians(float(rng.uniform(32.0, 178.0)))
            r_mat = axis_angle_to_matrix(axis, ang)
            lat = rng.normal(0.0, 3.2, size=3).astype(np.float32)
            lat = lat - float(np.dot(lat, u_axis_np)) * u_axis_np
            target_c = cdr_center + lat
            t_vec = target_c - (ag_center @ r_mat.T)
        else:
            ang = math.radians(float(rng.uniform(45.0, 180.0)))
            r_mat = axis_angle_to_matrix(axis, ang)
            lat = rng.normal(0.0, 4.0, size=3).astype(np.float32)
            target_c = fw_center + lat
            t_vec = target_c - (ag_center @ r_mat.T)

        rots[p] = r_mat
        trans[p] = t_vec

    rots_t = torch.from_numpy(rots)
    trans_t = torch.from_numpy(trans)

    # Two-stage fast analytical 5-bead ray-cast along u_axis for non-native poses (p > n_near)
    decoy_start = n_near + 1
    b_dec = n_poses - decoy_start
    if b_dec > 0:
        rots_d = rots_t[decoy_start:]
        trans_d = trans_t[decoy_start:]

        # Stage 1: coarse CA + SC ray-cast against antibody surface within 18 A of cdr/fw center
        rec_2b = rec_beads[:, [1, 4], :]  # [N_rec, 2, 3]
        lig_2b = (
            torch.einsum("nki,bji->bnkj", lig_beads0[:, [1, 4], :], rots_d)
            + trans_d[:, None, None, :]
        )
        rec_ca_t = rec_beads[:, 1]
        lig_ca_t = lig_2b[:, :, 0]
        rec_ca_perp = rec_ca_t - (rec_ca_t @ u_axis)[:, None] * u_axis
        lig_ca_perp = lig_ca_t - torch.einsum("bni,i->bn", lig_ca_t, u_axis)[:, :, None] * u_axis
        d_ca_perp = torch.cdist(rec_ca_perp.unsqueeze(0).expand(b_dec, -1, -1), lig_ca_perp)

        b_idx, r_idx, l_idx = torch.nonzero(d_ca_perp < 6.8, as_tuple=True)
        if len(b_idx) > 0:
            ab_b = rec_2b[r_idx]          # [P, 2, 3]
            ag_b = lig_2b[b_idx, l_idx]   # [P, 2, 3]
            diff = ag_b[:, None, :, :] - ab_b[:, :, None, :]  # [P, 2, 2, 3]
            z_diff = torch.einsum("pijk,k->pij", diff, u_axis)
            r_perp_sq = (diff ** 2).sum(dim=-1) - z_diff ** 2

            target_d = torch.from_numpy(rng.uniform(3.42, 4.12, size=b_dec).astype(np.float32))
            td_pair = target_d[b_idx][:, None, None]
            hit_mask = r_perp_sq < (td_pair ** 2)
            s_req = torch.where(
                hit_mask,
                -z_diff + torch.sqrt((td_pair ** 2 - r_perp_sq).clamp_min(0.0)),
                torch.full_like(z_diff, -1e4),
            )
            s_pair = s_req.amax(dim=(1, 2))
            s_pose = torch.full((b_dec,), -1e4, dtype=torch.float32)
            s_pose.scatter_reduce_(0, b_idx, s_pair, reduce="amax", include_self=True)
            valid_s = s_pose > -1e3
            trans_d = trans_d + torch.where(
                valid_s[:, None], s_pose[:, None] * u_axis[None, :], torch.zeros_like(trans_d)
            )

            # Stage 2: exact 5-bead refinement around Stage 1 contact interface (3D d_ca < 8.5 A)
            lig_5b = (
                torch.einsum("nki,bji->bnkj", lig_beads0, rots_d)
                + trans_d[:, None, None, :]
            )
            d_ca_3d = torch.cdist(
                rec_ca_t.unsqueeze(0).expand(b_dec, -1, -1), lig_5b[:, :, 1]
            )
            b2, r2, l2 = torch.nonzero(d_ca_3d < 8.5, as_tuple=True)
            if len(b2) > 0:
                ab_5 = rec_beads[r2]        # [P2, 5, 3]
                ag_5 = lig_5b[b2, l2]       # [P2, 5, 3]
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
                s_pose2 = torch.full((b_dec,), -1e4, dtype=torch.float32)
                s_pose2.scatter_reduce_(0, b2, s5, reduce="amax", include_self=True)
                valid2 = s_pose2 > -1e3
                trans_d = trans_d + torch.where(
                    valid2[:, None], s_pose2[:, None] * u_axis[None, :], torch.zeros_like(trans_d)
                )

        trans_t[decoy_start:] = trans_d

    # Also relieve any severe <3.10 A steric clashes on near-native poses 1..n_near
    lig_near = (
        torch.einsum("nki,bji->bnkj", lig_beads0, rots_t[1 : n_near + 1])
        + trans_t[1 : n_near + 1, None, None, :]
    )
    d_ca_near = torch.cdist(
        rec_beads[:, 1].unsqueeze(0).expand(n_near, -1, -1), lig_near[:, :, 1]
    )
    bn, rn, ln = torch.nonzero(d_ca_near < 8.0, as_tuple=True)
    if len(bn) > 0:
        ab_n = rec_beads[rn]
        ag_n = lig_near[bn, ln]
        bd_n = torch.cdist(ab_n.reshape(-1, 5, 3), ag_n.reshape(-1, 5, 3)).amin(dim=(1, 2))
        min_near = torch.full((n_near,), 10.0, dtype=torch.float32)
        min_near.scatter_reduce_(0, bn, bd_n, reduce="amin", include_self=True)
        push = F.relu(3.20 - min_near)
        trans_t[1 : n_near + 1] = trans_t[1 : n_near + 1] + push[:, None] * u_axis[None, :]

    # Compute L-RMSD quality labels for all poses
    lig_ca_all = (
        torch.einsum("ni,bji->bnj", lig_beads0[:, 1], rots_t) + trans_t[:, None, :]
    ).numpy()
    for p in range(1, n_poses):
        l_rmsd = float(np.sqrt(((lig_ca_all[p] - lig_ca) ** 2).sum(axis=-1).mean()))
        q_val = 1.0 / (1.0 + (l_rmsd / 4.0) ** 2)
        if l_rmsd > 10.0 or p > n_near:
            q_val = min(q_val, 0.08)
        qualities[p] = q_val

    return rots_t, trans_t, torch.from_numpy(qualities)

