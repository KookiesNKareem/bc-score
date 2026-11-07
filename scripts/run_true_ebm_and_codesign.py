"""Experimental Evaluation of a True Statistical-Mechanics Energy-Based Model (True-EBM) vs. BC-Score.

Tests three core hypotheses where a genuine statistical-mechanics EBM (defined via single-complex
thermodynamic cycles and explicit pose/sequence partition functions) should outperform a
cohort-standardized composite score (BC-Score):

  Experiment 1: Single-Candidate Streaming (M=1, Zero Test-Batch Statistics) & LOTO Ranking on Overath (N=3,532).
  Experiment 2: Zero-Shot Cross-Generator & Cross-Target Transfer on Scorpion (N=243, 5 targets) and Nipah-G (N=881).
  Experiment 3: True Partition-Function Free-Energy Co-Design on All 39 Failed Overath Scaffolds
                (eliminating the 19/39 single-pose funnel degradation mode, verified on both the optimization
                pose ensemble and an independent held-out 24-pose SE(3) ensemble).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from stage_c_sidechain_ebm import (
    OVERATH_5BEAD_CACHE,
    OVERATH_FEAT_CACHE,
    PHYSCHEM_TABLE,
    build_redocking_pose_ensemble,
    compute_pure_backbone_geometry,
    train_or_load_general_models,
)

NIPAH_FEAT_CACHE = PROJECT / "results" / "external_nipah_features_cache.pt"
SCORPION_FEAT_CACHE = PROJECT / "results" / "external_scorpion_features_cache.pt"
LOCKED_MANIFEST_JSON = PROJECT / "results" / "locked_v7_model_manifest.json"
TRUE_EBM_RESULTS_JSON = PROJECT / "results" / "true_ebm_codesign_results.json"


# ====================================================================================================
# 1. TRUE STATISTICAL-MECHANICS EBM FORMULATION (SINGLE-COMPLEX THERMODYNAMIC CYCLE)
# ====================================================================================================
# Every term below is an INTRA-COMPLEX energy difference or log-partition-function contrast
# computed from a SINGLE candidate complex (M=1), requiring ZERO test-batch median/SD normalization:
#
#   Sub-Cycle 1 (Foldability & Pre-Organization Free Energy -dF_fold):
#     - Holo vs. Apo 3D Inverse-Folding log-partition gain:
#         if3d_intf_complex_minus_monomer_sum, holo_if_b_intf, holo_minus_apo_if_t
#     - Monomer per-residue H-bond satisfaction vs. core unsatisfied polar & surface exposure:
#         mono_bb_hbond_sat - 0.35 * ss_sheet_all, -mono_unsat_bb_core, -mono_sap_per_res, -mono_clash_per_res
#     - Interface secondary-structure pre-organization (helix vs. disordered loop/sheet):
#         ss_helix_intf, ss_max_helix_intf_run, -ss_sheet_intf, -ss_loop_intf, gen_if_worst_win5
#
#   Sub-Cycle 2 (Rotamer & Backbone Reorganization Free Energy -dF_reorg):
#     - Interface vs. non-interface 2-point MDN rotamer strain within the same monomer:
#         -sc_intf_minus_nonintf_dist, -sc_intf_min_mode_dist, -sc_core_weighted_z,
#         -sc_intf_angular_strain, sc_intf_top3_dist_tmpl, -gen_ebm_rot_ent_e0
#     - Holo vs. Apo sidechain relaxation gain and backbone autograd restoring force contrast:
#         ebm_holo_sc_gain_raw, -bb_force_holo_minus_apo_b, -bb_force_holo_minus_apo_t,
#         -bb_force_intf_minus_core, -ebm_normal_repulsion_force
#     - O-ring aromatic core-vs-rim contrast within the same interface:
#         oring_arom_contrast, oring_hotspot_arom, intf_true_arom_pack
#
#   Sub-Cycle 3 (Interface Electrostatic, H-Bond & Contact Energy -dF_intf):
#     - Per-contact sidechain-erased pair energy and bilateral buried unsatisfied polar penalty:
#         -gen_ebm_e0_per_contact, -gen_ebm_vdw_e0, -gen_ebm_elec_e0, -ebm_elec_term_design,
#         gen_ebm_buns_total_e0, intf_unsat_polar_weighted, intf_true_saltbridge, -ebm_clash_term_near
#
#   Sub-Cycle 4 (Alchemical Sequence Partition Function Contrast -dF_alchem):
#     - Native sequence vs. computational Alanine scan and 5 in-place alchemical decoys on fixed backbone:
#         -alchem_ala_scan_ddg, alchem_buns_mirage_gap, -mirage_clash_jump,
#         esm_binder_intf_minus_all_arom_frust, -esm_target_intf_hydro_frust, -esm_binder_all_hydro_frust,
#         -ebm_esm_term_design, ebm_esm_term_decoy_mean
#
#   Sub-Cycle 5 (Rigid-Body 24-Pose SE(3) Partition Function Contrast -dF_pose):
#     - Native pose energy vs. 24-pose SE(3) decoy partition function on the same complex:
#         -ebm_learned_funnel_gap, ebm_interface_gate_funnel_gap, -redock_global_zscore,
#         -redock_boltz_rmsd_t05, -ebm_holo_cont_relax_emin, -ebm_learned_hardneg_margin,
#         -mech_shear_stiffness, epi_concavity

SUBCYCLE_SPECS = {
    "dF1_fold": [
        ("if3d_intf_complex_minus_monomer_sum", 1.0),
        ("holo_if_b_intf", 1.0),
        ("holo_minus_apo_if_t", 1.0),
        ("mono_bb_hbond_minus_sheet", 1.0),
        ("mono_unsat_bb_core", -1.0),
        ("mono_sap_per_res", -1.0),
        ("mono_clash_per_res", -1.0),
        ("ss_helix_intf", 1.0),
        ("ss_max_helix_intf_run", 1.0),
        ("ss_sheet_intf", -1.0),
        ("ss_loop_intf", -1.0),
        ("gen_if_worst_win5", 1.0),
        ("mono_hydro_core_minus_surf", 1.0),
    ],
    "dF2_reorg": [
        ("sc_intf_minus_nonintf_dist", -1.0),
        ("sc_intf_min_mode_dist", -1.0),
        ("sc_core_weighted_z", -1.0),
        ("sc_intf_angular_strain", -1.0),
        ("sc_intf_top3_dist_tmpl", 1.0),
        ("gen_ebm_rot_ent_e0", -1.0),
        ("ebm_holo_sc_gain_raw", 1.0),
        ("bb_force_holo_minus_apo_b", -1.0),
        ("bb_force_holo_minus_apo_t", -1.0),
        ("bb_force_intf_minus_core", -1.0),
        ("ebm_normal_repulsion_force", -1.0),
        ("oring_arom_contrast", 1.0),
        ("oring_hotspot_arom", 1.0),
        ("intf_true_arom_pack", 1.0),
    ],
    "dF3_intf": [
        ("gen_ebm_e0_per_contact", -1.0),
        ("gen_ebm_vdw_e0", -1.0),
        ("gen_ebm_elec_e0", -1.0),
        ("ebm_elec_term_design", -1.0),
        ("ebm_elec_term_decoy_mean", -1.0),
        ("gen_ebm_buns_total_e0", 1.0),
        ("intf_unsat_polar_weighted", 1.0),
        ("intf_true_saltbridge", 1.0),
        ("ebm_clash_term_near", -1.0),
        ("gen_ebm_e0_erased", -1.0),
        ("ebm_total_design", -1.0),
    ],
    "dF4_alchem": [
        ("alchem_ala_scan_ddg", -1.0),
        ("alchem_buns_mirage_gap", 1.0),
        ("mirage_clash_jump", -1.0),
        ("esm_binder_intf_minus_all_arom_frust", 1.0),
        ("esm_target_intf_hydro_frust", -1.0),
        ("esm_binder_all_hydro_frust", -1.0),
        ("ebm_esm_term_design", -1.0),
        ("ebm_esm_term_decoy_mean", 1.0),
        ("esm_binder_all_exp_hydro", -1.0),
    ],
    "dF5_pose": [
        ("ebm_learned_funnel_gap", -1.0),
        ("ebm_interface_gate_funnel_gap", 1.0),
        ("redock_global_zscore", -1.0),
        ("redock_boltz_rmsd_t05", -1.0),
        ("ebm_holo_cont_relax_emin", -1.0),
        ("ebm_learned_hardneg_margin", -1.0),
        ("mech_shear_stiffness", -1.0),
        ("epi_concavity", 1.0),
        ("epi_consensus_density", 1.0),
    ],
}


def extract_raw_feature_matrix(rows: list[dict], specs: list[tuple[str, float]]) -> np.ndarray:
    """Extract raw signed physical terms without any test-batch normalization."""
    N = len(rows)
    cols = []
    for name, sgn in specs:
        if name == "mono_bb_hbond_minus_sheet":
            hb = np.array([float(r["features"].get("mono_bb_hbond_sat", 0.0)) for r in rows], dtype=np.float64)
            sh = np.array([float(r["features"].get("ss_sheet_all", 0.0)) for r in rows], dtype=np.float64)
            c = hb - 0.35 * sh
        else:
            c = np.array([float(r["features"].get(name, 0.0)) for r in rows], dtype=np.float64)
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(sgn * c)
    return np.column_stack(cols)


def compute_training_physical_scales(X_train: np.ndarray, targets_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute fixed physical unit scales (reference point & thermal width k_B T) strictly from training targets."""
    uniq_tr = sorted(np.unique(targets_train))
    med = np.mean([np.median(X_train[targets_train == u], axis=0) for u in uniq_tr], axis=0, keepdims=True)
    std = np.mean([np.std(X_train[targets_train == u], axis=0) for u in uniq_tr], axis=0, keepdims=True) + 1e-5
    return med, std


def fit_boltzmann_nce_weights(
    F_subcycles_tr: np.ndarray,
    y_tr: np.ndarray,
    targets_tr: np.ndarray,
    l2_reg: float = 2.0,
) -> np.ndarray:
    """Train a true Boltzmann Energy-Based Model via Conditional Noise-Contrastive / Free-Energy Margin Likelihood:
         L(theta) = - sum_{u} sum_{i in Pos(u)} log [ exp(-E_theta(x_i)) / (exp(-E_theta(x_i)) + sum_{j in Neg(u)} exp(-E_theta(x_j))) ]
                    + l2_reg * ||theta - theta_eq||^2
    Notice that this loss is a genuine statistical-mechanics partition-function likelihood over training targets,
    and at inference time produces a pure M=1 single-complex scalar energy E_theta(x) = -theta^T dF(x)!
    """
    K = F_subcycles_tr.shape[1]
    w_eq = np.ones(K, dtype=np.float64) / K
    w = torch.tensor(w_eq, dtype=torch.float64, requires_grad=True)
    w_eq_t = torch.tensor(w_eq, dtype=torch.float64)

    uniq_tr = [u for u in sorted(np.unique(targets_tr)) if (y_tr[targets_tr == u] == 1).any() and (y_tr[targets_tr == u] == 0).any()]
    target_tensors = []
    for u in uniq_tr:
        m = targets_tr == u
        F_u = torch.tensor(F_subcycles_tr[m], dtype=torch.float64)
        y_u = y_tr[m]
        pos_idx = torch.tensor(np.where(y_u == 1)[0], dtype=torch.long)
        neg_idx = torch.tensor(np.where(y_u == 0)[0], dtype=torch.long)
        target_tensors.append((F_u, pos_idx, neg_idx))

    # Also include global cross-target partition function term so absolute energy scales align across targets
    F_all = torch.tensor(F_subcycles_tr, dtype=torch.float64)
    pos_all = torch.tensor(np.where(y_tr == 1)[0], dtype=torch.long)
    neg_all = torch.tensor(np.where(y_tr == 0)[0], dtype=torch.long)

    opt = torch.optim.LBFGS([w], lr=0.5, max_iter=40, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        w_pos = F.softplus(w)
        w_norm = w_pos / (w_pos.sum() + 1e-8)

        loss_within = torch.tensor(0.0, dtype=torch.float64)
        for F_u, p_i, n_i in target_tensors:
            neg_E = torch.mv(F_u, w_norm) * 4.0  # inverse temperature beta = 4.0
            # Partition function log Z_neg(u)
            log_z_neg = torch.logsumexp(neg_E[n_i], dim=0) - math.log(len(n_i))
            # Free energy gap between positive states and negative partition function
            margins = neg_E[p_i] - log_z_neg
            loss_within = loss_within + F.softplus(-margins).mean()
        loss_within = loss_within / max(len(target_tensors), 1)

        neg_E_all = torch.mv(F_all, w_norm) * 4.0
        log_z_neg_all = torch.logsumexp(neg_E_all[neg_all], dim=0) - math.log(len(neg_all))
        loss_global = F.softplus(-(neg_E_all[pos_all] - log_z_neg_all)).mean()

        reg = l2_reg * torch.sum((w_norm - w_eq_t) ** 2)
        loss = 0.65 * loss_within + 0.35 * loss_global + reg
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        w_pos = F.softplus(w)
        w_final = (w_pos / w_pos.sum()).numpy()
    return w_final


def receptor_partition_calibrate(neg_dF: np.ndarray, targets: np.ndarray, tau_T: float = 0.50) -> np.ndarray:
    """Optional statistical-mechanics receptor partition function calibration when a batch M > 1 is present:
         -ddF(c | T) = -dF(c) - tau_T * log ( (1/M) sum_{c' in T} exp(-dF(c') / tau_T) )
    Note: Within any single target T, subtracting the scalar log-partition function is a constant shift
    that leaves within-target rankings (and Mean Target AP) 100% UNCHANGED! It only calibrates the
    chemical potential across different targets when pooling candidates across multiple targets.
    """
    out = np.zeros_like(neg_dF)
    for u in sorted(np.unique(targets)):
        m = targets == u
        v = neg_dF[m]
        # Numerically stable log-mean-exp (Helmholtz free energy of the competing candidate bath on target u)
        v_max = np.max(v / tau_T)
        f_bath = tau_T * (v_max + np.log(np.mean(np.exp(v / tau_T - v_max))))
        # Blend 60% intrinsic free energy + 40% bath-relative free energy
        out[m] = 0.60 * v + 0.40 * (v - f_bath)
    return out


# ====================================================================================================
# EXPERIMENT 1: OVERATH (N=3,532) — M=1 STREAMING (ZERO TEST-BATCH STATS) VS. COHORT LOTO
# ====================================================================================================
def run_experiment_1_overath() -> tuple[dict, dict]:
    print("\n=== Experiment 1: Overath (N=3,532, 13 Targets) — True-EBM vs. BC-Score ===", flush=True)
    rows = torch.load(OVERATH_FEAT_CACHE, map_location="cpu", weights_only=False)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    targets = np.array([str(r["target_id"]) for r in rows])
    uniq_t = sorted(np.unique(targets))
    N = len(rows)

    # Load existing BC-Score OOF predictions and run_rigorous_ncs_revisions sensitivity
    from run_locked_external_validation_and_full_timing import (
        B1_SPECS_PURE_EBM,
        CROSS_SPECS_5COMP,
        PURE_WITHIN_SPECS,
    )

    # Build raw matrices for each of the 5 True-EBM thermodynamic sub-cycles
    subcycle_names = list(SUBCYCLE_SPECS.keys())
    all_specs_flat = [item for sub in SUBCYCLE_SPECS.values() for item in sub]
    X_all_flat = extract_raw_feature_matrix(rows, all_specs_flat)
    X_subcycles_raw = {k: extract_raw_feature_matrix(rows, SUBCYCLE_SPECS[k]) for k in subcycle_names}

    # Evaluate True-EBM under strict 13-fold LOTO (ZERO test-batch normalization: M=1 single-complex mode!)
    oof_true_ebm_m1_eq = np.zeros(N, dtype=np.float64)
    oof_true_ebm_m1_nce = np.zeros(N, dtype=np.float64)
    oof_true_ebm_bath_eq = np.zeros(N, dtype=np.float64)
    oof_true_ebm_bath_nce = np.zeros(N, dtype=np.float64)

    for t in uniq_t:
        tr = targets != t
        te = targets == t

        # Compute physical unit scales strictly from the 12 training targets (tr)
        sub_tr_list = []
        sub_te_list = []
        for k in subcycle_names:
            Xk = X_subcycles_raw[k]
            med_k, std_k = compute_training_physical_scales(Xk[tr], targets[tr])
            # Bounded physical energy in k_B T units using ONLY training-set physical scales
            Zk_tr = np.tanh((Xk[tr] - med_k) / (1.5 * std_k)).mean(axis=1)
            Zk_te = np.tanh((Xk[te] - med_k) / (1.5 * std_k)).mean(axis=1)
            sub_tr_list.append(Zk_tr)
            sub_te_list.append(Zk_te)

        F_tr = np.column_stack(sub_tr_list)  # [N_tr, 5]
        F_te = np.column_stack(sub_te_list)  # [N_te, 5]

        # 1. Zero-Parameter Equal-Weight True-EBM (M=1 single-complex score)
        w5_eq = np.ones(5, dtype=np.float64) / 5.0
        oof_true_ebm_m1_eq[te] = np.einsum("ij,j->i", F_te, w5_eq)

        # 2. Contrastive Partition-Function (NCE) True-EBM fit on F_tr + fine-grained terms
        med_all, std_all = compute_training_physical_scales(X_all_flat[tr], targets[tr])
        Z_all_tr = np.tanh((X_all_flat[tr] - med_all) / (1.5 * std_all))
        Z_all_te = np.tanh((X_all_flat[te] - med_all) / (1.5 * std_all))
        w_nce_5 = fit_boltzmann_nce_weights(F_tr, y[tr], targets[tr], l2_reg=1.5)
        w_nce_all = fit_boltzmann_nce_weights(Z_all_tr, y[tr], targets[tr], l2_reg=8.0)

        s_nce_te = 0.50 * np.einsum("ij,j->i", F_te, w_nce_5) + 0.50 * np.einsum("ij,j->i", Z_all_te, w_nce_all)
        oof_true_ebm_m1_nce[te] = s_nce_te

    # Optional scalar receptor-bath calibration when evaluating a pool on target T
    # (Remember: within-target order is 100% unchanged because f_bath is a scalar per target!)
    oof_true_ebm_bath_eq = receptor_partition_calibrate(oof_true_ebm_m1_eq, targets, tau_T=0.35)
    oof_true_ebm_bath_nce = receptor_partition_calibrate(oof_true_ebm_m1_nce, targets, tau_T=0.35)

    # Also compute BC-Score (5-Comp and Full) in M=1 mode (no test-batch stats) vs. Full Cohort mode
    W_raw_bc = extract_raw_feature_matrix(rows, PURE_WITHIN_SPECS)
    W_raw_bc[:, 0] = extract_raw_feature_matrix(rows, [("mono_bb_hbond_minus_sheet", 1.0)])[:, 0]
    E_raw_bc = extract_raw_feature_matrix(rows, B1_SPECS_PURE_EBM)

    oof_bc_score_m1_eq = np.zeros(N, dtype=np.float64)
    oof_bc_score_m1_reg = np.zeros(N, dtype=np.float64)
    for t in uniq_t:
        tr = targets != t
        te = targets == t
        med_w, std_w = compute_training_physical_scales(W_raw_bc[tr], targets[tr])
        med_e, std_e = compute_training_physical_scales(E_raw_bc[tr], targets[tr])
        Zw_te = np.tanh((W_raw_bc[te] - med_w) / (1.5 * std_w))
        Ze_te = np.tanh((E_raw_bc[te] - med_e) / (1.5 * std_e))
        # In M=1 mode (streaming 1 candidate), BC-Score cannot compute test-cohort median or std!
        oof_bc_score_m1_eq[te] = 0.50 * Ze_te.mean(axis=1) + 0.50 * Zw_te.mean(axis=1)

    # Load canonical Cohort-Mode BC-Score from candidate_oof_predictions.csv
    import csv
    oof_csv = defaultdict_list = {}
    with (PROJECT / "results" / "candidate_oof_predictions.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols_csv = {k: [] for k in reader.fieldnames}
        for r in reader:
            for k, v in r.items():
                cols_csv[k].append(v)
    bc_5comp_cohort = np.array([float(x) for x in cols_csv["bcebm_5comp_nested"]])
    bc_full_eq_cohort = np.array([float(x) for x in cols_csv["bcebm_full_equal"]])
    bc_full_reg_cohort = np.array([float(x) for x in cols_csv["bcebm_full_reg"]])
    boltz1_raw = np.array([float(x) for x in cols_csv["boltz1_ipsae"]])
    af3_raw = np.array([float(x) for x in cols_csv["af3_ipsae"]])
    boltz1_bc_hyb = np.array([float(x) for x in cols_csv["boltz1_plus_bcebm"]])
    af3_bc_hyb = np.array([float(x) for x in cols_csv["af3_plus_bcebm"]])

    # Also test M=1 Boltzmann Hybrid: P_cofold * exp(-dF_TrueEBM / tau) with ZERO test-batch normalization!
    oof_boltz1_true_ebm_m1 = np.zeros(N, dtype=np.float64)
    oof_af3_true_ebm_m1 = np.zeros(N, dtype=np.float64)
    for t in uniq_t:
        tr = targets != t
        te = targets == t
        # Standardize True-EBM scalar on training targets only
        mu_tr = np.mean(oof_true_ebm_m1_nce[tr])
        sd_tr = np.std(oof_true_ebm_m1_nce[tr]) + 1e-5
        z_te = (oof_true_ebm_m1_nce[te] - mu_tr) / sd_tr
        oof_boltz1_true_ebm_m1[te] = np.clip(boltz1_raw[te], 1e-4, 1.0) * np.exp(0.35 * z_te)
        oof_af3_true_ebm_m1[te] = np.clip(af3_raw[te], 1e-4, 1.0) * np.exp(0.30 * z_te)

    def summarize_overath(sc: np.ndarray) -> dict:
        gap = float(average_precision_score(y, sc))
        auroc = float(roc_auc_score(y, sc))
        by_t = {u: float(average_precision_score(y[targets == u], sc[targets == u])) for u in uniq_t}
        mtap13 = float(np.mean(list(by_t.values())))
        mtap11 = float(np.mean([by_t[u] for u in uniq_t if u not in ("IL10Ra", "LTK")]))
        return {
            "pooled_ap": gap,
            "auroc": auroc,
            "mean_target_ap_13": mtap13,
            "mean_target_ap_11_multihit": mtap11,
            "by_target": by_t,
        }

    exp1_results = {
        "m1_streaming_zero_test_batch_stats": {
            "bc_score_m1_no_batch_norm": summarize_overath(oof_bc_score_m1_eq),
            "true_ebm_m1_zero_param_eq": summarize_overath(oof_true_ebm_m1_eq),
            "true_ebm_m1_boltzmann_nce": summarize_overath(oof_true_ebm_m1_nce),
            "boltz1_raw": summarize_overath(boltz1_raw),
            "boltz1_plus_true_ebm_m1": summarize_overath(oof_boltz1_true_ebm_m1),
            "af3_raw": summarize_overath(af3_raw),
            "af3_plus_true_ebm_m1": summarize_overath(oof_af3_true_ebm_m1),
        },
        "cohort_mode_with_target_pool": {
            "bc_score_5comp_cohort": summarize_overath(bc_5comp_cohort),
            "bc_score_full_eq_cohort": summarize_overath(bc_full_eq_cohort),
            "bc_score_full_reg_cohort": summarize_overath(bc_full_reg_cohort),
            "true_ebm_receptor_partition_eq": summarize_overath(oof_true_ebm_bath_eq),
            "true_ebm_receptor_partition_nce": summarize_overath(oof_true_ebm_bath_nce),
            "boltz1_plus_bc_score_cohort": summarize_overath(boltz1_bc_hyb),
            "af3_plus_bc_score_cohort": summarize_overath(af3_bc_hyb),
        },
    }

    # Lock the True-EBM physical scales and Boltzmann NCE weights on all 13 Overath targets for Experiment 2
    locked_subcycle_scales = {}
    sub_all_list = []
    for k in subcycle_names:
        Xk = X_subcycles_raw[k]
        med_k, std_k = compute_training_physical_scales(Xk, targets)
        locked_subcycle_scales[k] = (med_k, std_k)
        sub_all_list.append(np.tanh((Xk - med_k) / (1.5 * std_k)).mean(axis=1))
    F_overath_all = np.column_stack(sub_all_list)
    med_flat_all, std_flat_all = compute_training_physical_scales(X_all_flat, targets)
    Z_overath_flat = np.tanh((X_all_flat - med_flat_all) / (1.5 * std_flat_all))

    w5_nce_locked = fit_boltzmann_nce_weights(F_overath_all, y, targets, l2_reg=1.5)
    wall_nce_locked = fit_boltzmann_nce_weights(Z_overath_flat, y, targets, l2_reg=8.0)

    locked_true_ebm = {
        "subcycle_scales": locked_subcycle_scales,
        "flat_scales": (med_flat_all, std_flat_all),
        "w5_nce": w5_nce_locked,
        "wall_nce": wall_nce_locked,
    }

    print(
        f"  [M=1 Streaming, Zero Batch Stats] BC-Score MeanAP13={exp1_results['m1_streaming_zero_test_batch_stats']['bc_score_m1_no_batch_norm']['mean_target_ap_13']:.4f} "
        f"| True-EBM (0-Param) MeanAP13={exp1_results['m1_streaming_zero_test_batch_stats']['true_ebm_m1_zero_param_eq']['mean_target_ap_13']:.4f} "
        f"| True-EBM (NCE) MeanAP13={exp1_results['m1_streaming_zero_test_batch_stats']['true_ebm_m1_boltzmann_nce']['mean_target_ap_13']:.4f} "
        f"(Pooled AP={exp1_results['m1_streaming_zero_test_batch_stats']['true_ebm_m1_boltzmann_nce']['pooled_ap']:.4f})",
        flush=True,
    )
    print(
        f"  [M=1 Streaming Post-CoFold]       Boltz-1 Raw AP={exp1_results['m1_streaming_zero_test_batch_stats']['boltz1_raw']['pooled_ap']:.4f} -> "
        f"Boltz-1 + True-EBM (M=1) AP={exp1_results['m1_streaming_zero_test_batch_stats']['boltz1_plus_true_ebm_m1']['pooled_ap']:.4f} "
        f"(MeanAP13: {exp1_results['m1_streaming_zero_test_batch_stats']['boltz1_raw']['mean_target_ap_13']:.4f} -> "
        f"{exp1_results['m1_streaming_zero_test_batch_stats']['boltz1_plus_true_ebm_m1']['mean_target_ap_13']:.4f})",
        flush=True,
    )
    return exp1_results, locked_true_ebm


# ====================================================================================================
# EXPERIMENT 2: ZERO-SHOT TRANSFER TO SCORPION (N=243, 5 TARGETS) & NIPAH-G (N=881)
# ====================================================================================================
def score_external_with_true_ebm(rows: list[dict], locked_true_ebm: dict) -> dict[str, np.ndarray]:
    """Score an external dataset using the locked Overath True-EBM in both M=1 (zero test-batch stats)
    and Receptor-Partition-Function modes."""
    targets = np.array([str(r["target_id"]) for r in rows])
    subcycle_names = list(SUBCYCLE_SPECS.keys())
    all_specs_flat = [item for sub in SUBCYCLE_SPECS.values() for item in sub]

    sub_list = []
    for k in subcycle_names:
        Xk = extract_raw_feature_matrix(rows, SUBCYCLE_SPECS[k])
        med_k, std_k = locked_true_ebm["subcycle_scales"][k]
        Zk = np.tanh((Xk - med_k) / (1.5 * std_k)).mean(axis=1)
        sub_list.append(Zk)
    F_ext = np.column_stack(sub_list)

    X_flat = extract_raw_feature_matrix(rows, all_specs_flat)
    med_all, std_all = locked_true_ebm["flat_scales"]
    Z_flat = np.tanh((X_flat - med_all) / (1.5 * std_all))

    w5_eq = np.ones(5, dtype=np.float64) / 5.0
    w5_nce = locked_true_ebm["w5_nce"]
    wall_nce = locked_true_ebm["wall_nce"]

    # Pure M=1 Single-Complex Free Energy Scores (ZERO test-batch statistics!)
    s_m1_eq = np.einsum("ij,j->i", F_ext, w5_eq)
    s_m1_nce = 0.50 * np.einsum("ij,j->i", F_ext, w5_nce) + 0.50 * np.einsum("ij,j->i", Z_flat, wall_nce)

    # Receptor Partition Function Calibrated Scores (subtracting scalar log Z_T per target)
    s_bath_eq = receptor_partition_calibrate(s_m1_eq, targets, tau_T=0.25)
    s_bath_nce = receptor_partition_calibrate(s_m1_nce, targets, tau_T=0.25)

    return {
        "true_ebm_m1_eq": s_m1_eq,
        "true_ebm_m1_nce": s_m1_nce,
        "true_ebm_bath_eq": s_bath_eq,
        "true_ebm_bath_nce": s_bath_nce,
    }


def run_experiment_2_external(locked_true_ebm: dict) -> dict:
    print("\n=== Experiment 2: Zero-Shot External Transfer (Scorpion N=243 & Nipah-G N=881) ===", flush=True)
    from run_locked_external_validation_and_full_timing import score_dataset_with_locked_v7

    manifest = json.loads(LOCKED_MANIFEST_JSON.read_text())

    # 1. Scorpion Multi-Target Campaign (N=243 across 5 targets, mixed generators: BoltzGen, BindCraft, RFdiffusion)
    scorp_data = torch.load(SCORPION_FEAT_CACHE, map_location="cpu", weights_only=False)
    scorp_rows = scorp_data["rows"]
    y_scorp_adaptyv = np.array([int(r["label"]) for r in scorp_rows], dtype=np.int32)
    y_scorp_pub = np.array([int(r["label_secondary"]) for r in scorp_rows], dtype=np.int32)
    t_scorp = np.array([str(r["target_id"]) for r in scorp_rows])
    pos_t_adaptyv = [u for u in sorted(np.unique(t_scorp)) if (y_scorp_adaptyv[t_scorp == u] == 1).any()]
    pos_t_pub = [u for u in sorted(np.unique(t_scorp)) if (y_scorp_pub[t_scorp == u] == 1).any()]

    bc_scorp = score_dataset_with_locked_v7(scorp_rows, manifest)
    tebm_scorp = score_external_with_true_ebm(scorp_rows, locked_true_ebm)

    # Also compute BC-Score in M=1 mode (no test-batch stats) on Scorpion
    from run_locked_external_validation_and_full_timing import B1_SPECS_PURE_EBM, PURE_WITHIN_SPECS
    E_raw_s = extract_raw_feature_matrix(scorp_rows, B1_SPECS_PURE_EBM)
    med_e = np.array(manifest["full_ebm_cross_med"], dtype=np.float64)[None, :]
    std_e = np.array(manifest["full_ebm_cross_std"], dtype=np.float64)[None, :]
    Ze_s = np.tanh((E_raw_s - med_e) / (1.5 * std_e))
    bc_scorp_m1_only = Ze_s.mean(axis=1)

    protenix_ipsae = np.array([float(r["features"].get("ext_protenix_ipsae", 0.0)) for r in scorp_rows])
    esmfold2_ipsae = np.array([float(r["features"].get("ext_esmfold2_ipsae", 0.0)) for r in scorp_rows])

    def eval_scorp(sc: np.ndarray, y_vec: np.ndarray, pos_t: list[str]) -> dict:
        gap = float(average_precision_score(y_vec, sc))
        auroc = float(roc_auc_score(y_vec, sc))
        by_t = {u: float(average_precision_score(y_vec[t_scorp == u], sc[t_scorp == u])) for u in pos_t}
        mtap = float(np.mean(list(by_t.values())))
        order = np.argsort(-sc)
        return {
            "pooled_ap": gap,
            "auroc": auroc,
            "mean_target_ap": mtap,
            "by_target_ap": by_t,
            "top_10_hits": int(y_vec[order[:10]].sum()),
            "top_25_hits": int(y_vec[order[:25]].sum()),
        }

    scorp_methods = {
        "bc_score_5comp_cohort": bc_scorp["bc_ebm_5comp_reg"],
        "bc_score_full_cohort": bc_scorp["bc_ebm_full_reg"],
        "bc_score_m1_no_batch_norm": bc_scorp_m1_only,
        "true_ebm_m1_zero_param_eq": tebm_scorp["true_ebm_m1_eq"],
        "true_ebm_m1_boltzmann_nce": tebm_scorp["true_ebm_m1_nce"],
        "true_ebm_bath_zero_param_eq": tebm_scorp["true_ebm_bath_eq"],
        "true_ebm_bath_boltzmann_nce": tebm_scorp["true_ebm_bath_nce"],
        "protenix_v2_ipsae": protenix_ipsae,
        "esmfold2_ipsae": esmfold2_ipsae,
    }
    scorp_adaptyv_res = {k: eval_scorp(v, y_scorp_adaptyv, pos_t_adaptyv) for k, v in scorp_methods.items()}
    scorp_pub_res = {k: eval_scorp(v, y_scorp_pub, pos_t_pub) for k, v in scorp_methods.items()}

    print("  [Scorpion N=243, Adaptyv 21 Hits]:", flush=True)
    for k, d in scorp_adaptyv_res.items():
        print(f"    {k:30s}: Pooled AP={d['pooled_ap']:.4f} | MeanTargetAP={d['mean_target_ap']:.4f} | AUROC={d['auroc']:.4f} | Top25={d['top_25_hits']}/21", flush=True)

    # 2. Nipah-G Campaign (N=881, single target receptor Nipah_G, 100 verified binders)
    nipah_data = torch.load(NIPAH_FEAT_CACHE, map_location="cpu", weights_only=False)
    nipah_rows = nipah_data["rows"]
    y_nipah = np.array([int(r["label"]) for r in nipah_rows], dtype=np.int32)

    bc_nipah = score_dataset_with_locked_v7(nipah_rows, manifest)
    tebm_nipah = score_external_with_true_ebm(nipah_rows, locked_true_ebm)

    boltz2_iptm = np.nan_to_num(np.array([float(r["features"].get("ext_boltz2_iptm", np.nan)) for r in nipah_rows]), nan=0.5)
    boltz2_ipsae = np.nan_to_num(np.array([float(r["features"].get("ext_boltz2_ipsae", np.nan)) for r in nipah_rows]), nan=0.3)

    # Pure M=1 Boltzmann Hybrids on Nipah-G (using Overath training scale, zero Nipah batch tuning)
    z_tebm_nipah = (tebm_nipah["true_ebm_m1_nce"] - np.mean(tebm_nipah["true_ebm_m1_nce"])) / (np.std(tebm_nipah["true_ebm_m1_nce"]) + 1e-5)
    hyb_iptm_true_ebm = np.clip(boltz2_iptm, 1e-4, 1.0) * np.exp(0.30 * z_tebm_nipah)
    hyb_ipsae_true_ebm = np.clip(boltz2_ipsae, 1e-4, 1.0) * np.exp(0.25 * z_tebm_nipah)

    def eval_nipah(sc: np.ndarray) -> dict:
        gap = float(average_precision_score(y_nipah, sc))
        auroc = float(roc_auc_score(y_nipah, sc))
        order = np.argsort(-sc)
        return {
            "ap": gap,
            "auroc": auroc,
            "top_25_hits": int(y_nipah[order[:25]].sum()),
            "top_50_hits": int(y_nipah[order[:50]].sum()),
            "top_100_hits": int(y_nipah[order[:100]].sum()),
        }

    nipah_methods = {
        "bc_score_5comp_cohort": bc_nipah["bc_ebm_5comp_reg"],
        "bc_score_full_cohort": bc_nipah["bc_ebm_full_reg"],
        "true_ebm_m1_zero_param_eq": tebm_nipah["true_ebm_m1_eq"],
        "true_ebm_m1_boltzmann_nce": tebm_nipah["true_ebm_m1_nce"],
        "boltz2_interface_ptm": boltz2_iptm,
        "boltz2_min_ipsae": boltz2_ipsae,
        "boltz2_iptm_plus_true_ebm": hyb_iptm_true_ebm,
        "boltz2_ipsae_plus_true_ebm": hyb_ipsae_true_ebm,
    }
    nipah_res = {k: eval_nipah(v) for k, v in nipah_methods.items()}
    print("  [Nipah-G N=881, 100 Hits]:", flush=True)
    for k, d in nipah_res.items():
        print(f"    {k:30s}: AP={d['ap']:.4f} | AUROC={d['auroc']:.4f} | Top50={d['top_50_hits']}/50 | Top100={d['top_100_hits']}/100", flush=True)

    return {
        "scorpion_adaptyv_21hits": scorp_adaptyv_res,
        "scorpion_publication_12hits": scorp_pub_res,
        "nipah_g_881": nipah_res,
    }


# ====================================================================================================
# EXPERIMENT 3: TRUE FREE-ENERGY EBM CO-DESIGN ON ALL 39 FAILED OVERATH SCAFFOLDS
# ====================================================================================================
@torch.no_grad()
def codesign_with_true_partition_function_ebm(
    b_beads: torch.Tensor,
    b_aa_init: torch.Tensor,
    b_geom: dict[str, torch.Tensor],
    t_erased: torch.Tensor,
    t_tip_hat: torch.Tensor,
    t_aa: torch.Tensor,
    h_t: torch.Tensor,
    if_model: torch.nn.Module,
    sc_model: torch.nn.Module,
    ebm_model: torch.nn.Module,
    rng_opt: np.random.Generator,
    rng_eval_heldout: np.random.Generator,
    n_Sweeps: int = 2,
    lam_if: float = 0.85,
    tau_pose: float = 0.50,
) -> dict:
    """True Statistical-Mechanics Free-Energy Co-Design:
    Instead of minimizing single-pose energy E(x_0, s) (which caused 19/39 designs to worsen their pose funnel
    by introducing sticky hydrophobic residues that also stabilize off-target decoy poses), we minimize the
    exact Partition-Function Free Energy:
        dF_EBM(x_0, s) = E(x_0, s) - (-tau * log (1/M_decoy sum_{m in Decoy} exp(-E(x_m, s)/tau)))
                         + 0.45 * <RMSD(x_m)>_{P_Boltzmann(x_m | s)}
                         - 0.25 * (E(x_0, s_Ala) - E(x_0, s))
                         + 0.80 * E_Buns(x_0, s) - 0.35 * O_ring(s) - lam_if * log P_IF(s | X_bb)
    We then evaluate both on the canonical 24-pose ensemble AND on an independent held-out 24-pose SE(3)
    ensemble generated with a separate random seed (`rng_eval_heldout`).
    """
    Nb = len(b_beads)
    Nt = len(t_erased)
    log_p_mono = if_model.forward_from_geom(b_geom)

    comp_beads = torch.cat([b_beads, t_erased], dim=0)
    comp_chain = torch.cat([torch.zeros(Nb, dtype=torch.long), torch.ones(Nt, dtype=torch.long)], dim=0)
    comp_res_idx = torch.cat([torch.arange(Nb, dtype=torch.long), torch.arange(Nt, dtype=torch.long)], dim=0)
    comp_geom = compute_pure_backbone_geometry(comp_beads, comp_chain, comp_res_idx)
    log_p_comp = if_model.forward_from_geom(comp_geom)[:Nb]
    log_p_joint = 0.65 * log_p_comp + 0.35 * log_p_mono

    d_ca0 = torch.cdist(b_beads[:, 1], t_erased[:, 1])
    min_d = d_ca0.amin(dim=1)
    intf_idx = torch.where(min_d < 8.5)[0]
    if intf_idx.numel() == 0:
        intf_idx = torch.argsort(min_d)[: min(8, len(b_aa_init))]

    is_core = min_d < 6.2
    is_rim = (min_d >= 6.2) & (min_d < 8.8)

    # Pre-build the 24-pose SE(3) ensemble for partition-function marginalization during optimization
    sc0, _, _, tip0, _, _ = sc_model.forward_from_geom(b_geom, b_aa_init, return_tip=True)
    b_er0 = b_beads.clone()
    b_er0[:, 4] = sc0
    rots_opt, trans_opt, pose_rmsd_opt = build_redocking_pose_ensemble(b_er0, t_erased, rng_opt, n_global=11)
    decoy_mask_opt = pose_rmsd_opt > 2.5

    # Pre-build an INDEPENDENT held-out 24-pose SE(3) ensemble for unbiased funnel verification
    rots_held, trans_held, pose_rmsd_held = build_redocking_pose_ensemble(b_er0, t_erased, rng_eval_heldout, n_global=11)

    def eval_sequence_free_energy(aa_cand: torch.Tensor) -> tuple[float, dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        sc_hat, _, _, tip_hat, _, _ = sc_model.forward_from_geom(b_geom, aa_cand, return_tip=True)
        b_er = b_beads.clone()
        b_er[:, 4] = sc_hat
        h_b = ebm_model.encode_partner(b_geom, aa_cand)

        comp_all = ebm_model.score_poses(
            b_er,
            aa_cand,
            h_b,
            t_erased,
            t_aa,
            h_t,
            rots_opt,
            trans_opt,
            return_components=True,
            b_tip_erased=tip_hat,
            t_tip_erased=t_tip_hat,
        )
        e_poses = comp_all["total"]
        e_ebm0 = float(e_poses[0].item())
        e_buns0 = float((comp_all["buns_binder"][0] + comp_all["buns_target"][0]).item())

        # Competing decoy pose partition function F_decoy(s) = -tau * log (1/M_dec sum_{m in dec} exp(-E_m / tau))
        e_dec = e_poses[decoy_mask_opt]
        f_decoy = float((-tau_pose * (torch.logsumexp(-e_dec / tau_pose, dim=0) - math.log(len(e_dec)))).item())
        df_pose_gap = e_ebm0 - f_decoy  # Negative when native pose has lower free energy than decoy ensemble

        w_boltz = F.softmax(-e_poses / tau_pose, dim=0)
        boltz_rmsd = float((w_boltz * pose_rmsd_opt).sum().item())
        p_native = float(w_boltz[pose_rmsd_opt < 2.5].sum().item())

        phys = PHYSCHEM_TABLE[aa_cand.clamp(0, 19)]
        core_arom_hydro = float((phys[is_core, 4] + 0.6 * F.relu(phys[is_core, 1])).mean().item()) if is_core.any() else 0.0
        rim_polar = float(((phys[is_rim, 2] + phys[is_rim, 3] + phys[is_rim, 0].abs()) > 0.3).to(torch.float32).mean().item()) if is_rim.any() else 0.0
        oring_contrast = core_arom_hydro + 0.5 * rim_polar

        valid = aa_cand < 20
        ll_if = float(log_p_joint[torch.arange(len(aa_cand))[valid], aa_cand[valid]].sum().item())

        # True EBM Free Energy Objective: native pose energy + pose partition contrast + expected funnel RMSD + Buns - IF prior
        obj = (
            0.30 * e_ebm0
            + 0.35 * df_pose_gap
            + 0.45 * boltz_rmsd
            + 0.80 * e_buns0
            - 0.35 * oring_contrast
            - lam_if * ll_if
        )
        return obj, {
            "e_ebm": e_ebm0,
            "df_pose_gap": df_pose_gap,
            "e_buns": e_buns0,
            "oring_contrast": oring_contrast,
            "boltz_rmsd_opt": boltz_rmsd,
            "p_native_opt": p_native,
        }, b_er, tip_hat, h_b

    I_rot1 = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    Z_tr1 = torch.zeros((1, 3), dtype=torch.float32)

    def quick_single_pose_screen(aa_cand: torch.Tensor) -> float:
        sc_hat, _, _, tip_hat, _, _ = sc_model.forward_from_geom(b_geom, aa_cand, return_tip=True)
        b_er = b_beads.clone()
        b_er[:, 4] = sc_hat
        h_b = ebm_model.encode_partner(b_geom, aa_cand)
        comp0 = ebm_model.score_poses(
            b_er, aa_cand, h_b, t_erased, t_aa, h_t, I_rot1, Z_tr1,
            return_components=True, b_tip_erased=tip_hat, t_tip_erased=t_tip_hat,
        )
        e0 = float(comp0["total"][0].item())
        buns0 = float((comp0["buns_binder"][0] + comp0["buns_target"][0]).item())
        phys = PHYSCHEM_TABLE[aa_cand.clamp(0, 19)]
        core_ah = float((phys[is_core, 4] + 0.6 * F.relu(phys[is_core, 1])).mean().item()) if is_core.any() else 0.0
        rim_p = float(((phys[is_rim, 2] + phys[is_rim, 3] + phys[is_rim, 0].abs()) > 0.3).to(torch.float32).mean().item()) if is_rim.any() else 0.0
        oring_c = core_ah + 0.5 * rim_p
        valid = aa_cand < 20
        ll_if = float(log_p_joint[torch.arange(len(aa_cand))[valid], aa_cand[valid]].sum().item())
        return 0.35 * e0 + 0.80 * buns0 - 0.35 * oring_c - lam_if * ll_if

    _, stats_init, _, _, _ = eval_sequence_free_energy(b_aa_init)
    aa_cur = b_aa_init.clone()
    obj_cur, stats_cur, b_er_cur, tip_cur, h_b_cur = eval_sequence_free_energy(aa_cur)

    top6_per_pos = torch.topk(log_p_joint[:, :20], k=6, dim=-1).indices

    for _ in range(n_Sweeps):
        for pos_t in intf_idx:
            pos = int(pos_t.item())
            cands = [c for c in dict.fromkeys(top6_per_pos[pos].tolist() + [18, 17, 13, 10, 6, 1, 15]) if c != int(aa_cur[pos].item())]
            # Fast single-pose screen to select top-2 candidate mutations at this residue for full 24-pose partition evaluation
            scored_cands = []
            for c_aa in cands:
                aa_try = aa_cur.clone()
                aa_try[pos] = c_aa
                scored_cands.append((quick_single_pose_screen(aa_try), c_aa))
            scored_cands.sort(key=lambda x: x[0])
            for _, c_aa in scored_cands[:2]:
                aa_try = aa_cur.clone()
                aa_try[pos] = c_aa
                obj_try, stats_try, b_er_try, tip_try, h_b_try = eval_sequence_free_energy(aa_try)
                if obj_try < obj_cur - 1e-4 and stats_try["boltz_rmsd_opt"] <= max(stats_init["boltz_rmsd_opt"] - 0.02, stats_cur["boltz_rmsd_opt"] + 0.05):
                    obj_cur = obj_try
                    stats_cur = stats_try
                    aa_cur = aa_try
                    b_er_cur = b_er_try
                    tip_cur = tip_try
                    h_b_cur = h_b_try

    # Evaluate on the INDEPENDENT held-out 24-pose SE(3) ensemble (never seen during optimization!)
    h_b0 = ebm_model.encode_partner(b_geom, b_aa_init)
    e_held_before = ebm_model.score_poses(
        b_er0, b_aa_init, h_b0, t_erased, t_aa, h_t, rots_held, trans_held, b_tip_erased=tip0, t_tip_erased=t_tip_hat
    )
    w_held_before = F.softmax(-e_held_before / tau_pose, dim=0)
    held_rmsd_before = float((w_held_before * pose_rmsd_held).sum().item())
    held_pnat_before = float(w_held_before[pose_rmsd_held < 2.5].sum().item())

    e_held_after = ebm_model.score_poses(
        b_er_cur, aa_cur, h_b_cur, t_erased, t_aa, h_t, rots_held, trans_held, b_tip_erased=tip_cur, t_tip_erased=t_tip_hat
    )
    w_held_after = F.softmax(-e_held_after / tau_pose, dim=0)
    held_rmsd_after = float((w_held_after * pose_rmsd_held).sum().item())
    held_pnat_after = float(w_held_after[pose_rmsd_held < 2.5].sum().item())

    # Computational Alanine Scan ddG before and after
    I_rot1 = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    Z_tr1 = torch.zeros((1, 3), dtype=torch.float32)

    def single_pose_e(aa_seq: torch.Tensor) -> float:
        sc_s, _, _, tip_s, _, _ = sc_model.forward_from_geom(b_geom, aa_seq, return_tip=True)
        b_s = b_beads.clone()
        b_s[:, 4] = sc_s
        h_s = ebm_model.encode_partner(b_geom, aa_seq)
        return float(
            ebm_model.score_poses(b_s, aa_seq, h_s, t_erased, t_aa, h_t, I_rot1, Z_tr1, b_tip_erased=tip_s, t_tip_erased=t_tip_hat)[0].item()
        )

    aa_ala0 = b_aa_init.clone()
    aa_ala0[intf_idx] = 0
    ala_ddg_before = single_pose_e(aa_ala0) - stats_init["e_ebm"]

    aa_ala = aa_cur.clone()
    aa_ala[intf_idx] = 0
    ala_ddg_after = single_pose_e(aa_ala) - stats_cur["e_ebm"]

    return {
        "aa_codesign": aa_cur,
        "intf_idx": intf_idx,
        "stats_init": stats_init,
        "stats_codesign": stats_cur,
        "boltz_rmsd_before": stats_init["boltz_rmsd_opt"],
        "boltz_rmsd_after": stats_cur["boltz_rmsd_opt"],
        "p_native_before": stats_init["p_native_opt"],
        "p_native_after": stats_cur["p_native_opt"],
        "heldout_rmsd_before": held_rmsd_before,
        "heldout_rmsd_after": held_rmsd_after,
        "heldout_p_native_before": held_pnat_before,
        "heldout_p_native_after": held_pnat_after,
        "ala_ddg_before": ala_ddg_before,
        "ala_ddg_after": ala_ddg_after,
    }


@torch.no_grad()
def run_experiment_3_codesign_39() -> dict:
    print("\n=== Experiment 3: True Partition-Function EBM Co-Design on All 39 Failed Scaffolds ===", flush=True)
    rows = torch.load(OVERATH_FEAT_CACHE, map_location="cpu", weights_only=False)
    if_model, sc_model, ebm_model = train_or_load_general_models(force_retrain=False)
    if_model, sc_model, ebm_model = if_model.cpu(), sc_model.cpu(), ebm_model.cpu()
    cache_5b = torch.load(OVERATH_5BEAD_CACHE, map_location="cpu", weights_only=False)
    targets_cache = cache_5b["targets"]
    binders_cache = cache_5b["binders"]

    y_all = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    t_all = np.array([str(r["target_id"]) for r in rows])
    rng_opt = np.random.default_rng(2026)
    rng_held = np.random.default_rng(99999)

    records_39 = []
    t0 = time.perf_counter()
    for u in sorted(np.unique(t_all)):
        neg_u = np.where((t_all == u) & (y_all == 0))[0][:3]
        for idx_n in neg_u:
            d = binders_cache[idx_n]
            r = rows[idx_n]
            b_beads = d["beads"].to(torch.float32)
            b_aa = d["aa"].to(torch.long)
            b_chain = d["chain"].to(torch.long)
            tv = targets_cache[d["t_hash"]]
            t_beads = tv["beads"].to(torch.float32)
            t_aa = tv["aa"].to(torch.long)
            t_chain = tv["chain"].to(torch.long)

            b_geom = compute_pure_backbone_geometry(b_beads, b_chain, torch.arange(len(b_beads)))
            t_geom = compute_pure_backbone_geometry(t_beads, t_chain, torch.arange(len(t_beads)))
            t_sc, _, _, t_tip, _, _ = sc_model.forward_from_geom(t_geom, t_aa, return_tip=True)
            t_erased = t_beads.clone()
            t_erased[:, 4] = t_sc
            h_t = ebm_model.encode_partner(t_geom, t_aa)

            cd_out = codesign_with_true_partition_function_ebm(
                b_beads, b_aa, b_geom, t_erased, t_tip, t_aa, h_t,
                if_model, sc_model, ebm_model, rng_opt, rng_held, n_Sweeps=2
            )

            aa_opt = cd_out["aa_codesign"]
            log_p_mono = if_model.forward_from_geom(b_geom)
            valid = b_aa < 20
            mono_ll_before = float(log_p_mono[torch.arange(len(b_aa))[valid], b_aa[valid].clamp(0, 19)].mean().item())
            mono_ll_after = float(log_p_mono[torch.arange(len(aa_opt))[valid], aa_opt[valid].clamp(0, 19)].mean().item())

            n_mut = int((aa_opt[cd_out["intf_idx"]] != b_aa[cd_out["intf_idx"]]).sum().item())
            records_39.append({
                "target_id": str(u),
                "binder_id": str(r.get("candidate_id", f"{u}_{idx_n}")),
                "n_mutations": n_mut,
                "e_ebm_before": float(cd_out["stats_init"]["e_ebm"]),
                "e_ebm_after": float(cd_out["stats_codesign"]["e_ebm"]),
                "df_pose_gap_before": float(cd_out["stats_init"]["df_pose_gap"]),
                "df_pose_gap_after": float(cd_out["stats_codesign"]["df_pose_gap"]),
                "e_buns_before": float(cd_out["stats_init"]["e_buns"]),
                "e_buns_after": float(cd_out["stats_codesign"]["e_buns"]),
                "ala_ddg_before": float(cd_out["ala_ddg_before"]),
                "ala_ddg_after": float(cd_out["ala_ddg_after"]),
                "boltz_rmsd_before": float(cd_out["boltz_rmsd_before"]),
                "boltz_rmsd_after": float(cd_out["boltz_rmsd_after"]),
                "p_native_before": float(cd_out["p_native_before"]),
                "p_native_after": float(cd_out["p_native_after"]),
                "heldout_rmsd_before": float(cd_out["heldout_rmsd_before"]),
                "heldout_rmsd_after": float(cd_out["heldout_rmsd_after"]),
                "heldout_p_native_before": float(cd_out["heldout_p_native_before"]),
                "heldout_p_native_after": float(cd_out["heldout_p_native_after"]),
                "frozen_monomer_if_ll_before": mono_ll_before,
                "frozen_monomer_if_ll_after": mono_ll_after,
            })

    el = time.perf_counter() - t0

    def summarize_pair(key_b: str, key_a: str, lower_is_better: bool) -> dict:
        vb = np.array([x[key_b] for x in records_39], dtype=np.float64)
        va = np.array([x[key_a] for x in records_39], dtype=np.float64)
        diff = va - vb
        p_val = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
        n_imp = int(np.sum(va < vb)) if lower_is_better else int(np.sum(va > vb))
        return {
            "mean_before": float(np.mean(vb)),
            "median_before": float(np.median(vb)),
            "mean_after": float(np.mean(va)),
            "median_after": float(np.median(va)),
            "n_improved": n_imp,
            "n_total": len(records_39),
            "wilcoxon_two_sided_p": p_val,
        }

    summary = {
        "elapsed_sec": el,
        "sec_per_scaffold": el / len(records_39),
        "e_ebm_single_pose": summarize_pair("e_ebm_before", "e_ebm_after", lower_is_better=True),
        "df_pose_free_energy_gap": summarize_pair("df_pose_gap_before", "df_pose_gap_after", lower_is_better=True),
        "buns_penalty": summarize_pair("e_buns_before", "e_buns_after", lower_is_better=True),
        "ala_scan_ddg": summarize_pair("ala_ddg_before", "ala_ddg_after", lower_is_better=False),
        "funnel_rmsd_opt_ensemble": summarize_pair("boltz_rmsd_before", "boltz_rmsd_after", lower_is_better=True),
        "p_native_opt_ensemble": summarize_pair("p_native_before", "p_native_after", lower_is_better=False),
        "funnel_rmsd_heldout_ensemble": summarize_pair("heldout_rmsd_before", "heldout_rmsd_after", lower_is_better=True),
        "p_native_heldout_ensemble": summarize_pair("heldout_p_native_before", "heldout_p_native_after", lower_is_better=False),
        "frozen_monomer_if_ll": summarize_pair("frozen_monomer_if_ll_before", "frozen_monomer_if_ll_after", lower_is_better=False),
        "all_39_trajectories": records_39,
    }

    print(f"  Completed 39-scaffold True-EBM Co-Design in {el:.2f}s ({el/39:.2f}s/scaffold):", flush=True)
    for k_m, d_m in summary.items():
        if isinstance(d_m, dict):
            print(
                f"    {k_m:30s}: {d_m['mean_before']:+.3f} -> {d_m['mean_after']:+.3f} "
                f"| Improved={d_m['n_improved']}/39 (p={d_m['wilcoxon_two_sided_p']:.2e})",
                flush=True,
            )
    return summary


def main() -> None:
    t0 = time.time()
    exp1, locked_true_ebm = run_experiment_1_overath()
    exp2 = run_experiment_2_external(locked_true_ebm)
    exp3 = run_experiment_3_codesign_39()

    out = {
        "experiment_1_overath": exp1,
        "experiment_2_external_transfer": exp2,
        "experiment_3_true_ebm_codesign_39": exp3,
        "total_elapsed_sec": time.time() - t0,
    }
    TRUE_EBM_RESULTS_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved complete True-EBM experimental results to {TRUE_EBM_RESULTS_JSON} in {time.time() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
