#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final no-target numerical evaluator for the charged-lepton mass hierarchy.

This script implements the closed [101]-ledger / affine-load version of the
charged-lepton hierarchy described in the final manuscript notation.

Baseline theory input
---------------------
Only one user-supplied theory input enters the prediction:

    α⁻¹

The comparison masses m_e, m_μ, m_τ and their experimental uncertainties are
held in a separate comparison-only block.  They are used only after the
mass ratios m_μ/m_e and m_τ/m_e have been predicted.

Final chain
-----------
    U(R)
      → Relator exponent slot
      → Γₙ^[101]
      → Tₙ^[101] = ln Γₙ^[101]
      → ΔΛ̂_{λC,n} = Tₙ/[β_Λ(A−Tₙ)]
      → ΔΛ̂ₙ = ΔΛ̂_{λC,n}/(1−ρₙ)
      → α_{R,n} = α(1 + β_Λ ΔΛ̂ₙ)
      → Lₙ = ln(mₙ/mₑ)

The compact final formula used for the prediction is

    Lₙ = A Tₙ / [ A(1−ρₙ) + ρₙ Tₙ ],
    A = π/(8α),
    ρₙ = C_UV^Gauss ln(Dₙ)/(n+1),
    C_UV^Gauss = ½(ln2 + γ_E),
    n = 2,3.

No ΔΛ̂_{λC,μ}, ΔΛ̂_{λC,τ}, T_μ, T_τ, D_μ, D_τ, L_μ, or L_τ is supplied as an
input.  They are all computed inside this script.

Interpretation guardrails
-------------------------
• Γₙ^[101] is a shell-admission ratio, not a mass ratio.
• Tₙ^[101] is a bare Relator-exponent-slot readout, not Lₙ.
• α_{R,n} is a Relator-exponent Alpha coordinate, not α_QED(q²).
• α_{R,n} is not substituted into path-dependent prefactors such as
  C_RLVM(α) or C_RLTM(α).
• The finite [101] odd-mode source ledger is treated as fixed Alpha-ledger
  data, not a target fit.

Dependencies
------------
    pip install rich mpmath numpy scipy
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import math
import mpmath as mp
import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.special import ellipe, ellipk, eval_legendre, roots_legendre

# =============================================================================
# Numerical precision
# =============================================================================

mp.mp.dps = 80


# =============================================================================
# THEORY INPUT: the only supplied theory input that enters the prediction
# =============================================================================


@dataclass(frozen=True)
class TheoryInput:
    """Only user-supplied theory input."""

    alpha_inv: mp.mpf = mp.mpf("137.035999177")


# =============================================================================
# COMPARISON-ONLY VALUES
# =============================================================================
# These values are not used to construct Γₙ, Tₙ, ΔΛ̂ₙ, Dₙ, ρₙ, Lₙ, or mₙ/mₑ.
# m_e is used only after the dimensionless ratios are complete, to express the
# predictions in MeV.


@dataclass(frozen=True)
class ComparisonValues:
    electron_mass_mev: mp.mpf = mp.mpf("0.51099895069")
    electron_sigma_mev: mp.mpf = mp.mpf("0.00000000016")

    muon_mass_exp_mev: mp.mpf = mp.mpf("105.6583755")
    muon_sigma_exp_mev: mp.mpf = mp.mpf("0.0000023")

    tau_mass_exp_mev: mp.mpf = mp.mpf("1776.93")
    tau_sigma_exp_mev: mp.mpf = mp.mpf("0.09")


# =============================================================================
# FIXED FORMULA / ALPHA-LEDGER DEFINITIONS
# =============================================================================
# These are fixed mathematical definitions or imported Alpha-ledger data.  They
# are not fitted to m_μ or m_τ.


@dataclass(frozen=True)
class FormulaLedger:
    """Fixed non-tunable formula/ledger settings used by the calculation."""

    shell_source_jmax: int = 101
    shell_source_gl_nodes: int = 512

    # Fifth-order scalar branch evaluator D_C(x) = Σ c_k x^k.  These are fixed
    # Alpha-sector ledger coefficients, not charged-lepton fit parameters.
    scalar_dc_coefficients: Tuple[mp.mpf, ...] = (
        mp.mpf("0"),
        mp.mpf("1"),
        mp.mpf("-1.37318423257715981904"),
        mp.mpf("3.80011555037140726825"),
        mp.mpf("-8.85785467402324147604"),
        mp.mpf("24.73523387578082466951"),
    )


# =============================================================================
# DIAGNOSTIC PERTURBATIONS FOR LOCAL SENSITIVITY ONLY
# =============================================================================
# Baseline values are exactly 1.  These are not inputs to the prediction.  They
# are used only to compute local elasticities around the baseline formula.


@dataclass(frozen=True)
class AuditPerturbation:
    alpha_inv_scale: mp.mpf = mp.mpf("1")
    c_uv_scale: mp.mpf = mp.mpf("1")
    p_ir_1_scale: mp.mpf = mp.mpf("1")
    p_ir_2_scale: mp.mpf = mp.mpf("1")
    p_ir_3_scale: mp.mpf = mp.mpf("1")
    out_1_scale: mp.mpf = mp.mpf("1")
    out_2_scale: mp.mpf = mp.mpf("1")
    out_3_scale: mp.mpf = mp.mpf("1")
    r_chi_1_scale: mp.mpf = mp.mpf("1")
    r_chi_2_scale: mp.mpf = mp.mpf("1")
    r_chi_3_scale: mp.mpf = mp.mpf("1")
    tail_scale: mp.mpf = mp.mpf("1")
    heavy_moment_scale: mp.mpf = mp.mpf("1")
    tau_endpoint_scale: mp.mpf = mp.mpf("1")
    electron_mass_scale: mp.mpf = mp.mpf("1")


@dataclass(frozen=True)
class OutputControls:
    output_text_file: str = "charged_lepton_hierarchy_final_no_tuning_output.txt"
    sensitivity_relative_step: mp.mpf = mp.mpf("1e-6")
    include_sensitivity: bool = True


# =============================================================================
# Formatting helpers
# =============================================================================


def fmt(x: object, digits: int = 16) -> str:
    value = mp.mpf(x)
    if value == 0:
        return "0"
    return mp.nstr(value, n=digits, strip_zeros=False)


def signed_fmt(x: object, digits: int = 14) -> str:
    value = mp.mpf(x)
    return ("+" if value >= 0 else "") + mp.nstr(value, n=digits, strip_zeros=False)


def percent(x: object, places: int = 8) -> str:
    return f"{float(mp.mpf(x) * 100):.{places}f}%"


def ppm(x: object, places: int = 6) -> str:
    return f"{float(mp.mpf(x) * mp.mpf('1e6')):.{places}f} ppm"


def make_table(title: str, columns: Iterable[Tuple[str, str]]) -> Table:
    table = Table(title=title, box=box.ROUNDED, title_style="bold cyan", show_lines=False)
    for header, justify in columns:
        table.add_column(header, justify=justify, no_wrap=False)
    return table


# =============================================================================
# Universal formulae
# =============================================================================

EULER_GAMMA = mp.euler


def effective_alpha_inv(theory: TheoryInput, perturb: AuditPerturbation) -> mp.mpf:
    return theory.alpha_inv * perturb.alpha_inv_scale


def alpha(theory: TheoryInput, perturb: AuditPerturbation = AuditPerturbation()) -> mp.mpf:
    return 1 / effective_alpha_inv(theory, perturb)


def x_star(theory: TheoryInput, perturb: AuditPerturbation = AuditPerturbation()) -> mp.mpf:
    return alpha(theory, perturb) / mp.pi


def A_relator(theory: TheoryInput, perturb: AuditPerturbation = AuditPerturbation()) -> mp.mpf:
    """A = π/(8α)."""
    return mp.pi / (8 * alpha(theory, perturb))


def theta_R() -> mp.mpf:
    return mp.pi / 8


def C_UV_Gauss(perturb: AuditPerturbation = AuditPerturbation()) -> mp.mpf:
    return perturb.c_uv_scale * mp.mpf("0.5") * (mp.log(2) + EULER_GAMMA)


def eta(n: int) -> mp.mpf:
    return 1 / (mp.mpf(n) * mp.pi)


def ell(n: int) -> mp.mpf:
    return 1 / (mp.mpf(n) * mp.pi * mp.sqrt(mp.pi))


def perturb_shell_scale(perturb: AuditPerturbation, prefix: str, n: int) -> mp.mpf:
    return getattr(perturb, f"{prefix}_{n}_scale")


# =============================================================================
# Near-IR TT-χ gate P_IR(ℓ)
# =============================================================================


def I_total() -> mp.mpf:
    return mp.mpf(1) / 6 - 1 / (4 * mp.pi ** 2)


@lru_cache(maxsize=8)
def P_IR_base(n: int) -> mp.mpf:
    """Compute the normalized near-IR TT-χ gate for shell n."""
    ell_value = ell(n)
    split = [
        mp.mpf("0"),
        mp.mpf("0.25"),
        mp.mpf("0.5"),
        mp.mpf("0.75"),
        mp.mpf("0.9"),
        mp.mpf("0.97"),
        mp.mpf("0.99"),
        mp.mpf("0.997"),
        mp.mpf("1"),
    ]

    def integrand(u: mp.mpf) -> mp.mpf:
        one_minus = 1 - u
        gate = 1 - mp.mpf(1) / 3 * one_minus ** 2 / (one_minus ** 2 + ell_value ** 2)
        return u ** 2 * mp.sin(mp.pi * u) ** 2 * gate * mp.e ** (-(one_minus / ell_value) ** 2)

    return mp.quad(integrand, split) / I_total()


def P_IR(n: int, perturb: AuditPerturbation = AuditPerturbation()) -> mp.mpf:
    return P_IR_base(n) * perturb_shell_scale(perturb, "p_ir", n)


# =============================================================================
# Exact Alpha-vector OUT subtraction at odd-mode finite ledger [101]
# =============================================================================


@lru_cache(maxsize=16)
def shell_source_J_modes_base(n: int, jmax: int, nodes: int) -> Tuple[Tuple[int, mp.mpf], ...]:
    """Return J_j source-vector modes for j=1,3,...,jmax.

    The source-vector OUT subtraction is evaluated by

        ΔΛ̂_out^{(n),[jmax]} = -½ Σ_j J_j(η_n)^2.

    The mode coefficients are obtained from the toroidal-shell field projection
    using Gauss-Legendre quadrature over u = cos θ.  The ledger is fixed before
    the charged-lepton comparison.
    """
    eta_float = float(eta(n))
    u, w = roots_legendre(nodes)
    rho = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    z = u

    denom = (eta_float + rho) ** 2 + z * z
    denom2 = (eta_float - rho) ** 2 + z * z
    k2 = 4.0 * eta_float * rho / denom
    K = ellipk(k2)
    E = ellipe(k2)

    # Gauss-Legendre nodes do not include endpoints, so rho>0 in practice.
    B_rho = z / (2.0 * math.pi * rho * np.sqrt(denom)) * (
            -K + (eta_float * eta_float + rho * rho + z * z) / denom2 * E
    )
    B_z = 1.0 / (2.0 * math.pi * np.sqrt(denom)) * (
            K + (eta_float * eta_float - rho * rho - z * z) / denom2 * E
    )
    b_tilde = z * B_rho - rho * B_z

    modes: List[Tuple[int, mp.mpf]] = []
    for j in range(1, jmax + 1, 2):
        Pj = eval_legendre(j, u)
        Pjm1 = eval_legendre(j - 1, u)
        Tj = j * (Pjm1 - u * Pj)
        Ij = 2.0 * j * (j + 1) / (2.0 * j + 1.0)
        a_sh = float(np.sum(w * b_tilde * Tj) / Ij)
        a_hat = 2.0 * (j + 1) * eta_float ** (-0.5) * a_sh
        Jj = math.sqrt(2.0 * math.pi / ((j + 1) * (2 * j + 1))) * a_hat
        modes.append((j, mp.mpf(Jj)))
    return tuple(modes)


def shell_source_J_modes(n: int, ledger: FormulaLedger = FormulaLedger()) -> Tuple[Tuple[int, mp.mpf], ...]:
    return shell_source_J_modes_base(n, ledger.shell_source_jmax, ledger.shell_source_gl_nodes)


def Delta_Lambda_out_exact_101(
        n: int,
        ledger: FormulaLedger = FormulaLedger(),
        perturb: AuditPerturbation = AuditPerturbation(),
) -> mp.mpf:
    modes = shell_source_J_modes(n, ledger)
    norm = mp.fsum(Jj ** 2 for _, Jj in modes)
    return -mp.mpf("0.5") * norm * perturb_shell_scale(perturb, "out", n)


def Delta_Lambda_out_flat(n: int) -> mp.mpf:
    """Flat closed approximation, reported only as a diagnostic."""
    e = eta(n)
    return -mp.pi * (mp.log(1 - e ** 4) / (2 * e) + mp.atanh(e) - mp.atan(e))


# =============================================================================
# Reduced shell-memory return R_χ,n^red
# =============================================================================


def Gamma_geom_shell(n: int) -> mp.mpf:
    e = eta(n)
    return mp.mpf("0.5") * mp.sinh(e) / e


def R_chi_reduced(
        n: int,
        ledger: FormulaLedger = FormulaLedger(),
        perturb: AuditPerturbation = AuditPerturbation(),
) -> mp.mpf:
    e = eta(n)
    P = P_IR(n, perturb)
    uv_load = C_UV_Gauss(perturb) * P
    gamma_geom = Gamma_geom_shell(n)
    A_ret = mp.log(2) * e ** 4
    B_ret = e ** 4 / (8 * mp.pi ** 2)

    beta_chi = P * gamma_geom + P * uv_load / (2 * (1 + A_ret))

    modes = shell_source_J_modes(n, ledger)
    norm = mp.fsum(Jj ** 2 for _, Jj in modes)
    F_chi = mp.mpf("0")
    for j, Jj in modes:
        weight_j = Jj ** 2 / norm
        m_j = mp.mpf(j - 1) / 2
        phi_j = (1 + mp.mpf("0.5") * B_ret * m_j) / (1 + A_ret * m_j)
        F_chi += weight_j * phi_j
    return (1 + beta_chi * F_chi) * perturb_shell_scale(perturb, "r_chi", n)


# =============================================================================
# Exact shell-admission source Γₙ^[101] and Tₙ^[101]
# =============================================================================


def Gamma_101(
        n: int,
        ledger: FormulaLedger = FormulaLedger(),
        perturb: AuditPerturbation = AuditPerturbation(),
) -> mp.mpf:
    if n == 1:
        return mp.mpf("1")
    numerator = abs(Delta_Lambda_out_exact_101(1, ledger, perturb) * R_chi_reduced(1, ledger, perturb))
    denominator = abs(Delta_Lambda_out_exact_101(n, ledger, perturb) * R_chi_reduced(n, ledger, perturb))
    return mp.mpf(n) * P_IR(1, perturb) / P_IR(n, perturb) * numerator / denominator


def T_101(
        n: int,
        ledger: FormulaLedger = FormulaLedger(),
        perturb: AuditPerturbation = AuditPerturbation(),
) -> mp.mpf:
    return mp.log(Gamma_101(n, ledger, perturb))


def T_components(
        n: int,
        ledger: FormulaLedger = FormulaLedger(),
        perturb: AuditPerturbation = AuditPerturbation(),
) -> Dict[str, mp.mpf]:
    geom = mp.log(n)
    ir = mp.log(P_IR(1, perturb) / P_IR(n, perturb))
    out = mp.log(
        abs(Delta_Lambda_out_exact_101(1, ledger, perturb) * R_chi_reduced(1, ledger, perturb))
        / abs(Delta_Lambda_out_exact_101(n, ledger, perturb) * R_chi_reduced(n, ledger, perturb))
    )
    total = geom + ir + out
    return {"log_n": geom, "log_P_ratio": ir, "log_out_R_ratio": out, "T": total}


# =============================================================================
# Alpha scalar branch and ALP affine response
# =============================================================================


@lru_cache(maxsize=1)
def K_ov() -> mp.mpf:
    return (mp.mpf(128) / mp.pi ** 6) * mp.nsum(lambda q: q ** 2 / (q ** 2 - 1) ** 5, [2, mp.inf])


def D_C(x: mp.mpf, ledger: FormulaLedger = FormulaLedger()) -> mp.mpf:
    value = mp.mpf("0")
    for power, coefficient in enumerate(ledger.scalar_dc_coefficients):
        value += coefficient * x ** power
    return value


def D_C_prime(x: mp.mpf, ledger: FormulaLedger = FormulaLedger()) -> mp.mpf:
    value = mp.mpf("0")
    for power, coefficient in enumerate(ledger.scalar_dc_coefficients):
        if power > 0:
            value += power * coefficient * x ** (power - 1)
    return value


def D_lock(Lambda_value: mp.mpf) -> mp.mpf:
    K = K_ov()
    return (mp.mpf(3) * K / 2) * Lambda_value * (1 + K * Lambda_value / (2 * mp.pi ** 2))


def D_lock_prime(Lambda_value: mp.mpf) -> mp.mpf:
    K = K_ov()
    return (mp.mpf(3) * K / 2) * (1 + K * Lambda_value / mp.pi ** 2)


def D_lock_inverse(D_value: mp.mpf) -> mp.mpf:
    K = K_ov()
    a = (mp.mpf(3) * K ** 2) / (4 * mp.pi ** 2)
    b = (mp.mpf(3) * K) / 2
    c = -D_value
    return (-b + mp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def Lambda_e(
        theory: TheoryInput,
        ledger: FormulaLedger,
        perturb: AuditPerturbation = AuditPerturbation(),
) -> mp.mpf:
    return D_lock_inverse(D_C(x_star(theory, perturb), ledger))


def beta_Lambda(
        theory: TheoryInput,
        ledger: FormulaLedger,
        perturb: AuditPerturbation = AuditPerturbation(),
) -> mp.mpf:
    x = x_star(theory, perturb)
    lam_e = Lambda_e(theory, ledger, perturb)
    return (mp.pi / alpha(theory, perturb)) * D_lock_prime(lam_e) / D_C_prime(x, ledger)


def A_affine(delta_hat: mp.mpf, theory: TheoryInput, ledger: FormulaLedger,
             perturb: AuditPerturbation = AuditPerturbation()) -> mp.mpf:
    beta = beta_Lambda(theory, ledger, perturb)
    return A_relator(theory, perturb) * beta * delta_hat / (1 + beta * delta_hat)


def A_affine_inverse(L: mp.mpf, theory: TheoryInput, ledger: FormulaLedger,
                     perturb: AuditPerturbation = AuditPerturbation()) -> mp.mpf:
    beta = beta_Lambda(theory, ledger, perturb)
    A = A_relator(theory, perturb)
    return L / (beta * (A - L))


def raw_ALP_displacement_from_affine_load(
        delta_hat: mp.mpf,
        theory: TheoryInput,
        ledger: FormulaLedger,
        perturb: AuditPerturbation = AuditPerturbation(),
) -> mp.mpf:
    """Return q_n, the raw ALP displacement corresponding to an affine load."""
    beta = beta_Lambda(theory, ledger, perturb)
    x = x_star(theory, perturb)
    target = D_C(x * (1 + beta * delta_hat), ledger)
    return D_lock_inverse(target) - Lambda_e(theory, ledger, perturb)


# =============================================================================
# Schur determinants, return fractions, and compact final ladder
# =============================================================================


def normal_collar_tail(n: int, perturb: AuditPerturbation = AuditPerturbation()) -> mp.mpf:
    el = ell(n)
    return perturb.tail_scale * C_UV_Gauss(perturb) * mp.log(2) * el ** 4 * (1 - el ** 2) ** (-3)


def D_tau(theory: TheoryInput, ledger: FormulaLedger, perturb: AuditPerturbation = AuditPerturbation()) -> Dict[
    str, mp.mpf]:
    static_base = mp.mpf("2")
    endpoint = (
            perturb.tau_endpoint_scale
            * beta_Lambda(theory, ledger, perturb)
            * D_C(x_star(theory, perturb), ledger)
            * mp.log(mp.mpf(4) / 3)
    )
    tail = normal_collar_tail(3, perturb)
    return {"static_base": static_base, "endpoint": endpoint, "tail": tail, "D": static_base + endpoint + tail}


def heavy_moment(L_mu: mp.mpf, L_tau: mp.mpf, perturb: AuditPerturbation = AuditPerturbation()) -> mp.mpf:
    """Heavy-pole closure for the numerical finite-ledger audit."""
    return perturb.heavy_moment_scale * mp.e ** (-2 * (L_tau - L_mu))


def D_mu(L_mu: mp.mpf, L_tau: mp.mpf, perturb: AuditPerturbation = AuditPerturbation()) -> Dict[str, mp.mpf]:
    h = heavy_moment(L_mu, L_tau, perturb)
    heavy_term = -mp.mpf("0.75") * h
    static_base = mp.mpf("2")
    tail = normal_collar_tail(2, perturb)
    return {"static_base": static_base, "h_mu_tau": h, "heavy_term": heavy_term, "tail": tail,
            "D": static_base + heavy_term + tail}


def rho(n: int, D_n: mp.mpf, perturb: AuditPerturbation = AuditPerturbation()) -> mp.mpf:
    return C_UV_Gauss(perturb) * mp.log(D_n) / (mp.mpf(n) + 1)


def compact_L(T: mp.mpf, rho_value: mp.mpf, theory: TheoryInput,
              perturb: AuditPerturbation = AuditPerturbation()) -> mp.mpf:
    A = A_relator(theory, perturb)
    return (A * T) / (A * (1 - rho_value) + rho_value * T)


# =============================================================================
# Results
# =============================================================================


@dataclass(frozen=True)
class BranchResult:
    symbol: str
    name: str
    n: int
    eta: mp.mpf
    ell: mp.mpf
    P_IR: mp.mpf
    Delta_out_flat: mp.mpf
    Delta_out_exact_101: mp.mpf
    R_chi_red: mp.mpf
    Gamma_101: mp.mpf
    T_101: mp.mpf
    T_log_n: mp.mpf
    T_log_P_ratio: mp.mpf
    T_log_out_R_ratio: mp.mpf
    Delta_hat_lambda_C: mp.mpf
    D_components: Mapping[str, mp.mpf]
    D: mp.mpf
    rho: mp.mpf
    Delta_hat_dressed: mp.mpf
    alpha_R: mp.mpf
    raw_q_ALP: mp.mpf
    L: mp.mpf
    mass_ratio: mp.mpf
    mass_mev: mp.mpf


@dataclass(frozen=True)
class ModelResult:
    theory: TheoryInput
    comparison: ComparisonValues
    ledger: FormulaLedger
    perturb: AuditPerturbation
    mu: BranchResult
    tau: BranchResult
    mu_iterations: int
    mu_residual: mp.mpf


def build_branch(
        symbol: str,
        n: int,
        name: str,
        theory: TheoryInput,
        comparison: ComparisonValues,
        ledger: FormulaLedger,
        perturb: AuditPerturbation,
        D_components: Mapping[str, mp.mpf],
        L_value: mp.mpf,
) -> BranchResult:
    comps = T_components(n, ledger, perturb)
    Tn = comps["T"]
    delta_lc = A_affine_inverse(Tn, theory, ledger, perturb)
    Dn = D_components["D"]
    rhon = rho(n, Dn, perturb)
    delta_dressed = delta_lc / (1 - rhon)
    beta = beta_Lambda(theory, ledger, perturb)
    alpha_R_value = alpha(theory, perturb) * (1 + beta * delta_dressed)
    q_raw = raw_ALP_displacement_from_affine_load(delta_dressed, theory, ledger, perturb)
    mass_ratio = mp.e ** L_value
    return BranchResult(
        symbol=symbol,
        name=name,
        n=n,
        eta=eta(n),
        ell=ell(n),
        P_IR=P_IR(n, perturb),
        Delta_out_flat=Delta_Lambda_out_flat(n),
        Delta_out_exact_101=Delta_Lambda_out_exact_101(n, ledger, perturb),
        R_chi_red=R_chi_reduced(n, ledger, perturb),
        Gamma_101=Gamma_101(n, ledger, perturb),
        T_101=Tn,
        T_log_n=comps["log_n"],
        T_log_P_ratio=comps["log_P_ratio"],
        T_log_out_R_ratio=comps["log_out_R_ratio"],
        Delta_hat_lambda_C=delta_lc,
        D_components=D_components,
        D=Dn,
        rho=rhon,
        Delta_hat_dressed=delta_dressed,
        alpha_R=alpha_R_value,
        raw_q_ALP=q_raw,
        L=L_value,
        mass_ratio=mass_ratio,
        mass_mev=comparison.electron_mass_mev * perturb.electron_mass_scale * mass_ratio,
    )


def solve_model(
        theory: TheoryInput = TheoryInput(),
        comparison: ComparisonValues = ComparisonValues(),
        ledger: FormulaLedger = FormulaLedger(),
        perturb: AuditPerturbation = AuditPerturbation(),
) -> ModelResult:
    # Tau is explicit once D_tau and T_tau are known.
    Dtau = D_tau(theory, ledger, perturb)
    rhotau = rho(3, Dtau["D"], perturb)
    Ttau = T_101(3, ledger, perturb)
    Ltau = compact_L(Ttau, rhotau, theory, perturb)

    # Muon is implicit because D_mu contains h_{μτ}=exp[-2(L_τ−L_μ)].
    Tmu = T_101(2, ledger, perturb)

    def residual(Lmu: mp.mpf) -> mp.mpf:
        Dmu = D_mu(Lmu, Ltau, perturb)
        rhomu = rho(2, Dmu["D"], perturb)
        return compact_L(Tmu, rhomu, theory, perturb) - Lmu

    lower = mp.mpf("0")
    upper = max(mp.mpf("1"), Ltau)
    f_lower = residual(lower)
    f_upper = residual(upper)
    expand_count = 0
    while f_lower * f_upper > 0 and expand_count < 100:
        upper *= 2
        f_upper = residual(upper)
        expand_count += 1
    if f_lower * f_upper > 0:
        raise RuntimeError("Could not bracket the μ fixed point.")

    iterations = 0
    for iterations in range(1, 400):
        mid = (lower + upper) / 2
        f_mid = residual(mid)
        if abs(f_mid) < mp.mpf("1e-50") or abs(upper - lower) < mp.mpf("1e-50"):
            Lmu = mid
            break
        if f_lower * f_mid <= 0:
            upper = mid
            f_upper = f_mid
        else:
            lower = mid
            f_lower = f_mid
    else:
        Lmu = (lower + upper) / 2

    Dmu = D_mu(Lmu, Ltau, perturb)
    mu = build_branch("μ", 2, "Muon", theory, comparison, ledger, perturb, Dmu, Lmu)
    tau = build_branch("τ", 3, "Tau", theory, comparison, ledger, perturb, Dtau, Ltau)
    return ModelResult(
        theory=theory,
        comparison=comparison,
        ledger=ledger,
        perturb=perturb,
        mu=mu,
        tau=tau,
        mu_iterations=iterations,
        mu_residual=residual(Lmu),
    )


# =============================================================================
# Sensitivity analysis
# =============================================================================


@dataclass(frozen=True)
class SensitivityItem:
    field_name: str
    display_name: str
    description: str
    branch_scope: str = "both"  # "μ", "τ", or "both"


SENSITIVITY_ITEMS: Tuple[SensitivityItem, ...] = (
    SensitivityItem("alpha_inv_scale", "α⁻¹", "only theory input"),
    SensitivityItem("p_ir_1_scale", "P_IR(ℓ₁)", "electron-shell IR gate"),
    SensitivityItem("p_ir_2_scale", "P_IR(ℓ₂)", "muon-shell IR gate", "μ"),
    SensitivityItem("p_ir_3_scale", "P_IR(ℓ₃)", "tau-shell IR gate", "τ"),
    SensitivityItem("out_1_scale", "ΔΛ̂_out⁽¹,[101]⁾", "electron-shell exact OUT source norm"),
    SensitivityItem("out_2_scale", "ΔΛ̂_out⁽²,[101]⁾", "muon-shell exact OUT source norm", "μ"),
    SensitivityItem("out_3_scale", "ΔΛ̂_out⁽³,[101]⁾", "tau-shell exact OUT source norm", "τ"),
    SensitivityItem("r_chi_1_scale", "R_χ,1^red", "electron-shell memory return"),
    SensitivityItem("r_chi_2_scale", "R_χ,2^red", "muon-shell memory return", "μ"),
    SensitivityItem("r_chi_3_scale", "R_χ,3^red", "tau-shell memory return", "τ"),
    SensitivityItem("c_uv_scale", "C_UV^Gauss", "Gaussian UV finite part"),
    SensitivityItem("tail_scale", "normal-collar tail", "Hilbert-series determinant tail"),
    SensitivityItem("heavy_moment_scale", "h_{μτ}", "muon heavy Stieltjes moment", "μ"),
    SensitivityItem("tau_endpoint_scale", "τ endpoint", "β_ΛD_C ln(4/3) determinant endpoint", "τ"),
    SensitivityItem("electron_mass_scale", "mₑ scale", "MeV conversion only"),
)


def perturb_scale(perturb: AuditPerturbation, field_name: str, factor: mp.mpf) -> AuditPerturbation:
    return replace(perturb, **{field_name: getattr(perturb, field_name) * factor})


def branch_mass(result: ModelResult, symbol: str) -> mp.mpf:
    return result.mu.mass_mev if symbol == "μ" else result.tau.mass_mev


def compute_sensitivity_rows(
        theory: TheoryInput,
        comparison: ComparisonValues,
        ledger: FormulaLedger,
        controls: OutputControls,
) -> Dict[str, List[Dict[str, object]]]:
    rows_by_branch: Dict[str, List[Dict[str, object]]] = {"μ": [], "τ": []}
    h = controls.sensitivity_relative_step
    baseline = solve_model(theory, comparison, ledger, AuditPerturbation())

    for branch in ("μ", "τ"):
        raw_rows: List[Dict[str, object]] = []
        for item in SENSITIVITY_ITEMS:
            if item.branch_scope not in ("both", branch):
                continue
            plus = solve_model(theory, comparison, ledger, perturb_scale(AuditPerturbation(), item.field_name, 1 + h))
            minus = solve_model(theory, comparison, ledger, perturb_scale(AuditPerturbation(), item.field_name, 1 - h))
            y_plus = branch_mass(plus, branch)
            y_minus = branch_mass(minus, branch)
            elasticity = (mp.log(y_plus) - mp.log(y_minus)) / (mp.log(1 + h) - mp.log(1 - h))

            plus_1pct = solve_model(theory, comparison, ledger,
                                    perturb_scale(AuditPerturbation(), item.field_name, mp.mpf("1.01")))
            pct_change = branch_mass(plus_1pct, branch) / branch_mass(baseline, branch) - 1
            raw_rows.append(
                {
                    "parameter": item.display_name,
                    "description": item.description,
                    "elasticity": elasticity,
                    "pct_change_plus_1pct": pct_change,
                }
            )
        total = mp.fsum(abs(mp.mpf(row["elasticity"])) for row in raw_rows)
        for row in raw_rows:
            row["influence_share"] = abs(mp.mpf(row["elasticity"])) / total if total else mp.mpf("0")
        rows_by_branch[branch] = raw_rows
    return rows_by_branch


# =============================================================================
# Rich report tables
# =============================================================================


def formulas_panel() -> Panel:
    lines = [
        "Only theory input: α⁻¹",
        "A = π/(8α)",
        "ηₙ = 1/(nπ),   ℓₙ = 1/(nπ√π)",
        "ΔΛ̂_out⁽ⁿ,[101]⁾ = −½Σ_{j=1,3,…,101} J_j(ηₙ)²",
        "Γₙ^[101] = n · P_IR(ℓ₁)/P_IR(ℓₙ) · |ΔΛ̂_out⁽¹,[101]⁾R_χ,1^red| / |ΔΛ̂_out⁽ⁿ,[101]⁾R_χ,n^red|",
        "Tₙ^[101] = ln Γₙ^[101]",
        "D_μ = 2 − ¾h_{μτ} + C_UV^Gauss ln2 · ℓ₂⁴(1−ℓ₂²)⁻³",
        "D_τ = 2 + β_ΛD_C(α/π)ln(4/3) + C_UV^Gauss ln2 · ℓ₃⁴(1−ℓ₃²)⁻³",
        "ρₙ = C_UV^Gauss ln(Dₙ)/(n+1)",
        "Lₙ = A Tₙ / [A(1−ρₙ) + ρₙTₙ]",
        "mₙ/mₑ = exp(Lₙ)",
    ]
    return Panel("\n".join(lines), title="Final compact formula map", border_style="green")


def constants_table(result: ModelResult) -> Table:
    theory, comp, ledger = result.theory, result.comparison, result.ledger
    table = make_table(
        "Inputs, comparison values, and fixed ledger status",
        [("Category", "left"), ("Symbol", "left"), ("Value", "right"), ("Role", "left")],
    )
    table.add_row("Theory input", "α⁻¹", fmt(theory.alpha_inv, 18), "only supplied theory input")
    table.add_row("Computed", "α", fmt(alpha(theory), 18), "1/α⁻¹")
    table.add_row("Computed", "A", fmt(A_relator(theory), 18), "π/(8α)")
    table.add_row("Computed", "D_C(α/π)", fmt(D_C(x_star(theory), ledger), 18), "Alpha scalar branch evaluator")
    table.add_row("Computed", "K_ov", fmt(K_ov(), 18), "closed overlap series")
    table.add_row("Computed", "Λ_e", fmt(Lambda_e(theory, ledger), 18), "electron Alpha anchor")
    table.add_row("Computed", "β_Λ", fmt(beta_Lambda(theory, ledger), 18), "ALP affine response")
    table.add_row("Fixed ledger", "[101]", str(ledger.shell_source_jmax), "finite odd-mode source-vector ledger")
    table.add_row("Fixed ledger", "GL nodes", str(ledger.shell_source_gl_nodes), "quadrature for J_χ source modes")
    table.add_row("Comparison only", "mₑ ± σₑ",
                  f"{fmt(comp.electron_mass_mev, 16)} ± {fmt(comp.electron_sigma_mev, 8)} MeV",
                  "MeV conversion and comparison only")
    table.add_row("Comparison only", "m_μ ± σ_μ",
                  f"{fmt(comp.muon_mass_exp_mev, 16)} ± {fmt(comp.muon_sigma_exp_mev, 8)} MeV", "not used upstream")
    table.add_row("Comparison only", "m_τ ± σ_τ",
                  f"{fmt(comp.tau_mass_exp_mev, 16)} ± {fmt(comp.tau_sigma_exp_mev, 8)} MeV", "not used upstream")
    return table


def universal_table() -> Table:
    table = make_table("Universal derived quantities", [("Quantity", "left"), ("Formula", "left"), ("Value", "right")])
    table.add_row("θ_R", "π/8", fmt(theta_R(), 18))
    table.add_row("C_UV^Gauss", "½(ln2 + γ_E)", fmt(C_UV_Gauss(), 18))
    for n in (1, 2, 3):
        table.add_row(f"η_{n}", f"1/({n}π)", fmt(eta(n), 18))
        table.add_row(f"ℓ_{n}", f"1/({n}π√π)", fmt(ell(n), 18))
        table.add_row(f"P_IR(ℓ_{n})", "normalized TT-χ gate integral", fmt(P_IR(n), 18))
        table.add_row(f"ΔΛ_out^flat({n})", "closed flat approximation (diagnostic only)",
                      fmt(Delta_Lambda_out_flat(n), 18))
        table.add_row(f"ΔΛ̂_out^[101]({n})", "−½||J_χ||²_[101]", fmt(Delta_Lambda_out_exact_101(n), 18))
        table.add_row(f"R_χ,{n}^red", "source-weighted shell return", fmt(R_chi_reduced(n), 18))
    return table


def shell_source_table(result: ModelResult) -> Table:
    table = make_table(
        "Exact [101] shell-admission source",
        [
            ("Branch", "left"),
            ("Formula", "left"),
            ("Γₙ^[101]", "right"),
            ("Tₙ^[101]", "right"),
            ("ΔΛ̂_{λC,n}", "right"),
            ("log n", "right"),
            ("log P ratio", "right"),
            ("log OUT×Rχ ratio", "right"),
        ],
    )
    for b in (result.mu, result.tau):
        table.add_row(
            b.symbol,
            "Γ=n·P₁/Pₙ·|OUT₁R₁|/|OUTₙRₙ|",
            fmt(b.Gamma_101, 16),
            fmt(b.T_101, 16),
            fmt(b.Delta_hat_lambda_C, 16),
            fmt(b.T_log_n, 16),
            fmt(b.T_log_P_ratio, 16),
            fmt(b.T_log_out_R_ratio, 16),
        )
    return table


def determinant_table(result: ModelResult) -> Table:
    table = make_table(
        "Schur determinant components",
        [("Branch", "left"), ("Formula", "left"), ("Static", "right"), ("Heavy / endpoint", "right"), ("Tail", "right"),
         ("Dₙ", "right"), ("ρₙ", "right")],
    )
    mu, tau = result.mu, result.tau
    table.add_row(
        "μ",
        "D=2−¾h+C_UV ln2·ℓ₂⁴(1−ℓ₂²)⁻³",
        fmt(mu.D_components["static_base"], 16),
        f"{fmt(mu.D_components['heavy_term'], 16)}\nh={fmt(mu.D_components['h_mu_tau'], 16)}",
        fmt(mu.D_components["tail"], 16),
        fmt(mu.D, 16),
        fmt(mu.rho, 16),
    )
    table.add_row(
        "τ",
        "D=2+β_ΛD_C ln(4/3)+C_UV ln2·ℓ₃⁴(1−ℓ₃²)⁻³",
        fmt(tau.D_components["static_base"], 16),
        fmt(tau.D_components["endpoint"], 16),
        fmt(tau.D_components["tail"], 16),
        fmt(tau.D, 16),
        fmt(tau.rho, 16),
    )
    return table


def determinant_share_table(result: ModelResult) -> Table:
    table = make_table(
        "Signed and absolute determinant-term shares",
        [("Branch", "left"), ("Term", "left"), ("Value", "right"), ("Signed share in Dₙ", "right"),
         ("Absolute share", "right")],
    )
    for b in (result.mu, result.tau):
        terms: List[Tuple[str, mp.mpf]] = [("Static base", b.D_components["static_base"])]
        if b.symbol == "μ":
            terms.append(("Heavy corridor −¾h_{μτ}", b.D_components["heavy_term"]))
        else:
            terms.append(("Tau endpoint β_ΛD_C ln(4/3)", b.D_components["endpoint"]))
        terms.append(("Normal-collar tail", b.D_components["tail"]))
        total_abs = mp.fsum(abs(v) for _, v in terms)
        for name, value in terms:
            table.add_row(b.symbol, name, fmt(value, 16), percent(value / b.D, 8), percent(abs(value) / total_abs, 8))
    return table


def load_mass_table(result: ModelResult) -> Table:
    table = make_table(
        "Load chain and final compact mass readout",
        [
            ("Branch", "left"),
            ("ΔΛ̂_{λC,n}=T/[β(A−T)]", "right"),
            ("ΔΛ̂ₙ=ΔΛ̂_{λC}/(1−ρ)", "right"),
            ("α_{R,n}", "right"),
            ("raw qₙ", "right"),
            ("Lₙ", "right"),
            ("mₙ/mₑ", "right"),
            ("mₙ [MeV]", "right"),
        ],
    )
    for b in (result.mu, result.tau):
        table.add_row(
            b.symbol,
            fmt(b.Delta_hat_lambda_C, 16),
            fmt(b.Delta_hat_dressed, 16),
            fmt(b.alpha_R, 16),
            fmt(b.raw_q_ALP, 16),
            fmt(b.L, 16),
            fmt(b.mass_ratio, 16),
            fmt(b.mass_mev, 16),
        )
    return table


def compact_formula_table(result: ModelResult) -> Table:
    table = make_table(
        "Direct compact formula check",
        [("Branch", "left"), ("A", "right"), ("Tₙ", "right"), ("ρₙ", "right"), ("A(1−ρ)+ρT", "right"), ("A·T", "right"),
         ("Lₙ", "right")],
    )
    A = A_relator(result.theory)
    for b in (result.mu, result.tau):
        denom = A * (1 - b.rho) + b.rho * b.T_101
        num = A * b.T_101
        table.add_row(b.symbol, fmt(A, 16), fmt(b.T_101, 16), fmt(b.rho, 16), fmt(denom, 16), fmt(num, 16),
                      fmt(num / denom, 16))
    return table


def comparison_table(result: ModelResult) -> Table:
    comp = result.comparison
    rows = [("μ", result.mu, comp.muon_mass_exp_mev, comp.muon_sigma_exp_mev),
            ("τ", result.tau, comp.tau_mass_exp_mev, comp.tau_sigma_exp_mev)]
    table = make_table(
        "External comparison audit (not used upstream)",
        [("Branch", "left"), ("Prediction [MeV]", "right"), ("Experiment [MeV]", "right"), ("σ_exp [MeV]", "right"),
         ("Abs. error [MeV]", "right"), ("Rel. error", "right"), ("ppm", "right"), ("Pull", "right")],
    )
    for symbol, b, exp_mass, sigma in rows:
        error = b.mass_mev - exp_mass
        rel = error / exp_mass
        pull = error / sigma
        table.add_row(symbol, fmt(b.mass_mev, 18), fmt(exp_mass, 18), fmt(sigma, 10), signed_fmt(error, 16),
                      percent(rel, 10), ppm(rel, 8), signed_fmt(pull, 12))
    return table


def schur_effect_table(result: ModelResult) -> Table:
    table = make_table(
        "Bare versus Schur-dressed logarithmic effect",
        [("Branch", "left"), ("Tₙ^[101] bare", "right"), ("Lₙ final", "right"), ("ΔL_Schur", "right"),
         ("Schur share", "right"), ("exp(ΔL_Schur)", "right")],
    )
    for b in (result.mu, result.tau):
        delta_L = b.L - b.T_101
        table.add_row(b.symbol, fmt(b.T_101, 16), fmt(b.L, 16), fmt(delta_L, 16), percent(delta_L / b.L, 8),
                      fmt(mp.e ** delta_L, 16))
    return table


def sensitivity_table(title: str, rows: List[Dict[str, object]]) -> Table:
    table = make_table(
        title,
        [("Parameter", "left"), ("Description", "left"), ("Elasticity ∂lnm/∂lnx", "right"), ("Δm/m for +1%", "right"),
         ("Influence share", "right")],
    )
    for row in rows:
        table.add_row(
            str(row["parameter"]),
            str(row["description"]),
            signed_fmt(row["elasticity"], 10),
            percent(row["pct_change_plus_1pct"], 8),
            percent(row["influence_share"], 8),
        )
    return table


# =============================================================================
# Report renderer and package builder
# =============================================================================


def render_report(
        theory: TheoryInput = TheoryInput(),
        comparison: ComparisonValues = ComparisonValues(),
        ledger: FormulaLedger = FormulaLedger(),
        controls: OutputControls = OutputControls(),
        perturb: AuditPerturbation = AuditPerturbation(),
        output_path: Optional[Path] = None,
) -> ModelResult:
    result = solve_model(theory, comparison, ledger, perturb)
    console = Console(record=True, width=180)

    console.print(Panel("Charged-lepton hierarchy — final no-target [101]-ledger evaluator", border_style="green",
                        style="bold white"))
    console.print(formulas_panel())
    console.print(constants_table(result))
    console.print(universal_table())
    console.print(shell_source_table(result))
    console.print(determinant_table(result))
    console.print(determinant_share_table(result))
    console.print(load_mass_table(result))
    console.print(compact_formula_table(result))
    console.print(schur_effect_table(result))
    console.print(comparison_table(result))

    solver = make_table("Muon fixed-point solver", [("Quantity", "left"), ("Value", "right")])
    solver.add_row("iterations", str(result.mu_iterations))
    solver.add_row("residual", fmt(result.mu_residual, 18))
    solver.add_row("equation", "L_μ = A T_μ/[A(1−ρ_μ(L_μ,L_τ))+ρ_μ(L_μ,L_τ)T_μ]")
    console.print(solver)

    if controls.include_sensitivity:
        sens = compute_sensitivity_rows(theory, comparison, ledger, controls)
        console.print(sensitivity_table("Local sensitivity of final μ mass", sens["μ"]))
        console.print(sensitivity_table("Local sensitivity of final τ mass", sens["τ"]))

    notes = "\n".join(
        [
            "Interpretation notes:",
            "• The only supplied theory input is α⁻¹.",
            "• ΔΛ̂_{λC,μ} and ΔΛ̂_{λC,τ} are computed from Tₙ^[101] via T/[β_Λ(A−T)]; they are not inputs.",
            "• ΔΛ̂_out^[101] is computed as −½||J_χ||² from odd source modes j=1,3,…,101.",
            "• The flat OUT expression is printed only as a diagnostic and is not used in Γₙ^[101].",
            "• Experimental masses and uncertainties are used only in the external comparison table.",
            "• Sensitivity scale factors are diagnostic perturbations around the baseline formula, not tunable baseline parameters.",
            "• α_{R,n} is the Relator-exponent Alpha coordinate; it is not α_QED(q²).",
        ]
    )
    console.print(Panel(notes, title="Guardrails", border_style="yellow"))

    if output_path is None:
        output_path = Path(__file__).with_name(controls.output_text_file)
    output_path.write_text(console.export_text(styles=False), encoding="utf-8")
    console.print(f"\nSaved full text report to: [bold]{output_path.resolve()}[/bold]")
    return result


def build_zip(script_path: Path, report_path: Path, zip_path: Path) -> None:
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        zf.write(script_path, arcname=script_path.name)
        zf.write(report_path, arcname=report_path.name)


if __name__ == "__main__":
    render_report()
