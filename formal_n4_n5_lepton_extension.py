#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full minimal five-shell closure for the charged-lepton hierarchy ledger.

Runtime inputs
--------------
Only two runtime inputs are exposed:

    --alpha-inv             inverse fine-structure constant
    --electron-mass-mev     electron mass-energy used only after dimensionless
                            ratios have been computed

Everything else is a fixed ledger definition in the calculation:
D_C polynomial, K_ALP, source-vector projection, shell-return rule, determinant
continuation, and numerical convergence settings.

Correct five-shell closure used here
------------------------------------
For an N=5 closure, every branch n has all lower corridors encoded in

    P_C(n) = (n-1) ln 2 - ln n,

which gives P_C(2)=0, P_C(3)=ln(4/3), P_C(4)=ln 2, and
P_C(5)=ln(16/5).  Every non-terminal branch n<N also has one upper heavy
corridor to the next branch n+1.  Thus

    D_n = 2 + beta_Lambda D_C(x_*) P_C(n)
              - 3/4 exp[-2(L_{n+1}-L_n)]
              + C_UV ln 2 ell_n^4 (1-ell_n^2)^(-3),     n=2,3,4,

and the terminal branch has no upper heavy subtraction:

    D_5 = 2 + beta_Lambda D_C(x_*) P_C(5)
              + C_UV ln 2 ell_5^4 (1-ell_5^2)^(-3).

The compact readout remains

    L_n = A T_n / [A(1-rho_n) + rho_n T_n],
    rho_n = C_UV ln(D_n)/(n+1),
    A = pi/(8 alpha).

The equations are solved backward from the terminal branch n=5.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import argparse
import math
import mpmath as mp
import numpy as np
from scipy.special import ellipe, ellipk, eval_legendre, roots_legendre

mp.mp.dps = 80

# Fixed internal numerical resolution. These are not runtime physical inputs.
JMAX = 401
GL_NODES = 1024
JMAX_CHECK = 201
GL_NODES_CHECK = 768

# Fixed Alpha / Relator ledger constants.
THETA_R = mp.pi / 8
CUV = mp.mpf("0.5") * (mp.log(2) + mp.euler)
DC_COEFFS = (
    mp.mpf("0"),
    mp.mpf("1"),
    mp.mpf("-1.37318423257715981904"),
    mp.mpf("3.80011555037140726825"),
    mp.mpf("-8.85785467402324147604"),
    mp.mpf("24.73523387578082466951"),
)


@dataclass(frozen=True)
class Inputs:
    alpha_inv: mp.mpf
    electron_mass_mev: mp.mpf

    @property
    def alpha(self) -> mp.mpf:
        return 1 / self.alpha_inv

    @property
    def xstar(self) -> mp.mpf:
        return self.alpha / mp.pi

    @property
    def A(self) -> mp.mpf:
        return mp.pi / (8 * self.alpha)


@dataclass(frozen=True)
class ShellLedger:
    n: int
    eta: mp.mpf
    ell: mp.mpf
    P_IR: mp.mpf
    out_hat: mp.mpf
    R_chi_red: mp.mpf
    Gamma: mp.mpf
    T: mp.mpf
    Dhat_LC: mp.mpf


@dataclass(frozen=True)
class BranchResult:
    n: int
    P_C: mp.mpf
    endpoint: mp.mpf
    heavy: mp.mpf
    tail: mp.mpf
    D: mp.mpf
    rho: mp.mpf
    L: mp.mpf
    ratio: mp.mpf
    mass_mev: mp.mpf
    mass_gev: mp.mpf
    Dhat_LC: mp.mpf
    Dhat: mp.mpf
    alpha_R: mp.mpf
    q_raw: mp.mpf
    residual: mp.mpf
    iterations: int


def fmt(x: mp.mpf, digits: int = 17) -> str:
    return mp.nstr(mp.mpf(x), digits, strip_zeros=False)


def D_C(x: mp.mpf) -> mp.mpf:
    return mp.fsum(c * x**k for k, c in enumerate(DC_COEFFS))


def D_C_prime(x: mp.mpf) -> mp.mpf:
    return mp.fsum(k * c * x ** (k - 1) for k, c in enumerate(DC_COEFFS) if k > 0)


def K_ALP() -> mp.mpf:
    return (150 * mp.pi**2 - 8 * mp.pi**4 - 315) / (180 * mp.pi**6)


def D_lock_inverse(D: mp.mpf) -> mp.mpf:
    K = K_ALP()
    return (mp.pi**2 / K) * (mp.sqrt(1 + 4 * D / (3 * mp.pi**2)) - 1)


def D_lock_prime(Lambda: mp.mpf) -> mp.mpf:
    K = K_ALP()
    return (3 * K / 2) * (1 + K * Lambda / mp.pi**2)


def Lambda_e(inp: Inputs) -> mp.mpf:
    return D_lock_inverse(D_C(inp.xstar))


def beta_Lambda(inp: Inputs) -> mp.mpf:
    return (mp.pi / inp.alpha) * D_lock_prime(Lambda_e(inp)) / D_C_prime(inp.xstar)


def eta(n: int) -> mp.mpf:
    return 1 / (mp.mpf(n) * mp.pi)


def ell(n: int) -> mp.mpf:
    return 1 / (mp.mpf(n) * mp.pi * mp.sqrt(mp.pi))


def I0_gate() -> mp.mpf:
    return mp.mpf(1) / 6 - 1 / (4 * mp.pi**2)


@lru_cache(maxsize=None)
def P_IR(n: int) -> mp.mpf:
    en = ell(n)

    def integrand(y: mp.mpf) -> mp.mpf:
        a = 1 - y
        bracket = 1 - mp.mpf(1) / 3 * a**2 / (a**2 + en**2)
        return y**2 * mp.sin(mp.pi * y) ** 2 * bracket * mp.e ** (-(a / en) ** 2)

    split = [
        mp.mpf("0"), mp.mpf("0.25"), mp.mpf("0.5"), mp.mpf("0.75"),
        mp.mpf("0.9"), mp.mpf("0.97"), mp.mpf("0.99"), mp.mpf("0.997"), mp.mpf("1"),
    ]
    return mp.quad(integrand, split) / I0_gate()


@lru_cache(maxsize=None)
def J_modes(n: int, jmax: int, nodes: int) -> Tuple[Tuple[int, mp.mpf], ...]:
    """Odd source-vector modes J_{chi,j}(eta_n)."""
    en = float(eta(n))
    u, w = roots_legendre(nodes)

    rho = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    z = u
    delta_p = (en + rho) ** 2 + z * z
    delta_m = (en - rho) ** 2 + z * z
    k2 = 4.0 * en * rho / delta_p

    K = ellipk(k2)
    E = ellipe(k2)
    B_rho = z / (2.0 * math.pi * rho * np.sqrt(delta_p)) * (
        -K + (en * en + rho * rho + z * z) / delta_m * E
    )
    B_z = 1.0 / (2.0 * math.pi * np.sqrt(delta_p)) * (
        K + (en * en - rho * rho - z * z) / delta_m * E
    )
    b_tilde = z * B_rho - rho * B_z

    modes: List[Tuple[int, mp.mpf]] = []
    for j in range(1, jmax + 1, 2):
        Pj = eval_legendre(j, u)
        Pjm1 = eval_legendre(j - 1, u)
        Tj = j * (Pjm1 - u * Pj)
        Ij = 2.0 * j * (j + 1) / (2.0 * j + 1.0)
        a_sh = float(np.sum(w * b_tilde * Tj) / Ij)
        a_hat = 2.0 * (j + 1) * en ** (-0.5) * a_sh
        Jj = math.sqrt(2.0 * math.pi / ((j + 1) * (2 * j + 1))) * a_hat
        modes.append((j, mp.mpf(Jj)))
    return tuple(modes)


def out_hat(n: int, jmax: int = JMAX, nodes: int = GL_NODES) -> mp.mpf:
    return -mp.mpf("0.5") * mp.fsum(Jj**2 for _, Jj in J_modes(n, jmax, nodes))


def R_chi_red(n: int, jmax: int = JMAX, nodes: int = GL_NODES) -> mp.mpf:
    en = eta(n)
    P = P_IR(n)
    U_chi = CUV * P
    gamma_geom = mp.mpf("0.5") * mp.sinh(en) / en
    A_ret = mp.log(2) * en**4
    B_ret = en**4 / (8 * mp.pi**2)
    beta_chi = P * gamma_geom + P * U_chi / (2 * (1 + A_ret))

    modes = J_modes(n, jmax, nodes)
    norm = mp.fsum(Jj**2 for _, Jj in modes)
    F = mp.mpf("0")
    for j, Jj in modes:
        w = Jj**2 / norm
        level = mp.mpf(j - 1) / 2
        Phi = (1 + mp.mpf("0.5") * B_ret * level) / (1 + A_ret * level)
        F += w * Phi
    return 1 + beta_chi * F


def Gamma_n(n: int, jmax: int = JMAX, nodes: int = GL_NODES) -> mp.mpf:
    if n == 1:
        return mp.mpf("1")
    num = abs(out_hat(1, jmax, nodes) * R_chi_red(1, jmax, nodes))
    den = abs(out_hat(n, jmax, nodes) * R_chi_red(n, jmax, nodes))
    return mp.mpf(n) * P_IR(1) / P_IR(n) * num / den


def affine_inverse(L: mp.mpf, inp: Inputs) -> mp.mpf:
    return L / (beta_Lambda(inp) * (inp.A - L))


def raw_ALP_from_dhat(dhat: mp.mpf, inp: Inputs) -> mp.mpf:
    beta = beta_Lambda(inp)
    return D_lock_inverse(D_C(inp.xstar * (1 + beta * dhat))) - Lambda_e(inp)


def build_shell(n: int, inp: Inputs, jmax: int = JMAX, nodes: int = GL_NODES) -> ShellLedger:
    G = Gamma_n(n, jmax, nodes)
    T = mp.log(G)
    return ShellLedger(
        n=n,
        eta=eta(n),
        ell=ell(n),
        P_IR=P_IR(n),
        out_hat=out_hat(n, jmax, nodes),
        R_chi_red=R_chi_red(n, jmax, nodes),
        Gamma=G,
        T=T,
        Dhat_LC=affine_inverse(T, inp),
    )


def P_C(n: int) -> mp.mpf:
    return (mp.mpf(n) - 1) * mp.log(2) - mp.log(n)


def endpoint(n: int, inp: Inputs) -> mp.mpf:
    return beta_Lambda(inp) * D_C(inp.xstar) * P_C(n)


def tail(n: int) -> mp.mpf:
    en = ell(n)
    return CUV * mp.log(2) * en**4 * (1 - en**2) ** (-3)


def rho_from_D(n: int, D: mp.mpf) -> mp.mpf:
    return CUV * mp.log(D) / (mp.mpf(n) + 1)


def compact_L(T: mp.mpf, rho: mp.mpf, inp: Inputs) -> mp.mpf:
    return inp.A * T / (inp.A * (1 - rho) + rho * T)


def determinant(n: int, L_n: mp.mpf | None, L_next: mp.mpf | None, inp: Inputs) -> Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    ep = endpoint(n, inp)
    ta = tail(n)
    hv = mp.mpf("0")
    if L_next is not None:
        if L_n is None:
            raise ValueError("L_n is required for a non-terminal heavy corridor")
        hv = -mp.mpf("0.75") * mp.e ** (-2 * (L_next - L_n))
    D = mp.mpf("2") + ep + hv + ta
    return D, ep, hv, ta


def solve_branch(n: int, L_next: mp.mpf, Tn: mp.mpf, inp: Inputs) -> Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf, mp.mpf, int, mp.mpf]:
    def residual(L: mp.mpf) -> mp.mpf:
        D, _, _, _ = determinant(n, L, L_next, inp)
        r = rho_from_D(n, D)
        return compact_L(Tn, r, inp) - L

    lower = mp.mpf("0")
    upper = L_next - mp.mpf("1e-30")
    f_lower = residual(lower)
    f_upper = residual(upper)
    if f_lower * f_upper > 0:
        upper = inp.A - mp.mpf("1e-20")
        f_upper = residual(upper)
    if f_lower * f_upper > 0:
        raise RuntimeError(f"Could not bracket fixed point for n={n}")

    for i in range(1, 700):
        mid = (lower + upper) / 2
        f_mid = residual(mid)
        if abs(f_mid) < mp.mpf("1e-50"):
            L = mid
            break
        if f_lower * f_mid <= 0:
            upper = mid
            f_upper = f_mid
        else:
            lower = mid
            f_lower = f_mid
    else:
        L = (lower + upper) / 2
        i = 700
        f_mid = residual(L)

    D, ep, hv, ta = determinant(n, L, L_next, inp)
    return L, D, ep, hv, ta, i, residual(L)


def solve_full_five_closure(inp: Inputs) -> Tuple[Dict[int, ShellLedger], Dict[int, BranchResult]]:
    shells = {n: build_shell(n, inp) for n in range(2, 6)}
    branches: Dict[int, BranchResult] = {}
    L: Dict[int, mp.mpf] = {}

    # Terminal branch n=5.
    D5, ep5, hv5, ta5 = determinant(5, None, None, inp)
    rho5 = rho_from_D(5, D5)
    L5 = compact_L(shells[5].T, rho5, inp)
    L[5] = L5

    for n in [5]:
        dhat_lc = shells[n].Dhat_LC
        dhat = dhat_lc / (1 - rho5)
        alpha_R = inp.alpha * (1 + beta_Lambda(inp) * dhat)
        q_raw = raw_ALP_from_dhat(dhat, inp)
        ratio = mp.e**L5
        branches[n] = BranchResult(
            n=n, P_C=P_C(n), endpoint=ep5, heavy=hv5, tail=ta5, D=D5, rho=rho5,
            L=L5, ratio=ratio, mass_mev=ratio * inp.electron_mass_mev,
            mass_gev=ratio * inp.electron_mass_mev / 1000,
            Dhat_LC=dhat_lc, Dhat=dhat, alpha_R=alpha_R, q_raw=q_raw,
            residual=mp.mpf("0"), iterations=0,
        )

    # Non-terminal branches solved backwards.
    for n in [4, 3, 2]:
        Ln, Dn, ep, hv, ta, it, res = solve_branch(n, L[n + 1], shells[n].T, inp)
        L[n] = Ln
        rn = rho_from_D(n, Dn)
        dhat_lc = shells[n].Dhat_LC
        dhat = dhat_lc / (1 - rn)
        alpha_R = inp.alpha * (1 + beta_Lambda(inp) * dhat)
        q_raw = raw_ALP_from_dhat(dhat, inp)
        ratio = mp.e**Ln
        branches[n] = BranchResult(
            n=n, P_C=P_C(n), endpoint=ep, heavy=hv, tail=ta, D=Dn, rho=rn,
            L=Ln, ratio=ratio, mass_mev=ratio * inp.electron_mass_mev,
            mass_gev=ratio * inp.electron_mass_mev / 1000,
            Dhat_LC=dhat_lc, Dhat=dhat, alpha_R=alpha_R, q_raw=q_raw,
            residual=res, iterations=it,
        )

    return shells, branches


def print_report(inp: Inputs) -> None:
    shells, branches = solve_full_five_closure(inp)

    print("\nFull minimal five-shell closure")
    print("=" * 88)
    print(f"alpha_inv                    = {fmt(inp.alpha_inv, 22)}")
    print(f"electron_mass_mev             = {fmt(inp.electron_mass_mev, 22)}")
    print(f"alpha                         = {fmt(inp.alpha, 22)}")
    print(f"A = pi/(8 alpha)              = {fmt(inp.A, 22)}")
    print(f"D_C(alpha/pi)                 = {fmt(D_C(inp.xstar), 22)}")
    print(f"K_ALP                         = {fmt(K_ALP(), 22)}")
    print(f"Lambda_e                      = {fmt(Lambda_e(inp), 22)}")
    print(f"beta_Lambda                   = {fmt(beta_Lambda(inp), 22)}")
    print(f"internal evaluator             JMAX={JMAX}, GL_NODES={GL_NODES}")

    print("\nShell ledger")
    print("-" * 88)
    print(" n  eta                  ell                  P_IR                 OUT_hat              R_chi_red            Gamma               T")
    for n in range(2, 6):
        s = shells[n]
        print(f"{n:2d}  {fmt(s.eta, 16):>18s}  {fmt(s.ell, 16):>18s}  {fmt(s.P_IR, 16):>18s}  {fmt(s.out_hat, 16):>18s}  {fmt(s.R_chi_red, 16):>18s}  {fmt(s.Gamma, 16):>18s}  {fmt(s.T, 16):>18s}")

    print("\nFull lower-corridor determinant ledger")
    print("-" * 88)
    print(" n  P_C(n)               endpoint             heavy term           tail                 D_n                 rho_n")
    for n in range(2, 6):
        b = branches[n]
        print(f"{n:2d}  {fmt(b.P_C, 16):>18s}  {fmt(b.endpoint, 16):>18s}  {fmt(b.heavy, 16):>18s}  {fmt(b.tail, 16):>18s}  {fmt(b.D, 16):>18s}  {fmt(b.rho, 16):>18s}")

    print("\nLoad chain and masses")
    print("-" * 88)
    print(" n  Dhat_LC              Dhat                 alpha_R              q_raw                L_n                 m_n/m_e             m_n [GeV]")
    for n in range(2, 6):
        b = branches[n]
        print(f"{n:2d}  {fmt(b.Dhat_LC, 16):>18s}  {fmt(b.Dhat, 16):>18s}  {fmt(b.alpha_R, 16):>18s}  {fmt(b.q_raw, 16):>18s}  {fmt(b.L, 16):>18s}  {fmt(b.ratio, 16):>18s}  {fmt(b.mass_gev, 16):>18s}")

    print("\nFixed-point residuals")
    print("-" * 88)
    for n in [4, 3, 2]:
        b = branches[n]
        print(f"n={n}: iterations={b.iterations}, residual={mp.nstr(b.residual, 14)}")

    print("\nNew-branch summary")
    print("-" * 88)
    for n in [4, 5]:
        b = branches[n]
        print(f"n={n}: L={fmt(b.L, 18)}, m/m_e={fmt(b.ratio, 18)}, m={fmt(b.mass_gev, 18)} GeV")

    print("\nConvergence diagnostic for shell ledger")
    print("-" * 88)
    print(f"comparison resolution JMAX={JMAX_CHECK}, GL_NODES={GL_NODES_CHECK}")
    print(" n  |Delta T|            |Delta Gamma/Gamma|  |Delta OUT/OUT|    |Delta Rchi/Rchi|")
    for n in range(2, 6):
        hi = shells[n]
        lo = build_shell(n, inp, JMAX_CHECK, GL_NODES_CHECK)
        print(f"{n:2d}  {mp.nstr(abs(hi.T-lo.T), 8):>18s}  {mp.nstr(abs(hi.Gamma-lo.Gamma)/abs(hi.Gamma), 8):>18s}  {mp.nstr(abs(hi.out_hat-lo.out_hat)/abs(hi.out_hat), 8):>18s}  {mp.nstr(abs(hi.R_chi_red-lo.R_chi_red)/abs(hi.R_chi_red), 8):>18s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the full minimal five-shell charged-lepton closure.")
    parser.add_argument("--alpha-inv", default="137.035999177", help="inverse fine-structure constant")
    parser.add_argument("--electron-mass-mev", default="0.51099895069", help="electron mass-energy in MeV for final conversion")
    args = parser.parse_args()
    inp = Inputs(alpha_inv=mp.mpf(args.alpha_inv), electron_mass_mev=mp.mpf(args.electron_mass_mev))
    print_report(inp)


if __name__ == "__main__":
    main()
