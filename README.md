# Charged-Lepton Mass Hierarchy Law

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/pajuhaan/Charged-Lepton-Mass-Hierarchy-Public)

Minimal repository companion for the numerical evaluator of the charged-lepton mass hierarchy law by **M. Pajuhaan**.

The calculation implements a no-target Relator/Alpha source-ledger model. The only supplied dimensionless theory input is

```math
\alpha^{-1}=137.035999177.
```

Measured muon and tau masses are used only for the final external comparison. The electron mass is used only as the MeV conversion scale after the dimensionless ratios have already been computed.

## Compact law

For shell number `n = 2, 3`, the final Schur-dressed mass logarithm is

```math
L_n \equiv \ln\frac{m_n}{m_e}
=
\frac{A T_n}{A(1-\rho_n)+\rho_n T_n},
\qquad
A=\frac{\pi}{8\alpha},
\qquad
\rho_n=\frac{C_{\rm UV}^{\rm Gauss}}{n+1}\ln D_n.
```

Here `T_n = ln Γ_n` is the bare Relator exponent-slot readout. It is not the physical mass logarithm. The determinant object `D_n` is not a mass and is not an additive mass-log correction.

## Main numerical results

Using `m_e = 0.51099895069 MeV` only for final unit conversion:

| Branch | `n` | `L_n = ln(m_n/m_e)` | `m_n/m_e` | Prediction [MeV] | External-audit pull |
|---|---:|---:|---:|---:|---:|
| muon | 2 | 5.33159877458397 | 206.768285919222 | 105.658377140693 | +0.713σ |
| tau | 3 | 8.15399066072771 | 3477.22785082421 | 1776.85978308121 | -0.780σ |

## Reproduce

Install the numerical dependencies:

```bash
pip install mpmath numpy scipy rich
```

Run the evaluator:

```bash
python "Check Point.py"
```

The script prints the compact formula map, upstream ledger quantities, Schur determinant components, final mass ratios, MeV conversion, and external comparison table.

## Guardrails

- `Γ_n` is a shell-admission ratio, not a mass ratio.
- `T_n` is a bare Relator exponent-slot readout, not `L_n`.
- `α_{R,n}` is an internal Relator exponent-slot coordinate, not `α_QED(q²)`.
- `D_n` is a determinant-sector object, not a mass and not an additive correction to `L_n`.
- The odd-mode source-vector ledger is fixed before comparison with charged-lepton masses.
- Finite numerical resolution is a convergence certificate, not a fitted cutoff.

## Citation and links

Please cite the manuscript and repository output when using these calculations.

- Zenodo: https://doi.org/10.5281/zenodo.17069630
- ResearchGate: https://www.researchgate.net/publication/396387184_ChargedLepton_Mass_Hierarchy

## Credit and copyright

Scientific framework, formulas, numerical ledger, and manuscript credit belong to **M. Pajuhaan**.

Copyright © 2026 **M. Pajuhaan**. All rights reserved unless a separate `LICENSE` file in this repository states otherwise.
