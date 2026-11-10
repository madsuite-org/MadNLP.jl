# Scalers of the matrix a sparse KKT system factorizes (see LinearSolvers/scaling.jl).

"""
    AbstractScaler

Symmetric scaling `W K W` of the matrix a sparse KKT system factorizes, set with the option
`kkt_scaler`. The KKT system itself is not scaled: the factorization is, and each solve is
unscaled to match, so iterative refinement works on the unscaled system.

Available scalers: `NoScaler`, `JacobiScaler`, `MaxNormScaler` and
`RuizScaler`.
"""
abstract type AbstractScaler end

"""
    NoScaler()

No scaling.
"""
struct NoScaler <: AbstractScaler end

"""
    JacobiScaler()

`W = diag(|K_ii|)^{-1/2}`; a zero diagonal entry gets `W_ii = 1`.
"""
struct JacobiScaler <: AbstractScaler end

"""
    MaxNormScaler()

One sweep of symmetric infinity-norm scaling: `W_ii = (max_j |K_ij|)^{-1/2}`.
"""
struct MaxNormScaler <: AbstractScaler end

"""
    RuizScaler(sweeps = 4)

Ruiz's symmetric equilibration: `sweeps` sweeps of `W_ii ← W_ii (max_j |W_ii K_ij W_jj|)^{-1/2}`,
which drive the infinity norm of every row of `W K W` towards one.
"""
@kwdef struct RuizScaler <: AbstractScaler
    sweeps::Int = 4
end

"""
    AugmentedRuizScaler(sweeps = 4)

Ruiz's equilibration of the augmented system, before condensation, for `SparseCondensedKKTSystem`
(on the host). The sweeps run on `[H + Σx  Jᵀ; J  Σd - Σs⁻¹]` and give a primal scaling `Wx` and a
constraint scaling `Wy`; the condensed matrix is scaled by `Wx` (`Wy` cancels in `Jᵀ D J`) and the
rows kept in the border by `Wy`. Unlike `RuizScaler` on the condensed matrix, a row whose `D` is
very large (a tightly relaxed constraint) is equilibrated on its own before it is summed into the
condensed block. With another KKT system the matrix is not scaled.

With `bounds = :symmetrized` the sweeps run on the symmetrized 3×3 system instead: each bound of a
variable is a row of its own, coupled to the variable by `√z` and with diagonal `x - xˡ` (`xᵘ - x`),
where `:folded` (the default) sums the bounds into `Σx`. The bounds of the slacks stay in `Σs`.
"""
@kwdef struct AugmentedRuizScaler <: AbstractScaler
    sweeps::Int = 4
    bounds::Symbol = :folded
end

scaler_sweeps(::NoScaler) = 0
scaler_sweeps(::AugmentedRuizScaler) = 0   # set by the KKT system, see update_kkt_scaling!
scaler_sweeps(::MaxNormScaler) = 1
scaler_sweeps(s::RuizScaler) = s.sweeps
