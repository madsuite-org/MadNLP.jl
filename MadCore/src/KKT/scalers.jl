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

scaler_sweeps(::NoScaler) = 0
scaler_sweeps(::MaxNormScaler) = 1
scaler_sweeps(s::RuizScaler) = s.sweeps
