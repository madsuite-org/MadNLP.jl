module MadCoreGPUSDSExt

import MadCore
import MadCoreGPU
import SparseDirectSolver

const SDS = SparseDirectSolver

"""
    SDSSolverOptions

Options of [`MadCoreGPU.SDSSolver`](@ref), passed as keywords of `madnlp`.

* `sds_factorization::Symbol = :cholesky`: `:cholesky` factors the KKT matrix as symmetric
  positive definite; `:ldl` as symmetric indefinite, reporting the inertia from the signs of
  `D`. Only `:ldl` can factor the condensed matrix with dense Jacobian rows kept as a border.
* `sds_pivoting::Symbol = :none`: `:ldl` only. `:none` factors in the identity pivot order,
  `:bk` uses in-front Bunch-Kaufman pivoting.
* `sds_matching::String = "none"`: `:ldl` only, the `matching_alg` of the analysis (`"none"`, or
  `"algo1"` to `"algo6"`; `"algo6"` is the automatic matching with scaling, as cuDSS's default).
* `sds_deferred_analysis::Bool = false`: run the analysis at the first factorization, so that a
  value-dependent analysis (matching, 2×2 pivot pairs) sees the first KKT matrix.
* `sds_pivot_threshold`: `:bk` only, the in-front acceptance threshold.
* `sds_pivot_epsilon`: `:ldl` only, pivots below this magnitude are replaced, with the sign of
  their block (positive for the primal, negative for the dual rows).
* `sds_ir::Int = 0`: the steps of iterative refinement inside SparseDirectSolver after every
  solve (as `cudss_ir` for cuDSS).
* `sds_reordering::String = "algo4"`: the `reordering_alg` of the analysis (`"algo4"` is nested
  dissection, `"algo3"` AMD, `"default"` the cost-model choice).
* `sds_tuning`: performance settings (a `SparseDirectSolver.SDSTuning`); `nothing` selects the
  profile of the GPU architecture.
* `sds_user_perm`: a 1-based permutation that replaces the reordering.
* `sds_fused`: run the fused pipeline (`true`) or the stock one (`false`); `nothing` lets
  SparseDirectSolver choose. `:bk` runs the stock pipeline; `sds_fused = true` with `:bk` raises
  `SparseDirectSolver.InvalidValueError`.

`nothing` keeps the default of SparseDirectSolver. The KKT system sets `n_primal` and
`n_dual`, the sizes of the primal and dual blocks.
"""
MadCore.@kwdef mutable struct SDSSolverOptions <: MadCore.AbstractOptions
    sds_factorization::Symbol = :cholesky
    sds_pivoting::Symbol = :none
    sds_matching::String = "none"
    sds_deferred_analysis::Bool = false
    sds_pivot_threshold::Union{Nothing, Float64} = nothing
    sds_pivot_epsilon::Union{Nothing, Float64} = nothing
    sds_ir::Int = 0
    sds_reordering::String = "algo4"
    sds_tuning::Union{Nothing, SDS.SDSTuning} = nothing
    sds_user_perm::Union{Nothing, Vector{Int}} = nothing
    sds_fused::Union{Nothing, Bool} = nothing
    n_primal::Int = -1
    n_dual::Int = 0
end

MadCore._factors_indefinite(opt::SDSSolverOptions) = opt.sds_factorization === :ldl

mutable struct SDSState{T, K <: SDS.KKTSolver, M} <: MadCore.AbstractLinearSolver{T}
    inner::K
    csc::M
    opt::SDSSolverOptions
    logger::MadCore.MadNLPLogger
end

# `csc` is the lower triangle of the KKT matrix in compressed columns, which SparseDirectSolver
# reads as the upper triangle in compressed rows.
function SDSState(csc; opt = SDSSolverOptions(), logger = MadCore.MadNLPLogger())
    n = size(csc, 1)
    ldl = opt.sds_factorization === :ldl
    bk = opt.sds_pivoting === :bk
    n_primal = opt.n_primal < 0 ? n : opt.n_primal
    inner = SDS.KKTSolver(
        csc.colPtr, csc.rowVal, csc.nzVal, 'U';
        factorization = opt.sds_factorization,
        pivoting = opt.sds_pivoting,
        fused = opt.sds_fused,
        pivot_threshold = bk ? opt.sds_pivot_threshold : nothing,
        # LDLᵀ options, ignored for Cholesky so that a preset that sets them still allows it
        matching = ldl ? opt.sds_matching : "none",
        pivot_epsilon = ldl ? opt.sds_pivot_epsilon : nothing,
        pivot_sign = ldl ? vcat(ones(Int8, n_primal), fill(Int8(-1), n - n_primal)) : nothing,
        ir_n_steps = opt.sds_ir,
        deferred_analysis = opt.sds_deferred_analysis,
        reordering = opt.sds_reordering,
        user_perm = opt.sds_user_perm,
        tuning = opt.sds_tuning,
    )
    return SDSState{eltype(MadCore.scaling_nzval(csc)), typeof(inner), typeof(csc)}(inner, csc, opt, logger)
end

function MadCore.factorize!(M::SDSState)
    SDS.factorize!(M.inner)
    return M
end

function MadCore.solve_linear_system!(M::SDSState, xb)
    SDS.solve!(M.inner, xb)
    return xb
end

MadCore.is_factorized(M::SDSState) = SDS.getparam(M.inner, "info") == 0
MadCore.is_inertia(::SDSState) = true
# A failed factorization is reported as an indefinite inertia, so that the interior-point
# method regularizes and factorizes again.
function MadCore.inertia(M::SDSState)
    n = size(M.csc, 1)
    MadCore.is_factorized(M) || return (n - 2, 1, 1)
    M.opt.sds_factorization === :ldl || return (n, 0, 0)
    return SDS.inertia(M.inner)
end
MadCore.improve!(::SDSState) = false
function MadCore.introduce(M::SDSState)
    pipeline = SDS.pipeline(M.inner) === :fused ? "fused" : "stock"
    return "SparseDirectSolver.jl ($(M.opt.sds_factorization), $(pipeline) pipeline)"
end

MadCore.default_options(::Type{SDSState}) = SDSSolverOptions()
MadCore.input_type(::Type{SDSState}) = :csc
MadCore.is_supported(::Type{SDSState}, ::Type{T}) where {T <: Union{Float32, Float64}} = true

MadCore.create_linear_solver(::Type{MadCoreGPU.SDSSolver}, A; kwargs...) = SDSState(A; kwargs...)
MadCore.default_options(::Type{MadCoreGPU.SDSSolver}) = SDSSolverOptions()
MadCore.input_type(::Type{MadCoreGPU.SDSSolver}) = :csc
MadCore.is_supported(::Type{MadCoreGPU.SDSSolver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(SDSState, T)

end
