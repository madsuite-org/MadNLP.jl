module MadCoreGPUSDSExt

import MadCore
import MadCoreGPU
import SparseDirectSolver
import KernelAbstractions: get_backend, synchronize

const SDS = SparseDirectSolver

"""
    SDSSolverOptions

Options of [`MadCoreGPU.SDSSolver`](@ref), passed as keywords of `madnlp`.

* `sds_factorization::Symbol = :cholesky`: `:cholesky` factors the KKT matrix as symmetric
  positive definite; `:ldl` as symmetric indefinite, reporting the inertia from the signs of
  `D`. Only `:ldl` can factor the condensed matrix with dense Jacobian rows kept as a border.
* `sds_pivoting::Symbol = :none`: `:ldl` only. `:none` factors in the identity pivot order,
  `:bk` uses in-front Bunch-Kaufman pivoting.
* `sds_pivot_threshold`: `:bk` only, the in-front acceptance threshold.
* `sds_pivot_epsilon`: `:ldl` with `:none` on the stock path, pivots below this magnitude are
  replaced.
* `sds_reordering::String = "algo4"`: the `reordering_alg` of the analysis (`"algo4"` is nested
  dissection, `"algo3"` AMD, `"default"` the cost-model choice).
* `sds_tuning`: performance settings of the fused pipeline; `nothing` selects the profile of
  the GPU architecture.
* `sds_user_perm`: a 1-based permutation that replaces the reordering.
* `sds_fused`: run the fused pipeline (`true`) or the stock one (`false`); `nothing` takes the
  default of the GPU backend. Metal uses the stock pipeline.

`nothing` keeps the default of SparseDirectSolver. The KKT system sets `n_primal` and
`n_dual`, the sizes of the primal and dual blocks.
"""
MadCore.@kwdef mutable struct SDSSolverOptions <: MadCore.AbstractOptions
    sds_factorization::Symbol = :cholesky
    sds_pivoting::Symbol = :none
    sds_pivot_threshold::Union{Nothing, Float64} = nothing
    sds_pivot_epsilon::Union{Nothing, Float64} = nothing
    sds_reordering::String = "algo4"
    sds_tuning::Union{Nothing, SDS.Fused.SDSTuning} = nothing
    sds_user_perm::Union{Nothing, Vector{Int}} = nothing
    sds_fused::Union{Nothing, Bool} = nothing
    n_primal::Int = -1
    n_dual::Int = 0
end

MadCore._factors_indefinite(opt::SDSSolverOptions) = opt.sds_factorization === :ldl

mutable struct SDSState{T, S <: SDS.AbstractDirectSolver, M, V} <: MadCore.AbstractLinearSolver{T}
    inner::S
    csc::M
    x::V
    b::V
    last_info::Int
    opt::SDSSolverOptions
    logger::MadCore.MadNLPLogger
end

# `csc` is the lower triangle of the KKT matrix in compressed columns, which SparseDirectSolver
# reads as the upper triangle in compressed rows.
function SDSState(csc; opt = SDSSolverOptions(), logger = MadCore.MadNLPLogger())
    colptr, rowval, nzval = csc.colPtr, csc.rowVal, csc.nzVal
    T = eltype(nzval)
    n = size(csc, 1)
    backend = get_backend(nzval)
    opt.sds_factorization in (:cholesky, :ldl) ||
        throw(ArgumentError("sds_factorization must be :cholesky or :ldl, got $(repr(opt.sds_factorization))"))
    opt.sds_pivoting in (:none, :bk) ||
        throw(ArgumentError("sds_pivoting must be :none or :bk, got $(repr(opt.sds_pivoting))"))
    ldl = opt.sds_factorization === :ldl
    bk = opt.sds_pivoting === :bk
    bk && !ldl && throw(ArgumentError("sds_pivoting = :bk needs sds_factorization = :ldl"))
    fused = something(opt.sds_fused, MadCoreGPU.sds_fused_default(backend)) && !bk
    inner = if fused
        SDS.FusedSolver(colptr, rowval, nzval, ldl ? "S" : "SPD", 'U'; tuning = opt.sds_tuning)
    else
        solver = SDS.DirectSolver(colptr, rowval, nzval, ldl ? "S" : "SPD", 'U')
        if opt.sds_tuning !== nothing
            SDS.Fused._apply_tuning!(solver, SDS.Fused._resolved_tuning(opt.sds_tuning, backend))
        end
        if bk
            opt.sds_pivot_threshold === nothing || SDS.setparam!(solver, "pivot_threshold", opt.sds_pivot_threshold)
        elseif ldl
            SDS.setparam!(solver, "pivot_type", 'N')
            SDS.setparam!(solver, "pivot_pairs", "none")
            opt.sds_pivot_epsilon === nothing || SDS.setparam!(solver, "pivot_epsilon", opt.sds_pivot_epsilon)
            n_primal = opt.n_primal < 0 ? n : opt.n_primal
            SDS.setparam!(solver, "pivot_sign", vcat(ones(Int8, n_primal), fill(Int8(-1), n - n_primal)))
        end
        solver
    end
    if opt.sds_user_perm !== nothing
        SDS.setparam!(inner, "user_perm", opt.sds_user_perm)
    else
        SDS.setparam!(inner, "reordering_alg", opt.sds_reordering)
    end
    x = similar(nzval, n)
    b = similar(nzval, n)
    SDS.analyze!(inner, x, b; asynchronous = false)
    return SDSState{T, typeof(inner), typeof(csc), typeof(x)}(inner, csc, x, b, 0, opt, logger)
end

function MadCore.factorize!(M::SDSState)
    SDS.refactorize!(M.inner, M.x, M.b; asynchronous = false)
    M.last_info = Int(SDS.getparam(M.inner, "info"))
    return M
end

function MadCore.solve_linear_system!(M::SDSState, xb)
    copyto!(M.b, xb)
    SDS.solve!(M.inner, M.x, M.b)
    copyto!(xb, M.x)
    synchronize(get_backend(xb))
    return xb
end

MadCore.is_factorized(M::SDSState) = M.last_info == 0
MadCore.is_inertia(::SDSState) = true
# A failed factorization is reported as an indefinite inertia, so that the interior-point
# method regularizes and factorizes again.
function MadCore.inertia(M::SDSState)
    n = size(M.csc, 1)
    M.last_info == 0 || return (n - 2, 1, 1)
    M.opt.sds_factorization === :ldl || return (n, 0, 0)
    npos, nneg = SDS.getparam(M.inner, "inertia")
    return (Int(npos), n - Int(npos) - Int(nneg), Int(nneg))
end
MadCore.improve!(::SDSState) = false
function MadCore.introduce(M::SDSState)
    pipeline = M.inner isa SDS.FusedSolver ? "fused" : "stock"
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
