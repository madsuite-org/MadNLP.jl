# MadCore linear solver interface to SparseDirectSolver.jl

const SDS = SparseDirectSolver
const SDSTuning = SDS.Fused.SDSTuning

"""
    SDSSolverOptions(; sds_factorization = :cholesky, sds_pivoting = :none,
                       sds_pivot_threshold = nothing, sds_reordering = "algo4",
                       sds_tuning = nothing, sds_user_perm = nothing)

Options of [`SDSSolver`](@ref).

* `sds_factorization`: `:cholesky` factors the KKT matrix as symmetric positive definite,
  `:ldl` as symmetric indefinite and reports the inertia from the signs of `D`.
* `sds_pivoting`: `:ldl` only. `:none` factors in the identity pivot order through the
  fused pipeline, `:bk` uses in-front Bunch-Kaufman pivoting.
* `sds_pivot_threshold`: `:bk` only, the in-front acceptance threshold; `nothing` keeps the
  default of SparseDirectSolver.
* `sds_reordering`: the `reordering_alg` of the analysis (`"algo4"` is nested dissection,
  `"algo3"` AMD, `"default"` the cost-model choice).
* `sds_tuning`: the performance settings; `nothing` selects the per-architecture profile.
* `sds_user_perm`: a 1-based permutation that replaces the reordering.
"""
MadCore.@kwdef mutable struct SDSSolverOptions <: MadCore.AbstractOptions
    sds_factorization::Symbol = :cholesky
    sds_pivoting::Symbol = :none
    sds_pivot_threshold::Union{Nothing, Float64} = nothing
    sds_reordering::String = "algo4"
    sds_tuning::Union{Nothing, SDSTuning} = nothing
    sds_user_perm::Union{Nothing, Vector{Int}} = nothing
end

"""
    SDSSolver(csc; opt = SDSSolverOptions(), logger = MadCore.MadNLPLogger())

Sparse direct linear solver on Intel GPUs, based on SparseDirectSolver.jl. `csc` is the lower
triangle of the KKT matrix as a `oneSparseMatrixCSC`. MadCore stores it column by column,
which SparseDirectSolver reads as the upper triangle in compressed row form.
"""
mutable struct SDSSolver{T, S <: SDS.AbstractDirectSolver, M, V} <: MadCore.AbstractLinearSolver{T}
    inner::S
    tril::M
    x::V
    b::V
    last_info::Int
    opt::SDSSolverOptions
    logger::MadCore.MadNLPLogger
end

function SDSSolver(
        csc;
        opt = SDSSolverOptions(),
        logger = MadCore.MadNLPLogger(),
    )
    colptr, rowval, nzval = csc.colPtr, csc.rowVal, csc.nzVal
    T = eltype(nzval)
    n = size(csc, 1)
    ldl = opt.sds_factorization === :ldl
    opt.sds_pivoting in (:none, :bk) ||
        throw(ArgumentError("sds_pivoting must be :none or :bk, got $(repr(opt.sds_pivoting))"))
    bk = opt.sds_pivoting === :bk
    bk && !ldl && throw(ArgumentError("sds_pivoting = :bk needs sds_factorization = :ldl"))
    inner = if bk
        tuning = SDS.Fused._resolved_tuning(opt.sds_tuning, SDS.KernelAbstractions.get_backend(nzval))
        solver = SDS.DirectSolver(colptr, rowval, nzval, "S", 'U')
        opt.sds_pivot_threshold !== nothing && SDS.setparam!(solver, "pivot_threshold", opt.sds_pivot_threshold)
        SDS.Fused._apply_tuning!(solver, tuning)
    else
        SDS.FusedSolver(colptr, rowval, nzval, ldl ? "S" : "SPD", 'U'; tuning = opt.sds_tuning)
    end
    if opt.sds_user_perm !== nothing
        SDS.setparam!(inner, "user_perm", opt.sds_user_perm)
    else
        SDS.setparam!(inner, "reordering_alg", opt.sds_reordering)
    end
    x = similar(nzval, n)
    b = similar(nzval, n)
    SDS.analyze!(inner, x, b; asynchronous = false)
    SDS.factorize!(inner, x, b; asynchronous = false)
    info = Int(SDS.getparam(inner, "info"))
    return SDSSolver{T, typeof(inner), typeof(csc), typeof(x)}(inner, csc, x, b, info, opt, logger)
end

function MadCore.factorize!(M::SDSSolver)
    SDS.refactorize!(M.inner, M.x, M.b; asynchronous = false)
    M.last_info = Int(SDS.getparam(M.inner, "info"))
    return M
end

function MadCore.solve_linear_system!(M::SDSSolver, xb)
    copyto!(M.b, xb)
    SDS.solve!(M.inner, M.x, M.b)
    copyto!(xb, M.x)
    oneAPI.synchronize()
    return xb
end

MadCore.is_factorized(M::SDSSolver) = M.last_info == 0
MadCore.is_inertia(::SDSSolver) = true
# A failed factorization is reported as an indefinite inertia, so that the interior-point
# method regularizes and factorizes again.
function MadCore.inertia(M::SDSSolver)
    n = size(M.tril, 1)
    M.last_info == 0 || return (n - 2, 1, 1)
    M.opt.sds_factorization === :ldl || return (n, 0, 0)
    npos, nneg = SDS.getparam(M.inner, "inertia")
    return (Int(npos), n - Int(npos) - Int(nneg), Int(nneg))
end
MadCore.improve!(::SDSSolver) = false
MadCore.introduce(M::SDSSolver) = M.inner isa SDS.FusedSolver ?
    "SparseDirectSolver.jl (fused pipeline)" : "SparseDirectSolver.jl (Bunch-Kaufman LDLᵀ)"

MadCore.create_linear_solver(::Type{SDSSolver}, A; kwargs...) = SDSSolver(A; kwargs...)
MadCore.default_options(::Type{SDSSolver}) = SDSSolverOptions()
MadCore.input_type(::Type{SDSSolver}) = :csc
MadCore.is_supported(::Type{SDSSolver}, ::Type{T}) where {T <: Union{Float32, Float64}} = true
