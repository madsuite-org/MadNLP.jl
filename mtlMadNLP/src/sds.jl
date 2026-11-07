# MadCore linear solver interface to SparseDirectSolver.jl

const SDS = SparseDirectSolver

"""
    SDSSolverOptions(; sds_factorization = :ldl, sds_pivot_epsilon = nothing)

Options of [`SDSSolver`](@ref).

* `sds_factorization`: `:ldl` factors the KKT matrix as symmetric indefinite without pivoting
  and reports the inertia from the signs of `D`; `:cholesky` as symmetric positive definite.
  `:cholesky` needs a positive definite matrix, so dense Jacobian rows kept as a border by
  `SparseCondensedKKTSystem` make it fail.
* `sds_pivot_epsilon`: `:ldl` only, pivots below this magnitude are replaced; `nothing` keeps
  the default of SparseDirectSolver.

The KKT system sets `n_primal` and `n_dual`, the sizes of the primal and dual blocks.
"""
MadCore.@kwdef mutable struct SDSSolverOptions <: MadCore.AbstractOptions
    sds_factorization::Symbol = :ldl
    sds_pivot_epsilon::Union{Nothing, Float64} = nothing
    n_primal::Int = -1
    n_dual::Int = 0
end

"""
    SDSSolver(csc; opt = SDSSolverOptions(), logger = MadCore.MadNLPLogger())

Sparse direct linear solver on Apple GPUs, based on SparseDirectSolver.jl. `csc` is the lower
triangle of the condensed KKT matrix. SparseDirectSolver reads it as the upper triangle in
compressed row form.
"""
mutable struct SDSSolver{T, S <: SDS.AbstractDirectSolver, M, V} <: MadCore.AbstractLinearSolver{T}
    inner::S
    csc::M
    x::V
    b::V
    last_info::Int
    opt::SDSSolverOptions
    logger::MadCore.MadNLPLogger
end

function SDSSolver(csc; opt = SDSSolverOptions(), logger = MadCore.MadNLPLogger())
    T = eltype(csc.nzVal)
    n = size(csc, 1)
    ldl = opt.sds_factorization === :ldl
    !ldl && opt.n_dual > 0 &&
        throw(ArgumentError("sds_factorization = :cholesky cannot factor a matrix with dense rows kept as a border"))
    inner = SDS.DirectSolver(csc.colPtr, csc.rowVal, csc.nzVal, ldl ? "S" : "SPD", 'U')
    if ldl
        SDS.setparam!(inner, "pivot_type", 'N')
        SDS.setparam!(inner, "pivot_pairs", "none")
        opt.sds_pivot_epsilon === nothing || SDS.setparam!(inner, "pivot_epsilon", opt.sds_pivot_epsilon)
        n_primal = opt.n_primal < 0 ? n : opt.n_primal
        SDS.setparam!(inner, "pivot_sign", vcat(ones(Int8, n_primal), fill(Int8(-1), n - n_primal)))
    end
    x = similar(csc.nzVal, n)
    b = similar(csc.nzVal, n)
    SDS.analyze!(inner, x, b; asynchronous = false)
    return SDSSolver{T, typeof(inner), typeof(csc), typeof(x)}(inner, csc, x, b, 0, opt, logger)
end

function MadCore.factorize!(M::SDSSolver)
    SDS.factorize!(M.inner, M.x, M.b; asynchronous = false)
    M.last_info = Int(SDS.getparam(M.inner, "info"))
    return M
end

function MadCore.solve_linear_system!(M::SDSSolver, xb)
    copyto!(M.b, xb)
    SDS.solve!(M.inner, M.x, M.b)
    copyto!(xb, M.x)
    Metal.synchronize()
    return xb
end

MadCore.is_factorized(M::SDSSolver) = M.last_info == 0
MadCore.is_inertia(::SDSSolver) = true
# A failed factorization is reported as an indefinite inertia, so that the interior-point
# method regularizes and factorizes again.
function MadCore.inertia(M::SDSSolver)
    n = size(M.csc, 1)
    M.last_info == 0 || return (n - 2, 1, 1)
    M.opt.sds_factorization === :ldl || return (n, 0, 0)
    npos, nneg = SDS.getparam(M.inner, "inertia")
    return (Int(npos), n - Int(npos) - Int(nneg), Int(nneg))
end
MadCore.improve!(::SDSSolver) = false
MadCore.introduce(M::SDSSolver) = M.opt.sds_factorization === :ldl ?
    "SparseDirectSolver.jl (LDLᵀ without pivoting)" : "SparseDirectSolver.jl (Cholesky)"

MadCore.create_linear_solver(::Type{SDSSolver}, A; kwargs...) = SDSSolver(A; kwargs...)
MadCore.default_options(::Type{SDSSolver}) = SDSSolverOptions()
MadCore.input_type(::Type{SDSSolver}) = :csc
MadCore.is_supported(::Type{SDSSolver}, ::Type{Float32}) = true
