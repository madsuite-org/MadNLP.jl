# Shared methods for all LAPACK-based dense linear solvers.
# The CPU and every GPU backend use the same solver type, `LapackSolver`; the library it calls
# follows its workspace, which `lapack_workspace` builds for the type of the factorization buffer.
# Each backend implements `lapack_workspace` and the per-algorithm primitives
# (setup_*/factorize_*/solve_*) for its workspace.

@kwdef mutable struct LapackOptions <: AbstractOptions
    lapack_algorithm::LinearFactorization = BUNCHKAUFMAN
end

"""
    LapackSolver{T, MT, Alg, M, V, W}

Dense LAPACK solver of the matrix `A::MT` with the factorization `Alg`. It copies `A` into
the dense buffer `fact::M` and factorizes it in place; `sol`, `tau` and `Λ` are vectors of
type `V` on the same device. The workspace `ws::W` holds what the library needs (pivots,
work arrays, status) and selects it by dispatch: [`lapack_workspace`](@ref) builds it for
the type of `fact`.

[`LapackCPUSolver`](@ref) is the instance on the host; the markers `LapackCUDASolver`,
`LapackROCmSolver` and `LapackOneMKLSolver` build it on the device of their matrix.
"""
struct LapackSolver{T, MT, Alg, M <: AbstractMatrix{T}, V <: AbstractVector{T}, W} <: AbstractLapackSolver{T, Alg}
    A::MT
    fact::M
    n::Int64
    sol::V
    tau::V
    Λ::V
    ws::W
    opt::LapackOptions
    logger::MadNLPLogger
end

"""
    lapack_workspace(fact::AbstractMatrix; kwargs...)

Return the workspace of a [`LapackSolver`](@ref) factorizing `fact` in place. Each backend
implements it for its matrix type, with a workspace type on which it implements the
setup, factorization and solve of each algorithm.
"""
function lapack_workspace end

function LapackSolver(A::AbstractMatrix, fact::AbstractMatrix; opt = LapackOptions(), logger = MadNLPLogger(), kwargs...)
    m, n = size(A)
    @assert m == n
    @assert size(fact) == (m, n)
    T = eltype(fact)
    sol = similar(fact, T, 0)
    tau = similar(fact, T, 0)
    Λ = similar(fact, T, 0)
    ws = lapack_workspace(fact; kwargs...)
    solver = LapackSolver{T, typeof(A), opt.lapack_algorithm, typeof(fact), typeof(sol), typeof(ws)}(
        A, fact, n, sol, tau, Λ, ws, opt, logger,
    )
    setup!(solver)
    return solver
end

# Factorize on the device of `A`: a dense matrix of the array type of `A`, or of its
# nonzeros for a sparse matrix.
LapackSolver(A::AbstractMatrix; kwargs...) = LapackSolver(A, _dense_similar(A); kwargs...)

_dense_similar(A::AbstractMatrix) = similar(A, eltype(A), size(A))
_dense_similar(A::AbstractSparseMatrix) = similar(nonzeros(A), eltype(A), size(A))

solve!(M::LapackSolver{T}, x::AbstractVector{T}) where {T} = _solve!(M, x)

# Per-algorithm primitives that each backend must implement.
function setup_cholesky! end
function setup_lu! end
function setup_qr! end
function setup_evd! end
function setup_bunchkaufman! end
function factorize_cholesky! end
function factorize_lu! end
function factorize_qr! end
function factorize_evd! end
function factorize_bunchkaufman! end
function solve_cholesky! end
function solve_lu! end
function solve_qr! end
function solve_bunchkaufman! end

"""
    transfer_matrix!(M::AbstractLapackSolver)

Copy the matrix `M.A` into the factorization buffer `M.fact`, with [`copy_to_dense!`](@ref).
"""
transfer_matrix!(M::AbstractLapackSolver) = copy_to_dense!(M.fact, M.A)

# Copy a sparse or dense matrix into a dense one. MadCoreGPUExt adds the GPU sparse case.
copy_to_dense!(y, x) = copyto!(y, x)

"""
    supports_bunchkaufman_inertia(M::AbstractLapackSolver)

Return `true` if the solver supports inertia computation for Bunch-Kaufman.
Only the CPU backend overrides this to `true`.
"""
supports_bunchkaufman_inertia(::AbstractLapackSolver) = false

# The status of the last factorization, 0 on success. Implemented for each workspace.
function _get_info end

# setup! dispatch
setup!(M::AbstractLapackSolver{T, BUNCHKAUFMAN}) where {T} = setup_bunchkaufman!(M)
setup!(M::AbstractLapackSolver{T, LU}) where {T} = setup_lu!(M)
setup!(M::AbstractLapackSolver{T, QR}) where {T} = setup_qr!(M)
setup!(M::AbstractLapackSolver{T, CHOLESKY}) where {T} = setup_cholesky!(M)
setup!(M::AbstractLapackSolver{T, EVD}) where {T} = setup_evd!(M)

# factorize! dispatch
factorize!(M::AbstractLapackSolver{T, BUNCHKAUFMAN}) where {T} = (transfer_matrix!(M); factorize_bunchkaufman!(M))
function factorize!(M::AbstractLapackSolver{T, LU}) where {T}
    transfer_matrix!(M)
    tril_to_full!(M.fact)
    return factorize_lu!(M)
end
function factorize!(M::AbstractLapackSolver{T, QR}) where {T}
    transfer_matrix!(M)
    tril_to_full!(M.fact)
    return factorize_qr!(M)
end
factorize!(M::AbstractLapackSolver{T, CHOLESKY}) where {T} = (transfer_matrix!(M); factorize_cholesky!(M))
factorize!(M::AbstractLapackSolver{T, EVD}) where {T} = (transfer_matrix!(M); factorize_evd!(M))

# solve! dispatch
_solve!(M::AbstractLapackSolver{T, BUNCHKAUFMAN}, x) where {T} = solve_bunchkaufman!(M, x)
_solve!(M::AbstractLapackSolver{T, LU}, x) where {T} = solve_lu!(M, x)
_solve!(M::AbstractLapackSolver{T, QR}, x) where {T} = solve_qr!(M, x)
_solve!(M::AbstractLapackSolver{T, CHOLESKY}, x) where {T} = solve_cholesky!(M, x)
_solve!(M::AbstractLapackSolver{T, EVD}, x) where {T} = solve_evd!(M, x)

function solve_linear_system!(M::AbstractLapackSolver, x::AbstractVector)
    isempty(M.sol) && resize!(M.sol, M.n)
    copyto!(M.sol, x)
    _solve!(M, M.sol)
    copyto!(x, M.sol)
    return x
end

improve!(M::AbstractLapackSolver) = false
input_type(::Type{<:AbstractLapackSolver}) = :dense
default_options(::Type{<:AbstractLapackSolver}) = LapackOptions()

for T in (:Float32, :Float64)
    @eval is_supported(::Type{<:AbstractLapackSolver}, ::Type{$T}) = true
end

is_inertia(::AbstractLapackSolver) = false
is_inertia(::AbstractLapackSolver{T, CHOLESKY}) where {T} = true
is_inertia(::AbstractLapackSolver{T, EVD}) where {T} = true
is_inertia(M::AbstractLapackSolver{T, BUNCHKAUFMAN}) where {T} = supports_bunchkaufman_inertia(M)

inertia(M::AbstractLapackSolver{T, CHOLESKY}) where {T} = inertia_cholesky(M)
inertia(M::AbstractLapackSolver{T, EVD}) where {T} = inertia_evd(M)
inertia(M::AbstractLapackSolver{T, BUNCHKAUFMAN}) where {T} = inertia_bunchkaufman(M)

function inertia_cholesky(M::AbstractLapackSolver)
    return _get_info(M) == 0 ? (M.n, 0, 0) : (0, M.n, 0)
end

# Solve with the eigendecomposition: x = V Λ⁻¹ Vᵀ x, with `tau` as buffer.
function solve_evd!(M::AbstractLapackSolver{T}, x::AbstractVector{T}) where {T}
    mul!(M.tau, transpose(M.fact), x)
    M.tau ./= M.Λ
    mul!(x, M.fact, M.tau)
    return x
end

function inertia_evd(M::AbstractLapackSolver)
    numpos = count(λ -> λ > 0, M.Λ)
    numneg = count(λ -> λ < 0, M.Λ)
    numzero = M.n - numpos - numneg
    return (numpos, numzero, numneg)
end
