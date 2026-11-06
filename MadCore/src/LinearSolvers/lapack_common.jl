# Shared methods for all LAPACK-based dense linear solvers.
# Each backend (CPU, CUDA, ROCm, oneAPI) subtypes AbstractLapackSolver
# and provides per-algorithm primitives (setup_*/factorize_*/solve_*).

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
function solve_evd! end
function solve_bunchkaufman! end

"""
    transfer_matrix!(M::AbstractLapackSolver)

Copy the matrix `M.A` into the factorization buffer `M.fact`.
GPU backends override this to use `gpu_transfer!`.
"""
transfer_matrix!(M::AbstractLapackSolver) = copyto!(M.fact, M.A)

"""
    supports_bunchkaufman_inertia(M::AbstractLapackSolver)

Return `true` if the solver supports inertia computation for Bunch-Kaufman.
Only the CPU backend overrides this to `true`.
"""
supports_bunchkaufman_inertia(::AbstractLapackSolver) = false

"""
    _get_info(M::AbstractLapackSolver)

Return the scalar info value from the factorization.
CPU uses `M.info[]` (Ref); GPU backends override to `sum(M.info)`.
"""
_get_info(M::AbstractLapackSolver) = M.info[]

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

function inertia_evd(M::AbstractLapackSolver)
    numpos = count(λ -> λ > 0, M.Λ)
    numneg = count(λ -> λ < 0, M.Λ)
    numzero = M.n - numpos - numneg
    return (numpos, numzero, numneg)
end
