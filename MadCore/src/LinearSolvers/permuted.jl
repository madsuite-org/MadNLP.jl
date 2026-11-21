# Fill-reducing permutation of the matrix a sparse KKT system hands to its linear solver.

"""
    PermutedSolver

The linear solver `create_kkt_linear_solver` returns for an ordering other than
[`NoOrdering`](@ref): it keeps the permuted copy `P A Pᵀ` of the matrix `A` (lower triangle,
CSC), factorizes it with the selected solver, and permutes each right-hand side and solution.
The permutation is computed on the host from the pattern of `A`; the permuted matrix lives on
the device of `A`.
"""
struct PermutedSolver{T, MT, VT, VI, LS, O <: AbstractOrdering} <: AbstractLinearSolver{T}
    A::MT          # the matrix, read only
    Ap::MT         # its permuted copy P A Pᵀ (lower triangle), factorized by `inner`
    inner::LS
    ordering::O
    perm::VI       # row i of Ap is row perm[i] of A
    nzmap::VI      # nonzero of Ap -> nonzero of A
    buf::VT
end

function PermutedSolver(linear_solver, A; opt, logger, kkt_ordering::AbstractOrdering)
    hasproperty(opt, :n_primal) && throw(
        ArgumentError(
            "the ordering $(nameof(typeof(kkt_ordering))) is not yet supported with $(linear_solver), " *
                "whose factorization depends on the primal and dual blocks of the KKT matrix.",
        ),
    )
    n = size(A, 1)
    colptr = Vector{Int}(host_array(getcolptr(A)))
    rows = Vector{Int}(host_array(rowvals(A)))
    perm = ordering_permutation(kkt_ordering, SparseMatrixCSC(n, n, colptr, rows, ones(length(rows))))
    (length(perm) == n && isperm(perm)) || error(logger, "the ordering $(nameof(typeof(kkt_ordering))) did not return a permutation of 1:$n")
    iperm = invperm(perm)
    # the lower triangle of P A Pᵀ, each nonzero mapped to its nonzero of A
    I = similar(rows)
    J = similar(rows)
    for j in 1:n, k in colptr[j]:(colptr[j + 1] - 1)
        r, c = iperm[rows[k]], iperm[j]
        I[k], J[k] = max(r, c), min(r, c)
    end
    S = sparse(I, J, collect(1:length(rows)), n, n)
    Ti = eltype(rowvals(A))
    dv(v) = device_array(A, v)
    nzmap = dv(Ti.(nonzeros(S)))
    Ap = _get_sparse_csc(size(A), dv(Ti.(S.colptr)), dv(Ti.(S.rowval)), similar(nonzeros(A)))
    nonzeros(Ap) .= view(nonzeros(A), nzmap)
    inner = create_linear_solver(linear_solver, Ap; opt, logger)
    buf = similar(nonzeros(A), n)
    return PermutedSolver{eltype(buf), typeof(A), typeof(buf), typeof(nzmap), typeof(inner), typeof(kkt_ordering)}(
        A, Ap, inner, kkt_ordering, dv(Ti.(perm)), nzmap, buf,
    )
end

# The linear solver of the matrix `A`, permuted by `kkt_ordering` unless it is `NoOrdering`.
_create_ordered_solver(linear_solver, A, ::NoOrdering; opt, logger) = create_linear_solver(linear_solver, A; opt, logger)
_create_ordered_solver(linear_solver, A, kkt_ordering::AbstractOrdering; opt, logger) =
    PermutedSolver(linear_solver, A; opt, logger, kkt_ordering)

_permute_values!(M::PermutedSolver) = (nonzeros(M.Ap) .= view(nonzeros(M.A), M.nzmap))

function factorize!(M::PermutedSolver)
    _permute_values!(M)
    factorize!(M.inner)
    return M
end

# an analysis that reads the values (cuDSS with matching) sees the current matrix
function analyze!(M::PermutedSolver)
    _permute_values!(M)
    return analyze!(M.inner)
end

# A x = b  ⇔  (P A Pᵀ)(P x) = P b
function solve_linear_system!(M::PermutedSolver, b::AbstractVector)
    M.buf .= view(b, M.perm)
    solve_linear_system!(M.inner, M.buf)
    view(b, M.perm) .= M.buf
    return b
end

is_inertia(M::PermutedSolver) = is_inertia(M.inner)
inertia(M::PermutedSolver) = inertia(M.inner)
improve!(M::PermutedSolver) = improve!(M.inner)
is_factorized(M::PermutedSolver) = is_factorized(M.inner)
is_async(M::PermutedSolver) = is_async(M.inner)
introduce(M::PermutedSolver) = "$(introduce(M.inner)) with $(ordering_name(M.ordering)) ordering"

ordering_name(::MetisOrdering) = "METIS"
ordering_name(::AMDOrdering) = "AMD"
ordering_name(::SymAMDOrdering) = "SYMAMD"
ordering_name(::ColAMDOrdering) = "COLAMD"
ordering_name(::UserOrdering) = "user"
ordering_name(o::AbstractOrdering) = string(nameof(typeof(o)))
