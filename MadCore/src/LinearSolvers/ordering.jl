# The fill-reducing ordering of the option `kkt_ordering`, handed to the linear solver of a
# sparse KKT system.

"""
    set_permutation!(opt, perm) -> opt

Hand the fill-reducing permutation `perm` of the option `kkt_ordering` to a linear solver
through its options `opt`: the solver factorizes its matrix `A` in the pivot order `perm`
(row and column `i` of the factorized `P A Pᵀ` are row and column `perm[i]` of `A`) instead
of computing an ordering of its own. Each solver with a user-permutation interface defines a
method for its options type, which sets its option `*_user_perm`; `opt` is a copy of the
options the KKT system received.

The fallback throws an `ArgumentError`: an ordering other than [`DefaultOrdering`](@ref) needs a
linear solver that accepts a user permutation.
"""
set_permutation!(opt, perm) = throw(
    ArgumentError(
        "the linear solver of the options $(nameof(typeof(opt))) does not accept a user permutation; " *
            "use kkt_ordering = DefaultOrdering() or a linear solver that does.",
    ),
)

# The permutation of the matrix `A` (lower triangle, CSC, on the host or a device) by
# `ordering`, computed on the host from its pattern.
function kkt_permutation(ordering::AbstractOrdering, A)
    n = size(A, 1)
    rows = Vector{Int}(host_array(rowvals(A)))
    perm = ordering_permutation(ordering, SparseMatrixCSC(n, n, Vector{Int}(host_array(getcolptr(A))), rows, ones(length(rows))))
    (length(perm) == n && isperm(perm)) ||
        throw(ArgumentError("the ordering $(nameof(typeof(ordering))) did not return a permutation of 1:$n"))
    return perm
end

# The linear solver of the matrix `A` in the order of `kkt_ordering`: with `DefaultOrdering`, its
# own; otherwise the permutation, handed to the solver on a copy of `opt`, so that `opt` is not
# changed for the other solvers built from it.
_create_ordered_solver(linear_solver, A, ::DefaultOrdering; opt, logger) = create_linear_solver(linear_solver, A; opt, logger)
function _create_ordered_solver(linear_solver, A, kkt_ordering::AbstractOrdering; opt, logger)
    perm_opt = set_permutation!(deepcopy(opt), kkt_permutation(kkt_ordering, A))
    return create_linear_solver(linear_solver, A; opt = perm_opt, logger)
end
