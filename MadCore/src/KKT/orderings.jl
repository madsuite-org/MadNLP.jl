# Fill-reducing orderings of the matrix a sparse KKT system hands to its linear solver,
# computed on the host (see LinearSolvers/ordering.jl).

"""
    AbstractOrdering

Symmetric fill-reducing permutation `P A Pᵀ` of the matrix a sparse KKT system factorizes,
set with the option `kkt_ordering`. The permutation is computed once on the host, from the
pattern of the KKT matrix, and replaces the ordering of the linear solver, which receives it
through its user-permutation interface (see [`set_permutation!`](@ref)) and factorizes the
KKT matrix in that order. A linear solver without such an interface supports only
`DefaultOrdering`.

Available orderings: [`DefaultOrdering`](@ref) (the linear solver orders the matrix itself),
[`MetisOrdering`](@ref) (with Metis loaded) and [`AMDOrdering`](@ref) (with AMD loaded).
"""
abstract type AbstractOrdering end

"""
    DefaultOrdering()

No ordering at the KKT level: the linear solver factorizes the KKT matrix as assembled, with
its own fill-reducing ordering.
"""
struct DefaultOrdering <: AbstractOrdering end

"""
    MetisOrdering()

Nested dissection of METIS. Implemented in the extension loaded by `Metis`.
"""
struct MetisOrdering <: AbstractOrdering end

"""
    AMDOrdering()

Approximate minimum degree ordering of AMD. Implemented in the extension loaded by `AMD`.
"""
struct AMDOrdering <: AbstractOrdering end

backend_package(::Type{MetisOrdering}) = "Metis"
backend_package(::Type{AMDOrdering}) = "AMD"

"""
    ordering_permutation(ordering::AbstractOrdering, A::SparseMatrixCSC) -> Vector{Int}

Return the permutation `perm` of the symmetric matrix whose lower triangle is `A`: row and
column `i` of the permuted matrix are row and column `perm[i]` of the original, so `perm[i]`
is the `i`-th pivot. Only the pattern of `A` is read. The methods of `MetisOrdering` and
`AMDOrdering` are defined in the extensions loaded by Metis and AMD.
"""
function ordering_permutation end

ordering_permutation(::DefaultOrdering, A::SparseMatrixCSC) = collect(1:size(A, 1))
