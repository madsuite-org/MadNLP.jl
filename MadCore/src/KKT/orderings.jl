# Fill-reducing orderings of the matrix a sparse KKT system hands to its linear solver,
# computed on the host (see LinearSolvers/permuted.jl).

"""
    AbstractOrdering

Symmetric fill-reducing permutation `P A Pᵀ` of the matrix a sparse KKT system factorizes,
set with the option `kkt_ordering`. The linear solver factorizes the permuted matrix, and each
solve is permuted back, so the KKT system and iterative refinement are unchanged. It applies
to every sparse linear solver; a solver that computes its own ordering reorders the permuted
matrix again.

Available orderings: [`NoOrdering`](@ref), [`MetisOrdering`](@ref) (with Metis loaded),
[`AMDOrdering`](@ref), [`SymAMDOrdering`](@ref) and [`ColAMDOrdering`](@ref) (with AMD
loaded), and [`UserOrdering`](@ref).
"""
abstract type AbstractOrdering end

"""
    NoOrdering()

No permutation: the linear solver factorizes the KKT matrix as assembled.
"""
struct NoOrdering <: AbstractOrdering end

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

"""
    SymAMDOrdering()

Symmetric approximate minimum degree ordering of COLAMD (`symamd`). Implemented in the
extension loaded by `AMD`.
"""
struct SymAMDOrdering <: AbstractOrdering end

"""
    ColAMDOrdering()

Column approximate minimum degree ordering of COLAMD (`colamd`), applied to the symmetric
matrix. Implemented in the extension loaded by `AMD`.
"""
struct ColAMDOrdering <: AbstractOrdering end

"""
    UserOrdering(perm::Vector{Int})

The permutation `perm`, given by the user: row and column `i` of the factorized matrix are
row and column `perm[i]` of the KKT matrix.
"""
struct UserOrdering <: AbstractOrdering
    perm::Vector{Int}
end

backend_package(::Type{MetisOrdering}) = "Metis"
backend_package(::Type{<:Union{AMDOrdering, SymAMDOrdering, ColAMDOrdering}}) = "AMD"

"""
    ordering_permutation(ordering::AbstractOrdering, A::SparseMatrixCSC) -> Vector{Int}

Return the permutation `perm` of the symmetric matrix whose lower triangle is `A`: row and
column `i` of the permuted matrix are row and column `perm[i]` of the original. Only the
pattern of `A` is read. The methods of `MetisOrdering` and of the AMD orderings are defined
in the extensions loaded by Metis and AMD.
"""
function ordering_permutation end

ordering_permutation(::NoOrdering, A::SparseMatrixCSC) = collect(1:size(A, 1))

function ordering_permutation(o::UserOrdering, A::SparseMatrixCSC)
    (length(o.perm) == size(A, 1) && isperm(o.perm)) ||
        throw(ArgumentError("UserOrdering needs a permutation of 1:$(size(A, 1)), the size of the KKT matrix."))
    return o.perm
end
