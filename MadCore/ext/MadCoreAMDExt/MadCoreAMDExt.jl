module MadCoreAMDExt

import MadCore
import AMD
import SparseArrays: SparseMatrixCSC, getcolptr, rowvals, nnz

# AMD orders the pattern of A + Aᵀ and SYMAMD reads the lower triangle, so both take the lower
# triangle `A` as it is; COLAMD orders the columns of the full symmetric matrix.
MadCore.ordering_permutation(::MadCore.AMDOrdering, A::SparseMatrixCSC) = Vector{Int}(AMD.amd(A))
MadCore.ordering_permutation(::MadCore.SymAMDOrdering, A::SparseMatrixCSC) = Vector{Int}(AMD.symamd(A))
function MadCore.ordering_permutation(::MadCore.ColAMDOrdering, A::SparseMatrixCSC)
    L = SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), ones(nnz(A)))
    return Vector{Int}(AMD.colamd(L + L'))
end

end # module
