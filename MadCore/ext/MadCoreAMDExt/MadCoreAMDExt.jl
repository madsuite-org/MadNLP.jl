module MadCoreAMDExt

import MadCore
import AMD
import SparseArrays: SparseMatrixCSC

# AMD orders the pattern of A + Aᵀ, so it takes the lower triangle `A` as it is.
MadCore.ordering_permutation(::MadCore.AMDOrdering, A::SparseMatrixCSC) = Vector{Int}(AMD.amd(A))

end # module
