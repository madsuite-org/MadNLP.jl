module MadCoreAMDExt

import MadCore
import AMD
import SparseArrays: SparseMatrixCSC

MadCore.ordering_permutation(::Val{MadCore.AMD_ORDERING}, A::SparseMatrixCSC) = AMD.amd(A)
MadCore.ordering_permutation(::Val{MadCore.SYMAMD_ORDERING}, A::SparseMatrixCSC) = AMD.symamd(A)
MadCore.ordering_permutation(::Val{MadCore.COLAMD_ORDERING}, A::SparseMatrixCSC) = AMD.colamd(A)

end # module
