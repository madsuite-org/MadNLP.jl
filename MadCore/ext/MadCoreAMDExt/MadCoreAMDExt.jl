module MadCoreAMDExt

import MadCore
import AMD
import SparseArrays: SparseMatrixCSC

MadCore.ordering_permutation(::Val{MadCore.AMD_ORDERING}, A, perm) = AMD.amd(SparseMatrixCSC(A))
MadCore.ordering_permutation(::Val{MadCore.SYMAMD_ORDERING}, A, perm) = AMD.symamd(SparseMatrixCSC(A))
MadCore.ordering_permutation(::Val{MadCore.COLAMD_ORDERING}, A, perm) = AMD.colamd(SparseMatrixCSC(A))

end # module
