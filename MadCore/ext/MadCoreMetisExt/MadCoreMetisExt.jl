module MadCoreMetisExt

import MadCore
import Metis
import LinearAlgebra
import SparseArrays: SparseMatrixCSC

function MadCore.ordering_permutation(::Val{MadCore.METIS_ORDERING}, A::SparseMatrixCSC)
    A = A + A' - LinearAlgebra.Diagonal(A)
    G = Metis.graph(A, check_hermitian = false)
    perm, _ = Metis.permutation(G)
    return perm
end

end # module
