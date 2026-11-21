module MadCoreMetisExt

import MadCore
import Metis
import LinearAlgebra
import SparseArrays: SparseMatrixCSC

function MadCore.ordering_permutation(::Val{MadCore.METIS_ORDERING}, A, perm)
    L = SparseMatrixCSC(A)
    G = Metis.graph(L + L' - LinearAlgebra.Diagonal(L), check_hermitian = false)
    p, _ = Metis.permutation(G)
    return p
end

end # module
