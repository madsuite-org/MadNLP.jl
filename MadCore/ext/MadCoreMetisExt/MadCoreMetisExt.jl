module MadCoreMetisExt

import MadCore
import Metis
import LinearAlgebra: Symmetric
import SparseArrays: SparseMatrixCSC

# nested dissection of the graph of the symmetric matrix whose lower triangle is `A`
function MadCore.ordering_permutation(::MadCore.MetisOrdering, A::SparseMatrixCSC)
    perm, _ = Metis.permutation(Symmetric(A, :L))
    return Vector{Int}(perm)
end

end # module
