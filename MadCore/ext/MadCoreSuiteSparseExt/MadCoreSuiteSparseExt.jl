module MadCoreSuiteSparseExt

import MadCore
import SuiteSparse
for name in MadCore.public_names()
    @eval import MadCore: $name
end
import MadCore: improve!, input_type, is_factorized, is_inertia
import Base: @kwdef, finalize, size
import LinearAlgebra: issuccess
import MadCore: CHOLESKY, InertiaException, LDL, LinearFactorization, get_tril_to_full
import SparseArrays: SparseMatrixCSC, nnz, sparse

# UMFPACK and CHOLMOD are only available if Julia was built with the GPL libraries.
@static if Base.USE_GPL_LIBS
    import SuiteSparse: UMFPACK, CHOLMOD
    include("umfpack.jl")
    include("cholmod.jl")
    MadCore.solver_type(::Type{MadCore.UmfpackSolver}) = UmfpackState
    MadCore.solver_type(::Type{MadCore.CHOLMODSolver}) = CHOLMODState
end

end # module
