module MadCorePardisoExt

import MadCore
import Pardiso
for name in MadCore.public_names()
    @eval import MadCore: $name
end
import Base: @kwdef
import MadCore: BUNCHKAUFMAN, CHOLESKY, FactorizationException, LDL, LinearFactorization,
    PanuaPardisoSolver, SymbolicException, blas_num_threads
import SparseArrays: SparseMatrixCSC

include("pardiso.jl")

MadCore.solver_type(::Type{MadCore.PardisoMKLSolver}) = PardisoMKLState
MadCore.solver_type(::Type{MadCore.PanuaPardisoSolver}) = PanuaPardisoState

end # module
