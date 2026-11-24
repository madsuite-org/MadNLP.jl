module MadCoreLDLExt

import MadCore
import LDLFactorizations
for name in MadCore.public_names()
    @eval import MadCore: $name
end
import Base: @kwdef, size
import LinearAlgebra: ldiv!
import MadCore: get_tril_to_full
import SparseArrays: SparseMatrixCSC

include("ldl.jl")

MadCore.solver_type(::Type{MadCore.LDLSolver}) = LDLState

end # module
