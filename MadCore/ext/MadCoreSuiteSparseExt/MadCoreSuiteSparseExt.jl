module MadCoreSuiteSparseExt

import MadCore
import SuiteSparse
for name in MadCore.internal_names()
    @eval import MadCore: $name
end

# UMFPACK and CHOLMOD are only available if Julia was built with the GPL libraries.
@static if Base.USE_GPL_LIBS
    import SuiteSparse: UMFPACK, CHOLMOD
    include("umfpack.jl")
    include("cholmod.jl")
    MadCore.solver_type(::Type{MadCore.UmfpackSolver}) = UmfpackState
    MadCore.solver_type(::Type{MadCore.CHOLMODSolver}) = CHOLMODState
end

end # module
