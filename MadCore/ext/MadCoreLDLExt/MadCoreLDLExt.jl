module MadCoreLDLExt

import MadCore
import LDLFactorizations
for name in MadCore.internal_names()
    @eval import MadCore: $name
end

include("ldl.jl")

MadCore.solver_type(::Type{MadCore.LDLSolver}) = LDLState

end # module
