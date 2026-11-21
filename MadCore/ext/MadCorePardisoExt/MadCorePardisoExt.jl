module MadCorePardisoExt

import MadCore
import Pardiso
for name in MadCore.internal_names()
    @eval import MadCore: $name
end

include("pardiso.jl")

MadCore.solver_type(::Type{MadCore.PardisoMKLSolver}) = PardisoMKLState
MadCore.solver_type(::Type{MadCore.PanuaPardisoSolver}) = PanuaPardisoState

end # module
