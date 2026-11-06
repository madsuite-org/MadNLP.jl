module MadCoreKrylovExt

import MadCore
import Krylov
for name in MadCore.internal_names()
    @eval import MadCore: $name
end

include("krylov.jl")

(::Type{MadCore.KrylovIterator})(kkt; kwargs...) = KrylovState(kkt; kwargs...)
MadCore.default_options(::Type{MadCore.KrylovIterator}, tol) = MadCore.default_options(KrylovState, tol)

end # module
