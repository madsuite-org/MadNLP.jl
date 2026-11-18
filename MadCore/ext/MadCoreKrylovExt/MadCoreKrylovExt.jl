module MadCoreKrylovExt

import MadCore
import Krylov
for name in MadCore.internal_names()
    @eval import MadCore: $name
end

include("krylov.jl")
include("minres.jl")

(::Type{MadCore.MinresIterator})(kkt; kwargs...) = MinresState(kkt; kwargs...)
MadCore.default_options(::Type{MadCore.MinresIterator}, tol) = MadCore.default_options(MinresState, tol)

(::Type{MadCore.KrylovIterator})(kkt; kwargs...) = KrylovState(kkt; kwargs...)
MadCore.default_options(::Type{MadCore.KrylovIterator}, tol) = MadCore.default_options(KrylovState, tol)

end # module
