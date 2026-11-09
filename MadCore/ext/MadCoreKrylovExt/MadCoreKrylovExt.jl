module MadCoreKrylovExt

import MadCore
import Krylov
for name in MadCore.internal_names()
    @eval import MadCore: $name
end

include("krylov.jl")
include("symcg.jl")

(::Type{MadCore.KrylovIterator})(kkt; kwargs...) = KrylovState(kkt; kwargs...)
MadCore.default_options(::Type{MadCore.KrylovIterator}, tol) = MadCore.default_options(KrylovState, tol)
(::Type{MadCore.SymCGIterator})(kkt; kwargs...) = SymCGState(kkt; kwargs...)
MadCore.default_options(::Type{MadCore.SymCGIterator}, tol) = MadCore.default_options(SymCGState, tol)

end # module
