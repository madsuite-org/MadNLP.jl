module MadCoreKrylovExt

import MadCore
import Krylov
for name in MadCore.public_names()
    @eval import MadCore: $name
end
import Base: @kwdef, copyto!, size
import LinearAlgebra: ldiv!, mul!, norm
import MadCore: AbstractIterator, solve_refine!
import Printf: @sprintf

include("krylov.jl")

(::Type{MadCore.KrylovIterator})(kkt; kwargs...) = KrylovState(kkt; kwargs...)
MadCore.default_options(::Type{MadCore.KrylovIterator}, tol) = MadCore.default_options(KrylovState, tol)

end # module
