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
    MadCore.create_linear_solver(::Type{MadCore.UmfpackSolver}, A; kwargs...) = UmfpackState(A; kwargs...)
    MadCore.default_options(::Type{MadCore.UmfpackSolver}) = MadCore.default_options(UmfpackState)
    MadCore.input_type(::Type{MadCore.UmfpackSolver}) = MadCore.input_type(UmfpackState)
    MadCore.is_supported(::Type{MadCore.UmfpackSolver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(UmfpackState, T)
    MadCore.create_linear_solver(::Type{MadCore.CHOLMODSolver}, A; kwargs...) = CHOLMODState(A; kwargs...)
    MadCore.default_options(::Type{MadCore.CHOLMODSolver}) = MadCore.default_options(CHOLMODState)
    MadCore.input_type(::Type{MadCore.CHOLMODSolver}) = MadCore.input_type(CHOLMODState)
    MadCore.is_supported(::Type{MadCore.CHOLMODSolver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(CHOLMODState, T)
end

end # module
