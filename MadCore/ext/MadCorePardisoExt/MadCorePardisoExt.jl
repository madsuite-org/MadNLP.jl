module MadCorePardisoExt

import MadCore
import Pardiso
for name in MadCore.internal_names()
    @eval import MadCore: $name
end

include("pardiso.jl")

MadCore.create_linear_solver(::Type{MadCore.PardisoMKLSolver}, A; kwargs...) = PardisoMKLState(A; kwargs...)
MadCore.default_options(::Type{MadCore.PardisoMKLSolver}) = MadCore.default_options(PardisoMKLState)
MadCore.input_type(::Type{MadCore.PardisoMKLSolver}) = MadCore.input_type(PardisoMKLState)
MadCore.is_supported(::Type{MadCore.PardisoMKLSolver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(PardisoMKLState, T)
MadCore.create_linear_solver(::Type{MadCore.PanuaPardisoSolver}, A; kwargs...) = PanuaPardisoState(A; kwargs...)
MadCore.default_options(::Type{MadCore.PanuaPardisoSolver}) = MadCore.default_options(PanuaPardisoState)
MadCore.input_type(::Type{MadCore.PanuaPardisoSolver}) = MadCore.input_type(PanuaPardisoState)
MadCore.is_supported(::Type{MadCore.PanuaPardisoSolver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(PanuaPardisoState, T)

end # module
