module MadCoreLDLExt

import MadCore
import LDLFactorizations
for name in MadCore.public_names()
    @eval import MadCore: $name
end
import Base: @kwdef, size
import LinearAlgebra: ldiv!
import MadCore: SubVector
import SparseArrays: SparseMatrixCSC

include("ldl.jl")

MadCore.create_linear_solver(::Type{MadCore.LDLSolver}, A; kwargs...) = LDLState(A; kwargs...)
MadCore.default_options(::Type{MadCore.LDLSolver}) = MadCore.default_options(LDLState)
MadCore.input_type(::Type{MadCore.LDLSolver}) = MadCore.input_type(LDLState)
MadCore.is_supported(::Type{MadCore.LDLSolver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(LDLState, T)

end # module
