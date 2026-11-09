module MadCoreGPUOneAPIExt

import LinearAlgebra
import SparseArrays: SparseMatrixCSC
import LinearAlgebra: Symmetric

import MadCore
import MadCoreGPU

using oneAPI
using oneAPI.oneMKL, oneAPI.Support

include("oneapi_sparse.jl")
include("onemkl.jl")
include("oneapi.jl")
include("scaling.jl")

MadCore.create_linear_solver(::Type{MadCoreGPU.LapackOneMKLSolver}, A; kwargs...) = LapackOneMKLState(A; kwargs...)
MadCore.default_options(::Type{MadCoreGPU.LapackOneMKLSolver}) = MadCore.default_options(LapackOneMKLState)
MadCore.input_type(::Type{MadCoreGPU.LapackOneMKLSolver}) = MadCore.input_type(LapackOneMKLState)
MadCore.is_supported(::Type{MadCoreGPU.LapackOneMKLSolver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(LapackOneMKLState, T)

end
