module MadCoreGPUAMDGPUExt

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric

import MadCore
import MadCoreGPU

import KernelAbstractions: synchronize, get_backend

using AMDGPU
using AMDGPU.rocBLAS, AMDGPU.rocSOLVER, AMDGPU.rocSPARSE

include("rocm_sparse.jl")
include("rocsolver.jl")
include("rocm.jl")
include("timer.jl")

MadCore.create_linear_solver(::Type{MadCoreGPU.LapackROCmSolver}, A; kwargs...) = LapackROCmState(A; kwargs...)
MadCore.default_options(::Type{MadCoreGPU.LapackROCmSolver}) = MadCore.default_options(LapackROCmState)
MadCore.input_type(::Type{MadCoreGPU.LapackROCmSolver}) = MadCore.input_type(LapackROCmState)
MadCore.is_supported(::Type{MadCoreGPU.LapackROCmSolver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(LapackROCmState, T)

end
