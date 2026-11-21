module MadCoreAMDGPUExt

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric

import MadCore

import KernelAbstractions: synchronize, get_backend

using AMDGPU
using AMDGPU.rocBLAS, AMDGPU.rocSOLVER, AMDGPU.rocSPARSE

include("rocm_sparse.jl")
include("rocsolver.jl")
include("rocm.jl")
include("timer.jl")

MadCore.create_linear_solver(::Type{MadCore.LapackROCmSolver}, A; kwargs...) = LapackROCmState(A; kwargs...)
MadCore.default_options(::Type{MadCore.LapackROCmSolver}) = MadCore.default_options(LapackROCmState)
MadCore.input_type(::Type{MadCore.LapackROCmSolver}) = MadCore.input_type(LapackROCmState)
MadCore.is_supported(::Type{MadCore.LapackROCmSolver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(LapackROCmState, T)

end
