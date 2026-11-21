module MadCoreAMDGPUExt

import LinearAlgebra
import SparseArrays: nonzeros, nnz
import LinearAlgebra: Symmetric

import MadCore

import KernelAbstractions: synchronize, get_backend

using AMDGPU
using AMDGPU.rocBLAS, AMDGPU.rocSOLVER, AMDGPU.rocSPARSE

include("rocm_sparse.jl")
include("rocsolver.jl")
include("rocm.jl")
include("timer.jl")

MadCore.solver_type(::Type{MadCore.LapackROCmSolver}) = MadCore.LapackSolver
MadCore.default_options(::Type{MadCore.LapackROCmSolver}) = MadCore.LapackOptions(MadCore.EVD)

# Default linear solvers of models on AMD GPUs (SDSSolver needs SparseDirectSolver loaded).
MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:ROCVector{T}}) where {T} = MadCore.SDSSolver
MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:ROCVector{T}}) where {T} = MadCore.LapackROCmSolver

end
