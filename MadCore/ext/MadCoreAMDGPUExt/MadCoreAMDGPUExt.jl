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

MadCore.solver_type(::Type{MadCore.LapackROCmSolver}) = LapackROCmState

end
