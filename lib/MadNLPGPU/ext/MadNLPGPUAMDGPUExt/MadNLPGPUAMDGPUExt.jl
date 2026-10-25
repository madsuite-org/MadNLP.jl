module MadNLPGPUAMDGPUExt

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric

import MadNLP
import MadNLPGPU

import KernelAbstractions: synchronize, get_backend

using AMDGPU
using AMDGPU.rocBLAS, AMDGPU.rocSOLVER, AMDGPU.rocSPARSE
import PrecompileTools

function __init__()
    setglobal!(MadNLPGPU, :LapackROCmSolver, LapackROCmSolver)
    return
end

include("rocm_sparse.jl")
include("rocsolver.jl")
include("rocm.jl")
include("precompile.jl")

end
