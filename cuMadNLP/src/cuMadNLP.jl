"""
    cuMadNLP

Entry point of MadNLP on NVIDIA GPUs. Loads MadNLP with the GPU KKT systems of
MadCoreGPU and the cuSOLVER and cuDSS linear solvers. Models on `CuVector` are
solved with the condensed KKT system and cuDSS by default.
"""
module cuMadNLP

import MadCore, MadCoreGPU, MadNLPCore, MadNLP
import CUDA, CUDSS, cuBLAS, cuSOLVER, cuSPARSE
using CUDACore: CuVector
import PrecompileTools

using MadNLP
for name in names(MadNLP)
    name === :MadNLP && continue
    @eval export $name
end
using MadCoreGPU: LapackCUDASolver, CUDSSSolver
export LapackCUDASolver, CUDSSSolver

include("options.jl")
include("precompile.jl")

end # module
