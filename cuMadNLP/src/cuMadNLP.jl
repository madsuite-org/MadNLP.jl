"""
    cuMadNLP

Entry point of MadNLP on NVIDIA GPUs. Loads MadNLPCore with the GPU KKT systems of
MadCoreGPU and the cuSOLVER and cuDSS linear solvers. Models on `CuVector` are
solved with the condensed KKT system and cuDSS by default.
"""
module cuMadNLP

import MadCore, MadCoreGPU, MadNLPCore
import CUDA, CUDSS, cuBLAS, cuSOLVER, cuSPARSE
using CUDACore: CuVector
import PrecompileTools

using MadNLPCore
for name in names(MadNLPCore)
    name === :MadNLPCore && continue
    @eval export $name
end
using MadCoreGPU: LapackCUDASolver, CUDSSSolver
export LapackCUDASolver, CUDSSSolver

include("options.jl")
include("precompile.jl")

end # module
