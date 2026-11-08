"""
    cuMadNLP

Entry point of MadNLP on NVIDIA GPUs. Loads MadNLPCore with the GPU KKT systems of
MadCoreGPU and the cuSOLVER and cuDSS linear solvers. Models on `CuVector` are
solved by default with the augmented KKT system and cuDSS, after Ruiz scaling and a dual
shift of the factorized matrix, with GMRES refinement, and with the quality-function barrier
rule and slack reset that the GPU option preset sets for the augmented system.
"""
module cuMadNLP

import MadCore, MadCoreGPU, MadNLPCore
import CUDA, CUDSS, cuBLAS, cuSOLVER, cuSPARSE
using CUDACore: CuVector, CUDABackend
import PrecompileTools
import Krylov   # GMRES refinement (KrylovIterator) of the default options

using MadNLPCore
for name in names(MadNLPCore)
    name === :MadNLPCore && continue
    @eval export $name
end
using MadCoreGPU: LapackCUDASolver, CUDSSSolver
export LapackCUDASolver, CUDSSSolver, CUDABackend

include("options.jl")
include("precompile.jl")

end # module
