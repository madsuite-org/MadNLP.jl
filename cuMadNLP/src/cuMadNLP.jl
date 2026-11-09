"""
    cuMadNLP

Entry point of MadNLP on NVIDIA GPUs. Loads MadNLPCore with the GPU KKT systems of
MadCoreGPU and the cuSOLVER and cuDSS linear solvers. Models on `CuVector` are
solved by default with the augmented KKT system and cuDSS, after Ruiz scaling and a dual
shift of the factorized matrix, with conjugate-gradient refinement, the quality-function barrier rule and
slack reset: the default preset, [`ShiftedPreset`](@ref).
"""
module cuMadNLP

import MadCore, MadCoreGPU, MadNLPCore
import CUDA, CUDSS, cuBLAS, cuSOLVER, cuSPARSE
using CUDACore: CuVector, CUDABackend
import PrecompileTools
import Krylov   # GMRES refinement (KrylovIterator)

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
