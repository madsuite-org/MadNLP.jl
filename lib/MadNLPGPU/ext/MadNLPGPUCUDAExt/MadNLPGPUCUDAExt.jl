module MadNLPGPUCUDAExt

import MadNLP
import MadNLPGPU
import MadCoreGPU: CUDSSSolver, LapackCUDASolver
using CUDACore
import CUDSS, cuBLAS, cuSOLVER, cuSPARSE
import PrecompileTools

include("options.jl")
include("precompile.jl")

end # module
