"""
    rocMadNLP

Entry point of MadNLP on AMD GPUs. Loads MadNLPCore with the GPU KKT systems of
MadCoreGPU and the rocSOLVER solvers. Models on `ROCVector` are solved with the
condensed KKT system and rocSOLVER by default.
"""
module rocMadNLP

import MadCore, MadCoreGPU, MadNLPCore
import AMDGPU
using AMDGPU: ROCVector
import PrecompileTools

using MadNLPCore
for name in names(MadNLPCore)
    name === :MadNLPCore && continue
    @eval export $name
end
using MadCoreGPU: LapackROCmSolver
export LapackROCmSolver

include("options.jl")
include("precompile.jl")

end # module
