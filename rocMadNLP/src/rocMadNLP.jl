"""
    rocMadNLP

Entry point of MadNLP on AMD GPUs. Loads MadNLP with the GPU KKT systems of
MadCoreGPU and the rocSOLVER solvers. Models on `ROCVector` are solved with the
condensed KKT system and rocSOLVER by default.
"""
module rocMadNLP

import MadCore, MadCoreGPU, MadNLPCore, MadNLP
import AMDGPU
using AMDGPU: ROCVector
import PrecompileTools

using MadNLP
for name in names(MadNLP)
    name === :MadNLP && continue
    @eval export $name
end
using MadCoreGPU: LapackROCmSolver
export LapackROCmSolver

include("options.jl")
include("precompile.jl")

end # module
