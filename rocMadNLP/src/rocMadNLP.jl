"""
    rocMadNLP

Entry point of MadNLP on AMD GPUs. Loads MadNLPCore with the GPU KKT systems of
MadCoreGPU, the rocSOLVER solvers and SparseDirectSolver.jl. Models on `ROCVector` are
solved with the condensed KKT system and [`SDSSolver`](@ref) by default.
"""
module rocMadNLP

import MadCore, MadCoreGPU, MadNLPCore
import AMDGPU
using AMDGPU: ROCVector, ROCBackend
import PrecompileTools
# SparseDirectSolver loads MadCoreGPUSDSExt, which implements SDSSolver.
import SparseDirectSolver
import Krylov   # GMRES refinement (KrylovIterator) of ShiftedPreset

using MadNLPCore
for name in names(MadNLPCore)
    name === :MadNLPCore && continue
    @eval export $name
end
using MadCoreGPU: LapackROCmSolver, SDSSolver
export LapackROCmSolver, SDSSolver, ROCBackend

include("options.jl")
include("precompile.jl")

end # module
