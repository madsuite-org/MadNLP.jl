"""
    rocMadNLP

Entry point of MadNLP on AMD GPUs. Loads MadNLPCore with the GPU KKT systems of
MadCoreGPU, the rocSOLVER solvers and SparseDirectSolver.jl. Models on `ROCVector` are
solved with [`ShiftedPreset`](@ref MadNLPCore.ShiftedPreset) by default: the augmented KKT
system with the static LDLᵀ factorization of [`SDSSolver`](@ref).
"""
module rocMadNLP

import MadCore, MadCoreGPU, MadNLPCore
import AMDGPU
using AMDGPU: ROCVector, ROCBackend
import PrecompileTools
# SparseDirectSolver loads MadCoreGPUSDSExt, which implements SDSSolver.
import SparseDirectSolver
import Krylov   # CG refinement (SymCGIterator) of ShiftedPreset

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
