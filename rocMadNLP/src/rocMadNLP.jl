"""
    rocMadNLP

Entry point of MadNLP on AMD GPUs. Loads MadNLPCore with the GPU KKT systems of
MadCore's GPU extensions, the rocSOLVER solvers and SparseDirectSolver.jl. Models on `ROCVector` are
solved with [`ShiftedPreset`](@ref MadNLPCore.ShiftedPreset) by default: the augmented KKT
system with the LDLᵀ factorization of [`SDSSolver`](@ref) with diagonal pivoting.
"""
module rocMadNLP

import MadCore, MadNLPCore
import AMDGPU
using AMDGPU: ROCVector, ROCBackend
import PrecompileTools
# SparseDirectSolver loads MadCoreSDSExt, which implements SDSSolver.
import SparseDirectSolver
import Krylov   # GMRES refinement (KrylovIterator) of ShiftedPreset

using MadNLPCore
for name in names(MadNLPCore)
    name === :MadNLPCore && continue
    @eval export $name
end
using MadCore: LapackROCmSolver, SDSSolver
export LapackROCmSolver, SDSSolver, ROCBackend

include("precompile.jl")

end # module
