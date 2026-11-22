"""
    oneMadNLP

Entry point of MadNLP on Intel GPUs. Loads MadNLPCore with the GPU KKT systems of
MadCore's GPU extensions, the oneMKL solvers and SparseDirectSolver.jl. Models on `oneVector` are solved
with [`ShiftedPreset`](@ref MadNLPCore.ShiftedPreset) by default: the augmented KKT system
with the LDLᵀ factorization of [`SDSSolver`](@ref) with diagonal pivoting.
"""
module oneMadNLP

import MadCore, MadNLPCore
import oneAPI
using oneAPI: oneVector, oneAPIBackend
# SparseDirectSolver loads MadCoreSDSExt, which implements SDSSolver.
import SparseDirectSolver
import Krylov   # GMRES refinement (KrylovIterator) of ShiftedPreset

using MadNLPCore
for name in names(MadNLPCore)
    name === :MadNLPCore && continue
    @eval export $name
end
using MadCore: LapackOneMKLSolver, SDSSolver
export LapackOneMKLSolver, SDSSolver, oneAPIBackend


end # module
