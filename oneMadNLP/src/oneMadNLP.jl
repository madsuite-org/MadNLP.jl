"""
    oneMadNLP

Entry point of MadNLP on Intel GPUs. Loads MadNLPCore with the GPU KKT systems of
MadCoreGPU, the oneMKL solvers and SparseDirectSolver.jl. Models on `oneVector` are solved
with the condensed KKT system and [`SDSSolver`](@ref) by default.
"""
module oneMadNLP

import MadCore, MadCoreGPU, MadNLPCore
import oneAPI
using oneAPI: oneVector, oneAPIBackend
# SparseDirectSolver loads MadCoreGPUSDSExt, which implements SDSSolver.
import SparseDirectSolver
import Krylov   # GMRES refinement (KrylovIterator) of ShiftedPreset

using MadNLPCore
for name in names(MadNLPCore)
    name === :MadNLPCore && continue
    @eval export $name
end
using MadCoreGPU: LapackOneMKLSolver, SDSSolver
export LapackOneMKLSolver, SDSSolver, oneAPIBackend

include("options.jl")

end # module
