"""
    mtlMadNLP

Entry point of MadNLP on Apple GPUs. Loads MadNLPCore with the GPU KKT systems of MadCoreGPU
and SparseDirectSolver.jl. Models on `MtlVector` are solved with
[`ShiftedPreset`](@ref MadNLPCore.ShiftedPreset) by default: the augmented KKT system with the
static LDLᵀ factorization of [`SDSSolver`](@ref). Metal has no `Float64`, so models use
`Float32`, with the shift and the pivot bound of
[`preset_perturbation`](@ref MadNLPCore.preset_perturbation)`(Float32)`.
"""
module mtlMadNLP

import MadCore, MadCoreGPU, MadNLPCore
# GPUArrays is a trigger of MadCoreGPUMetalExt, which defines the sparse matrix type.
import GPUArrays, Metal
using Metal: MtlVector, MetalBackend
# SparseDirectSolver loads MadCoreGPUSDSExt, which implements SDSSolver.
import SparseDirectSolver
import Krylov   # GMRES refinement (KrylovIterator) of ShiftedPreset

using MadNLPCore
for name in names(MadNLPCore)
    name === :MadNLPCore && continue
    @eval export $name
end
using MadCoreGPU: SDSSolver
export SDSSolver, MetalBackend

include("options.jl")

end # module
