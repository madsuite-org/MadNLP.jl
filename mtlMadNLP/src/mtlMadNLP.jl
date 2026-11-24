"""
    mtlMadNLP

Entry point of MadNLP on Apple GPUs. Loads MadNLPCore with the GPU KKT systems of MadCore
and SparseDirectSolver.jl. Models on `MtlVector` are solved with
[`ShiftedPreset`](@ref MadNLPCore.ShiftedPreset) by default: the augmented KKT system with the
LDLᵀ factorization of [`SDSSolver`](@ref) with diagonal pivoting. Metal has no `Float64`, so models use
`Float32`, with the shift and the pivot bound of
[`preset_perturbation`](@ref MadNLPCore.preset_perturbation)`(Float32)`.
Dense models are solved with `LapackCPUSolver`, which factorizes on the host. Load
AppleAccelerate to run it on Accelerate, and build dense models in `Metal.SharedStorage` to
avoid copies to the host (see the Apple silicon tutorial).
"""
module mtlMadNLP

import MadCore, MadNLPCore
# GPUArrays is a trigger of MadCoreMetalExt, which defines the sparse matrix type.
import GPUArrays, Metal
using Metal: MtlVector, MetalBackend
# SparseDirectSolver loads MadCoreSDSExt, which implements SDSSolver.
import SparseDirectSolver
import Krylov   # GMRES refinement (KrylovIterator) of ShiftedPreset

using MadNLPCore
for name in names(MadNLPCore)
    name === :MadNLPCore && continue
    @eval export $name
end
using MadCore: SDSSolver
export SDSSolver, MetalBackend


end # module
