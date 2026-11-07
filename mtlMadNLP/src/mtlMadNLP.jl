"""
    mtlMadNLP

Entry point of MadNLP on Apple GPUs. Loads MadNLPCore with the GPU KKT systems of MadCoreGPU
and SparseDirectSolver.jl. Models on `MtlVector` are solved with the condensed KKT system and
[`SDSSolver`](@ref). Metal has no `Float64`, so models use `Float32`.
"""
module mtlMadNLP

import MadCore, MadCoreGPU, MadNLPCore
# GPUArrays is a trigger of MadCoreGPUMetalExt, which defines the sparse matrix type.
import GPUArrays, Metal
using Metal: MtlVector, MetalBackend
import SparseDirectSolver

using MadNLPCore
for name in names(MadNLPCore)
    name === :MadNLPCore && continue
    @eval export $name
end
export SDSSolver, SDSSolverOptions, MetalBackend

include("sds.jl")
include("options.jl")

end # module
