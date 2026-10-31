module MadNLPGPU

import MadCore, MadCoreGPU, MadNLP
import PrecompileTools

# MadNLPGPU bundles MadNLP with the GPU KKT systems and linear solvers of MadCoreGPU.
# For backward compatibility, every name of MadCoreGPU also resolves as
# `MadNLPGPU.name`, and MadNLPGPU re-exports the names exported by MadNLP.
for name in MadCore.internal_names(MadCoreGPU)
    name === :MadCore && continue
    @eval import MadCoreGPU: $name
end
for name in names(MadNLP)
    name === :MadNLP && continue
    @eval using MadNLP: $name
    @eval export $name
end
export LapackCUDASolver, CUDSSSolver, LapackROCmSolver

end # module
