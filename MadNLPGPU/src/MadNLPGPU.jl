"""
    MadNLPGPU

Deprecated: use `cuMadNLP` on NVIDIA GPUs and `rocMadNLP` on AMD GPUs. MadNLPGPU
loads cuMadNLP and keeps the names of MadNLPGPU resolving for one release.
"""
module MadNLPGPU

import MadCore, cuMadNLP
# The MadNLPGPU of before the split loaded CUDA; keep that, so code relying on it
# (an ExaModels model on CUDABackend(), say) runs unchanged.
import CUDA

# The names of MadNLPGPU before the split resolve as `MadNLPGPU.name`: the GPU
# internals of MadCore's GPU extension, and the solver markers and orderings of MadCore.
const _GPUExt = Base.get_extension(MadCore, :MadCoreGPUExt)
for name in MadCore.defined_names(_GPUExt)
    name === :MadCore && continue
    @eval import ._GPUExt: $name
end
for name in (
        :CUDSSSolver, :LapackCUDASolver, :LapackROCmSolver, :LapackOneMKLSolver, :SDSSolver,
        :ORDERING, :DEFAULT_ORDERING, :METIS_ORDERING, :AMD_ORDERING, :USER_ORDERING,
        :SYMAMD_ORDERING, :COLAMD_ORDERING, :EventTimer,
    )
    @eval import MadCore: $name
end
using cuMadNLP
for name in names(cuMadNLP)
    name === :cuMadNLP && continue
    @eval export $name
end

function __init__()
    Base.depwarn("MadNLPGPU is deprecated; use cuMadNLP (NVIDIA) or rocMadNLP (AMD).", :MadNLPGPU)
    return
end

end # module
