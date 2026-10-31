"""
    MadNLPGPU

Deprecated: use `cuMadNLP` on NVIDIA GPUs and `rocMadNLP` on AMD GPUs. MadNLPGPU
loads cuMadNLP and keeps the names of MadNLPGPU resolving for one release.
"""
module MadNLPGPU

import MadCore, MadCoreGPU, cuMadNLP

# Every name of MadCoreGPU resolves as `MadNLPGPU.name`, as before the split.
for name in MadCore.internal_names(MadCoreGPU)
    name === :MadCore && continue
    @eval import MadCoreGPU: $name
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
