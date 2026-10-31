module MadNLPGPUAMDGPUExt

import MadNLP
import MadNLPGPU
using AMDGPU
import PrecompileTools

include("options.jl")
include("precompile.jl")

end # module
