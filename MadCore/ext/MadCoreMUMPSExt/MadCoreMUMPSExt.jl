module MadCoreMUMPSExt

import MadCore
import MUMPS_seq_jll, OpenBLAS32_jll
import LinearAlgebra: BLAS
for name in MadCore.internal_names()
    @eval import MadCore: $name
end

include("mumps.jl")

function __init__()
    # MUMPS_seq_jll calls BLAS with 32-bit integers through libblastrampoline.
    config = BLAS.lbt_get_config()
    if !any(lib -> lib.interface == :lp64, config.loaded_libs)
        BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
    end
    return
end

MadCore.solver_type(::Type{MadCore.MumpsSolver}) = MumpsState

end # module
