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

MadCore.create_linear_solver(::Type{MadCore.MumpsSolver}, A; kwargs...) = MumpsState(A; kwargs...)
MadCore.default_options(::Type{MadCore.MumpsSolver}) = MadCore.default_options(MumpsState)
MadCore.input_type(::Type{MadCore.MumpsSolver}) = MadCore.input_type(MumpsState)
MadCore.is_supported(::Type{MadCore.MumpsSolver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(MumpsState, T)

end # module
