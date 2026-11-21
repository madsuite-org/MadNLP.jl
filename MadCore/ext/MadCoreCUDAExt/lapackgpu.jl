# Workspace of a `MadCore.LapackSolver` on an NVIDIA GPU, which calls cuSOLVER. With
# `legacy = false`, the factorizations use the 64-bit API of cuSOLVER (`cusolverDnX*`).
mutable struct CuSolverWorkspace
    const work_gpu::CuVector{UInt8}
    lwork_gpu::Csize_t
    const work_cpu::Vector{UInt8}
    lwork_cpu::Csize_t
    const info::CuVector{Cint}
    const ipiv::CuVector{Cint}
    const ipiv64::CuVector{Int64}
    const legacy::Bool
    const params::CuSolverParameters
end

function MadCore.lapack_workspace(::CuMatrix; legacy::Bool = true)
    return CuSolverWorkspace(
        CuVector{UInt8}(undef, 0), 0, Vector{UInt8}(undef, 0), 0,
        CuVector{Cint}(undef, 1), CuVector{Cint}(undef, 0), CuVector{Int64}(undef, 0),
        legacy, CuSolverParameters(),
    )
end

const CuLapackSolver{T} = MadCore.LapackSolver{T, <:Any, <:Any, <:CuMatrix{T}, <:Any, CuSolverWorkspace}

MadCore._get_info(M::CuLapackSolver) = sum(M.ws.info)
MadCore.introduce(M::CuLapackSolver) = "cuSOLVER v$(cuSOLVER.version()) -- ($(M.opt.lapack_algorithm))"
