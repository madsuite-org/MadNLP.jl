mutable struct LapackCUDASolver{T, MT, Alg} <: MadNLP.AbstractLapackSolver{T, Alg}
    A::MT
    fact::CuMatrix{T}
    n::Int64
    sol::CuVector{T}
    tau::CuVector{T}
    Λ::CuVector{T}
    work_gpu::CuVector{UInt8}
    lwork_gpu::Csize_t
    work_cpu::Vector{UInt8}
    lwork_cpu::Csize_t
    info::CuVector{Cint}
    ipiv::CuVector{Cint}
    ipiv64::CuVector{Int64}
    opt::MadNLP.LapackOptions
    logger::MadNLP.MadNLPLogger
    legacy::Bool
    params::CuSolverParameters

    function LapackCUDASolver(
            A::MT;
            option_dict::Dict{Symbol, Any} = Dict{Symbol, Any}(),
            opt = MadNLP.LapackOptions(),
            logger = MadNLP.MadNLPLogger(),
            legacy::Bool = true,
            kwargs...,
        ) where {MT <: AbstractMatrix}
        MadNLP.set_options!(opt, option_dict, kwargs...)
        T = eltype(A)
        m, n = size(A)
        @assert m == n
        fact = CuMatrix{T}(undef, m, n)
        sol = CuVector{T}(undef, 0)
        tau = CuVector{T}(undef, 0)
        Λ = CuVector{T}(undef, 0)
        work_gpu = CuVector{UInt8}(undef, 0)
        lwork_gpu = zero(Int64)
        work_cpu = Vector{UInt8}(undef, 0)
        lwork_cpu = zero(Int64)
        info = CuVector{Cint}(undef, 1)
        ipiv = CuVector{Cint}(undef, 0)
        ipiv64 = CuVector{Int64}(undef, 0)
        params = CuSolverParameters()
        alg = opt.lapack_algorithm
        solver = new{T, MT, alg}(
            A, fact, n, sol, tau, Λ, work_gpu, lwork_gpu, work_cpu, lwork_cpu,
            info, ipiv, ipiv64, opt, logger, legacy, params
        )
        MadNLP.setup!(solver)
        return solver
    end
end

MadNLP.transfer_matrix!(M::LapackCUDASolver) = MadNLPGPU.gpu_transfer!(M.fact, M.A)
MadNLP._get_info(M::LapackCUDASolver) = sum(M.info)
MadNLP.solve!(M::LapackCUDASolver{T}, x::CuVector{T}) where {T} = MadNLP._solve!(M, x)
MadNLP.introduce(M::LapackCUDASolver) = "cuSOLVER v$(cuSOLVER.version()) -- ($(M.opt.lapack_algorithm))"
