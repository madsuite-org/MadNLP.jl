mutable struct LapackOneMKLState{T, MT, Alg} <: MadCore.AbstractLapackSolver{T, Alg}
    A::MT
    fact::oneMatrix{T}
    n::Int64
    sol::oneVector{T}
    tau::oneVector{T}
    Λ::oneVector{T}
    info::Base.RefValue{Cint}
    ipiv::oneVector{Int64}
    scratchpad::oneVector{T}
    scratchpad_size::Int64
    device_queue::SYCL.syclQueue_t
    alpha::Base.RefValue{T}
    beta::Base.RefValue{T}
    opt::MadCore.LapackOptions
    logger::MadCore.MadNLPLogger

    function LapackOneMKLState(
            A::MT;
            option_dict::Dict{Symbol, Any} = Dict{Symbol, Any}(),
            opt = MadCore.LapackOptions(),
            logger = MadCore.MadNLPLogger(),
            kwargs...,
        ) where {MT <: AbstractMatrix}
        MadCore.set_options!(opt, option_dict, kwargs...)
        T = eltype(A)
        m, n = size(A)
        @assert m == n
        fact = oneMatrix{T}(undef, m, n)
        sol = oneVector{T}(undef, 0)
        tau = oneVector{T}(undef, 0)
        Λ = oneVector{T}(undef, 0)
        info = Ref{Cint}(0)
        ipiv = oneVector{Int64}(undef, 0)
        scratchpad = oneVector{T}(undef, 0)
        queue = oneAPI.global_queue(oneAPI.context(fact), oneAPI.device(fact))
        device_queue = oneAPI.sycl_queue(queue)
        alpha = Ref{T}(1)
        beta = Ref{T}(0)
        alg = opt.lapack_algorithm
        solver = new{T, MT, alg}(A, fact, n, sol, tau, Λ, info, ipiv, scratchpad, 0, device_queue, alpha, beta, opt, logger)
        MadCore.setup!(solver)
        return solver
    end
end

MadCore.transfer_matrix!(M::LapackOneMKLState) = MadCoreGPU.gpu_transfer!(M.fact, M.A)
MadCore.default_options(::Type{LapackOneMKLState}) = MadCore.LapackOptions(MadCore.EVD)
MadCore.introduce(M::LapackOneMKLState) = "oneMKL -- ($(M.opt.lapack_algorithm))"
MadCore.solve!(M::LapackOneMKLState{T}, x::oneVector{T}) where {T} = MadCore._solve!(M, x)

for (potrf, potrf_buffer, potrs, potrs_buffer, T) in
    (
        (:onemklDpotrf, :onemklDpotrf_scratchpad_size, :onemklDpotrs, :onemklDpotrs_scratchpad_size, :Float64),
        (:onemklSpotrf, :onemklSpotrf_scratchpad_size, :onemklSpotrs, :onemklSpotrs_scratchpad_size, :Float32),
    )
    @eval begin
        function MadCore.setup_cholesky!(M::LapackOneMKLState{$T})
            potrf_scratchpad_size = Support.$potrf_buffer(M.device_queue, 'L', M.n, M.n)
            potrs_scratchpad_size = Support.$potrs_buffer(M.device_queue, 'L', M.n, one(Int64), M.n, M.n)
            M.scratchpad_size = max(potrf_scratchpad_size, potrs_scratchpad_size)
            resize!(M.scratchpad, M.scratchpad_size)
            return M
        end

        function MadCore.factorize_cholesky!(M::LapackOneMKLState{$T})
            M.info[] = Support.$potrf(
                M.device_queue,
                'L',
                M.n,
                M.fact,
                M.n,
                M.scratchpad,
                M.scratchpad_size,
            )
            return M
        end

        function MadCore.solve_cholesky!(M::LapackOneMKLState{$T}, x::oneVector{$T})
            Support.$potrs(
                M.device_queue,
                'L',
                M.n,
                one(Int64),
                M.fact,
                M.n,
                x,
                M.n,
                M.scratchpad,
                M.scratchpad_size,
            )
            return x
        end
    end
end

for (getrf, getrf_buffer, getrs, getrs_buffer, T) in
    (
        (:onemklDgetrf, :onemklDgetrf_scratchpad_size, :onemklDgetrs, :onemklDgetrs_scratchpad_size, :Float64),
        (:onemklSgetrf, :onemklSgetrf_scratchpad_size, :onemklSgetrs, :onemklSgetrs_scratchpad_size, :Float32),
    )
    @eval begin
        function MadCore.setup_lu!(M::LapackOneMKLState{$T})
            resize!(M.ipiv, M.n)
            getrf_scratchpad_size = Support.$getrf_buffer(M.device_queue, M.n, M.n, M.n)
            getrs_scratchpad_size = Support.$getrs_buffer(M.device_queue, 'N', M.n, one(Int64), M.n, M.n)
            M.scratchpad_size = max(getrf_scratchpad_size, getrs_scratchpad_size)
            resize!(M.scratchpad, M.scratchpad_size)
            return M
        end

        function MadCore.factorize_lu!(M::LapackOneMKLState{$T})
            M.info[] = Support.$getrf(
                M.device_queue,
                M.n,
                M.n,
                M.fact,
                M.n,
                M.ipiv,
                M.scratchpad,
                M.scratchpad_size,
            )
            return M
        end

        function MadCore.solve_lu!(M::LapackOneMKLState{$T}, x::oneVector{$T})
            Support.$getrs(
                M.device_queue,
                'N',
                M.n,
                one(Int64),
                M.fact,
                M.n,
                M.ipiv,
                x,
                M.n,
                M.scratchpad,
                M.scratchpad_size,
            )
            return x
        end
    end
end

for (geqrf, geqrf_buffer, ormqr, ormqr_buffer, trsv, T) in
    (
        (:onemklDgeqrf, :onemklDgeqrf_scratchpad_size, :onemklDormqr, :onemklDormqr_scratchpad_size, :onemklDtrsv, :Float64),
        (:onemklSgeqrf, :onemklSgeqrf_scratchpad_size, :onemklSormqr, :onemklSormqr_scratchpad_size, :onemklStrsv, :Float32),
    )
    @eval begin
        function MadCore.setup_qr!(M::LapackOneMKLState{$T})
            resize!(M.tau, M.n)
            geqrf_scratchpad_size = Support.$geqrf_buffer(M.device_queue, M.n, M.n, M.n)
            ormqr_scratchpad_size = Support.$ormqr_buffer(M.device_queue, 'L', 'T', M.n, one(Int64), M.n, M.n, M.n)
            M.scratchpad_size = max(geqrf_scratchpad_size, ormqr_scratchpad_size)
            resize!(M.scratchpad, M.scratchpad_size)
            return M
        end

        function MadCore.factorize_qr!(M::LapackOneMKLState{$T})
            M.info[] = Support.$geqrf(
                M.device_queue,
                M.n,
                M.n,
                M.fact,
                M.n,
                M.tau,
                M.scratchpad,
                M.scratchpad_size,
            )
            return M
        end

        function MadCore.solve_qr!(M::LapackOneMKLState{$T}, x::oneVector{$T})
            # Apply Q^T to x: x = Q^T * x
            Support.$ormqr(
                M.device_queue,
                'L',          # side (left multiplication)
                'T',          # trans (transpose)
                M.n,          # m
                one(Int64),   # n (single RHS)
                M.n,          # k
                M.fact,       # A
                M.n,          # lda
                M.tau,        # tau
                x,            # c (the RHS vector)
                M.n,          # ldc
                M.scratchpad,
                M.scratchpad_size,
            )
            # Solve R*x = Q^T*b using triangular solve
            oneMKL.trsv!('U', 'N', 'N', M.fact, x)  # upper, no-trans, non-unit diagonal
            return x
        end
    end
end

for (syevd, syevd_buffer, gemv, T) in
    (
        (:onemklDsyevd, :onemklDsyevd_scratchpad_size, :onemklDgemv, :Float64),
        (:onemklSsyevd, :onemklSsyevd_scratchpad_size, :onemklSgemv, :Float32),
    )
    @eval begin
        function MadCore.setup_evd!(M::LapackOneMKLState{$T})
            resize!(M.tau, M.n)
            resize!(M.Λ, M.n)
            M.scratchpad_size = Support.$syevd_buffer(M.device_queue, 'V', 'L', M.n, M.n)
            resize!(M.scratchpad, M.scratchpad_size)
            return M
        end

        function MadCore.factorize_evd!(M::LapackOneMKLState{$T})
            M.info[] = Support.$syevd(
                M.device_queue,
                'V',
                'L',
                M.n,
                M.fact,
                M.n,
                M.Λ,
                M.scratchpad,
                M.scratchpad_size,
            )
            return M
        end

        function MadCore.solve_evd!(M::LapackOneMKLState{$T}, x::oneVector{$T})
            Support.$gemv(
                M.device_queue,
                'T',
                M.n,
                M.n,
                M.alpha,
                M.fact,
                M.n,
                x,
                one(Int64),
                M.beta,
                M.tau,
                one(Int64),
            )
            M.tau ./= M.Λ
            Support.$gemv(
                M.device_queue,
                'N',
                M.n,
                M.n,
                M.alpha,
                M.fact,
                M.n,
                M.tau,
                one(Int64),
                M.beta,
                x,
                one(Int64),
            )
            return x
        end
    end
end
