# Workspace of a `MadCore.LapackSolver` on an Intel GPU, which calls oneMKL.
mutable struct OneMKLWorkspace{T}
    const info::Base.RefValue{Cint}
    const ipiv::oneVector{Int64}
    const scratchpad::oneVector{T}
    scratchpad_size::Int64
    # oneMKL calls take `sycl_queue(queue)` each time: it orders them after the Julia work
    # queued before, and the Julia work queued after them.
    const queue::oneAPI.oneL0.ZeCommandQueue
end

function MadCore.lapack_workspace(fact::oneMatrix{T}) where {T}
    queue = oneAPI.global_queue(oneAPI.context(fact), oneAPI.device(fact))
    return OneMKLWorkspace{T}(Ref{Cint}(0), oneVector{Int64}(undef, 0), oneVector{T}(undef, 0), 0, queue)
end

const OneMKLLapackSolver{T} = MadCore.LapackSolver{T, <:Any, <:Any, <:oneMatrix{T}, <:Any, OneMKLWorkspace{T}}

MadCore._get_info(M::OneMKLLapackSolver) = M.ws.info[]
MadCore.introduce(M::OneMKLLapackSolver) = "oneMKL -- ($(M.opt.lapack_algorithm))"

for (potrf, potrf_buffer, potrs, potrs_buffer, T) in
    (
        (:onemklDpotrf, :onemklDpotrf_scratchpad_size, :onemklDpotrs, :onemklDpotrs_scratchpad_size, :Float64),
        (:onemklSpotrf, :onemklSpotrf_scratchpad_size, :onemklSpotrs, :onemklSpotrs_scratchpad_size, :Float32),
    )
    @eval begin
        function MadCore.setup_cholesky!(M::OneMKLLapackSolver{$T})
            potrf_scratchpad_size = Support.$potrf_buffer(oneAPI.sycl_queue(M.ws.queue), 'L', M.n, M.n)
            potrs_scratchpad_size = Support.$potrs_buffer(oneAPI.sycl_queue(M.ws.queue), 'L', M.n, one(Int64), M.n, M.n)
            M.ws.scratchpad_size = max(potrf_scratchpad_size, potrs_scratchpad_size)
            resize!(M.ws.scratchpad, M.ws.scratchpad_size)
            return M
        end

        function MadCore.factorize_cholesky!(M::OneMKLLapackSolver{$T})
            M.ws.info[] = Support.$potrf(
                oneAPI.sycl_queue(M.ws.queue),
                'L',
                M.n,
                M.fact,
                M.n,
                M.ws.scratchpad,
                M.ws.scratchpad_size,
            )
            return M
        end

        function MadCore.solve_cholesky!(M::OneMKLLapackSolver{$T}, x::oneVector{$T})
            Support.$potrs(
                oneAPI.sycl_queue(M.ws.queue),
                'L',
                M.n,
                one(Int64),
                M.fact,
                M.n,
                x,
                M.n,
                M.ws.scratchpad,
                M.ws.scratchpad_size,
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
        function MadCore.setup_lu!(M::OneMKLLapackSolver{$T})
            resize!(M.ws.ipiv, M.n)
            getrf_scratchpad_size = Support.$getrf_buffer(oneAPI.sycl_queue(M.ws.queue), M.n, M.n, M.n)
            getrs_scratchpad_size = Support.$getrs_buffer(oneAPI.sycl_queue(M.ws.queue), 'N', M.n, one(Int64), M.n, M.n)
            M.ws.scratchpad_size = max(getrf_scratchpad_size, getrs_scratchpad_size)
            resize!(M.ws.scratchpad, M.ws.scratchpad_size)
            return M
        end

        function MadCore.factorize_lu!(M::OneMKLLapackSolver{$T})
            M.ws.info[] = Support.$getrf(
                oneAPI.sycl_queue(M.ws.queue),
                M.n,
                M.n,
                M.fact,
                M.n,
                M.ws.ipiv,
                M.ws.scratchpad,
                M.ws.scratchpad_size,
            )
            return M
        end

        function MadCore.solve_lu!(M::OneMKLLapackSolver{$T}, x::oneVector{$T})
            Support.$getrs(
                oneAPI.sycl_queue(M.ws.queue),
                'N',
                M.n,
                one(Int64),
                M.fact,
                M.n,
                M.ws.ipiv,
                x,
                M.n,
                M.ws.scratchpad,
                M.ws.scratchpad_size,
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
        function MadCore.setup_qr!(M::OneMKLLapackSolver{$T})
            resize!(M.tau, M.n)
            geqrf_scratchpad_size = Support.$geqrf_buffer(oneAPI.sycl_queue(M.ws.queue), M.n, M.n, M.n)
            ormqr_scratchpad_size = Support.$ormqr_buffer(oneAPI.sycl_queue(M.ws.queue), 'L', 'T', M.n, one(Int64), M.n, M.n, M.n)
            M.ws.scratchpad_size = max(geqrf_scratchpad_size, ormqr_scratchpad_size)
            resize!(M.ws.scratchpad, M.ws.scratchpad_size)
            return M
        end

        function MadCore.factorize_qr!(M::OneMKLLapackSolver{$T})
            M.ws.info[] = Support.$geqrf(
                oneAPI.sycl_queue(M.ws.queue),
                M.n,
                M.n,
                M.fact,
                M.n,
                M.tau,
                M.ws.scratchpad,
                M.ws.scratchpad_size,
            )
            return M
        end

        function MadCore.solve_qr!(M::OneMKLLapackSolver{$T}, x::oneVector{$T})
            # Apply Q^T to x: x = Q^T * x
            Support.$ormqr(
                oneAPI.sycl_queue(M.ws.queue),
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
                M.ws.scratchpad,
                M.ws.scratchpad_size,
            )
            # Solve R*x = Q^T*b using triangular solve
            oneMKL.trsv!('U', 'N', 'N', M.fact, x)  # upper, no-trans, non-unit diagonal
            return x
        end
    end
end

for (syevd, syevd_buffer, T) in
    (
        (:onemklDsyevd, :onemklDsyevd_scratchpad_size, :Float64),
        (:onemklSsyevd, :onemklSsyevd_scratchpad_size, :Float32),
    )
    @eval begin
        function MadCore.setup_evd!(M::OneMKLLapackSolver{$T})
            resize!(M.tau, M.n)
            resize!(M.Λ, M.n)
            M.ws.scratchpad_size = Support.$syevd_buffer(oneAPI.sycl_queue(M.ws.queue), 'V', 'L', M.n, M.n)
            resize!(M.ws.scratchpad, M.ws.scratchpad_size)
            return M
        end

        function MadCore.factorize_evd!(M::OneMKLLapackSolver{$T})
            M.ws.info[] = Support.$syevd(
                oneAPI.sycl_queue(M.ws.queue),
                'V',
                'L',
                M.n,
                M.fact,
                M.n,
                M.Λ,
                M.ws.scratchpad,
                M.ws.scratchpad_size,
            )
            return M
        end
    end
end
