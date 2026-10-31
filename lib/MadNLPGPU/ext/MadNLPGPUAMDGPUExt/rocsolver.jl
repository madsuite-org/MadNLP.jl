mutable struct LapackROCmSolver{T, MT, Alg} <: MadNLP.AbstractLapackSolver{T, Alg}
    A::MT
    fact::ROCMatrix{T}
    n::Int64
    sol::ROCVector{T}
    tau::ROCVector{T}
    Λ::ROCVector{T}
    # The `_64` rocSOLVER routines (getrf, potrf) write a 64-bit `info`;
    # syevd uses the 32-bit API and writes a `rocblas_int`.
    info::ROCVector{Int64}
    info32::ROCVector{Cint}
    ipiv::ROCVector{Int64}
    alpha::Base.RefValue{T}
    beta::Base.RefValue{T}
    opt::MadNLP.LapackOptions
    logger::MadNLP.MadNLPLogger

    function LapackROCmSolver(
        A::MT;
        option_dict::Dict{Symbol,Any} = Dict{Symbol,Any}(),
        opt = MadNLP.LapackOptions(),
        logger = MadNLP.MadNLPLogger(),
        kwargs...,
    ) where {MT<:AbstractMatrix}
        MadNLP.set_options!(opt, option_dict, kwargs...)
        T = eltype(A)
        m,n = size(A)
        @assert m == n
        fact = ROCMatrix{T}(undef, m, n)
        sol = ROCVector{T}(undef, 0)
        tau = ROCVector{T}(undef, 0)
        Λ = ROCVector{T}(undef, 0)
        info = AMDGPU.zeros(Int64, 1)
        info32 = AMDGPU.zeros(Cint, 1)
        ipiv = ROCVector{Int64}(undef, 0)
        alpha = Ref{T}(1)
        beta = Ref{T}(0)
        alg = opt.lapack_algorithm
        solver = new{T, MT, alg}(A, fact, n, sol, tau, Λ, info, info32, ipiv, alpha, beta, opt, logger)
        MadNLP.setup!(solver)
        return solver
    end
end

MadNLP.transfer_matrix!(M::LapackROCmSolver) = MadNLPGPU.gpu_transfer!(M.fact, M.A)
MadNLP._get_info(M::LapackROCmSolver) = sum(M.info)
MadNLP.default_options(::Type{LapackROCmSolver}) = MadNLP.LapackOptions(MadNLP.EVD)
MadNLP.introduce(M::LapackROCmSolver) = "rocSOLVER v$(rocSOLVER.version()) -- ($(M.opt.lapack_algorithm))"
MadNLP.solve!(M::LapackROCmSolver{T}, x::ROCVector{T}) where {T} = MadNLP._solve!(M, x)

for (potrf, potrs, T) in
    ((:rocsolver_dpotrf_64, :rocsolver_dpotrs_64, :Float64),
     (:rocsolver_spotrf_64, :rocsolver_spotrs_64, :Float32))
    @eval begin
        MadNLP.setup_cholesky!(M::LapackROCmSolver{$T}) = M

        function MadNLP.factorize_cholesky!(M::LapackROCmSolver{$T})
            rocSOLVER.$potrf(
                rocBLAS.handle(),
                rocBLAS.rocblas_fill_lower,
                M.n,
                M.fact,
                M.n,
                M.info,
            )
            return M
        end

        function MadNLP.solve_cholesky!(M::LapackROCmSolver{$T}, x::ROCVector{$T})
            rocSOLVER.$potrs(
                rocBLAS.handle(),
                rocBLAS.rocblas_fill_lower,
                M.n,
                one(Int64),
                M.fact,
                M.n,
                x,
                M.n,
            )
            return x
        end
    end
end

for (getrf, getrs, T) in
    ((:rocsolver_dgetrf_64, :rocsolver_dgetrs_64, :Float64),
     (:rocsolver_sgetrf_64, :rocsolver_sgetrs_64, :Float32))
    @eval begin
        function MadNLP.setup_lu!(M::LapackROCmSolver{$T})
            resize!(M.ipiv, M.n)
            return M
        end

        function MadNLP.factorize_lu!(M::LapackROCmSolver{$T})
            rocSOLVER.$getrf(
                rocBLAS.handle(),
                M.n,
                M.n,
                M.fact,
                M.n,
                M.ipiv,
                M.info,
            )
            return M
        end

        function MadNLP.solve_lu!(M::LapackROCmSolver{$T}, x::ROCVector{$T})
            rocSOLVER.$getrs(
                rocBLAS.handle(),
                rocBLAS.rocblas_operation_none,
                M.n,
                one(Int64),
                M.fact,
                M.n,
                M.ipiv,
                x,
                M.n,
            )
            return x
        end
    end
end

for (geqrf, ormqr, trsv, T) in
    ((:rocsolver_dgeqrf_64, :rocsolver_dormqr, :rocblas_dtrsv_64, :Float64),
     (:rocsolver_sgeqrf_64, :rocsolver_sormqr, :rocblas_strsv_64, :Float32))
    @eval begin
        function MadNLP.setup_qr!(M::LapackROCmSolver{$T})
            resize!(M.tau, M.n)
            return M
        end

        function MadNLP.factorize_qr!(M::LapackROCmSolver{$T})
            rocSOLVER.$geqrf(
                rocBLAS.handle(),
                M.n,
                M.n,
                M.fact,
                M.n,
                M.tau,
            )
            return M
        end

        function MadNLP.solve_qr!(M::LapackROCmSolver{$T}, x::ROCVector{$T})
            rocSOLVER.$ormqr(
                rocBLAS.handle(),
                rocBLAS.rocblas_side_left,
                rocBLAS.rocblas_operation_transpose,
                Cint(M.n),
                Cint(1),
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.tau,
                x,
                Cint(M.n),
            )
            rocBLAS.$trsv(
                rocBLAS.handle(),
                rocBLAS.rocblas_fill_upper,
                rocBLAS.rocblas_operation_none,
                rocBLAS.rocblas_diagonal_non_unit,
                M.n,
                M.fact,
                M.n,
                x,
                one(Int64),
            )
            return x
        end
    end
end

for (syevd, gemv, T) in
    ((:rocsolver_dsyevd, :rocblas_dgemv_64, :Float64),
     (:rocsolver_ssyevd, :rocblas_sgemv_64, :Float32))
    @eval begin
        function MadNLP.setup_evd!(M::LapackROCmSolver{$T})
            resize!(M.tau, M.n)
            resize!(M.Λ, M.n)
            return M
        end

        function MadNLP.factorize_evd!(M::LapackROCmSolver{$T})
            rocSOLVER.$syevd(
                rocBLAS.handle(),
                rocSOLVER.rocblas_evect_original,
                rocBLAS.rocblas_fill_lower,
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.Λ,
                M.tau,
                M.info32,
            )
            return M
        end

        function MadNLP.solve_evd!(M::LapackROCmSolver{$T}, x::ROCVector{$T})
            rocBLAS.$gemv(
                rocBLAS.handle(),
                rocBLAS.rocblas_operation_transpose,
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
            rocBLAS.$gemv(
                rocBLAS.handle(),
                rocBLAS.rocblas_operation_none,
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
