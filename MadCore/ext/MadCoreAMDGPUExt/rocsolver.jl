# Workspace of a `MadCore.LapackSolver` on an AMD GPU, which calls rocSOLVER.
struct RocSolverWorkspace
    # The `_64` rocSOLVER routines (getrf, potrf) write a 64-bit `info`;
    # syevd uses the 32-bit API and writes a `rocblas_int`.
    info::ROCVector{Int64}
    info32::ROCVector{Cint}
    ipiv::ROCVector{Int64}
end

MadCore.lapack_workspace(::ROCMatrix) = RocSolverWorkspace(AMDGPU.zeros(Int64, 1), AMDGPU.zeros(Cint, 1), ROCVector{Int64}(undef, 0))

const RocLapackSolver{T} = MadCore.LapackSolver{T, <:Any, <:Any, <:ROCMatrix{T}, <:Any, RocSolverWorkspace}

MadCore._get_info(M::RocLapackSolver) = sum(M.ws.info)
MadCore.introduce(M::RocLapackSolver) = "rocSOLVER v$(rocSOLVER.version()) -- ($(M.opt.lapack_algorithm))"

for (potrf, potrs, T) in
    (
        (:rocsolver_dpotrf_64, :rocsolver_dpotrs_64, :Float64),
        (:rocsolver_spotrf_64, :rocsolver_spotrs_64, :Float32),
    )
    @eval begin
        MadCore.setup_cholesky!(M::RocLapackSolver{$T}) = M

        function MadCore.factorize_cholesky!(M::RocLapackSolver{$T})
            rocSOLVER.$potrf(
                rocBLAS.handle(),
                rocBLAS.rocblas_fill_lower,
                M.n,
                M.fact,
                M.n,
                M.ws.info,
            )
            return M
        end

        function MadCore.solve_cholesky!(M::RocLapackSolver{$T}, x::ROCVector{$T})
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
    (
        (:rocsolver_dgetrf_64, :rocsolver_dgetrs_64, :Float64),
        (:rocsolver_sgetrf_64, :rocsolver_sgetrs_64, :Float32),
    )
    @eval begin
        function MadCore.setup_lu!(M::RocLapackSolver{$T})
            resize!(M.ws.ipiv, M.n)
            return M
        end

        function MadCore.factorize_lu!(M::RocLapackSolver{$T})
            rocSOLVER.$getrf(
                rocBLAS.handle(),
                M.n,
                M.n,
                M.fact,
                M.n,
                M.ws.ipiv,
                M.ws.info,
            )
            return M
        end

        function MadCore.solve_lu!(M::RocLapackSolver{$T}, x::ROCVector{$T})
            rocSOLVER.$getrs(
                rocBLAS.handle(),
                rocBLAS.rocblas_operation_none,
                M.n,
                one(Int64),
                M.fact,
                M.n,
                M.ws.ipiv,
                x,
                M.n,
            )
            return x
        end
    end
end

for (geqrf, ormqr, trsv, T) in
    (
        (:rocsolver_dgeqrf_64, :rocsolver_dormqr, :rocblas_dtrsv_64, :Float64),
        (:rocsolver_sgeqrf_64, :rocsolver_sormqr, :rocblas_strsv_64, :Float32),
    )
    @eval begin
        function MadCore.setup_qr!(M::RocLapackSolver{$T})
            resize!(M.tau, M.n)
            return M
        end

        function MadCore.factorize_qr!(M::RocLapackSolver{$T})
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

        function MadCore.solve_qr!(M::RocLapackSolver{$T}, x::ROCVector{$T})
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

for (syevd, T) in
    (
        (:rocsolver_dsyevd, :Float64),
        (:rocsolver_ssyevd, :Float32),
    )
    @eval begin
        function MadCore.setup_evd!(M::RocLapackSolver{$T})
            resize!(M.tau, M.n)
            resize!(M.Λ, M.n)
            return M
        end

        function MadCore.factorize_evd!(M::RocLapackSolver{$T})
            rocSOLVER.$syevd(
                rocBLAS.handle(),
                rocSOLVER.rocblas_evect_original,
                rocBLAS.rocblas_fill_lower,
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.Λ,
                M.tau,
                M.ws.info32,
            )
            return M
        end
    end
end
