for (potrf, potrf_buffer, potrs, nbytes, T) in
    (
        (:cusolverDnDpotrf, :cusolverDnDpotrf_bufferSize, :cusolverDnDpotrs, :8, :Float64),
        (:cusolverDnSpotrf, :cusolverDnSpotrf_bufferSize, :cusolverDnSpotrs, :4, :Float32),
    )
    @eval begin
        function MadCore.setup_cholesky!(M::CuLapackSolver{$T})
            if M.ws.legacy
                potrf_lwork_gpu = Ref{Cint}(0)
                cuSOLVER.$potrf_buffer(
                    dense_handle(),
                    CUBLAS_FILL_MODE_LOWER,
                    Cint(M.n),
                    M.fact,
                    Cint(M.n),
                    potrf_lwork_gpu,
                )
                M.ws.lwork_gpu = potrf_lwork_gpu[] * $nbytes
                resize!(M.ws.work_gpu, M.ws.lwork_gpu |> Int64)
            else
                potrf_lwork_gpu = Ref{Csize_t}(0)
                potrf_lwork_cpu = Ref{Csize_t}(0)
                cuSOLVER.cusolverDnXpotrf_bufferSize(
                    dense_handle(),
                    M.ws.params,
                    CUBLAS_FILL_MODE_LOWER,
                    M.n,
                    $T,
                    M.fact,
                    M.n,
                    $T,
                    potrf_lwork_gpu,
                    potrf_lwork_cpu,
                )
                M.ws.lwork_cpu = potrf_lwork_cpu[]
                M.ws.lwork_gpu = potrf_lwork_gpu[]
                resize!(M.ws.work_cpu, M.ws.lwork_cpu |> Int64)
                resize!(M.ws.work_gpu, M.ws.lwork_gpu |> Int64)
            end
            return M
        end

        function MadCore.factorize_cholesky!(M::CuLapackSolver{$T})
            if M.ws.legacy
                cuSOLVER.$potrf(
                    dense_handle(),
                    CUBLAS_FILL_MODE_LOWER,
                    Cint(M.n),
                    M.fact,
                    Cint(M.n),
                    M.ws.work_gpu,
                    Cint(M.ws.lwork_gpu ÷ $nbytes),
                    M.ws.info,
                )
            else
                cuSOLVER.cusolverDnXpotrf(
                    dense_handle(),
                    M.ws.params,
                    CUBLAS_FILL_MODE_LOWER,
                    M.n,
                    $T,
                    M.fact,
                    M.n,
                    $T,
                    M.ws.work_gpu,
                    M.ws.lwork_gpu,
                    M.ws.work_cpu,
                    M.ws.lwork_cpu,
                    M.ws.info,
                )
            end
            return M
        end

        function MadCore.solve_cholesky!(M::CuLapackSolver{$T}, x::CuVector{$T})
            if M.ws.legacy
                cuSOLVER.$potrs(
                    dense_handle(),
                    CUBLAS_FILL_MODE_LOWER,
                    Cint(M.n),
                    one(Cint),
                    M.fact,
                    Cint(M.n),
                    x,
                    Cint(M.n),
                    M.ws.info,
                )
            else
                cuSOLVER.cusolverDnXpotrs(
                    dense_handle(),
                    M.ws.params,
                    CUBLAS_FILL_MODE_LOWER,
                    M.n,
                    one(Int64),
                    $T,
                    M.fact,
                    M.n,
                    $T,
                    x,
                    M.n,
                    M.ws.info,
                )
            end
            return x
        end
    end
end

for (sytrf_buffer, sytrf, nbytes, T) in
    (
        (:cusolverDnDsytrf_bufferSize, :cusolverDnDsytrf, :8, :Float64),
        (:cusolverDnSsytrf_bufferSize, :cusolverDnSsytrf, :4, :Float32),
    )
    @eval begin
        function MadCore.setup_bunchkaufman!(M::CuLapackSolver{$T})
            resize!(M.ws.ipiv, M.n)
            resize!(M.ws.ipiv64, M.n)
            sytrf_lwork_gpu = Ref{Cint}(0)
            cuSOLVER.$sytrf_buffer(
                dense_handle(),
                Cint(M.n),
                M.fact,
                Cint(M.n),
                sytrf_lwork_gpu,
            )
            sytrs_lwork_cpu = Ref{Csize_t}(0)
            sytrs_lwork_gpu = Ref{Csize_t}(0)
            cuSOLVER.cusolverDnXsytrs_bufferSize(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                M.n,
                one(Int64),
                $T,
                M.fact,
                M.n,
                M.ws.ipiv64,
                $T,
                M.tau, # We can use any vector of the same length as the solution, which is M.n
                M.n,
                sytrs_lwork_gpu,
                sytrs_lwork_cpu,
            )
            M.ws.lwork_cpu = sytrs_lwork_cpu[]
            M.ws.lwork_gpu = max(sytrs_lwork_gpu[], sytrf_lwork_gpu[] * $nbytes)
            resize!(M.ws.work_cpu, M.ws.lwork_cpu |> Int64)
            resize!(M.ws.work_gpu, M.ws.lwork_gpu |> Int64)
            return M
        end

        function MadCore.factorize_bunchkaufman!(M::CuLapackSolver{$T})
            # We only have the legacy API for sytrf
            cuSOLVER.$sytrf(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.ws.ipiv,
                M.ws.work_gpu,
                Cint(M.ws.lwork_gpu ÷ $nbytes),
                M.ws.info,
            )
            return M
        end

        function MadCore.solve_bunchkaufman!(M::CuLapackSolver{$T}, x::CuVector{$T})
            copyto!(M.ws.ipiv64, M.ws.ipiv)  # No workaround possible until NVIDIA implements cusolverDnXsytrf
            cuSOLVER.cusolverDnXsytrs(
                dense_handle(),
                CUBLAS_FILL_MODE_LOWER,
                M.n,
                one(Int64),
                $T,
                M.fact,
                M.n,
                M.ws.ipiv64,
                $T,
                x,
                M.n,
                M.ws.work_gpu,
                M.ws.lwork_gpu,
                M.ws.work_cpu,
                M.ws.lwork_cpu,
                M.ws.info,
            )
            return x
        end
    end
end

for (getrf, getrf_buffer, getrs, nbytes, T) in
    (
        (:cusolverDnDgetrf, :cusolverDnDgetrf_bufferSize, :cusolverDnDgetrs, :8, :Float64),
        (:cusolverDnSgetrf, :cusolverDnSgetrf_bufferSize, :cusolverDnSgetrs, :4, :Float32),
    )
    @eval begin
        function MadCore.setup_lu!(M::CuLapackSolver{$T})
            if M.ws.legacy
                resize!(M.ws.ipiv, M.n)
                getrf_lwork_gpu = Ref{Cint}(0)
                cuSOLVER.$getrf_buffer(
                    dense_handle(),
                    Cint(M.n),
                    Cint(M.n),
                    M.fact,
                    Cint(M.n),
                    getrf_lwork_gpu,
                )
                M.ws.lwork_gpu = getrf_lwork_gpu[] * $nbytes
                resize!(M.ws.work_gpu, M.ws.lwork_gpu |> Int64)
            else
                resize!(M.ws.ipiv64, M.n)
                getrf_lwork_cpu = Ref{Csize_t}(0)
                getrf_lwork_gpu = Ref{Csize_t}(0)
                cuSOLVER.cusolverDnXgetrf_bufferSize(
                    dense_handle(),
                    M.ws.params,
                    M.n,
                    M.n,
                    $T,
                    M.fact,
                    M.n,
                    $T,
                    getrf_lwork_gpu,
                    getrf_lwork_cpu,
                )
                M.ws.lwork_cpu = getrf_lwork_cpu[]
                M.ws.lwork_gpu = getrf_lwork_gpu[]
                resize!(M.ws.work_cpu, M.ws.lwork_cpu |> Int64)
                resize!(M.ws.work_gpu, M.ws.lwork_gpu |> Int64)
            end
            return M
        end

        function MadCore.factorize_lu!(M::CuLapackSolver{$T})
            if M.ws.legacy
                cuSOLVER.$getrf(
                    dense_handle(),
                    Cint(M.n),
                    Cint(M.n),
                    M.fact,
                    Cint(M.n),
                    M.ws.work_gpu,
                    M.ws.ipiv,
                    M.ws.info,
                )
            else
                cuSOLVER.cusolverDnXgetrf(
                    dense_handle(),
                    M.ws.params,
                    M.n,
                    M.n,
                    $T,
                    M.fact,
                    M.n,
                    M.ws.ipiv64,
                    $T,
                    M.ws.work_gpu,
                    M.ws.lwork_gpu,
                    M.ws.work_cpu,
                    M.ws.lwork_cpu,
                    M.ws.info,
                )
            end
            return M
        end

        function MadCore.solve_lu!(M::CuLapackSolver{$T}, x::CuVector{$T})
            if M.ws.legacy
                cuSOLVER.$getrs(
                    dense_handle(),
                    CUBLAS_OP_N,
                    Cint(M.n),
                    one(Cint),
                    M.fact,
                    Cint(M.n),
                    M.ws.ipiv,
                    x,
                    Cint(M.n),
                    M.ws.info,
                )
            else
                cuSOLVER.cusolverDnXgetrs(
                    dense_handle(),
                    M.ws.params,
                    CUBLAS_OP_N,
                    M.n,
                    one(Int64),
                    $T,
                    M.fact,
                    M.n,
                    M.ws.ipiv64,
                    $T,
                    x,
                    M.n,
                    M.ws.info,
                )
            end
            return x
        end
    end
end

for (geqrf, geqrf_buffer, ormqr, ormqr_buffer, trsv, nbytes, T) in
    (
        (:cusolverDnDgeqrf, :cusolverDnDgeqrf_bufferSize, :cusolverDnDormqr, :cusolverDnDormqr_bufferSize, :cublasDtrsv_v2_64, :8, :Float64),
        (:cusolverDnSgeqrf, :cusolverDnSgeqrf_bufferSize, :cusolverDnSormqr, :cusolverDnSormqr_bufferSize, :cublasStrsv_v2_64, :4, :Float32),
    )
    @eval begin
        function MadCore.setup_qr!(M::CuLapackSolver{$T})
            resize!(M.tau, M.n)
            ormqr_lwork_gpu = Ref{Cint}(0)
            cuSOLVER.$ormqr_buffer(
                dense_handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_OP_T,
                Cint(M.n),
                one(Cint),
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.tau,
                M.tau,  # We can use any vector of the same length as the solution, which is M.n
                Cint(M.n),
                ormqr_lwork_gpu,
            )
            if M.ws.legacy
                geqrf_lwork_gpu = Ref{Cint}(0)
                cuSOLVER.$geqrf_buffer(
                    dense_handle(),
                    Cint(M.n),
                    Cint(M.n),
                    M.fact,
                    Cint(M.n),
                    geqrf_lwork_gpu,
                )
                M.ws.lwork_gpu = max(geqrf_lwork_gpu[], ormqr_lwork_gpu[]) * $nbytes
                resize!(M.ws.work_gpu, M.ws.lwork_gpu |> Int64)
            else
                geqrf_lwork_cpu = Ref{Csize_t}(0)
                geqrf_lwork_gpu = Ref{Csize_t}(0)
                cuSOLVER.cusolverDnXgeqrf_bufferSize(
                    dense_handle(),
                    M.ws.params,
                    M.n,
                    M.n,
                    $T,
                    M.fact,
                    M.n,
                    $T,
                    M.tau,
                    $T,
                    geqrf_lwork_gpu,
                    geqrf_lwork_cpu,
                )
                M.ws.lwork_cpu = geqrf_lwork_cpu[]
                M.ws.lwork_gpu = max(ormqr_lwork_gpu[] * $nbytes, geqrf_lwork_gpu[])
                resize!(M.ws.work_cpu, M.ws.lwork_cpu |> Int64)
                resize!(M.ws.work_gpu, M.ws.lwork_gpu |> Int64)
            end
            return M
        end

        function MadCore.factorize_qr!(M::CuLapackSolver{$T})
            if M.ws.legacy
                cuSOLVER.$geqrf(
                    dense_handle(),
                    Cint(M.n),
                    Cint(M.n),
                    M.fact,
                    Cint(M.n),
                    M.tau,
                    M.ws.work_gpu,
                    Cint(M.ws.lwork_gpu ÷ $nbytes),
                    M.ws.info,
                )
            else
                cuSOLVER.cusolverDnXgeqrf(
                    dense_handle(),
                    M.ws.params,
                    M.n,
                    M.n,
                    $T,
                    M.fact,
                    M.n,
                    $T,
                    M.tau,
                    $T,
                    M.ws.work_gpu,
                    M.ws.lwork_gpu,
                    M.ws.work_cpu,
                    M.ws.lwork_cpu,
                    M.ws.info,
                )
            end
            return M
        end

        function MadCore.solve_qr!(M::CuLapackSolver{$T}, x::CuVector{$T})
            # We only have the legacy API for ormqr
            cuSOLVER.$ormqr(
                dense_handle(),
                CUBLAS_SIDE_LEFT,
                CUBLAS_OP_T,
                Cint(M.n),
                one(Cint),
                Cint(M.n),
                M.fact,
                Cint(M.n),
                M.tau,
                x,
                Cint(M.n),
                M.ws.work_gpu,
                Cint(M.ws.lwork_gpu ÷ $nbytes),
                M.ws.info,
            )
            cuBLAS.$trsv(
                handle(),
                CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
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

for (syevd, syevd_buffer, nbytes, T) in
    (
        (:cusolverDnDsyevd, :cusolverDnDsyevd_bufferSize, :8, :Float64),
        (:cusolverDnSsyevd, :cusolverDnSsyevd_bufferSize, :4, :Float32),
    )
    @eval begin
        function MadCore.setup_evd!(M::CuLapackSolver{$T})
            resize!(M.tau, M.n)
            resize!(M.Λ, M.n)
            if M.ws.legacy
                syevd_lwork_gpu = Ref{Cint}(0)
                cuSOLVER.$syevd_buffer(
                    dense_handle(),
                    CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER,
                    Cint(M.n),
                    M.fact,
                    Cint(M.n),
                    M.Λ,
                    syevd_lwork_gpu,
                )
                M.ws.lwork_gpu = syevd_lwork_gpu[] * $nbytes
                resize!(M.ws.work_gpu, M.ws.lwork_gpu |> Int64)
            else
                syevd_lwork_cpu = Ref{Csize_t}(0)
                syevd_lwork_gpu = Ref{Csize_t}(0)
                cuSOLVER.cusolverDnXsyevd_bufferSize(
                    dense_handle(),
                    M.ws.params,
                    CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER,
                    M.n,
                    $T,
                    M.fact,
                    M.n,
                    $T,
                    M.Λ,
                    $T,
                    syevd_lwork_gpu,
                    syevd_lwork_cpu,
                )
                M.ws.lwork_cpu = syevd_lwork_cpu[]
                M.ws.lwork_gpu = syevd_lwork_gpu[]
                resize!(M.ws.work_cpu, M.ws.lwork_cpu |> Int64)
                resize!(M.ws.work_gpu, M.ws.lwork_gpu |> Int64)
            end
            return M
        end

        function MadCore.factorize_evd!(M::CuLapackSolver{$T})
            if M.ws.legacy
                cuSOLVER.$syevd(
                    dense_handle(),
                    CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER,
                    Cint(M.n),
                    M.fact,
                    Cint(M.n),
                    M.Λ,
                    M.ws.work_gpu,
                    Cint(M.ws.lwork_gpu ÷ $nbytes),
                    M.ws.info,
                )
            else
                cuSOLVER.cusolverDnXsyevd(
                    dense_handle(),
                    M.ws.params,
                    CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER,
                    M.n,
                    $T,
                    M.fact,
                    M.n,
                    $T,
                    M.Λ,
                    $T,
                    M.ws.work_gpu,
                    M.ws.lwork_gpu,
                    M.ws.work_cpu,
                    M.ws.lwork_cpu,
                    M.ws.info,
                )
            end
            return M
        end
    end
end
