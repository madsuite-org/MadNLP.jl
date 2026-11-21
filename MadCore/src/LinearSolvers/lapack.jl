mutable struct LapackCPUWorkspace{T}
    work::Vector{T}
    lwork::BlasInt
    iwork::Vector{BlasInt}
    liwork::BlasInt
    const info::Base.RefValue{BlasInt}
    const ipiv::Vector{BlasInt}
end

"""
    LapackCPUSolver

Dense LAPACK solver on the host: a [`LapackSolver`](@ref) whose workspace calls the LAPACK of
`libblastrampoline`. `LapackCPUSolver(A; opt, logger)` copies `A`, which may be a host or a
GPU array, into a host matrix before each factorization.
"""
const LapackCPUSolver{T, MT, Alg} = LapackSolver{T, MT, Alg, Matrix{T}, Vector{T}, LapackCPUWorkspace{T}}

lapack_workspace(::Matrix{T}) where {T} = LapackCPUWorkspace{T}(
    Vector{T}(undef, 1), BlasInt(-1), Vector{BlasInt}(undef, 1), BlasInt(-1), Ref{BlasInt}(0), Vector{BlasInt}(undef, 0),
)

LapackCPUSolver(A::AbstractMatrix; kwargs...) = LapackSolver(A, Matrix{eltype(A)}(undef, size(A)); kwargs...)

_get_info(M::LapackCPUSolver) = M.ws.info[]
supports_bunchkaufman_inertia(::LapackCPUSolver) = true
inertia_bunchkaufman(M::LapackCPUSolver) = inertia(M.fact, M.ws.ipiv, M.ws.info[])

for (potrf, potrs, getrf, getrs, sytrf, sytrs, geqrf, ormqr, trsv, syevd, T) in
    (
        (:spotrf_, :spotrs_, :sgetrf_, :sgetrs_, :ssytrf_, :ssytrs_, :sgeqrf_, :sormqr_, :strsv_, :ssyevd_, :Float32),
        (:dpotrf_, :dpotrs_, :dgetrf_, :dgetrs_, :dsytrf_, :dsytrs_, :dgeqrf_, :dormqr_, :dtrsv_, :dsyevd_, :Float64),
    )
    @eval begin
        # potrf
        function $potrf(uplo, n, a, lda, info)
            return ccall((@blasfunc($potrf), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong),
                          uplo, n, a, lda, info, 1)
        end

        # potrs
        function $potrs(uplo, n, nrhs, a, lda, b, ldb, info)
            return ccall((@blasfunc($potrs), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T},
                          Ref{BlasInt}, Ref{BlasInt}, Clong),
                          uplo, n, nrhs, a, lda, b, ldb, info, 1)
        end

        # getrf
        function $getrf(m, n, a, lda, ipiv, info)
            return ccall((@blasfunc($getrf), libblastrampoline), Cvoid,
                         (Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
                          m, n, a, lda, ipiv, info)
        end

        # getrs
        function $getrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info)
            return ccall((@blasfunc($getrs), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt},
                          Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong),
                          trans, n, nrhs, a, lda, ipiv, b, ldb, info, 1)
        end

        # sytrf
        function $sytrf(uplo, n, a, lda, ipiv, work, lwork, info)
            return ccall((@blasfunc($sytrf), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$T},
                          Ref{BlasInt}, Ref{BlasInt}, Clong),
                          uplo, n, a, lda, ipiv, work, lwork, info, 1)
        end

        # sytrs
        function $sytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb, info)
            return ccall((@blasfunc($sytrs), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt},
                          Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong), uplo, n, nrhs, a, lda, ipiv,
                          b, ldb, info, 1)
        end

        # geqrf
        function $geqrf(m, n, a, lda, tau, work, lwork, info)
            return ccall((@blasfunc($geqrf), libblastrampoline), Cvoid,
                         (Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ptr{$T},
                          Ref{BlasInt}, Ref{BlasInt}),
                          m, n, a, lda, tau, work, lwork, info)
        end

        # ormqr
        function $ormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info)
            return ccall((@blasfunc($ormqr), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T},
                          Ref{BlasInt}, Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                          Ref{BlasInt}, Clong, Clong),
                          side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, 1, 1)
        end

        # trsv
        function $trsv(uplo, trans, diag, n, a, lda, x, incx)
            return ccall((@blasfunc($trsv), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                          Ptr{$T}, Ref{BlasInt}, Clong, Clong, Clong),
                          uplo, trans, diag, n, a, lda, x, incx, 1, 1, 1)
        end

        # syevd
        function $syevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info)
            return ccall((@blasfunc($syevd), libblastrampoline), Cvoid,
                         (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T},
                          Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
                          jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, 1, 1)
        end

        function setup_cholesky!(M::LapackCPUSolver{$T})
            return M
        end

        function factorize_cholesky!(M::LapackCPUSolver{$T})
            $potrf('L', M.n, M.fact, M.n, M.ws.info)
            return M
        end

        function solve_cholesky!(M::LapackCPUSolver{$T}, x::Vector{$T})
            $potrs('L', M.n, one(BlasInt), M.fact, M.n, x, M.n, M.ws.info)
            return x
        end

        function setup_bunchkaufman!(M::LapackCPUSolver{$T})
            resize!(M.ws.ipiv, M.n)
            $sytrf('L', M.n, M.fact, M.n, M.ws.ipiv, M.ws.work, M.ws.lwork, M.ws.info)
            buffer_size_sytrf = M.ws.work[1] |> BlasInt
            M.ws.lwork = buffer_size_sytrf
            resize!(M.ws.work, M.ws.lwork)
            return M
        end

        function factorize_bunchkaufman!(M::LapackCPUSolver{$T})
            $sytrf('L', M.n, M.fact, M.n, M.ws.ipiv, M.ws.work, M.ws.lwork, M.ws.info)
            return M
        end

        function solve_bunchkaufman!(M::LapackCPUSolver{$T}, x::Vector{$T})
            $sytrs('L', M.n, one(BlasInt), M.fact, M.n, M.ws.ipiv, x, M.n, M.ws.info)
            return x
        end

        function setup_lu!(M::LapackCPUSolver{$T})
            resize!(M.ws.ipiv, M.n)
            return M
        end

        function factorize_lu!(M::LapackCPUSolver{$T})
            $getrf(M.n, M.n, M.fact, M.n, M.ws.ipiv, M.ws.info)
            return M
        end

        function solve_lu!(M::LapackCPUSolver{$T}, x::Vector{$T})
            $getrs('N', M.n, one(BlasInt), M.fact, M.n, M.ws.ipiv, x, M.n, M.ws.info)
            return x
        end

        function setup_qr!(M::LapackCPUSolver{$T})
            resize!(M.tau, M.n)
            $geqrf(M.n, M.n, M.fact, M.n, M.tau, M.ws.work, M.ws.lwork, M.ws.info)
            buffer_size_geqrf = M.ws.work[1] |> BlasInt
            $ormqr('L', 'T', M.n, one(BlasInt), M.n, M.fact, M.n, M.tau, M.tau, M.n, M.ws.work, M.ws.lwork, M.ws.info)
            buffer_size_ormqr = M.ws.work[1] |> BlasInt
            M.ws.lwork = max(buffer_size_geqrf, buffer_size_ormqr)
            resize!(M.ws.work, M.ws.lwork)
            return M
        end

        function factorize_qr!(M::LapackCPUSolver{$T})
            $geqrf(M.n, M.n, M.fact, M.n, M.tau, M.ws.work, M.ws.lwork, M.ws.info)
            return M
        end

        function solve_qr!(M::LapackCPUSolver{$T}, x::Vector{$T})
            $ormqr('L', 'T', M.n, one(BlasInt), M.n, M.fact, M.n, M.tau, x, M.n, M.ws.work, M.ws.lwork, M.ws.info)
            $trsv('U', 'N', 'N' , M.n, M.fact, M.n, x, one(BlasInt))
            return x
        end

        function setup_evd!(M::LapackCPUSolver{$T})
            resize!(M.tau, M.n)
            resize!(M.Λ, M.n)
            $syevd('V', 'L', M.n, M.fact, M.n, M.Λ, M.ws.work, M.ws.lwork, M.ws.iwork, M.ws.liwork, M.ws.info)
            buffer_size_syevd = M.ws.work[1] |> BlasInt
            buffer2_size_syevd = M.ws.iwork[1] |> BlasInt
            M.ws.lwork = buffer_size_syevd
            M.ws.liwork = buffer2_size_syevd
            resize!(M.ws.work, M.ws.lwork)
            resize!(M.ws.iwork, M.ws.liwork)
            return M
        end

        function factorize_evd!(M::LapackCPUSolver{$T})
            $syevd('V', 'L', M.n, M.fact, M.n, M.Λ, M.ws.work, M.ws.lwork, M.ws.iwork, M.ws.liwork, M.ws.info)
            return M
        end
    end
end

introduce(M::LapackCPUSolver) = "Lapack-CPU ($(M.opt.lapack_algorithm))"

function inertia(fact, ipiv, info)
    numneg = num_neg_ev(size(fact,1), fact, ipiv)
    numzero = info > 0 ? 1 : 0
    numpos = size(fact,1) - numneg - numzero
    return (numpos, numzero, numneg)
end

function num_neg_ev(n, D, ipiv)
    numneg = 0
    t = 0
    for k=1:n
        d = D[k,k]
        if ipiv[k] < 0
            if t == 0
                t = abs(D[k+1,k])
                d = (d/t) * D[k+1,k+1] - t
            else
                d = t
                t = 0
            end
        end
        d < 0 && (numneg += 1)
        if d == 0
            numneg = -1
            break
        end
    end
    return numneg
end

create_linear_solver(::Type{LapackCPUSolver}, A; kwargs...) = LapackCPUSolver(A; kwargs...)
