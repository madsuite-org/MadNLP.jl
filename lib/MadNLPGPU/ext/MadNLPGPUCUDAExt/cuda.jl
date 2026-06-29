#=
    MadNLP.MadNLPOptions
=#

function MadNLP.MadNLPOptions{T}(
        nlp::MadNLP.AbstractNLPModel{T, VT};
        dense_callback = MadNLP.is_dense_callback(nlp),
        callback = dense_callback ? MadNLP.DenseCallback : MadNLP.SparseCallback,
        kkt_system = dense_callback ? MadNLP.DenseCondensedKKTSystem : MadNLP.SparseCondensedKKTSystem,
        linear_solver = dense_callback ? LapackCUDASolver : CUDSSSolver,
        tol = MadNLP.get_tolerance(T, kkt_system),
        # See MadNLP.MadNLPOptions in src/IPM/options.jl for why condensed systems (Sparse + Schur)
        # relax by `tol` while the rest use 1e-8. Kept identical on CPU / CUDA / ROCm.
        bound_relax_factor = (kkt_system <: Union{MadNLP.SparseCondensedKKTSystem, MadNLP.SchurComplementCondensedKKTSystem}) ? tol : T(1.0e-8),
    ) where {T, VT <: CuVector{T}}
    return MadNLP.MadNLPOptions{T}(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
    )
end

function MadNLP.default_options(::MadNLP.AbstractNLPModel{T, VT}, ::Type{MadNLP.SparseCondensedKKTSystem}, linear_solver::Type{CUDSSSolver}) where {T, VT <: CuVector{T}}
    opt = MadNLP.default_options(linear_solver)
    # MadNLP.set_options!(opt, Dict(:cudss_algorithm => MadNLP.CHOLESKY)) # commented out due to issue #539

    return opt
end

#=
    SparseMatrixCSC to CuSparseMatrixCSC
=#

function cuSPARSE.CuSparseMatrixCSC{Tv, Ti}(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    return cuSPARSE.CuSparseMatrixCSC{Tv, Ti}(
        CuVector(A.colptr),
        CuVector(A.rowval),
        CuVector(A.nzval),
        size(A),
    )
end

#=
    CuSparseMatrixCSC to CuMatrix
=#

MadNLPGPU.gpu_transfer!(y::CuMatrix{T}, x::cuSPARSE.CuSparseMatrixCSC{T}) where {T} =
    MadNLPGPU._gpu_sparse_csc_to_dense!(y, x)

#=
    MadNLP._ger!
=#

MadNLP._ger!(alpha::T, x::CuVector{T}, y::CuVector{T}, A::CuMatrix{T}) where {T} = cuBLAS.ger!(alpha, x, y, A)

#=
    MadNLP._syr!
=#

MadNLP._syr!(uplo::Char, alpha::T, x::CuVector{T}, A::CuMatrix{T}) where {T} = cuBLAS.syr!(uplo, alpha, x, A)

#=
    MadNLP._symv!
=#

MadNLP._symv!(uplo::Char, alpha::T, A::CuMatrix{T}, x::CuVector{T}, beta::T, y::CuVector{T}) where {T} = cuBLAS.symv!(uplo, alpha, A, x, beta, y)

#=
    MadNLP._syrk!
=#

MadNLP._syrk!(uplo::Char, trans::Char, alpha::T, A::CuMatrix{T}, beta::T, C::CuMatrix{T}) where {T} = cuBLAS.syrk!(uplo, trans, alpha, A, beta, C)

#=
    MadNLP._trsm!
=#

MadNLP._trsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::T, A::CuMatrix{T}, B::CuMatrix{T}) where {T} = cuBLAS.trsm!(side, uplo, transa, diag, alpha, A, B)

#=
    MadNLP._dgmm!
=#

MadNLP._dgmm!(side::Char, A::CuMatrix{T}, x::CuVector{T}, B::CuMatrix{T}) where {T} = cuBLAS.dgmm!(side, A, x, B)
