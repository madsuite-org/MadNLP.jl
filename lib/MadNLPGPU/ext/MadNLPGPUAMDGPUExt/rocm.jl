#=
    MadNLP.MadNLPOptions
=#

function MadNLP.MadNLPOptions{T}(
    nlp::MadNLP.AbstractNLPModel{T,VT};
    dense_callback = MadNLP.is_dense_callback(nlp),
    callback = dense_callback ? MadNLP.DenseCallback : MadNLP.SparseCallback,
    kkt_system = dense_callback ? MadNLP.DenseCondensedKKTSystem : MadNLP.SparseCondensedKKTSystem,
    linear_solver = MadNLPGPU.LapackROCmSolver,
    tol = MadNLP.get_tolerance(T,kkt_system),
    # See MadNLP.MadNLPOptions in src/IPM/options.jl for why condensed systems (Sparse + Schur)
    # relax by `tol` while the rest use 1e-8. Kept identical on CPU / CUDA / ROCm.
    bound_relax_factor = (kkt_system <: Union{MadNLP.SparseCondensedKKTSystem, MadNLP.SchurComplementCondensedKKTSystem}) ? tol : T(1.0e-8),
) where {T, VT <: ROCVector{T}}
    return MadNLP.MadNLPOptions{T}(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
    )
end

#=
    SparseMatrixCSC to ROCSparseMatrixCSC
=#

function rocSPARSE.ROCSparseMatrixCSC{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    return rocSPARSE.ROCSparseMatrixCSC{Tv,Ti}(
        ROCVector(A.colptr),
        ROCVector(A.rowval),
        ROCVector(A.nzval),
        size(A),
    )
end

#=
    ROCSparseMatrixCSC to ROCMatrix
=#

MadNLPGPU.gpu_transfer!(y::ROCMatrix{T}, x::rocSPARSE.ROCSparseMatrixCSC{T}) where {T} =
    MadNLPGPU._gpu_sparse_csc_to_dense!(y, x)

#=
    MadNLP._ger!
=#

MadNLP._ger!(alpha::T, x::ROCVector{T}, y::ROCVector{T}, A::ROCMatrix{T}) where T = rocBLAS.ger!(alpha, x, y, A)

#=
    MadNLP._syr!
=#

MadNLP._syr!(uplo::Char, alpha::T, x::ROCVector{T}, A::ROCMatrix{T}) where T = rocBLAS.syr!(uplo, alpha, x, A)

#=
    MadNLP._symv!
=#

MadNLP._symv!(uplo::Char, alpha::T, A::ROCMatrix{T}, x::ROCVector{T}, beta::T, y::ROCVector{T}) where T = rocBLAS.symv!(uplo, alpha, A, x, beta, y)

#=
    MadNLP._syrk!
=#

MadNLP._syrk!(uplo::Char, trans::Char, alpha::T, A::ROCMatrix{T}, beta::T, C::ROCMatrix{T}) where T = rocBLAS.syrk!(uplo, trans, alpha, A, beta, C)

#=
    MadNLP._trsm!
=#

MadNLP._trsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::T, A::ROCMatrix{T}, B::ROCMatrix{T}) where T = rocBLAS.trsm!(side, uplo, transa, diag, alpha, A, B)

#=
    MadNLP._dgmm!
=#

MadNLP._dgmm!(side::Char, A::ROCMatrix{T}, x::ROCVector{T}, B::ROCMatrix{T}) where T = rocBLAS.dgmm!(side, A, x, B)
