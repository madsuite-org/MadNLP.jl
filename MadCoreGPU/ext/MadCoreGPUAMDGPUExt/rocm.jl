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

MadCoreGPU.gpu_transfer!(y::ROCMatrix{T}, x::rocSPARSE.ROCSparseMatrixCSC{T}) where {T} =
    MadCoreGPU._gpu_sparse_csc_to_dense!(y, x)

#=
    MadCore._ger!
=#

MadCore._ger!(alpha::T, x::ROCVector{T}, y::ROCVector{T}, A::ROCMatrix{T}) where T = rocBLAS.ger!(alpha, x, y, A)

#=
    MadCore._syr!
=#

MadCore._syr!(uplo::Char, alpha::T, x::ROCVector{T}, A::ROCMatrix{T}) where T = rocBLAS.syr!(uplo, alpha, x, A)

#=
    MadCore._symv!
=#

MadCore._symv!(uplo::Char, alpha::T, A::ROCMatrix{T}, x::ROCVector{T}, beta::T, y::ROCVector{T}) where T = rocBLAS.symv!(uplo, alpha, A, x, beta, y)

#=
    MadCore._syrk!
=#

MadCore._syrk!(uplo::Char, trans::Char, alpha::T, A::ROCMatrix{T}, beta::T, C::ROCMatrix{T}) where T = rocBLAS.syrk!(uplo, trans, alpha, A, beta, C)

#=
    MadCore._trsm!
=#

MadCore._trsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::T, A::ROCMatrix{T}, B::ROCMatrix{T}) where T = rocBLAS.trsm!(side, uplo, transa, diag, alpha, A, B)

#=
    MadCore._dgmm!
=#

MadCore._dgmm!(side::Char, A::ROCMatrix{T}, x::ROCVector{T}, B::ROCMatrix{T}) where T = rocBLAS.dgmm!(side, A, x, B)
