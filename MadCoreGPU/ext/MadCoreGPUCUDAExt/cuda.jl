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

MadCoreGPU.gpu_transfer!(y::CuMatrix{T}, x::cuSPARSE.CuSparseMatrixCSC{T}) where {T} =
    MadCoreGPU._gpu_sparse_csc_to_dense!(y, x)

#=
    MadCore._ger!
=#

MadCore._ger!(alpha::T, x::CuVector{T}, y::CuVector{T}, A::CuMatrix{T}) where {T} = cuBLAS.ger!(alpha, x, y, A)

#=
    MadCore._syr!
=#

MadCore._syr!(uplo::Char, alpha::T, x::CuVector{T}, A::CuMatrix{T}) where {T} = cuBLAS.syr!(uplo, alpha, x, A)

#=
    MadCore._symv!
=#

MadCore._symv!(uplo::Char, alpha::T, A::CuMatrix{T}, x::CuVector{T}, beta::T, y::CuVector{T}) where {T} = cuBLAS.symv!(uplo, alpha, A, x, beta, y)

#=
    MadCore._syrk!
=#

MadCore._syrk!(uplo::Char, trans::Char, alpha::T, A::CuMatrix{T}, beta::T, C::CuMatrix{T}) where {T} = cuBLAS.syrk!(uplo, trans, alpha, A, beta, C)

#=
    MadCore._trsm!
=#

MadCore._trsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::T, A::CuMatrix{T}, B::CuMatrix{T}) where {T} = cuBLAS.trsm!(side, uplo, transa, diag, alpha, A, B)

#=
    MadCore._dgmm!
=#

MadCore._dgmm!(side::Char, A::CuMatrix{T}, x::CuVector{T}, B::CuMatrix{T}) where {T} = cuBLAS.dgmm!(side, A, x, B)
