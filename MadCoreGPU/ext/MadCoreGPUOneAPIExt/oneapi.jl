#=
    oneSparseMatrixCSC to oneMatrix
=#

MadCoreGPU.gpu_transfer!(y::oneMatrix{T}, x::oneMKL.oneSparseMatrixCSC{T}) where {T} =
    MadCoreGPU._gpu_sparse_csc_to_dense!(y, x)

#=
    BLAS wrappers
=#

MadCore._ger!(alpha::T, x::oneVector{T}, y::oneVector{T}, A::oneMatrix{T}) where {T} = oneMKL.ger!(alpha, x, y, A)
MadCore._syr!(uplo::Char, alpha::T, x::oneVector{T}, A::oneMatrix{T}) where {T} = oneMKL.syr!(uplo, alpha, x, A)
MadCore._symv!(uplo::Char, alpha::T, A::oneMatrix{T}, x::oneVector{T}, beta::T, y::oneVector{T}) where {T} = oneMKL.symv!(uplo, alpha, A, x, beta, y)
MadCore._syrk!(uplo::Char, trans::Char, alpha::T, A::oneMatrix{T}, beta::T, C::oneMatrix{T}) where {T} = oneMKL.syrk!(uplo, trans, alpha, A, beta, C)
MadCore._trsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::T, A::oneMatrix{T}, B::oneMatrix{T}) where {T} = oneMKL.trsm!(side, uplo, transa, diag, alpha, A, B)
MadCore._dgmm!(side::Char, A::oneMatrix{T}, x::oneVector{T}, B::oneMatrix{T}) where {T} = oneMKL.dgmm!(side, A, x, B)
