#=
    SparseMatrixCSC to oneSparseMatrixCSC
=#

function oneMKL.oneSparseMatrixCSC{Tv, Ti}(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    return oneMKL.oneSparseMatrixCSC{Tv, Ti}(
        oneVector(A.colptr),
        oneVector(A.rowval),
        oneVector(A.nzval),
        size(A),
    )
end

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

#=
    LinearAlgebra.norm on oneArray.

    The norm of GPUArrays calls `norm` on scalars inside `mapreduce`, which does not compile
    to SPIR-V. The method on `SubArray` also avoids scalar indexing on views.
=#

function _onenorm(v, p::Real)
    isempty(v) && return float(zero(eltype(v)))
    if p == Inf
        return maximum(abs, v)
    elseif p == -Inf
        return minimum(abs, v)
    elseif p == 1
        return mapreduce(abs, +, v)
    elseif p == 2
        return sqrt(mapreduce(abs2, +, v))
    elseif p == 0
        return float(count(!iszero, v))
    else
        spp = float(p)
        return mapreduce(x -> abs(x)^spp, +, v)^inv(spp)
    end
end

LinearAlgebra.norm(v::oneArray{<:Number}, p::Real = 2) = _onenorm(v, p)
LinearAlgebra.norm(v::SubArray{<:Number, <:Any, <:oneArray}, p::Real = 2) = _onenorm(v, p)
