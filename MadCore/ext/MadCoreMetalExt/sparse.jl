#=
    Sparse matrix on Metal

    Metal has no sparse matrix type. `MtlSparseMatrixCSC` stores the matrix in compressed
    columns and keeps a compressed row view of the same nonzeros, so that both `A * x` and
    `Aᵀ * x` run without atomics.
=#

struct MtlSparseMatrixCSC{Tv, Ti} <: GPUArrays.AbstractGPUSparseMatrixCSC{Tv, Ti}
    colPtr::MtlVector{Ti}
    rowVal::MtlVector{Ti}
    nzVal::MtlVector{Tv}
    dims::NTuple{2, Int}
    rowPtr::MtlVector{Ti}
    colIdx::MtlVector{Ti}
    perm::MtlVector{Ti} # the k-th entry in compressed rows is nzVal[perm[k]]
end

function MtlSparseMatrixCSC(colptr::AbstractVector, rowval::AbstractVector, nzval::MtlVector{Tv}, dims) where {Tv}
    Ti = eltype(colptr)
    cp, rv = Array(colptr), Array(rowval)
    m, n = dims
    positions = SparseMatrixCSC(m, n, cp, rv, collect(Ti, 1:length(rv)))
    rows = copy(transpose(positions))
    return MtlSparseMatrixCSC{Tv, Ti}(
        MtlVector(cp), MtlVector(rv), nzval, (Int(m), Int(n)),
        MtlVector(Ti.(rows.colptr)), MtlVector(Ti.(rows.rowval)), MtlVector(Ti.(rows.nzval)),
    )
end

function MtlSparseMatrixCSC{Tv, Ti}(colptr, rowval, nzval, dims) where {Tv, Ti}
    return MtlSparseMatrixCSC(Ti.(Array(colptr)), Ti.(Array(rowval)), MtlVector{Tv}(nzval), dims)
end
MtlSparseMatrixCSC{Tv, Ti}(A::SparseMatrixCSC) where {Tv, Ti} =
    MtlSparseMatrixCSC{Tv, Ti}(A.colptr, A.rowval, A.nzval, size(A))

Base.size(A::MtlSparseMatrixCSC) = A.dims
SparseArrays.nnz(A::MtlSparseMatrixCSC) = length(A.nzVal)
SparseArrays.nonzeros(A::MtlSparseMatrixCSC) = A.nzVal
Base.copy(A::MtlSparseMatrixCSC{Tv, Ti}) where {Tv, Ti} = MtlSparseMatrixCSC{Tv, Ti}(
    copy(A.colPtr), copy(A.rowVal), copy(A.nzVal), A.dims, copy(A.rowPtr), copy(A.colIdx), copy(A.perm),
)
SparseArrays.SparseMatrixCSC(A::MtlSparseMatrixCSC) =
    SparseMatrixCSC(A.dims..., Array(A.colPtr), Array(A.rowVal), Array(A.nzVal))

@kernel function _rows_mul_kernel!(y, @Const(rowptr), @Const(colidx), @Const(perm), @Const(nzval), @Const(x), alpha, beta, beta0::Bool)
    i = @index(Global)
    s = zero(eltype(y))
    @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
        s += nzval[perm[k]] * x[colidx[k]]
    end
    @inbounds y[i] = beta0 ? alpha * s : alpha * s + beta * y[i]
end

@kernel function _columns_mul_kernel!(y, @Const(colptr), @Const(rowval), @Const(nzval), @Const(x), alpha, beta, beta0::Bool)
    j = @index(Global)
    s = zero(eltype(y))
    @inbounds for k in colptr[j]:(colptr[j + 1] - 1)
        s += nzval[k] * x[rowval[k]]
    end
    @inbounds y[j] = beta0 ? alpha * s : alpha * s + beta * y[j]
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::MtlSparseMatrixCSC{T}, x::AbstractVector, alpha::Number, beta::Number) where {T}
    size(A, 1) == 0 && return y
    _rows_mul_kernel!(MetalBackend())(
        y, A.rowPtr, A.colIdx, A.perm, A.nzVal, x, T(alpha), T(beta), iszero(beta);
        ndrange = size(A, 1),
    )
    return y
end

function LinearAlgebra.mul!(
        y::AbstractVector{T},
        At::Union{Adjoint{<:Any, <:MtlSparseMatrixCSC{T}}, Transpose{<:Any, <:MtlSparseMatrixCSC{T}}},
        x::AbstractVector,
        alpha::Number,
        beta::Number,
    ) where {T}
    A = parent(At)
    size(A, 2) == 0 && return y
    _columns_mul_kernel!(MetalBackend())(
        y, A.colPtr, A.rowVal, A.nzVal, x, T(alpha), T(beta), iszero(beta);
        ndrange = size(A, 2),
    )
    return y
end

# Metal sorts plain numbers only. The symbolic setup sorts vectors of tuples once, on the host.
MadCore._sort_tuples!(v::MtlVector; kw...) = (length(v) > 0 && copyto!(v, sort!(Array(v); kw...)); v)
MadCore._sortperm_tuples(v::MtlVector; kw...) = MtlVector(sortperm(Array(v); kw...))
MadCore._permute_tuples!(v::MtlVector, p::MtlVector) = (length(v) > 0 && copyto!(v, v[p]); v)
