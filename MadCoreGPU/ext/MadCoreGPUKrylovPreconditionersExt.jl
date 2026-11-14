# Sparse matrix-vector products through KrylovPreconditioners' KrylovOperator, which keeps the
# sparse library's matrix descriptor, work buffer and preprocessing across calls (cuSPARSE,
# rocSPARSE, oneMKL). The sparse libraries otherwise rebuild them for every product, which on a
# small KKT system costs more than the product itself.
module MadCoreGPUKrylovPreconditionersExt

import MadCore
import LinearAlgebra: mul!, Adjoint, Transpose
import SparseArrays: AbstractSparseMatrix, nonzeros
import GPUArraysCore: AbstractGPUVector
import KrylovPreconditioners: KrylovOperator

# a device vector, or a contiguous view of one
const DenseGPUVector{T} = Union{
    AbstractGPUVector{T},
    SubArray{T, 1, <:AbstractGPUVector{T}, <:Tuple{UnitRange{Int}}, true},
}

# the sparse libraries take whole vectors: a contiguous view is passed as a vector that shares
# its memory
_whole(x::AbstractGPUVector) = x
_whole(x::SubArray) = unsafe_wrap(typeof(parent(x)), pointer(x), length(x))

# Operators by matrix identity: the matrix is not kept alive, and its finalizer marks the table
# so that the next lookup drops the entries of collected matrices. An operator is rebuilt when
# the matrix's nonzeros array is replaced. `nothing` records a matrix the package has no
# operator for.
mutable struct Entry
    matrix::WeakRef
    nzval::WeakRef
    ops::Dict{Char, Any}
end
const _OPERATORS = Dict{UInt, Entry}()
const _DIRTY = Threads.Atomic{Bool}(false)
const _LOCK = ReentrantLock()

function _operator(A::AbstractSparseMatrix, transa::Char)
    return lock(_LOCK) do
        if _DIRTY[]
            _DIRTY[] = false
            filter!(kv -> kv[2].matrix.value !== nothing, _OPERATORS)
        end
        key = objectid(A)
        entry = get(_OPERATORS, key, nothing)
        if entry === nothing || entry.matrix.value !== A || entry.nzval.value !== nonzeros(A)
            entry = Entry(WeakRef(A), WeakRef(nonzeros(A)), Dict{Char, Any}())
            _OPERATORS[key] = entry
            finalizer(_ -> (_DIRTY[] = true), A)
        end
        get!(entry.ops, transa) do
            applicable(KrylovOperator, A) || return nothing
            op = KrylovOperator(A; transa)
            # KrylovPreconditioners 0.4.0 sizes a transposed operator as A
            m, n = size(A)
            size(op) == (transa == 'N' ? (m, n) : (n, m)) ? op : nothing
        end
    end
end

function _product!(y, A::AbstractSparseMatrix, transa::Char, x, alpha, beta, Aop)
    op = _operator(A, transa)
    wy, wx = _whole(y), _whole(x)
    # y = α A x + β y needs a KrylovPreconditioners newer than 0.4.0
    (op === nothing || !applicable(mul!, wy, op, wx, alpha, beta)) && return mul!(y, Aop, x, alpha, beta)
    mul!(wy, op, wx, alpha, beta)
    return y
end

MadCore._spmv!(y::DenseGPUVector{T}, A::AbstractSparseMatrix{T}, x::DenseGPUVector{T}, alpha, beta) where {T} =
    _product!(y, A, 'N', x, alpha, beta, A)
MadCore._spmv!(y::DenseGPUVector{T}, A::Transpose{T, <:AbstractSparseMatrix{T}}, x::DenseGPUVector{T}, alpha, beta) where {T} =
    _product!(y, parent(A), 'T', x, alpha, beta, A)
MadCore._spmv!(y::DenseGPUVector{T}, A::Adjoint{T, <:AbstractSparseMatrix{T}}, x::DenseGPUVector{T}, alpha, beta) where {T} =
    _product!(y, parent(A), 'C', x, alpha, beta, A)

end # module
