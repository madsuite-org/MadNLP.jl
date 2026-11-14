# Products with the GPU KKT matrices through KrylovPreconditioners' KrylovOperator, which keeps
# the sparse library's matrix descriptor, work buffer and preprocessing across calls (cuSPARSE,
# rocSPARSE, oneMKL). The sparse libraries otherwise rebuild them for every product, which on a
# small KKT system costs more than the product itself.
module MadCoreGPUKrylovPreconditionersExt

import MadCoreGPU
import LinearAlgebra: LinearAlgebra, mul!
import SparseArrays: AbstractSparseMatrix, nonzeros
import GPUArraysCore: AbstractGPUVector
import KrylovPreconditioners: KrylovOperator

"""
    KeptOperator

A `KrylovOperator` that also takes contiguous views of device vectors, as the KKT systems pass
them: the operator itself takes whole vectors only.
"""
struct KeptOperator{O}
    op::O
end

_whole(x::AbstractGPUVector) = x
_whole(x::SubArray) = unsafe_wrap(typeof(parent(x)), pointer(x), length(x))

function LinearAlgebra.mul!(y::AbstractVector, A::KeptOperator, x::AbstractVector, alpha::Number, beta::Number)
    mul!(_whole(y), A.op, _whole(x), alpha, beta)
    return y
end
LinearAlgebra.mul!(y::AbstractVector, A::KeptOperator, x::AbstractVector) = mul!(y, A, x, true, false)

function MadCoreGPU.kept_op(A::AbstractSparseMatrix{T}, transa::Char) where {T}
    fallback = transa == 'N' ? A : transa == 'T' ? transpose(A) : adjoint(A)
    applicable(KrylovOperator, A) || return fallback
    op = KrylovOperator(A; transa)
    m, n = size(A)
    vec = similar(nonzeros(A), 0)
    # y = α A x + β y and transposed operators of rectangular matrices need
    # KrylovPreconditioners newer than 0.4.0
    (size(op) == (transa == 'N' ? (m, n) : (n, m)) && hasmethod(mul!, Tuple{typeof(vec), typeof(op), typeof(vec), T, T})) ||
        return fallback
    return KeptOperator(op)
end

end # module
