# Products with the GPU KKT matrices through KrylovPreconditioners' KrylovOperator, which keeps
# the sparse library's matrix descriptor, work buffer and preprocessing across calls (cuSPARSE,
# rocSPARSE, oneMKL). The sparse libraries otherwise rebuild them for every product, which on a
# small KKT system costs more than the product itself.
module MadCoreGPUKrylovPreconditionersExt

import MadCoreGPU
import LinearAlgebra: LinearAlgebra, mul!
import SparseArrays: AbstractSparseMatrix
import GPUArraysCore: AbstractGPUVector
import KrylovPreconditioners: KrylovOperator

"""
    KrylovViewOperator

A `KrylovOperator` that also takes contiguous views of device vectors, as the KKT systems pass
them: the operator itself takes whole vectors only.
"""
struct KrylovViewOperator{O}
    op::O
end

_whole(x::AbstractGPUVector) = x
_whole(x::SubArray) = unsafe_wrap(typeof(parent(x)), pointer(x), length(x))

function LinearAlgebra.mul!(y::AbstractVector, A::KrylovViewOperator, x::AbstractVector, alpha::Number, beta::Number)
    mul!(_whole(y), A.op, _whole(x), alpha, beta)
    return y
end
LinearAlgebra.mul!(y::AbstractVector, A::KrylovViewOperator, x::AbstractVector) = mul!(y, A, x, true, false)

function MadCoreGPU.gpu_operator(A::AbstractSparseMatrix, transa::Char)
    fallback = transa == 'N' ? A : transpose(A)
    applicable(KrylovOperator, A) || return fallback
    return KrylovViewOperator(KrylovOperator(A; transa))
end

end # module
