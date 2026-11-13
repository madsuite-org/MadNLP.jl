# Sparse matrix-vector products with a plan kept across calls. cuSPARSE's mv! builds the
# matrix descriptor, allocates the work buffer and repeats the preprocessing on every call;
# on a small KKT system those cost more than the product itself. A plan holds all three for
# one matrix and one operation, and is rebuilt when the matrix arrays change.

struct SpMVPlan{T}
    desc::cuSPARSE.CuSparseMatrixDescriptor
    op::Char
    buffer::CuVector{UInt8}
    arrays::NTuple{3, CuPtr{Cvoid}}   # colPtr, rowVal, nzVal of the matrix the plan was built for
end

# Plans by matrix identity (WeakKeyDict would hash the matrix by value, which reads it on
# the host). A plan does not keep its matrix alive: the matrix's finalizer marks the table,
# and the next lookup drops the entries whose matrix is gone.
const _SPMV_PLANS = Dict{UInt, Tuple{WeakRef, Dict{Bool, SpMVPlan}}}()
const _SPMV_DIRTY = Threads.Atomic{Bool}(false)
const _SPMV_LOCK = ReentrantLock()

function _spmv_plans(A)
    if _SPMV_DIRTY[]
        _SPMV_DIRTY[] = false
        filter!(kv -> kv[2][1].value !== nothing, _SPMV_PLANS)
    end
    key = objectid(A)
    entry = get(_SPMV_PLANS, key, nothing)
    entry !== nothing && entry[1].value === A && return entry[2]
    plans = Dict{Bool, SpMVPlan}()
    _SPMV_PLANS[key] = (WeakRef(A), plans)
    finalizer(_ -> (_SPMV_DIRTY[] = true), A)
    return plans
end

_arrays(A::CuSparseMatrixCSC) = (
    reinterpret(CuPtr{Cvoid}, pointer(A.colPtr)),
    reinterpret(CuPtr{Cvoid}, pointer(rowvals(A))),
    reinterpret(CuPtr{Cvoid}, pointer(nonzeros(A))),
)

function _spmv_plan(A::CuSparseMatrixCSC{T}, transposed::Bool, y, x) where {T}
    plans = lock(() -> _spmv_plans(A), _SPMV_LOCK)
    plan = get(plans, transposed, nothing)
    plan !== nothing && plan.arrays == _arrays(A) && return plan::SpMVPlan{T}
    # Aᵀ x runs as the non-transposed product with the CSR form of Aᵀ, which shares A's arrays
    desc = transposed ? cuSPARSE.CuSparseMatrixDescriptor(A, 'O'; transposed = true) : cuSPARSE.CuSparseMatrixDescriptor(A, 'O')
    op = 'N'
    descX = cuSPARSE.CuDenseVectorDescriptor(x)
    descY = cuSPARSE.CuDenseVectorDescriptor(y)
    alpha = Ref{T}(one(T)); beta = Ref{T}(one(T))
    algo = cuSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
    sz = Ref{Csize_t}()
    cuSPARSE.cusparseSpMV_bufferSize(cuSPARSE.handle(), op, alpha, desc, descX, beta, descY, T, algo, sz)
    buffer = CuVector{UInt8}(undef, max(sz[], 1))
    cuSPARSE.cusparseSpMV_preprocess(cuSPARSE.handle(), op, alpha, desc, descX, beta, descY, T, algo, buffer)
    plan = SpMVPlan{T}(desc, op, buffer, _arrays(A))
    plans[transposed] = plan
    return plan
end

function _spmv_run!(plan::SpMVPlan{T}, y, x, alpha, beta) where {T}
    descX = cuSPARSE.CuDenseVectorDescriptor(x)
    descY = cuSPARSE.CuDenseVectorDescriptor(y)
    cuSPARSE.cusparseSpMV(cuSPARSE.handle(), plan.op, Ref{T}(alpha), plan.desc, descX, Ref{T}(beta), descY, T,
        cuSPARSE.CUSPARSE_SPMV_ALG_DEFAULT, plan.buffer)
    return y
end

const _DenseCuVec{T} = Union{CuVector{T}, SubArray{T, 1, <:CuVector{T}, <:Tuple{UnitRange{Int}}, true}}

function MadCore._spmv!(y::_DenseCuVec{T}, A::CuSparseMatrixCSC{T}, x::_DenseCuVec{T}, alpha, beta) where {T <: Union{Float32, Float64}}
    return _spmv_run!(_spmv_plan(A, false, y, x), y, x, alpha, beta)
end
function MadCore._spmv!(
    y::_DenseCuVec{T}, A::LinearAlgebra.Adjoint{T, <:CuSparseMatrixCSC{T}}, x::_DenseCuVec{T}, alpha, beta,
) where {T <: Union{Float32, Float64}}
    return _spmv_run!(_spmv_plan(parent(A), true, y, x), y, x, alpha, beta)
end
