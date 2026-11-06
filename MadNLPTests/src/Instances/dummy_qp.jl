struct DenseDummyQP{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    VI <: AbstractVector{Int}
    } <: NLPModels.AbstractNLPModel{T,VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    P::MT # primal hessian
    A::MT # constraint jacobian
    q::VT
    buffer::VT
    hrows::VI
    hcols::VI
    jrows::VI
    jcols::VI
    counters::NLPModels.Counters
end

function NLPModels.jac_structure!(qp::DenseDummyQP, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, qp.jrows)
    copyto!(J, qp.jcols)
end
function NLPModels.hess_structure!(qp::DenseDummyQP, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, qp.hrows)
    copyto!(J, qp.hcols)
end

function NLPModels.obj(qp::DenseDummyQP{T}, x::AbstractVector{T}) where T
    mul!(qp.buffer, qp.P, x)
    return T(0.5) * dot(x, qp.buffer) + dot(qp.q, x)
end

function NLPModels.grad!(qp::DenseDummyQP, x::AbstractVector, g::AbstractVector)
    mul!(g, qp.P, x)
    g .+= qp.q
    return
end

function NLPModels.cons!(qp::DenseDummyQP, x::AbstractVector, c::AbstractVector)
    mul!(c, qp.A, x)
end

# Jacobian: sparse callback
function NLPModels.jac_coord!(qp::DenseDummyQP, x::AbstractVector, J::AbstractVector)
    index = 1
    for (i, j) in zip(qp.jrows, qp.jcols)
        J[index] = qp.A[i, j]
        index += 1
    end
end

function NLPModels.jprod!(qp::DenseDummyQP, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
    mul!(jv, qp.A, v)
    return jv
end

function NLPModels.jtprod!(qp::DenseDummyQP, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
    mul!(jv, qp.A', v)
    return jv
end

# Jacobian: dense callback
NLPModels.jac_dense!(qp::DenseDummyQP, x, J::AbstractMatrix) = copyto!(J, qp.A)

# Hessian: sparse callback
function NLPModels.hess_coord!(qp::DenseDummyQP{T},x, l, hess::AbstractVector; obj_weight=one(T)) where T
    index = 1
    for i in 1:NLPModels.get_nvar(qp) , j in 1:i
        hess[index] = obj_weight * qp.P[j, i]
        index += 1
    end
end

# Hessian: dense callback
function NLPModels.hess_dense!(qp::DenseDummyQP{T}, x, l, hess::AbstractMatrix; obj_weight=one(T)) where T
    copyto!(hess, obj_weight .* qp.P)
end

function DenseDummyQP(
    x0::AbstractVector{T} = zeros(T,100);
    m=10, fixed_variables=similar(x0,Int,0), equality_cons=similar(x0,Int,0)
    ) where {T}

    n = length(x0)

    if m >= n
        error("The number of constraints `m` should be less than the number of variable `n`.")
    end

    Random.seed!(1)
    # Generate random values.
    # N.B.: we need to allocate the matrix P_ right after the vector
    # q_ if we want to obtain a deterministic behavior: the seed is not working
    # if we call the function `randn` after allocating vectors on the device.
    q_ = randn(n)
    P_ = randn(n, n)

    y0 = fill!(similar(x0, m), zero(T))
    q = copyto!(similar(x0, n), q_)
    buffer = similar(x0, n)

    # Bound constraints
    xl = fill!(similar(x0, n), zero(T))
    xu = fill!(similar(x0, n), one(T))
    gl = fill!(similar(x0, m), zero(T))
    gu = fill!(similar(x0, m), one(T))

    # Update gu to load equality constraints
    gu[equality_cons] .= zero(T)
    xl[fixed_variables] .= @view(xu[fixed_variables])

    # Build QP problem 0.5 * x' * P * x + q' * x
    P = copyto!(similar(x0, n , n), P_)
    P = P*P' # P is symmetric
    P += T(100.0) * I


    # Build constraints gl <= Ax <= gu
    A = fill!(similar(x0, m, n), zero(T))
    A[1:m+1:m^2] .= one(T)
    A[m+1:m+1:m^2+m] .=-one(T)

    nnzh = div(n * (n + 1), 2)
    hrows = copyto!(similar(x0, Int, nnzh), [i for i in 1:n for j in 1:i])
    hcols = copyto!(similar(x0, Int, nnzh), [j for i in 1:n for j in 1:i])

    nnzj = n * m
    jrows = copyto!(similar(x0, Int, nnzj), [j for i in 1:n for j in 1:m])
    jcols = copyto!(similar(x0, Int, nnzj), [i for i in 1:n for j in 1:m])

    return DenseDummyQP(
        NLPModels.NLPModelMeta(
            n,
            ncon = m,
            nnzj = nnzj,
            nnzh = nnzh,
            x0 = x0,
            y0 = y0,
            lvar = xl,
            uvar = xu,
            lcon = gl,
            ucon = gu,
            sparse_jacobian = false,
            sparse_hessian = false,
            minimize = true
        ),
        P,A,q,buffer,
        hrows,hcols,jrows,jcols,
        NLPModels.Counters()
    )
end
