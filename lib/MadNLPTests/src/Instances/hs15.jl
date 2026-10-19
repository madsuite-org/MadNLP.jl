struct HS15Model{T} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
end

function HS15Model(;T = Float64, x0=zeros(T,2), y0=zeros(T,2))
    return HS15Model(
        NLPModels.NLPModelMeta(
            2,     #nvar
            ncon = 2,
            nnzj = 4,
            nnzh = 3,
            x0 = x0,
            y0 = y0,
            lvar = T[-Inf, -Inf],
            uvar = T[0.5, Inf],
            lcon = T[1.0, 0.0],
            ucon = T[Inf, Inf],
            sparse_jacobian = false,
            sparse_hessian = false,
            minimize = true
        ),
        NLPModels.Counters()
    )
end

function NLPModels.obj(nlp::HS15Model{T}, x::AbstractVector) where T
    return T(100) * (x[2] - x[1]^2)^2 + (one(T) - x[1])^2
end

function NLPModels.grad!(nlp::HS15Model{T}, x::AbstractVector, g::AbstractVector) where T
    z = x[2] - x[1]^2
    g[1] = T(-400) * z * x[1] - T(2) * (one(T) - x[1])
    g[2] = T(200) * z
    return g
end

function NLPModels.cons!(nlp::HS15Model, x::AbstractVector, c::AbstractVector)
    c[1] = x[1] * x[2]
    c[2] = x[1] + x[2]^2
    return c
end

function NLPModels.jac_structure!(nlp::HS15Model, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, [1, 1, 2, 2])
    copyto!(J, [1, 2, 1, 2])
    return I, J
end

function NLPModels.jac_coord!(nlp::HS15Model{T}, x::AbstractVector, J::AbstractVector) where T
    J[1] = x[2]       # (1, 1)
    J[2] = x[1]       # (1, 2)
    J[3] = one(T)     # (2, 1)
    J[4] = T(2)*x[2]  # (2, 2)
    return J
end

function NLPModels.jprod!(nlp::HS15Model{T}, x::AbstractVector, v::AbstractVector, jv::AbstractVector) where T
    jv[1] = x[2] * v[1] + x[1] * v[2]
    jv[2] = v[1] + T(2) * x[2] * v[2]
    return jv
end

function NLPModels.jtprod!(nlp::HS15Model{T}, x::AbstractVector, v::AbstractVector, jv::AbstractVector) where T
    jv[1] = x[2] * v[1] + v[2]
    jv[2] = x[1] * v[1] + T(2) * x[2] * v[2]
    return jv
end

function NLPModels.jac_dense!(nlp::HS15Model{T}, x::AbstractVector, J::AbstractMatrix) where T
    J[1, 1] = x[2]       # (1, 1)
    J[1, 2] = x[1]       # (1, 2)
    J[2, 1] = one(T)     # (2, 1)
    J[2, 2] = T(2)*x[2]  # (2, 2)
    return J
end

function NLPModels.hess_structure!(nlp::HS15Model, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, [1, 2, 2])
    copyto!(J, [1, 1, 2])
    return I, J
end

function NLPModels.hess_coord!(nlp::HS15Model{T}, x, y, H::AbstractVector; obj_weight=one(T)) where T
    # Objective
    H[1] = obj_weight * (T(-400) * x[2] + T(1200) * x[1]^2 + T(2))
    H[2] = obj_weight * (T(-400) * x[1])
    H[3] = obj_weight * T(200)
    # First constraint
    H[2] += y[1]
    # Second constraint
    H[3] += y[2] * T(2)
    return H
end

function NLPModels.hess_dense!(nlp::HS15Model{T}, x, y, H::AbstractMatrix; obj_weight=one(T)) where T
    H[1, 1] = obj_weight * (T(-400) * x[2] + T(1200) * x[1]^2 + T(2))
    H[2, 1] = obj_weight * (T(-400) * x[1])
    H[2, 2] = obj_weight * T(200)
    # First constraint
    H[2, 1] += y[1]
    # Second constraint
    H[2, 2] += y[2] * T(2)
    H[1, 2] = H[2, 1]
    return H
end

