struct HS15NoHessianModel{T} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
end

function HS15NoHessianModel(;T = Float64, x0=zeros(T,2), y0=zeros(T,2))
    return HS15NoHessianModel(
        NLPModels.NLPModelMeta(
            2,     #nvar
            ncon = 2,
            nnzj = 4,
            nnzh = 0,
            x0 = x0,
            y0 = y0,
            lvar = T[-Inf, -Inf],
            uvar = T[0.5, Inf],
            lcon = T[1.0, 0.0],
            ucon = T[Inf, Inf],
            minimize = true
        ),
        NLPModels.Counters()
    )
end

function NLPModels.obj(nlp::HS15NoHessianModel, x::AbstractVector)
    return 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2
end

function NLPModels.grad!(nlp::HS15NoHessianModel, x::AbstractVector, g::AbstractVector)
    z = x[2] - x[1]^2
    g[1] = -400.0 * z * x[1] - 2.0 * (1.0 - x[1])
    g[2] = 200.0 * z
    return
end

function NLPModels.cons!(nlp::HS15NoHessianModel, x::AbstractVector, c::AbstractVector)
    c[1] = x[1] * x[2]
    c[2] = x[1] + x[2]^2
end

function NLPModels.jac_structure!(nlp::HS15NoHessianModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, [1, 1, 2, 2])
    copyto!(J, [1, 2, 1, 2])
end

function NLPModels.jac_coord!(nlp::HS15NoHessianModel, x::AbstractVector, J::AbstractVector)
    J[1] = x[2]    # (1, 1)
    J[2] = x[1]    # (1, 2)
    J[3] = 1.0     # (2, 1)
    J[4] = 2*x[2]  # (2, 2)
    return J
end

function NLPModels.jprod!(nlp::HS15NoHessianModel, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
    jv[1] = x[2] * v[1] + x[1] * v[2]
    jv[2] = v[1] + 2 * x[2] * v[2]
    return jv
end

function NLPModels.jtprod!(nlp::HS15NoHessianModel, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
    jv[1] = x[2] * v[1] + v[2]
    jv[2] = x[1] * v[1] + 2 * x[2] * v[2]
    return jv
end

