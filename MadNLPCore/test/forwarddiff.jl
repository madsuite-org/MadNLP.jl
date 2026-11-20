# Forward-mode differentiation through the solve: HS15 with the parameter `p` of
#     min p (x₂ - x₁²)² + (1 - x₁)²   s.t.  x₁x₂ ≥ 1,  x₁ + x₂² ≥ 0,  x₁ ≤ 0.5
# solved in `ForwardDiff.Dual` numbers. The model has infinite bounds on the variables
# and on the slacks, where `Inf * 0` gives a `NaN` derivative unless the solver leaves
# them out of its arithmetic.
using ForwardDiff

struct ParametricHS15{T} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
    p::T
end

ParametricHS15(p::T) where {T} = ParametricHS15(
    NLPModels.NLPModelMeta(
        2; ncon = 2, nnzj = 4, nnzh = 3, x0 = zeros(T, 2), y0 = zeros(T, 2),
        lvar = T[-Inf, -Inf], uvar = T[0.5, Inf], lcon = T[1, 0], ucon = T[Inf, Inf],
    ),
    NLPModels.Counters(),
    p,
)

NLPModels.obj(nlp::ParametricHS15, x::AbstractVector) = nlp.p * (x[2] - x[1]^2)^2 + (1 - x[1])^2

function NLPModels.grad!(nlp::ParametricHS15, x::AbstractVector, g::AbstractVector)
    z = x[2] - x[1]^2
    g[1] = -4 * nlp.p * z * x[1] - 2 * (1 - x[1])
    g[2] = 2 * nlp.p * z
    return g
end

function NLPModels.cons!(nlp::ParametricHS15, x::AbstractVector, c::AbstractVector)
    c[1] = x[1] * x[2]
    c[2] = x[1] + x[2]^2
    return c
end

function NLPModels.jac_structure!(nlp::ParametricHS15, I::AbstractVector, J::AbstractVector)
    copyto!(I, [1, 1, 2, 2])
    copyto!(J, [1, 2, 1, 2])
    return I, J
end

function NLPModels.jac_coord!(nlp::ParametricHS15, x::AbstractVector, J::AbstractVector)
    J[1] = x[2]
    J[2] = x[1]
    J[3] = 1
    J[4] = 2 * x[2]
    return J
end

function NLPModels.hess_structure!(nlp::ParametricHS15, I::AbstractVector, J::AbstractVector)
    copyto!(I, [1, 2, 2])
    copyto!(J, [1, 1, 2])
    return I, J
end

function NLPModels.hess_coord!(nlp::ParametricHS15, x, y, H::AbstractVector; obj_weight = 1)
    p = nlp.p
    H[1] = obj_weight * (-4 * p * x[2] + 12 * p * x[1]^2 + 2)
    H[2] = obj_weight * (-4 * p * x[1]) + y[1]
    H[3] = obj_weight * 2 * p + 2 * y[2]
    return H
end
