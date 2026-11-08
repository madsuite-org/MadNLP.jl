# The example of Wächter and Biegler (2000) on which a filter line search with a
# Newton-type step is stuck at an infeasible point. From x0 = (-2, 1, 1) the solution is
# x = (1, 0, 0.5), f = 1.
using NLPModels

struct WachBiegModel{T} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
end

WachBiegModel(; T = Float64) = WachBiegModel(
    NLPModels.NLPModelMeta(3; ncon = 2, nnzj = 4, nnzh = 1, x0 = T[-2, 1, 1],
        lvar = T[-Inf, 0, 0], uvar = T[Inf, Inf, Inf], lcon = T[0, 0], ucon = T[0, 0]),
    NLPModels.Counters())

NLPModels.obj(::WachBiegModel, x::AbstractVector) = x[1]
NLPModels.grad!(::WachBiegModel, x::AbstractVector, g::AbstractVector) = (g .= 0; g[1] = 1; g)
function NLPModels.cons!(::WachBiegModel, x::AbstractVector, c::AbstractVector)
    c[1] = x[1]^2 - x[2] - 1
    c[2] = x[1] - x[3] - 0.5
    return c
end
NLPModels.jac_structure!(::WachBiegModel, I::AbstractVector, J::AbstractVector) =
    (copyto!(I, [1, 1, 2, 2]); copyto!(J, [1, 2, 1, 3]); (I, J))
NLPModels.jac_coord!(::WachBiegModel, x::AbstractVector, J::AbstractVector) =
    (J[1] = 2x[1]; J[2] = -1; J[3] = 1; J[4] = -1; J)
NLPModels.hess_structure!(::WachBiegModel, I::AbstractVector, J::AbstractVector) =
    (I[1] = 1; J[1] = 1; (I, J))
NLPModels.hess_coord!(::WachBiegModel, x, y, H::AbstractVector; obj_weight = 1.0) =
    (H[1] = 2y[1]; H)
