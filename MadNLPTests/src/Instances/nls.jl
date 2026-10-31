struct NLSModel <: NLPModels.AbstractNLSModel{Float64,Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    nls_meta::NLPModels.NLSMeta{Float64, Vector{Float64}}
    counters::NLPModels.NLSCounters
end

function NLSModel()
    x0 = [-1.2; 1.0]
    return NLSModel(
        NLPModels.NLPModelMeta(
            2,     #nvar
            ncon = 0,
            nnzj = 0,
            nnzh = 3,
            x0 = x0,
            lvar = zeros(2),
            uvar = ones(2),
            minimize = true
        ),
        NLPModels.NLSMeta(2, 2; nnzj=3, nnzh=4),
        NLPModels.NLSCounters()
    )
end

function NLPModels.residual!(nls::NLSModel, x, Fx)
    Fx[1] = x[1] - 1.0
    Fx[2] = 10.0 * (x[2] - x[1]^2)
    return Fx
end

function NLPModels.jac_structure_residual!(
    nls::NLSModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    copyto!(rows, [1, 2, 2])
    copyto!(cols, [1, 1, 2])
    return rows, cols
end

function NLPModels.jac_coord_residual!(nls::NLSModel, x::AbstractVector, vals::AbstractVector)
    vals[1] = 1.0
    vals[2] = -20.0 * x[1]
    vals[3] = 10.0
    return vals
end

function NLPModels.jprod_residual!(nls::NLSModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    Jv[1] = v[1]
    Jv[2] = -20.0 * x[1] * v[1] + 10.0 * v[2]
    return Jv
end

function NLPModels.jtprod_residual!(nls::NLSModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
    Jtv[1] = v[1] - 20.0 * x[1] * v[2]
    Jtv[2] = 10.0 * v[2]
    return Jtv
end

function NLPModels.hess_structure_residual!(
    nls::NLSModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    rows[1] = 1
    cols[1] = 1
    return rows, cols
end

function NLPModels.hess_coord_residual!(
    nls::NLSModel,
    x::AbstractVector,
    v::AbstractVector,
    vals::AbstractVector,
)
    vals[1] = -20.0 * v[2]
    return vals
end

function NLPModels.hess_structure!(nlp::NLSModel, rows::AbstractVector{T}, cols::AbstractVector{T}) where T
    copyto!(rows, [1, 2, 2])
    copyto!(cols, [1, 1, 2])
    return rows, cols
end

function NLPModels.hess_coord!(nlp::NLSModel, x, y, H::AbstractVector; obj_weight=1.0)
    # Objective
    H[1] = obj_weight * (1.0 - 200.0 * x[2] + 600 * x[1]^2)
    H[2] = obj_weight * (-200.0 * x[1])
    H[3] = obj_weight * 100.0
    return H
end

