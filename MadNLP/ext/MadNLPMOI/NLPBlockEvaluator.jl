# A legacy NLPBlock appends constraints to the backend model and can override
# its objective. Each evaluator receives its own objective weight and row slice.
mutable struct _NLPBlockEvaluator{E, B} <: MOI.AbstractNLPEvaluator
    backend::E
    block::B
    has_objective::Bool
    has_constraints::Bool
    objective_sign::Float64
    backend_ncon::Int
    backend_nnzj::Int
    backend_nnzh::Int
    # Scratch for adding the two variable-space derivative products.
    work::Vector{Float64}
end

function _block_needs(d::_NLPBlockEvaluator, feature)
    if feature == :Grad
        return d.has_objective
    elseif feature in (:Jac, :JacVec)
        return d.has_constraints
    end
    return true
end

function MOI.features_available(d::_NLPBlockEvaluator)
    block = MOI.features_available(d.block)
    return filter(MOI.features_available(d.backend)) do feature
        return !_block_needs(d, feature) || feature in block
    end
end

function MOI.initialize(d::_NLPBlockEvaluator, features)
    MOI.initialize(d.backend, features)
    MOI.initialize(d.block, filter(f -> _block_needs(d, f), features))
    d.backend_nnzj = length(MOI.jacobian_structure(d.backend))
    if :Hess in features
        d.backend_nnzh = length(MOI.hessian_lagrangian_structure(d.backend))
    end
    return
end

function MOI.eval_objective(d::_NLPBlockEvaluator, x)
    if d.has_objective
        return d.objective_sign * MOI.eval_objective(d.block, x)
    end
    return MOI.eval_objective(d.backend, x)
end

function MOI.eval_objective_gradient(d::_NLPBlockEvaluator, g, x)
    if d.has_objective
        MOI.eval_objective_gradient(d.block, g, x)
        g .*= d.objective_sign
        return
    end
    return MOI.eval_objective_gradient(d.backend, g, x)
end

function MOI.eval_constraint(d::_NLPBlockEvaluator, g, x)
    MOI.eval_constraint(d.backend, view(g, 1:d.backend_ncon), x)
    if d.has_constraints
        MOI.eval_constraint(d.block, view(g, (d.backend_ncon + 1):length(g)), x)
    end
    return
end

function MOI.jacobian_structure(d::_NLPBlockEvaluator)
    J = MOI.jacobian_structure(d.backend)
    if !d.has_constraints
        return J
    end
    return vcat(J, [(r + d.backend_ncon, c) for (r, c) in MOI.jacobian_structure(d.block)])
end

function MOI.eval_constraint_jacobian(d::_NLPBlockEvaluator, J, x)
    MOI.eval_constraint_jacobian(d.backend, view(J, 1:d.backend_nnzj), x)
    if d.has_constraints
        MOI.eval_constraint_jacobian(d.block, view(J, (d.backend_nnzj + 1):length(J)), x)
    end
    return
end

function MOI.eval_constraint_jacobian_product(d::_NLPBlockEvaluator, y, x, v)
    MOI.eval_constraint_jacobian_product(d.backend, view(y, 1:d.backend_ncon), x, v)
    if d.has_constraints
        MOI.eval_constraint_jacobian_product(d.block, view(y, (d.backend_ncon + 1):length(y)), x, v)
    end
    return
end

function MOI.eval_constraint_jacobian_transpose_product(d::_NLPBlockEvaluator, y, x, v)
    MOI.eval_constraint_jacobian_transpose_product(d.backend, y, x, view(v, 1:d.backend_ncon))
    if d.has_constraints
        MOI.eval_constraint_jacobian_transpose_product(d.block, d.work, x, view(v, (d.backend_ncon + 1):length(v)))
        y .+= d.work
    end
    return
end

function MOI.hessian_lagrangian_structure(d::_NLPBlockEvaluator)
    return vcat(MOI.hessian_lagrangian_structure(d.backend), MOI.hessian_lagrangian_structure(d.block))
end

function MOI.eval_hessian_lagrangian(d::_NLPBlockEvaluator, H, x, σ, μ)
    MOI.eval_hessian_lagrangian(
        d.backend, view(H, 1:d.backend_nnzh), x,
        d.has_objective ? 0.0 : σ, view(μ, 1:d.backend_ncon),
    )
    return MOI.eval_hessian_lagrangian(
        d.block, view(H, (d.backend_nnzh + 1):length(H)), x,
        d.has_objective ? d.objective_sign * σ : 0.0, view(μ, (d.backend_ncon + 1):length(μ)),
    )
end

function MOI.eval_hessian_lagrangian_product(d::_NLPBlockEvaluator, y, x, v, σ, μ)
    MOI.eval_hessian_lagrangian_product(
        d.backend, y, x, v, d.has_objective ? 0.0 : σ, view(μ, 1:d.backend_ncon),
    )
    MOI.eval_hessian_lagrangian_product(
        d.block, d.work, x, v,
        d.has_objective ? d.objective_sign * σ : 0.0, view(μ, (d.backend_ncon + 1):length(μ)),
    )
    y .+= d.work
    return
end
