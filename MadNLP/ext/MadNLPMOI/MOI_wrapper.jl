"""
    Optimizer()

Create a new MadNLP optimizer.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    solver::Union{Nothing,MadNLP.MadNLPSolver}
    nlp::Union{Nothing,NLPModels.AbstractNLPModel}
    result::Union{Nothing,MadNLP.MadNLPExecutionStats{Float64}}

    name::String
    invalid_model::Bool
    silent::Bool
    options::Dict{Symbol,Any}
    solve_time::Float64
    solve_iterations::Int

    model::MOI.ModelLike
    variable_primal_start::Vector{Union{Nothing,Float64}}
    variable_names::Dict{MOI.VariableIndex, String}
    constraint_names::Dict{MOI.ConstraintIndex, String}
    name_to_variable::Union{Nothing, Dict{String, Union{Nothing, MOI.VariableIndex}}}
    name_to_constraint_index::Union{Nothing, Dict{String, Union{Nothing, MOI.ConstraintIndex}}}

    nlp_data::MOI.NLPBlockData
    uses_nlp_block::Bool
    nlp_dual_start::Union{Nothing, Vector{Float64}}

    # The evaluator of `model`, rebuilt in `_setup_model`.
    evaluator::Union{Nothing, MOI.AbstractNLPEvaluator}
    ad_backend::MOI.Nonlinear.AbstractAutomaticDifferentiation

    jrows::Vector{Int}
    jcols::Vector{Int}
    hrows::Vector{Int}
    hcols::Vector{Int}
    needs_new_nlp::Bool
    has_only_linear_constraints::Bool
    objective_type::Type
    jprod_available::Bool
    hprod_available::Bool
    hess_available::Bool
end

function Optimizer(; kwargs...)
    option_dict = Dict{Symbol, Any}()
    for (name, value) in kwargs
        option_dict[name] = value
    end
    backend = MOI.Nonlinear.SparseReverseMode()
    return Optimizer(
        nothing,
        nothing,
        nothing,
        "",
        false,
        false,
        option_dict,
        NaN,
        0,
        MOI.Nonlinear.model(backend),
        Union{Nothing,Float64}[],
        Dict{MOI.VariableIndex, String}(),
        Dict{MOI.ConstraintIndex, String}(),
        nothing,
        nothing,
        MOI.NLPBlockData([], _EmptyNLPEvaluator(), false),
        false,
        nothing,
        nothing,
        backend,
        Int[],
        Int[],
        Int[],
        Int[],
        true,
        true,
        Nothing,
        false,
        false,
        false,
    )
end

const _SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

const _FUNCTIONS = Union{
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
    MOI.ScalarNonlinearFunction,
}

MOI.get(::Optimizer, ::MOI.SolverVersion) = MadNLP.version()

### _EmptyNLPEvaluator

struct _EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end

MOI.features_available(::_EmptyNLPEvaluator) = [:Grad, :Jac, :Hess, :JacVec, :HessVec]
MOI.initialize(::_EmptyNLPEvaluator, ::Any) = nothing
MOI.eval_constraint(::_EmptyNLPEvaluator, g, x) = nothing
MOI.jacobian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.hessian_lagrangian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.eval_constraint_jacobian(::_EmptyNLPEvaluator, J, x) = nothing
MOI.eval_hessian_lagrangian(::_EmptyNLPEvaluator, H, x, σ, μ) = nothing
MOI.eval_constraint_jacobian_product(d::_EmptyNLPEvaluator, Jv, x, v) = nothing
MOI.eval_constraint_jacobian_transpose_product(::_EmptyNLPEvaluator, Jtv, x, v) = nothing
MOI.eval_hessian_lagrangian_product(::_EmptyNLPEvaluator, Hv, x, v, σ, μ) = nothing

function MOI.empty!(model::Optimizer)
    model.solver = nothing
    model.nlp = nothing
    model.result = nothing
    model.invalid_model = false
    model.solve_time = NaN
    model.solve_iterations = 0
    model.model = MOI.Nonlinear.model(model.ad_backend)
    empty!(model.variable_primal_start)
    empty!(model.variable_names)
    empty!(model.constraint_names)
    model.name_to_variable = nothing
    model.name_to_constraint_index = nothing
    model.nlp_data = MOI.NLPBlockData([], _EmptyNLPEvaluator(), false)
    model.uses_nlp_block = false
    model.nlp_dual_start = nothing
    model.evaluator = nothing
    # SKIP: model.ad_backend
    empty!(model.jrows)
    empty!(model.jcols)
    empty!(model.hrows)
    empty!(model.hcols)
    model.needs_new_nlp = true
    model.has_only_linear_constraints = true
    model.objective_type = Nothing
    model.jprod_available = false
    model.hprod_available = false
    model.hess_available = false
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.model) && !model.uses_nlp_block
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(model, src)
end

MOI.get(::Optimizer, ::MOI.SolverName) = "MadNLP"

function _check_no_nlp_block(model)
    if model.uses_nlp_block
        error("Cannot mix the new and legacy nonlinear APIs")
    end
    return
end

function MOI.supports_add_constrained_variable(model::Optimizer, S::Type{MOI.Parameter{Float64}})
    return MOI.supports_add_constrained_variable(model.model, S)
end

function MOI.add_constrained_variable(model::Optimizer, set::MOI.Parameter{Float64})
    _check_no_nlp_block(model)
    result = MOI.add_constrained_variable(model.model, set)
    model.solver = nothing
    return result
end

function MOI.supports_constraint(
        model::Optimizer,
        F::Type{<:MOI.AbstractFunction},
        S::Type{<:MOI.AbstractSet},
)
    return MOI.supports_constraint(model.model, F, S)
end

function MOI.get(model::Optimizer, attr::MOI.ListOfConstraintTypesPresent)
    return MOI.get(model.model, attr)
end

### MOI.Name

MOI.supports(::Optimizer, ::MOI.Name) = true

function MOI.set(model::Optimizer, ::MOI.Name, value::String)
    model.name = value
    return
end

MOI.get(model::Optimizer, ::MOI.Name) = model.name

### MOI.Silent

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

### MOI.TimeLimitSec

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    MOI.set(model, MOI.RawOptimizerAttribute("max_cpu_time"), Float64(value))
    return
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    delete!(model.options, :max_cpu_time)
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return get(model.options, :max_cpu_time, nothing)
end

### MOI.RawOptimizerAttribute

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.set(model::Optimizer, p::MOI.RawOptimizerAttribute, value)
    model.options[Symbol(p.name)] = value
    # No need to reset model.solver because this gets handled in optimize!.
    return
end

function MOI.get(model::Optimizer, p::MOI.RawOptimizerAttribute)
    if !haskey(model.options, p.name)
        msg = "RawOptimizerAttribute with name $(p.name) is not set."
        throw(MOI.GetAttributeNotAllowed(p, msg))
    end
    return model.options[p.name]
end

### Variables

"""
    column(x::MOI.VariableIndex)

Return the column associated with a variable.
"""
column(x::MOI.VariableIndex) = x.value

function MOI.add_variable(model::Optimizer)
    x = MOI.add_variable(model.model)
    push!(model.variable_primal_start, nothing)
    model.solver = nothing
    return x
end

function MOI.is_valid(
        model::Optimizer,
        index::Union{MOI.VariableIndex, MOI.ConstraintIndex},
    )
    return MOI.is_valid(model.model, index)
end
function MOI.get(model::Optimizer, attr::MOI.ListOfVariableIndices)
    return MOI.get(model.model, attr)
end
function MOI.get(model::Optimizer, attr::MOI.NumberOfVariables)
    return MOI.get(model.model, attr)
end
function MOI.get(
    model::Optimizer,
        attr::Union{MOI.NumberOfConstraints, MOI.ListOfConstraintIndices},
    )
    return MOI.get(model.model, attr)
end
function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ConstraintFunction,MOI.ConstraintSet},
        ci::MOI.ConstraintIndex,
    )
    return MOI.get(model.model, attr, ci)
end

_is_linear(F::Type) = F <: Union{Nothing, MOI.VariableIndex, MOI.ScalarAffineFunction}

function MOI.add_constraint(model::Optimizer, f::MOI.AbstractFunction, set::MOI.AbstractSet)
    if f isa MOI.ScalarNonlinearFunction
        _check_no_nlp_block(model)
    end
    ci = MOI.add_constraint(model.model, f, set)
    model.has_only_linear_constraints &= _is_linear(typeof(f))
    model.solver = nothing
    return ci
end

function MOI.delete(model::Optimizer, ci::MOI.ConstraintIndex{MOI.VariableIndex, <:_SETS})
    # MOI 1.55 does not forward bound deletion through its QP layer.
    if model.model isa MOI.Nonlinear.ModelWithQuad
        MOI.delete(model.model.variables, ci)
    else
        MOI.delete(model.model, ci)
    end
    model.solver = nothing
    return
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintSet, ci::MOI.ConstraintIndex, set)
    MOI.set(model.model, attr, ci, set)
    model.needs_new_nlp = true
    return
end

function MOI.supports(
        model::Optimizer,
        attr::Union{MOI.ConstraintDualStart, MOI.LagrangeMultiplierStart},
        CI::Type{<:MOI.ConstraintIndex},
)
    return MOI.supports(model.model, attr, CI)
end

function MOI.get(
    model::Optimizer,
        attr::Union{MOI.ConstraintDualStart, MOI.LagrangeMultiplierStart},
        ci::MOI.ConstraintIndex,
)
    return MOI.get(model.model, attr, ci)
end

function MOI.set(
    model::Optimizer,
        attr::Union{MOI.ConstraintDualStart, MOI.LagrangeMultiplierStart},
        ci::MOI.ConstraintIndex,
        value,
)
    MOI.set(model.model, attr, ci, value)
    model.needs_new_nlp = true
    return
end

function MOI.supports(model::Optimizer, attr::MOI.UserDefinedFunction)
    return MOI.supports(model.model, attr)
end

function MOI.set(model::Optimizer, attr::MOI.UserDefinedFunction, value)
    _check_no_nlp_block(model)
    return MOI.set(model.model, attr, value)
end

function MOI.get(model::Optimizer, attr::MOI.ListOfSupportedNonlinearOperators)
    _check_no_nlp_block(model)
    return MOI.get(model.model, attr)
end

function _is_parameter(model::Optimizer, vi::MOI.VariableIndex)
    ci = MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{Float64}}(vi.value)
    return MOI.is_valid(model.model, ci)
end

### MOI.VariablePrimalStart

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
)
    if _is_parameter(model, vi)
        throw(MOI.GetAttributeNotAllowed(attr, "Variable is a Parameter"))
    end
    MOI.throw_if_not_valid(model, vi)
    return model.variable_primal_start[column(vi)]
end

function MOI.set(
    model::Optimizer,
    attr::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    if _is_parameter(model, vi)
        throw(MOI.SetAttributeNotAllowed(attr, "Variable is a Parameter"))
    end
    MOI.throw_if_not_valid(model, vi)
    model.variable_primal_start[column(vi)] = value
    model.needs_new_nlp = true
    return
end

### MOI.VariableName

MOI.supports(::Optimizer, ::MOI.VariableName, ::Type{MOI.VariableIndex}) = true

function MOI.set(
        model::Optimizer,
        ::MOI.VariableName,
        vi::MOI.VariableIndex,
        name::String,
    )
    MOI.throw_if_not_valid(model, vi)
    if isempty(name)
        delete!(model.variable_names, vi)
    else
        model.variable_names[vi] = name
    end
    # Names are non-numeric metadata: do not invalidate the NLP, only the
    # lazily-built reverse lookup.
    model.name_to_variable = nothing
    return
end

function MOI.get(model::Optimizer, ::MOI.VariableName, vi::MOI.VariableIndex)
    MOI.throw_if_not_valid(model, vi)
    return get(model.variable_names, vi, "")
end

function MOI.get(model::Optimizer, ::Type{MOI.VariableIndex}, name::String)
    if model.name_to_variable === nothing
        _rebuild_name_to_variable(model)
    end
    if haskey(model.name_to_variable, name)
        vi = model.name_to_variable[name]
        if vi === nothing
            error("Duplicate variable name detected: $(name)")
        end
        return vi
    end
    return nothing
end

function _rebuild_name_to_variable(model::Optimizer)
    model.name_to_variable = Dict{String, Union{Nothing, MOI.VariableIndex}}()
    for (vi, name) in model.variable_names
        if haskey(model.name_to_variable, name)
            model.name_to_variable[name] = nothing
        else
            model.name_to_variable[name] = vi
        end
    end
    return
end

### MOI.ConstraintName

# The constraint types that may carry a name: the scalar function/set blocks,
# which map one ConstraintIndex to exactly one KKT row. `VariableIndex`-in-set
# (bound) constraints are excluded so that MOI's fallback throws
# `VariableIndexConstraintNameError`, mirroring Gurobi.jl/Xpress.jl. Vector
# constraints (VectorNonlinearOracle) are excluded on purpose: a single name
# would label a whole block of `output_dimension` rows.
const _NAMED_CONSTRAINT = MOI.ConstraintIndex{<:_FUNCTIONS, <:_SETS}

MOI.supports(::Optimizer, ::MOI.ConstraintName, ::Type{<:_NAMED_CONSTRAINT}) = true

function MOI.set(
        model::Optimizer,
        ::MOI.ConstraintName,
        ci::_NAMED_CONSTRAINT,
        name::String,
    )
    MOI.throw_if_not_valid(model, ci)
    if isempty(name)
        delete!(model.constraint_names, ci)
    else
        model.constraint_names[ci] = name
    end
    model.name_to_constraint_index = nothing
    return
end

function MOI.get(model::Optimizer, ::MOI.ConstraintName, ci::_NAMED_CONSTRAINT)
    MOI.throw_if_not_valid(model, ci)
    return get(model.constraint_names, ci, "")
end

function MOI.get(model::Optimizer, ::Type{MOI.ConstraintIndex}, name::String)
    if model.name_to_constraint_index === nothing
        _rebuild_name_to_constraint_index(model)
    end
    if haskey(model.name_to_constraint_index, name)
        ci = model.name_to_constraint_index[name]
        if ci === nothing
            error("Duplicate constraint name detected: $(name)")
        end
        return ci
    end
    return nothing
end

function _rebuild_name_to_constraint_index(model::Optimizer)
    model.name_to_constraint_index = Dict{String, Union{Nothing, MOI.ConstraintIndex}}()
    for (ci, name) in model.constraint_names
        if haskey(model.name_to_constraint_index, name)
            model.name_to_constraint_index[name] = nothing
        else
            model.name_to_constraint_index[name] = ci
        end
    end
    return
end

### MOI.NLPBlockDualStart

const _BuiltinAD = Union{
    MOI.Nonlinear.ExprGraphOnly,
    MOI.Nonlinear.SparseReverseMode,
    MOI.Nonlinear.SymbolicMode,
}

function MOI.supports(model::Optimizer, ::Union{MOI.NLPBlock, MOI.NLPBlockDualStart})
    return model.ad_backend isa _BuiltinAD
end

function MOI.set(
        model::Optimizer,
        attr::MOI.NLPBlockDualStart,
        values::Union{Nothing, Vector},
    )
    if !MOI.supports(model, attr)
        throw(MOI.UnsupportedAttribute(attr))
    end
    model.nlp_dual_start = values
    model.needs_new_nlp = true
    return
end

MOI.get(model::Optimizer, ::MOI.NLPBlockDualStart) = model.nlp_dual_start

### MOI.NLPBlock

function MOI.get(model::Optimizer, attr::MOI.NLPBlock)
    if !MOI.supports(model, attr)
        throw(MOI.UnsupportedAttribute(attr))
    end
    if model.uses_nlp_block || model.solver === nothing
        return model.nlp_data
    end
    return MOI.NLPBlockData(model.evaluator)
end

function MOI.set(model::Optimizer, attr::MOI.NLPBlock, data::MOI.NLPBlockData)
    if !MOI.supports(model, attr)
        throw(MOI.UnsupportedAttribute(attr))
    end
    types = MOI.get(model.model, MOI.ListOfConstraintTypesPresent())
    if model.objective_type == MOI.ScalarNonlinearFunction ||
            any(F == MOI.ScalarNonlinearFunction || S == MOI.Parameter{Float64} for (F, S) in types)
        error("Cannot mix the new and legacy nonlinear APIs")
    end
    model.nlp_data = data
    model.uses_nlp_block = !(data.evaluator isa _EmptyNLPEvaluator)
    model.solver = nothing
    return
end

# Insert the supplied evaluator into MOI's built-in oracle and quadratic layers.
function _legacy_evaluator(model::Optimizer, vars)
    data = model.nlp_data
    quad = model.model
    block_model = MOI.Nonlinear.Model()
    if data.has_objective
        MOI.set(block_model, MOI.ObjectiveSense(), MOI.get(model, MOI.ObjectiveSense()))
        qp = quad.qp
        # Keep the overridden objective out of the QP Hessian.
        if model.objective_type !== Nothing
            qp = deepcopy(qp)
            F = MOI.ScalarAffineFunction{Float64}
            MOI.set(qp, MOI.ObjectiveFunction{F}(), zero(F))
        end
        quad = MOI.Nonlinear.ModelWithQuad{Float64}(
            qp, quad.inner; objective_sink = MOI.Nonlinear._INNER,
        )
    end
    block = MOI.Nonlinear.Evaluator(block_model, data.evaluator)
    oracles = MOI.Nonlinear.EvaluatorWithOracles(quad.inner, block, vars)
    return MOI.Nonlinear.EvaluatorWithQuad(quad, oracles)
end

### Objective forwarding

function MOI.supports(model::Optimizer, attr::Union{MOI.ObjectiveSense, MOI.ObjectiveFunction})
    return MOI.supports(model.model, attr)
end

function MOI.get(
    model::Optimizer,
        attr::Union{MOI.ObjectiveSense, MOI.ObjectiveFunctionType, MOI.ObjectiveFunction},
    )
    return MOI.get(model.model, attr)
end

function MOI.set(model::Optimizer, attr::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    MOI.set(model.model, attr, sense)
    model.solver = nothing
    return
end

function MOI.set(model::Optimizer, attr::MOI.ObjectiveFunction, f)
    if f isa MOI.ScalarNonlinearFunction
        _check_no_nlp_block(model)
    end
    MOI.set(model.model, attr, f)
    model.objective_type = typeof(f)
    model.solver = nothing
    return
end

# MOI's backend evaluators express every objective as a minimization.
function _objective_sign(model::Optimizer)
    sense = MOI.get(model, MOI.ObjectiveSense())
    return sense == MOI.MAX_SENSE ? -1.0 : sense == MOI.MIN_SENSE ? 1.0 : 0.0
end

function MOI.eval_objective(model::Optimizer, x)
    return MOI.eval_objective(model.evaluator, x)
end

function MOI.eval_objective_gradient(model::Optimizer, grad, x)
    return MOI.eval_objective_gradient(model.evaluator, grad, x)
end

function MOI.eval_constraint(model::Optimizer, g, x)
    return MOI.eval_constraint(model.evaluator, g, x)
end

function MOI.eval_constraint_jacobian(model::Optimizer, values, x)
    return MOI.eval_constraint_jacobian(model.evaluator, values, x)
end

function MOI.eval_constraint_jacobian_transpose_product(model::Optimizer, Jtv, x, v)
    return MOI.eval_constraint_jacobian_transpose_product(model.evaluator, Jtv, x, v)
end

function MOI.eval_constraint_jacobian_product(model::Optimizer, Jv, x, v)
    return MOI.eval_constraint_jacobian_product(model.evaluator, Jv, x, v)
end

function MOI.hessian_lagrangian_structure(model::Optimizer)
    return MOI.hessian_lagrangian_structure(model.evaluator)
end

MOI.jacobian_structure(model::Optimizer) = MOI.jacobian_structure(model.evaluator)

function MOI.eval_hessian_lagrangian(model::Optimizer, H, x, σ, μ)
    return MOI.eval_hessian_lagrangian(model.evaluator, H, x, σ, μ)
end

function MOI.eval_hessian_lagrangian_product(model::Optimizer, Hv, x, v, σ, μ)
    return MOI.eval_hessian_lagrangian_product(model.evaluator, Hv, x, v, σ, μ)
end

### MOI.AutomaticDifferentiationBackend

MOI.supports(::Optimizer, ::MOI.AutomaticDifferentiationBackend) = true

function MOI.get(model::Optimizer, ::MOI.AutomaticDifferentiationBackend)
    return model.ad_backend
end

function MOI.set(
    model::Optimizer,
    ::MOI.AutomaticDifferentiationBackend,
    backend::MOI.Nonlinear.AbstractAutomaticDifferentiation,
)
    if model.uses_nlp_block && !(backend isa _BuiltinAD)
        error("Cannot set a custom AD backend with an NLPBlock.")
    end
    new_model = MOI.Nonlinear.model(backend)
    if typeof(new_model) != typeof(model.model)
        if !MOI.is_empty(model)
            error(
                "The automatic-differentiation backend must be set before " *
                    "adding model data.",
            )
        end
        model.model = new_model
    end
    model.solver = nothing
    model.ad_backend = backend
    return
end

### NLPModels wrapper
struct MOIModel{T} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    model::Optimizer
    counters::NLPModels.Counters
end

NLPModels.obj(nlp::MOIModel, x::AbstractVector{Float64}) = MOI.eval_objective(nlp.model,x)

function NLPModels.grad!(nlp::MOIModel, x::AbstractVector{Float64}, g::AbstractVector{Float64})
    MOI.eval_objective_gradient(nlp.model, g, x)
end

function NLPModels.cons!(nlp::MOIModel, x::AbstractVector{Float64}, c::AbstractVector{Float64})
    MOI.eval_constraint(nlp.model, c, x)
end

function NLPModels.jac_coord!(nlp::MOIModel, x::AbstractVector{Float64}, jac::AbstractVector{Float64})
    MOI.eval_constraint_jacobian(nlp.model, jac, x)
end

function NLPModels.jprod!(nlp::MOIModel, x::AbstractVector{Float64}, v::AbstractVector{Float64}, Jv::AbstractVector{Float64})
    MOI.eval_constraint_jacobian_product(nlp.model, Jv, x, v)
end

function NLPModels.jtprod!(nlp::MOIModel, x::AbstractVector{Float64}, v::Vector{Float64}, Jtv::AbstractVector{Float64})
    MOI.eval_constraint_jacobian_transpose_product(nlp.model, Jtv, x, v)
end

function NLPModels.hess_coord!(nlp::MOIModel, x::AbstractVector{Float64}, l::AbstractVector{Float64}, hess::AbstractVector{Float64}; obj_weight::Float64=1.0)
    MOI.eval_hessian_lagrangian(nlp.model, hess, x, obj_weight, l)
end

function NLPModels.hprod!(nlp::MOIModel, x::AbstractVector{Float64}, l::AbstractVector{Float64}, v::AbstractVector{Float64}, Hv::AbstractVector{Float64}; obj_weight::Float64=1.0)
  MOI.eval_hessian_lagrangian_product(nlp.model, Hv, x, v, obj_weight, l)
end

function NLPModels.hess_structure!(nlp::MOIModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
    @assert length(I) == length(J) == length(nlp.model.hrows) == length(nlp.model.hcols)
    copyto!(I, nlp.model.hrows)
    copyto!(J, nlp.model.hcols)
    return
end

function NLPModels.jac_structure!(nlp::MOIModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
    @assert length(I) == length(J) == length(nlp.model.jrows) == length(nlp.model.jcols)
    copyto!(I, nlp.model.jrows)
    copyto!(J, nlp.model.jcols)
    return
end

### MOI.optimize!

function _setup_model(model::Optimizer)
    nvar = length(model.variable_primal_start)
    model.invalid_model = iszero(nvar)
    if model.invalid_model
        return
    end
    vars = MOI.VariableIndex.(1:nvar)
    model.evaluator = if model.uses_nlp_block
        _legacy_evaluator(model, vars)
    else
        MOI.Nonlinear.Evaluator(model.model, model.ad_backend, vars)
    end
    ncon = length(MOIU.constraint_bounds(model.model).lower)
    features = MOI.features_available(model.evaluator)
    has_hessian = :Hess in features
    has_jacobian_operator = :JacVec in features
    has_hessian_operator = :HessVec in features

    model.jprod_available = has_jacobian_operator
    model.hprod_available = has_hessian_operator
    model.hess_available = has_hessian

    # Initialize evaluator using model's structure.
    init_feat = [:Grad]
    if has_hessian
        push!(init_feat, :Hess)
    end
    if has_hessian_operator
        push!(init_feat, :HessVec)
    end
    if ncon > 0 || !isempty(model.nlp_data.constraint_bounds)
        push!(init_feat, :Jac)
    end
    if has_jacobian_operator
        push!(init_feat, :JacVec)
    end
    MOI.initialize(model.evaluator, init_feat)

    # Sparsity
    jacobian_sparsity = MOI.jacobian_structure(model)
    nnzj = length(jacobian_sparsity)
    jrows = Vector{Int}(undef, nnzj)
    jcols = Vector{Int}(undef, nnzj)
    for k in 1:nnzj
        jrows[k], jcols[k] = jacobian_sparsity[k]
    end
    model.jrows = jrows
    model.jcols = jcols

    hessian_sparsity = has_hessian ? MOI.hessian_lagrangian_structure(model) : Tuple{Int,Int}[]
    nnzh = length(hessian_sparsity)
    hrows = Vector{Int}(undef, nnzh)
    hcols = Vector{Int}(undef, nnzh)
    for k in 1:nnzh
        hrows[k], hcols[k] = hessian_sparsity[k]
    end
    model.hrows = hrows
    model.hcols = hcols

    model.needs_new_nlp = true
    return
end

function _has_linear_constraints(model::Optimizer)
    return model.has_only_linear_constraints && isempty(model.nlp_data.constraint_bounds)
end

function _setup_nlp(model::Optimizer; array_type = nothing)
    if !model.needs_new_nlp
        return model.nlp
    end

    # Number of nonzeros for the jacobian and hessian of the Lagrangian
    nnzj = length(model.jrows)
    nnzh = length(model.hrows)

    # Initial variable
    variable_bounds = MOIU.variable_bounds(model.model)
    nvar = length(variable_bounds.lower)
    x0 = zeros(Float64, nvar)
    for i in 1:length(model.variable_primal_start)
        x0[i] = if model.variable_primal_start[i] !== nothing
            model.variable_primal_start[i]
        else
            clamp(0.0, variable_bounds.lower[i], variable_bounds.upper[i])
        end
    end

    bounds = MOIU.constraint_bounds(model.model)
    g_L, g_U = bounds.lower, bounds.upper
    if model.uses_nlp_block
        legacy = model.nlp_data.constraint_bounds
        g_L = vcat(g_L, getfield.(legacy, :lower))
        g_U = vcat(g_U, getfield.(legacy, :upper))
    end
    ncon = length(g_L)
    starts = MOI.Nonlinear.constraint_dual_starts(model.model)
    y0 = zeros(Float64, ncon)
    for (i, start) in enumerate(starts)
        y0[i] = -something(start, 0.0)
    end
    if model.uses_nlp_block && model.nlp_dual_start !== nothing
        y0[(length(starts) + 1):end] .= .-model.nlp_dual_start
    end

    islp = _has_linear_constraints(model) && _is_linear(model.objective_type) &&
        !model.nlp_data.has_objective
    nlp = MOIModel(
        NLPModels.NLPModelMeta(
            nvar,
            x0 = x0,
            lvar = variable_bounds.lower,
            uvar = variable_bounds.upper,
            ncon = ncon,
            y0 = y0,
            lcon = g_L,
            ucon = g_U,
            nnzj = nnzj,
            nnzh = nnzh,
            minimize = true,
            islp = islp,
            name = "MOIModel",
            variable_bounds_analysis=false,
            constraint_bounds_analysis=false,
            jprod_available = model.jprod_available,
            hprod_available = model.hprod_available,
            hess_available = model.hess_available,
        ),
        model,
        NLPModels.Counters(),
    )

    model.nlp = if isnothing(array_type)
        nlp
    else
        MadNLP.SparseWrapperModel(array_type, nlp)
    end

    model.needs_new_nlp = false
    return model.nlp
end

function MOI.optimize!(model::Optimizer)
    if model.solver === nothing
        _setup_model(model)
    end
    if model.invalid_model
        return
    end

    array_type = pop!(model.options, :array_type, nothing)
    _setup_nlp(model; array_type = array_type)

    if model.silent
        model.options[:print_level] = MadNLP.ERROR
    end
    options = copy(model.options)
    # Specific options depending on problem's structure.
    if !model.hess_available
        options[:hessian_approximation] = MadNLP.CompactLBFGS
    end
    # Set Jacobian to constant if all constraints are linear.
    if _has_linear_constraints(model)
        options[:jacobian_constant] = true
    end
    # Instantiate MadNLP.
    model.solver = MadNLP.MadNLPSolver(model.nlp; options...)
    result = MadNLP.solve!(model.solver)
    model.result = if isnothing(array_type)
        result
    else
        model.options[:array_type] = array_type
        copy_result_to_cpu(result)
    end
    model.solve_time = model.solver.cnt.total_time
    model.solve_iterations = model.solver.cnt.k
    return
end


function copy_result_to_cpu(stats::MadNLP.MadNLPExecutionStats)
    return MadNLP.MadNLPExecutionStats(
        stats.options,
        stats.status,
        Array(stats.solution),
        stats.objective,
        Array(stats.constraints),
        stats.dual_feas,
        stats.primal_feas,
        Array(stats.multipliers),
        Array(stats.multipliers_L),
        Array(stats.multipliers_U),
        stats.iter,
        stats.counters,
    )
end

const _STATUS_CODES = Dict{MadNLP.Status,MOI.TerminationStatusCode}(
    MadNLP.SOLVE_SUCCEEDED => MOI.LOCALLY_SOLVED,
    MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL => MOI.ALMOST_LOCALLY_SOLVED,
    MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL => MOI.SLOW_PROGRESS,
    MadNLP.DIVERGING_ITERATES => MOI.INFEASIBLE_OR_UNBOUNDED,
    MadNLP.INFEASIBLE_PROBLEM_DETECTED => MOI.LOCALLY_INFEASIBLE,
    MadNLP.MAXIMUM_ITERATIONS_EXCEEDED => MOI.ITERATION_LIMIT,
    MadNLP.MAXIMUM_WALLTIME_EXCEEDED => MOI.TIME_LIMIT,
    MadNLP.INITIAL => MOI.OPTIMIZE_NOT_CALLED,
    MadNLP.RESTORATION_FAILED => MOI.NUMERICAL_ERROR,
    MadNLP.INVALID_NUMBER_DETECTED => MOI.INVALID_MODEL,
    MadNLP.ERROR_IN_STEP_COMPUTATION => MOI.NUMERICAL_ERROR,
    MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM => MOI.INVALID_MODEL,
    MadNLP.USER_REQUESTED_STOP => MOI.INTERRUPTED,
    MadNLP.INTERNAL_ERROR => MOI.OTHER_ERROR,
    MadNLP.INVALID_NUMBER_OBJECTIVE => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_GRADIENT => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_CONSTRAINTS => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_JACOBIAN => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN => MOI.INVALID_MODEL,
)

### MOI.ResultCount

# Ipopt always has an iterate available.
function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return (model.solver !== nothing) ? 1 : 0
end

### MOI.TerminationStatus

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.invalid_model
        return MOI.INVALID_MODEL
    elseif model.solver === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    return get(_STATUS_CODES, model.result.status, MOI.OTHER_ERROR)
end

### MOI.RawStatusString

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.invalid_model
        return "The model has no variable"
    elseif model.solver === nothing
        return "Optimize not called"
    end
    return MadNLP.get_status_output(model.result.status, model.result.options)
end


### MOI.PrimalStatus

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    status = model.result.status
    if status == MadNLP.SOLVE_SUCCEEDED
        return MOI.FEASIBLE_POINT
    elseif status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        return MOI.NEARLY_FEASIBLE_POINT
    elseif status == MadNLP.INFEASIBLE_PROBLEM_DETECTED
        return MOI.INFEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

### MOI.DualStatus

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    status = model.result.status
    if status == MadNLP.SOLVE_SUCCEEDED
        return MOI.FEASIBLE_POINT
    elseif status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        return MOI.NEARLY_FEASIBLE_POINT
    elseif status == MadNLP.INFEASIBLE_PROBLEM_DETECTED
        return MOI.INFEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

### MOI.SolveTimeSec

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

### MOI.BarrierIterations

MOI.get(model::Optimizer,::MOI.BarrierIterations) = model.solve_iterations

### MOI.ObjectiveValue

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return _objective_sign(model) * model.result.objective
end

### MOI.VariablePrimal

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    if _is_parameter(model, vi)
        ci = MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{Float64}}(vi.value)
        return MOI.get(model.model, MOI.ConstraintSet(), ci).value
    end
    return model.result.solution[vi.value]
end

### MOI.ConstraintPrimal

function MOI.get(
    model::Optimizer,
        attr::MOI.ConstraintPrimal,
        ci::MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.VectorNonlinearOracle{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    f = MOI.get(model.model, MOI.ConstraintFunction(), ci)
    return MOI.get.(model, MOI.VariablePrimal(attr.result_index), f.variables)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{<:_FUNCTIONS,<:_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.result.constraints[MOIU.rows(model.model, ci)]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.result.solution[ci.value]
end

### MOI.ConstraintDual

function MOI.get(
        model::Optimizer,
        attr::MOI.LagrangeMultiplier,
        ci::MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.VectorNonlinearOracle{Float64}},
    )
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return -model.result.multipliers[MOIU.rows(model.model, ci)]
end

function MOI.get(
        model::Optimizer,
        attr::MOI.ConstraintDual,
        ci::MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.VectorNonlinearOracle{Float64}},
    )
    λ = MOI.get(model, MOI.LagrangeMultiplier(attr.result_index), ci)
    f = MOI.get(model.model, MOI.ConstraintFunction(), ci)
    set = MOI.get(model.model, MOI.ConstraintSet(), ci)
    x = MOI.get.(model, MOI.VariablePrimal(attr.result_index), f.variables)
    values = zeros(length(set.jacobian_structure))
    set.eval_jacobian(values, x)
    dual = zeros(length(x))
    for ((row, col), value) in zip(set.jacobian_structure, values)
        dual[col] += λ[row] * value
    end
    return dual
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{<:_FUNCTIONS,<:_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return -model.result.multipliers[MOIU.rows(model.model, ci)]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return -model.result.multipliers_U[ci.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.result.multipliers_L[ci.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.result.multipliers_L[ci.value] - model.result.multipliers_U[ci.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Interval{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.result.multipliers_L[ci.value] - model.result.multipliers_U[ci.value]
end

### MOI.NLPBlockDual

function MOI.get(model::Optimizer, attr::MOI.NLPBlockDual)
    MOI.check_result_index_bounds(model, attr)
    if !model.uses_nlp_block
        return -model.result.multipliers
    end
    offset = length(model.result.multipliers) - length(model.nlp_data.constraint_bounds)
    return -model.result.multipliers[(offset + 1):end]
end
