mutable struct _VectorNonlinearOracleCache
    set::MOI.VectorNonlinearOracle{Float64}
    x::Vector{Float64}
    J_nzval::Vector{Float64}
    start::Union{Nothing,Vector{Float64}}
    eval_f_timer::Float64
    eval_jacobian_timer::Float64
    eval_hessian_lagrangian_timer::Float64

    function _VectorNonlinearOracleCache(set::MOI.VectorNonlinearOracle{Float64})
        nnzJ = length(set.jacobian_structure)
        return new(set, zeros(set.input_dimension), zeros(nnzJ), nothing, 0.0, 0.0, 0.0)
    end
end

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
    sense::MOI.OptimizationSense

    model::MOI.Nonlinear.ModelWithQuad{Float64,MOI.Nonlinear.Model}
    variable_primal_start::Vector{Union{Nothing,Float64}}
    variable_names::Dict{MOI.VariableIndex, String}
    constraint_names::Dict{MOI.ConstraintIndex, String}
    name_to_variable::Union{Nothing, Dict{String, Union{Nothing, MOI.VariableIndex}}}
    name_to_constraint_index::Union{Nothing, Dict{String, Union{Nothing, MOI.ConstraintIndex}}}

    nlp_data::MOI.NLPBlockData
    # Whether `nlp_data` was set through the legacy `MOI.NLPBlock` API, in
    # which case it must not be rebuilt from the inner nonlinear model.
    uses_nlp_block::Bool
    nlp_dual_start::Union{Nothing,Vector{Float64}}
    mult_g_nlp::Dict{MOI.Nonlinear.ConstraintIndex,Float64}

    # The evaluator of `model`, rebuilt in `_setup_model`.
    evaluator::Union{Nothing,MOI.Nonlinear.EvaluatorWithQuad{Float64}}
    ad_backend::MOI.Nonlinear.AbstractAutomaticDifferentiation
    vector_nonlinear_oracle_constraints::Vector{Tuple{MOI.VectorOfVariables,_VectorNonlinearOracleCache}}

    jrows::Vector{Int}
    jcols::Vector{Int}
    hrows::Vector{Int}
    hcols::Vector{Int}
    needs_new_nlp::Bool
    has_only_linear_constraints::Bool
    islp::Bool
    jprod_available::Bool
    hprod_available::Bool
    hess_available::Bool
end

function Optimizer(; kwargs...)
    option_dict = Dict{Symbol, Any}()
    for (name, value) in kwargs
        option_dict[name] = value
    end
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
        MOI.FEASIBILITY_SENSE,
        MOI.Nonlinear.ModelWithQuad(MOI.Nonlinear.Model()),
        Union{Nothing,Float64}[],
        Dict{MOI.VariableIndex, String}(),
        Dict{MOI.ConstraintIndex, String}(),
        nothing,
        nothing,
        MOI.NLPBlockData([], _EmptyNLPEvaluator(), false),
        false,
        nothing,
        Dict{MOI.Nonlinear.ConstraintIndex,Float64}(),
        nothing,
        MOI.Nonlinear.SparseReverseMode(),
        Tuple{MOI.VectorOfVariables,_VectorNonlinearOracleCache}[],
        Int[],
        Int[],
        Int[],
        Int[],
        true,
        false,
        false,
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
    model.sense = MOI.FEASIBILITY_SENSE
    model.model = MOI.Nonlinear.ModelWithQuad(MOI.Nonlinear.Model())
    empty!(model.variable_primal_start)
    empty!(model.variable_names)
    empty!(model.constraint_names)
    model.name_to_variable = nothing
    model.name_to_constraint_index = nothing
    model.nlp_data = MOI.NLPBlockData([], _EmptyNLPEvaluator(), false)
    model.uses_nlp_block = false
    model.nlp_dual_start = nothing
    empty!(model.mult_g_nlp)
    model.evaluator = nothing
    # SKIP: model.ad_backend
    empty!(model.vector_nonlinear_oracle_constraints)
    empty!(model.jrows)
    empty!(model.jcols)
    empty!(model.hrows)
    empty!(model.hcols)
    model.needs_new_nlp = true
    model.has_only_linear_constraints = false
    model.islp = false
    model.jprod_available = false
    model.hprod_available = false
    model.hess_available = false
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.model.variables) &&
           isempty(model.variable_primal_start) &&
           model.nlp_data.evaluator isa _EmptyNLPEvaluator &&
           model.sense == MOI.FEASIBILITY_SENSE &&
           isempty(model.vector_nonlinear_oracle_constraints)
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(model, src)
end

MOI.get(::Optimizer, ::MOI.SolverName) = "MadNLP"

# The nonlinear model is `model.model.inner` and always exists; this guard
# only rejects mixing it with the legacy `MOI.NLPBlock` API.
function _check_no_nlp_block(model)
    if model.uses_nlp_block
        error("Cannot mix the new and legacy nonlinear APIs")
    end
    return
end

function MOI.supports_add_constrained_variable(
    ::Optimizer,
    ::Type{MOI.Parameter{Float64}},
)
    return true
end

function MOI.add_constrained_variable(
    model::Optimizer,
    set::MOI.Parameter{Float64},
)
    _check_no_nlp_block(model)
    p, ci = MOI.add_constrained_variable(model.model, set)
    return p, ci
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{Float64}}
)
    p = MOI.VariableIndex(ci.value)
    return MOI.is_valid(model.model, ci)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ListOfConstraintIndices{F,S},
) where {F<:MOI.VariableIndex,S<:MOI.Parameter{Float64}}
    return MOI.get(model.model, attr)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.NumberOfConstraints{MOI.VariableIndex,MOI.Parameter{Float64}},
)
    return MOI.get(model.model, attr)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{Float64}},
    set::MOI.Parameter{Float64},
)
    MOI.set(model.model, attr, ci, set)
    return
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:Union{MOI.VariableIndex,_FUNCTIONS}},
    ::Type{<:_SETS},
)
    return true
end

### MOI.ListOfConstraintTypesPresent

function _add_scalar_nonlinear_constraints(ret, nlp_model::MOI.Nonlinear.Model)
    for v in values(nlp_model.constraints)
        F, S = MOI.ScalarNonlinearFunction, typeof(v.set)
        if !((F, S) in ret)
            push!(ret, (F, S))
        end
    end
    return
end

function MOI.get(model::Optimizer, attr::MOI.ListOfConstraintTypesPresent)
    ret = MOI.get(model.model.variables, attr)
    append!(ret, MOI.get(model.model, attr))
    _add_scalar_nonlinear_constraints(ret, model.model.inner)
    if !isempty(model.vector_nonlinear_oracle_constraints)
        push!(ret, (MOI.VectorOfVariables, MOI.VectorNonlinearOracle{Float64}))
    end
    if !isempty(model.model.qp.parameters)
        push!(ret, (MOI.VariableIndex, MOI.Parameter{Float64}))
    end
    return ret
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
    push!(model.variable_primal_start, nothing)
    model.solver = nothing
    x = MOI.add_variable(model.model.variables)
    return x
end

function MOI.is_valid(model::Optimizer, x::MOI.VariableIndex)
    return MOI.is_valid(model.model, x)
end

function MOI.get(model::Optimizer, attr::MOI.ListOfVariableIndices)
    return MOI.get(model.model, attr)
end

function MOI.get(model::Optimizer, attr::MOI.NumberOfVariables)
    return MOI.get(model.model, attr)
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    return MOI.is_valid(model.model.variables, ci)
end

function MOI.get(
    model::Optimizer,
    attr::Union{
        MOI.NumberOfConstraints{MOI.VariableIndex,<:_SETS},
        MOI.ListOfConstraintIndices{MOI.VariableIndex,<:_SETS},
    },
)
    return MOI.get(model.model.variables, attr)
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ConstraintFunction,MOI.ConstraintSet},
    c::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    return MOI.get(model.model.variables, attr, c)
end

function MOI.add_constraint(model::Optimizer, x::MOI.VariableIndex, set::_SETS)
    index = MOI.add_constraint(model.model.variables, x, set)
    model.solver = nothing
    return index
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
    set::S,
) where {S<:_SETS}
    MOI.set(model.model.variables, MOI.ConstraintSet(), ci, set)
    model.needs_new_nlp = true
    return
end

function MOI.delete(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    MOI.delete(model.model.variables, ci)
    model.solver = nothing
    return
end

### ScalarAffineFunction and ScalarQuadraticFunction constraints

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F,<:_SETS},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
}
    return MOI.is_valid(model.model, ci)
end

function MOI.add_constraint(
    model::Optimizer,
    func::Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    set::_SETS,
)
    index = MOI.add_constraint(model.model, func, set)
    model.solver = nothing
    return index
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.NumberOfConstraints{F,S},MOI.ListOfConstraintIndices{F,S}},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S<:_SETS,
}
    return MOI.get(model.model, attr)
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ConstraintFunction,MOI.ConstraintSet},
    c::MOI.ConstraintIndex{F,<:_SETS},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
}
    return MOI.get(model.model, attr, c)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{F,S},
    set::S,
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
    S<:_SETS,
}
    MOI.set(model.model, MOI.ConstraintSet(), ci, set)
    model.needs_new_nlp = true
    return
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{<:MOI.ConstraintIndex{F,<:_SETS}},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
}
    return true
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDualStart,
    c::MOI.ConstraintIndex{F,<:_SETS},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
}
    return MOI.get(model.model, attr, c)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{F,<:_SETS},
    value::Union{Real,Nothing},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
}
    MOI.throw_if_not_valid(model, ci)
    MOI.set(model.model, attr, ci, value)
    model.needs_new_nlp = true
    return
end

### ScalarNonlinearFunction

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,<:_SETS},
)
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    return MOI.is_valid(model.model.inner, index)
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.ScalarNonlinearFunction,
    s::_SETS,
)
    _check_no_nlp_block(model)
    index = MOI.Nonlinear.add_constraint(model.model, f, s)
    model.solver = nothing
    return MOI.ConstraintIndex{typeof(f),typeof(s)}(index.value)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ListOfConstraintIndices{F,S},
) where {F<:MOI.ScalarNonlinearFunction,S<:_SETS}
    ret = MOI.ConstraintIndex{F,S}[]
    for (k, v) in model.model.inner.constraints
        if v.set isa S
            push!(ret, MOI.ConstraintIndex{F,S}(k.value))
        end
    end
    return ret
end

function MOI.get(
    model::Optimizer,
    attr::MOI.NumberOfConstraints{F,S},
) where {F<:MOI.ScalarNonlinearFunction,S<:_SETS}
    return count(v.set isa S for v in values(model.model.inner.constraints))
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
)
    return true
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
    func::MOI.ScalarNonlinearFunction,
)
    _check_no_nlp_block(model)
    MOI.Nonlinear.set_objective(model.model, func)
    model.solver = nothing
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,<:_SETS},
)
    MOI.throw_if_not_valid(model, ci)
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    return model.model.inner[index].set
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,S},
    set::S,
) where {S<:_SETS}
    MOI.throw_if_not_valid(model, ci)
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    func = model.model.inner[index].expression
    model.model.inner.constraints[index] = MOI.Nonlinear.Constraint(func, set)
    model.needs_new_nlp = true
    return
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{<:MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,<:_SETS}},
)
    return true
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,<:_SETS},
)
    MOI.throw_if_not_valid(model, ci)
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    return get(model.mult_g_nlp, index, nothing)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,<:_SETS},
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, ci)
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    if value === nothing
        delete!(model.mult_g_nlp, index)
    else
        model.mult_g_nlp[index] = convert(Float64, value)
    end
    model.needs_new_nlp = true
    return
end

### MOI.VectorOfVariables in MOI.VectorNonlinearOracle{Float64}

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorOfVariables},
    ::Type{MOI.VectorNonlinearOracle{Float64}},
)
    return true
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{
        MOI.VectorOfVariables,
        MOI.VectorNonlinearOracle{Float64},
    },
)
    return 1 <= ci.value <= length(model.vector_nonlinear_oracle_constraints)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ListOfConstraintIndices{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    n = length(model.vector_nonlinear_oracle_constraints)
    return MOI.ConstraintIndex{F,S}.(1:n)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.NumberOfConstraints{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    return length(model.vector_nonlinear_oracle_constraints)
end

function MOI.add_constraint(
    model::Optimizer,
    f::F,
    s::S,
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    model.solver = nothing
    cache = _VectorNonlinearOracleCache(s)
    push!(model.vector_nonlinear_oracle_constraints, (f, cache))
    n = length(model.vector_nonlinear_oracle_constraints)
    return MOI.ConstraintIndex{F,S}(n)
end

function row(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    offset = length(model.model)
    for i in 1:(ci.value-1)
        _, s = model.vector_nonlinear_oracle_constraints[i]
        offset += s.set.output_dimension
    end
    _, s = model.vector_nonlinear_oracle_constraints[ci.value]
    return offset .+ (1:s.set.output_dimension)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    f, _ = model.vector_nonlinear_oracle_constraints[ci.value]
    return MOI.get.(model, MOI.VariablePrimal(attr.result_index), f.variables)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    sign = -_dual_multiplier(model)
    f, s = model.vector_nonlinear_oracle_constraints[ci.value]
    λ = model.result.multipliers[row(model, ci)]
    λ .*= sign
    dual = zeros(MOI.dimension(s.set))
    # dual = λ' * J(x)
    _eval_constraint_transpose_jacobian_product(dual, model.result.solution, 0, f, s, λ)
    return dual
end

function MOI.get(
    model::Optimizer,
    attr::MOI.LagrangeMultiplier,
    ci::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    sign = -_dual_multiplier(model)
    return sign * model.result.multipliers[row(model, ci)]
end

function MOI.supports(
    ::Optimizer,
    ::MOI.LagrangeMultiplierStart,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    return true
end

function MOI.get(
    model::Optimizer,
    attr::MOI.LagrangeMultiplierStart,
    ci::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    _, cache = model.vector_nonlinear_oracle_constraints[ci.value]
    return cache.start
end

function MOI.set(
    model::Optimizer,
    attr::MOI.LagrangeMultiplierStart,
    ci::MOI.ConstraintIndex{F,S},
    start::Union{Nothing,Vector{Float64}},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    _, cache = model.vector_nonlinear_oracle_constraints[ci.value]
    cache.start = start
    model.needs_new_nlp = true
    return
end

### UserDefinedFunction

MOI.supports(model::Optimizer, ::MOI.UserDefinedFunction) = true

function MOI.set(model::Optimizer, attr::MOI.UserDefinedFunction, args)
    _check_no_nlp_block(model)
    MOI.Nonlinear.register_operator(
        model.model.inner,
        attr.name,
        attr.arity,
        args...,
    )
    return
end

### ListOfSupportedNonlinearOperators

function MOI.get(model::Optimizer, attr::MOI.ListOfSupportedNonlinearOperators)
    _check_no_nlp_block(model)
    return MOI.get(model.model.inner, attr)
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
    if MOI.Nonlinear._is_parameter(vi)
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
    if MOI.Nonlinear._is_parameter(vi)
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

### MOI.ConstraintDualStart

_dual_start(::Optimizer, ::Nothing, ::Int = 1) = 0.0

function _dual_start(model::Optimizer, value::Real, scale::Int = 1)
    return _dual_multiplier(model) * value * scale
end

### MOI.NLPBlockDualStart

MOI.supports(::Optimizer, ::MOI.NLPBlockDualStart) = true

function MOI.set(
    model::Optimizer,
    ::MOI.NLPBlockDualStart,
    values::Union{Nothing,Vector},
)
    model.nlp_dual_start = values
    model.needs_new_nlp = true
    return
end

MOI.get(model::Optimizer, ::MOI.NLPBlockDualStart) = model.nlp_dual_start

### MOI.NLPBlock

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

# This may also be set by `optimize!` and contain the block created from
# ScalarNonlinearFunction
MOI.get(model::Optimizer, ::MOI.NLPBlock) = model.nlp_data

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    if !MOI.is_empty(model.model.inner)
        error("Cannot mix the new and legacy nonlinear APIs")
    end
    model.nlp_data = nlp_data
    model.uses_nlp_block = !(nlp_data.evaluator isa _EmptyNLPEvaluator)
    model.solver = nothing
    return
end

### ObjectiveSense

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    model.sense = sense
    model.needs_new_nlp = true
    return
end

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

### ObjectiveFunction

function MOI.get(model::Optimizer, attr::MOI.ObjectiveFunctionType)
    if model.model.inner.objective !== nothing
        return MOI.ScalarNonlinearFunction
    end
    return MOI.get(model.model, attr)
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{
        <:Union{
            MOI.VariableIndex,
            MOI.ScalarAffineFunction{Float64},
            MOI.ScalarQuadraticFunction{Float64},
        },
    },
)
    return true
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{F},
) where {
    F<:Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
}
    return convert(F, MOI.get(model.model, attr))
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{F},
    func::F,
) where {
    F<:Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
}
    # This also clears any objective of the inner nonlinear model.
    MOI.Nonlinear.set_objective(model.model, func)
    model.solver = nothing
    return
end

### Eval_F_CB

function MOI.eval_objective(model::Optimizer, x)
    if model.sense == MOI.FEASIBILITY_SENSE
        return 0.0
    elseif model.nlp_data.has_objective
        return MOI.eval_objective(model.nlp_data.evaluator, x)
    end
    return MOI.eval_objective(model.evaluator, x)
end

### Eval_Grad_F_CB

function MOI.eval_objective_gradient(model::Optimizer, grad, x)
    if model.sense == MOI.FEASIBILITY_SENSE
        grad .= zero(eltype(grad))
    elseif model.nlp_data.has_objective
        MOI.eval_objective_gradient(model.nlp_data.evaluator, grad, x)
    else
        MOI.eval_objective_gradient(model.evaluator, grad, x)
    end
    return
end

### _OracleNLPEvaluator

"""
    _OracleNLPEvaluator(model::Optimizer)

The inner evaluator of the `MOI.Nonlinear.EvaluatorWithQuad` of `model`: the
rows of the `MOI.VectorNonlinearOracle` constraints followed by the rows of
the `MOI.NLPBlock` evaluator.
"""
struct _OracleNLPEvaluator{E<:MOI.AbstractNLPEvaluator} <:
       MOI.AbstractNLPEvaluator
    oracles::Vector{Tuple{MOI.VectorOfVariables,_VectorNonlinearOracleCache}}
    nlp::E
    nlp_bounds::Vector{MOI.NLPBoundsPair}
    has_objective::Bool
end

function _OracleNLPEvaluator(model::Optimizer)
    return _OracleNLPEvaluator(
        model.vector_nonlinear_oracle_constraints,
        model.nlp_data.evaluator,
        model.nlp_data.constraint_bounds,
        model.nlp_data.has_objective,
    )
end

_ncon_oracle(d::_OracleNLPEvaluator) = sum(s.set.output_dimension for (_, s) in d.oracles; init = 0)

function MOI.features_available(d::_OracleNLPEvaluator)
    features = MOI.features_available(d.nlp)
    if any(s.set.eval_hessian_lagrangian === nothing for (_, s) in d.oracles)
        features = setdiff(features, [:Hess, :HessVec])
    end
    if !isempty(d.oracles)
        # The oracles do not implement the Jacobian and Hessian products.
        features = setdiff(features, [:JacVec, :HessVec])
    end
    return features
end

function MOI.initialize(d::_OracleNLPEvaluator, features::Vector{Symbol})
    return MOI.initialize(d.nlp, features)
end

MOI.eval_objective(d::_OracleNLPEvaluator, x) = MOI.eval_objective(d.nlp, x)

function MOI.eval_objective_gradient(d::_OracleNLPEvaluator, grad, x)
    return MOI.eval_objective_gradient(d.nlp, grad, x)
end

function MOI.Nonlinear._constraint_bounds(d::_OracleNLPEvaluator)
    bounds = MOI.NLPBoundsPair[]
    for (_, s) in d.oracles
        for i in 1:s.set.output_dimension
            push!(bounds, MOI.NLPBoundsPair(s.set.l[i], s.set.u[i]))
        end
    end
    return append!(bounds, d.nlp_bounds)
end

MOI.Nonlinear._has_objective(d::_OracleNLPEvaluator) = d.has_objective

### Eval_G_CB

function _eval_constraint(
    g::AbstractVector,
    offset::Int,
    x::AbstractVector,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracleCache,
)
    for i in 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
    end
    ret = view(g, offset .+ (1:s.set.output_dimension))
    s.eval_f_timer += @elapsed s.set.eval_f(ret, s.x)
    return offset + s.set.output_dimension
end

function MOI.eval_constraint(d::_OracleNLPEvaluator, g, x)
    offset = 0
    for (f, s) in d.oracles
        offset = _eval_constraint(g, offset, x, f, s)
    end
    MOI.eval_constraint(d.nlp, view(g, (offset+1):length(g)), x)
    return
end

function MOI.eval_constraint(model::Optimizer, g, x)
    return MOI.eval_constraint(model.evaluator, g, x)
end

### Eval_Jac_G_CB

function _jacobian_structure(
    ret::AbstractVector,
    row_offset::Int,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracleCache,
)
    for (i, j) in s.set.jacobian_structure
        push!(ret, (row_offset + i, f.variables[j].value))
    end
    return row_offset + s.set.output_dimension
end

function MOI.jacobian_structure(d::_OracleNLPEvaluator)
    J = Tuple{Int,Int}[]
    offset = 0
    for (f, s) in d.oracles
        offset = _jacobian_structure(J, offset, f, s)
    end
    if length(d.nlp_bounds) > 0
        J_nlp = MOI.jacobian_structure(d.nlp)::Vector{Tuple{Int64,Int64}}
        for (row, col) in J_nlp
            push!(J, (row + offset, col))
        end
    end
    return J
end

MOI.jacobian_structure(model::Optimizer) = MOI.jacobian_structure(model.evaluator)

function _eval_constraint_jacobian(
    values::AbstractVector,
    offset::Int,
    x::AbstractVector,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracleCache,
)
    for i in 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
    end
    nnz = length(s.set.jacobian_structure)
    s.eval_jacobian_timer +=
        @elapsed s.set.eval_jacobian(view(values, offset .+ (1:nnz)), s.x)
    return offset + nnz
end

function MOI.eval_constraint_jacobian(d::_OracleNLPEvaluator, values, x)
    offset = 0
    for (f, s) in d.oracles
        offset = _eval_constraint_jacobian(values, offset, x, f, s)
    end
    nlp_values = view(values, (offset+1):length(values))
    MOI.eval_constraint_jacobian(d.nlp, nlp_values, x)
    return
end

function MOI.eval_constraint_jacobian(model::Optimizer, values, x)
    return MOI.eval_constraint_jacobian(model.evaluator, values, x)
end

# N.B.: VectorNonlinearOracle does not support transposed-Jacobian vector-product by default.
# This function uses the original Jacobian when we have to compute the product in MadNLP.
# This can be slow on large-scale instances.
function _eval_constraint_transpose_jacobian_product(
    Jtv::AbstractVector,
    x::AbstractVector,
    offset::Integer,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracleCache,
    v::AbstractVector,
)
    for i = 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
    end
    s.set.eval_jacobian(s.J_nzval, s.x)
    k = 0
    for (r, c) in s.set.jacobian_structure
        k += 1
        row = offset + r
        col = f.variables[c].value
        Jtv[col] += s.J_nzval[k] * v[row]
    end
    return
end

function MOI.eval_constraint_jacobian_transpose_product(
    d::_OracleNLPEvaluator,
    Jtv,
    x,
    v,
)
    # Evaluate jtprod for the nonlinear expressions FIRST: implementations of
    # MOI.eval_constraint_jacobian_transpose_product are allowed to overwrite
    # their output (ReverseAD's does), so it must be called before the
    # accumulating contributions of the oracles.
    offset = _ncon_oracle(d)
    v_nlp = view(v, (offset+1):length(v))
    MOI.eval_constraint_jacobian_transpose_product(d.nlp, Jtv, x, v_nlp)
    # Evaluate jtprod for all VectorNonlinearOracle.
    offset = 0
    for (f, s) in d.oracles
        _eval_constraint_transpose_jacobian_product(Jtv, x, offset, f, s, v)
        offset += s.set.output_dimension
    end
    return
end

function MOI.eval_constraint_jacobian_transpose_product(model::Optimizer, Jtv, x, v)
    return MOI.eval_constraint_jacobian_transpose_product(model.evaluator, Jtv, x, v)
end

function MOI.eval_constraint_jacobian_product(d::_OracleNLPEvaluator, Jv, x, v)
    @assert isempty(d.oracles)
    return MOI.eval_constraint_jacobian_product(d.nlp, Jv, x, v)
end

function MOI.eval_constraint_jacobian_product(model::Optimizer, Jv, x, v)
    return MOI.eval_constraint_jacobian_product(model.evaluator, Jv, x, v)
end

### Eval_H_CB

function _hessian_lagrangian_structure(
    ret::AbstractVector,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracleCache,
)
    for (i, j) in s.set.hessian_lagrangian_structure
        push!(ret, (f.variables[i].value, f.variables[j].value))
    end
    return
end

function MOI.hessian_lagrangian_structure(d::_OracleNLPEvaluator)
    H = Tuple{Int,Int}[]
    for (f, s) in d.oracles
        _hessian_lagrangian_structure(H, f, s)
    end
    append!(H, MOI.hessian_lagrangian_structure(d.nlp))
    return H
end

function MOI.hessian_lagrangian_structure(model::Optimizer)
    return MOI.hessian_lagrangian_structure(model.evaluator)
end

function _eval_hessian_lagrangian(
    H::AbstractVector,
    H_offset::Int,
    x::AbstractVector,
    μ::AbstractVector,
    μ_offset::Int,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracleCache,
)
    for i in 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
    end
    H_nnz = length(s.set.hessian_lagrangian_structure)
    H_view = view(H, H_offset .+ (1:H_nnz))
    μ_view = view(μ, μ_offset .+ (1:s.set.output_dimension))
    s.eval_hessian_lagrangian_timer +=
        @elapsed s.set.eval_hessian_lagrangian(H_view, s.x, μ_view)
    return H_offset + H_nnz, μ_offset + s.set.output_dimension
end

function MOI.eval_hessian_lagrangian(d::_OracleNLPEvaluator, H, x, σ, μ)
    offset, μ_offset = 0, 0
    for (f, s) in d.oracles
        offset, μ_offset =
            _eval_hessian_lagrangian(H, offset, x, μ, μ_offset, f, s)
    end
    H_nlp = view(H, (offset+1):length(H))
    μ_nlp = view(μ, (μ_offset+1):length(μ))
    MOI.eval_hessian_lagrangian(d.nlp, H_nlp, x, σ, μ_nlp)
    return
end

function MOI.eval_hessian_lagrangian(model::Optimizer, H, x, σ, μ)
    return MOI.eval_hessian_lagrangian(model.evaluator, H, x, σ, μ)
end

function MOI.eval_hessian_lagrangian_product(d::_OracleNLPEvaluator, Hv, x, v, σ, μ)
    @assert isempty(d.oracles)
    return MOI.eval_hessian_lagrangian_product(d.nlp, Hv, x, v, σ, μ)
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
    # Setting the backend will invalidate the model if it is different. But we
    # don't requrire == for `::MOI.Nonlinear.AutomaticDifferentiationBackend` so
    # act defensive and invalidate regardless.
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
    vars = MOI.get(model.model.variables, MOI.ListOfVariableIndices())
    if isempty(vars)
        model.invalid_model = true
        return
    end
    # Create NLP backend.
    # Rebuild even when the inner model is empty: a previous solve may have
    # left a stale `nlp_data` (for example, a nonlinear objective replaced by
    # a quadratic one).
    if !model.uses_nlp_block
        evaluator = MOI.Nonlinear.Evaluator(model.model.inner, model.ad_backend, vars)
        model.nlp_data = MOI.NLPBlockData(evaluator)
    end
    # Check model's structure.
    has_oracle = !isempty(model.vector_nonlinear_oracle_constraints)
    model.evaluator = MOI.Nonlinear.EvaluatorWithQuad(
        model.model,
        _OracleNLPEvaluator(model),
    )
    has_quadratic_constraints = any(
        isequal(MOI.Nonlinear._kFunctionTypeScalarQuadratic),
        model.model.qp.function_type,
    )
    has_nlp_constraints = !isempty(model.nlp_data.constraint_bounds) || has_oracle
    has_nlp_objective = model.nlp_data.has_objective
    features = MOI.features_available(model.evaluator)
    has_hessian = :Hess in features
    has_jacobian_operator = :JacVec in features
    has_hessian_operator = :HessVec in features

    model.has_only_linear_constraints = !has_quadratic_constraints && !has_nlp_constraints
    model.islp = model.has_only_linear_constraints && !has_nlp_objective
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
    if has_nlp_constraints
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

function _setup_nlp(model::Optimizer; array_type = nothing)
    if !model.needs_new_nlp
        return model.nlp
    end

    # Number of nonzeros for the jacobian and hessian of the Lagrangian
    nnzj = length(model.jrows)
    nnzh = length(model.hrows)

    # Initial variable
    nvar = length(model.model.variables.lower)
    x0 = zeros(Float64, nvar)
    for i in 1:length(model.variable_primal_start)
        x0[i] = if model.variable_primal_start[i] !== nothing
            model.variable_primal_start[i]
        else
            clamp(0.0, model.model.variables.lower[i], model.model.variables.upper[i])
        end
    end

    # Constraints bounds
    bounds = MOI.Nonlinear._constraint_bounds(model.evaluator)
    g_L = Float64[b.lower for b in bounds]
    g_U = Float64[b.upper for b in bounds]
    ncon = length(g_L)

    # Dual multipliers
    y0 = zeros(Float64, ncon)
    for (i, start) in enumerate(model.model.qp.mult_g)
        y0[i] = _dual_start(model, start, -1)
    end
    offset = length(model.model.qp.mult_g)
    if model.nlp_dual_start === nothing
        # First there is VectorNonlinearOracle...
        for (_, cache) in model.vector_nonlinear_oracle_constraints
            if cache.start !== nothing
                for i in 1:cache.set.output_dimension
                    y0[offset+i] = _dual_start(model, cache.start[i], -1)
                end
            end
            offset += cache.set.output_dimension
        end
        # ...then come the ScalarNonlinearFunctions
        for (key, val) in model.mult_g_nlp
            y0[offset+key.value] = _dual_start(model, val, -1)
        end
    else
        for (i, start) in enumerate(model.nlp_dual_start::Vector{Float64})
            y0[offset+i] = _dual_start(model, start, -1)
        end
    end

    nlp = MOIModel(
        NLPModels.NLPModelMeta(
            nvar,
            x0 = x0,
            lvar = model.model.variables.lower,
            uvar = model.model.variables.upper,
            ncon = ncon,
            y0 = y0,
            lcon = g_L,
            ucon = g_U,
            nnzj = nnzj,
            nnzh = nnzh,
            minimize = model.sense == MOI.MIN_SENSE,
            islp = model.islp,
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
    if model.has_only_linear_constraints
        options[:jacobian_constant] = true
    end
    # Clear timers
    for (_, s) in model.vector_nonlinear_oracle_constraints
        s.eval_f_timer = 0.0
        s.eval_jacobian_timer = 0.0
        s.eval_hessian_lagrangian_timer = 0.0
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
    return model.result.objective
end

### MOI.VariablePrimal

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    if MOI.Nonlinear._is_parameter(vi)
        p = MOI.Nonlinear.ParameterIndex(vi.value - MOI.Nonlinear._PARAMETER_OFFSET)
        return model.model.inner[p]
    end
    return model.result.solution[vi.value]
end

### MOI.ConstraintPrimal

function row(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
}
    return ci.value
end

function row(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction},
)
    offset = length(model.model)
    for (_, s) in model.vector_nonlinear_oracle_constraints
        offset += s.set.output_dimension
    end
    return offset + ci.value
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{<:_FUNCTIONS,<:_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.result.constraints[row(model, ci)]
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

_dual_multiplier(model::Optimizer) = model.sense == MOI.MIN_SENSE ? 1.0 : -1.0

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{<:_FUNCTIONS,<:_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    s = -_dual_multiplier(model)
    return s * model.result.multipliers[row(model, ci)]
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
    s = -_dual_multiplier(model)
    offset = length(model.model)
    return s .* model.result.multipliers[(offset+1):end]
end
