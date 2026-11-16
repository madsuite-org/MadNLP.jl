#=
    MadNLP wrappers

    MadNLP adapts any AbstractNLPModel to avoid numerical issues.
    The `AbstractNLPModel` is wrapped as a `SparseCallback` or as a `DenseCallback`
    (if both nlp.meta.sparse_jacobian and nlp.meta.sparse_hessian are set to false).

    The wrapper reformulates the model by:
    1. scaling the objective and the constraints.
    2. removing the fixed variables from the formulation.

    The scaling can be switched off by setting `nlp_scaling=false` in the options.

    Four cases can occur for the fixed variables:

    Case #1. The problem doesn't have any fixed variable
        MadNLP sets `fixed_variable_treatment=NoFixedVariables()`
        and falls back to the default callbacks (that just apply the scaling).

    Case #2. `fixed_variable_treatment=RelaxBound`
        MadNLP slightly relaxes the bounds of the fixed variables during
        the initialization. The wrapper falls back to the default callbacks, similarly as with Case #1.

    Case #3. `callback=DenseCallback` and `fixed_variable_treatment=MakeParameter`
        Reformulate the fixed variables as dummy variables that are kept
        at their bounds. The problem's dimension is not modified. The wrapper
        modifies the Jacobian by filling the fixed columns with 0,
        and the Hessian by filling the fixed columns and rows with 0 for
        the non-diagonal elements, 1 for the diagonal ones.

    Case #4. `callback=SparseCallback` and `fixed_variable_treatment=MakeParameter`
        Remove the fixed variables from the model. As a consequence, the problem's
        dimension is reduced after the reformulation.
=#

"""
    AbstractFixedVariableTreatment

Abstract type to define the reformulation of the fixed variables inside MadNLP.
"""
abstract type AbstractFixedVariableTreatment end

"""
    NoFixedVariables <: AbstractFixedVariableTreatment

Do nothing if the problem has no fixed variables.
"""
struct NoFixedVariables <: AbstractFixedVariableTreatment end

"""
    MakeParameter{T, VT, VI} <: AbstractFixedVariableTreatment

Remove the fixed variables from the optimization variables and
define them as problem's parameters.

"""
struct MakeParameter{T, VT,VI} <: AbstractFixedVariableTreatment
    free::VI
    fixed::VI
    ind_jac_free::VI
    ind_hess_free::VI
    hash_x::Base.RefValue{T}
    x_full::VT
    g_full::VT
end

"""
    RelaxBound <: AbstractFixedVariableTreatment

Relax the fixed variables ``x = x_{fixed}`` as bounded
variables ``x_{fixed} - ϵ ≤ x ≤ x_{fixed} + ϵ``, with
``ϵ`` a small-enough parameter.

"""
struct RelaxBound <: AbstractFixedVariableTreatment end

"""
    AbstractEqualityTreatment

Abstract type to define the reformulation of the equality
constraints inside MadNLP.
"""
abstract type AbstractEqualityTreatment end

"""
    EnforceEquality <: AbstractEqualityTreatment

Keep the equality constraints intact.

The solution returned by MadNLP will respect the equality constraints.
"""
struct EnforceEquality <: AbstractEqualityTreatment end

"""
    RelaxEquality <: AbstractEqualityTreatment

Relax the equality constraints ``g(x) = 0`` with two
inequality constraints, as ``-ϵ ≤ g(x) ≤ ϵ``. The parameter
``ϵ`` is usually small.

The solution returned by MadNLP will satisfy the equality
constraints only up to a tolerance ``ϵ``.

"""
struct RelaxEquality <: AbstractEqualityTreatment end

"""
    AbstractCallback{T, VT, FH}

Wrap the `AbstractNLPModel` passed by the user in a form amenable to MadNLP.

An `AbstractCallback` handles the scaling of the problem and the
reformulations of the equality constraints and fixed variables.

"""
abstract type AbstractCallback{T, VT, FH} end

"""
    create_callback(
        ::Type{Callback},
        nlp::AbstractNLPModel{T, VT};
        fixed_variable_treatment=MakeParameter,
        equality_treatment=EnforceEquality,
    ) where {T, VT}

Wrap the nonlinear program `nlp` using the callback wrapper
with type `Callback`. The option `fixed_variable_treatment`
decides if the fixed variables are relaxed (`RelaxBound`)
or removed (`MakeParameter`). The option `equality_treatment`
decides if the equality constraints are kept as is
(`EnforceEquality`) or relaxed (`RelaxEquality`).

"""
function create_callback end

"""
    SparseCallback{T, VT} <: AbstractCallback{T, VT}

Wrap an `AbstractNLPModel` using sparse structures.

"""
struct SparseCallback{
    T,
    VT<:AbstractVector{T},
    VI<:AbstractVector{Int},
    I<:AbstractNLPModel{T,VT},
    FH<:AbstractFixedVariableTreatment,
    EH<:AbstractEqualityTreatment,
} <: AbstractCallback{T,VT, FH}
    nlp::I
    nvar::Int
    ncon::Int
    nnzj::Int
    nnzh::Int

    con_buffer::VT
    jac_buffer::VT
    grad_buffer::VT
    hess_buffer::VT

    jac_I::VI
    jac_J::VI
    hess_I::VI
    hess_J::VI

    obj_scale::Base.RefValue{T}
    obj_sign::T
    con_scale::VT
    jac_scale::VT
    # Variable scaling, set by `nlp_scaling_method = :ruiz`: the solver works with x̃ = x ./ var_scale.
    var_scale::VT
    hess_scale::VT
    x_buffer::VT
    var_scaled::Base.RefValue{Bool}

    fixed_handler::FH
    equality_handler::EH

    ind_eq::VI
    ind_ineq::VI
    ind_fixed::VI
    ind_lb::VI
    ind_ub::VI
    ind_llb::VI
    ind_uub::VI
end

"""
    DenseCallback{T, VT} <: AbstractCallback{T, VT}

Wrap an `AbstractNLPModel` using dense structures.

"""
struct DenseCallback{
    T,
    VT<:AbstractVector{T},
    VI<:AbstractVector{Int},
    I<:AbstractNLPModel{T,VT},
    FH<:AbstractFixedVariableTreatment,
    EH<:AbstractEqualityTreatment,
} <: AbstractCallback{T,VT, FH}
    nlp::I
    nvar::Int
    ncon::Int

    con_buffer::VT
    grad_buffer::VT

    obj_scale::Base.RefValue{T}
    obj_sign::T
    con_scale::VT

    fixed_handler::FH
    equality_handler::EH

    ind_eq::VI
    ind_ineq::VI
    ind_fixed::VI
    ind_lb::VI
    ind_ub::VI
    ind_llb::VI
    ind_uub::VI
end

create_array(cb::AbstractCallback, args...) = similar(get_x0(cb.nlp), args...)

#=
    Scaling
=#
function set_obj_scale!(obj_scale, f::VT, max_gradient) where {T,VT<:AbstractVector{T}}
    return obj_scale[] = min(one(T), max_gradient / norm(f, Inf))
end

function set_con_scale_sparse!(
    con_scale::VT,
    jac_I,
    jac_buffer,
    max_gradient,
) where {T,VT<:AbstractVector{T}}
    fill!(con_scale, one(T))
    _set_con_scale_sparse!(con_scale, jac_I, jac_buffer)
    return map!(x -> min(one(T), max_gradient / x), con_scale, con_scale)
end
function _set_con_scale_sparse!(con_scale, jac_I, jac_buffer)
    @inbounds @simd for i in 1:length(jac_I)
        row = jac_I[i]
        con_scale[row] = max(con_scale[row], abs(jac_buffer[i]))
    end
end

function set_jac_scale_sparse!(
    jac_scale::VT,
    con_scale,
    jac_I,
) where {T,VT<:AbstractVector{T}}
    return copyto!(jac_scale, @view(con_scale[jac_I]))
end

function set_con_scale_dense!(
    con_scale::VT,
    jac_buffer,
    max_gradient,
) where {T,VT<:AbstractVector{T}}
    return con_scale .=
        min.(
            one(T),
            max_gradient ./ mapreduce(abs, max, jac_buffer, dims = 2, init = one(T)),
        )
end

function create_sparse_fixed_handler(
    fixed_variable_treatment::Type{MakeParameter},
    nlp,
    jac_I,
    jac_J,
    hess_I,
    hess_J,
    hess_buffer,
)
    n = get_nvar(nlp)
    lvar = get_lvar(nlp)
    uvar = get_uvar(nlp)
    nnzj = get_nnzj(nlp.meta)
    nnzh = get_nnzh(nlp.meta)

    isfixed = (lvar .== uvar)
    isfree = (lvar .< uvar)

    fixed = findall(isfixed)
    nfixed = length(fixed)

    if nfixed == 0
        return NoFixedVariables(), n, nnzj, nnzh
    end

    free = findall(isfree)
    nx = length(free)
    map_full_to_free = similar(jac_I, n) ; fill!(map_full_to_free, -1)
    map_full_to_free[free] .= 1:nx

    ind_jac_free = findall(@view(isfree[jac_J]))
    ind_hess_free = findall(@view(isfree[hess_I]) .&& @view(isfree[hess_J]))

    nnzh = length(ind_hess_free)
    Hi, Hj = similar(hess_I, nnzh), similar(hess_J, nnzh)
    copyto!(Hi, map_full_to_free[hess_I[ind_hess_free]])
    copyto!(Hj, map_full_to_free[hess_J[ind_hess_free]])
    resize!(hess_I, nnzh)
    resize!(hess_J, nnzh)
    copyto!(hess_I, Hi)
    copyto!(hess_J, Hj)

    nnzj = length(ind_jac_free)
    Ji, Jj = similar(jac_I, nnzj), similar(jac_J, nnzj)
    copyto!(Ji, jac_I[ind_jac_free])
    copyto!(Jj, map_full_to_free[jac_J[ind_jac_free]])
    resize!(jac_I, nnzj)
    resize!(jac_J, nnzj)
    copyto!(jac_I, Ji)
    copyto!(jac_J, Jj)

    x_full = copy(lvar)

    fixed_handler = MakeParameter(
        free,
        fixed,
        ind_jac_free,
        ind_hess_free,
        Ref(NaN),
        x_full,
        similar(lvar, n),
    )

    return fixed_handler, nx, nnzj, nnzh
end


function create_dense_fixed_handler(fixed_variable_treatment::Type{MakeParameter}, nlp)
    n = get_nvar(nlp)
    lvar = get_lvar(nlp)
    uvar = get_uvar(nlp)
    isfixed = (lvar .== uvar)
    fixed = findall(lvar .== uvar)
    if length(fixed) == 0
        return NoFixedVariables()
    else
        free = findall(lvar .< uvar)
        return MakeParameter(
            free,
            fixed,
            similar(fixed, 0),
            similar(fixed, 0),
            Ref(NaN),
            similar(lvar, 0),
            similar(lvar, n),
        )
    end
end

function create_sparse_fixed_handler(
    fixed_variable_treatment::Type{RelaxBound},
    nlp,
    jac_I,
    jac_J,
    hess_I,
    hess_J,
    hess_buffer,
)
    n = get_nvar(nlp)
    fixed_handler = RelaxBound()
    return fixed_handler, n, get_nnzj(nlp.meta), get_nnzh(nlp.meta)
end

function _parse_indexes(lvar::AbstractVector{T}, uvar, lcon, ucon, equality_treatment) where {T}
    m = length(lcon)
    if m > 0
        if equality_treatment == EnforceEquality
            is_equality = lcon .== ucon
            ind_eq = findall(is_equality)
            ind_ineq = findall(~, is_equality)
        else
            ind_eq = similar(lvar, Int, 0)
            ind_ineq = similar(lvar, Int, m) .= 1:m
        end
        xl = [lvar; @inbounds(view(lcon, ind_ineq))]
        xu = [uvar; @inbounds(view(ucon, ind_ineq))]
    else
        ind_eq = similar(lvar, Int, 0)
        ind_ineq = similar(lvar, Int, 0)
        xl = lvar
        xu = uvar
    end

    is_upper_unbounded = uvar .== T(Inf)
    is_lower_unbounded = lvar .== -T(Inf)
    ind_llb = findall((.~is_lower_unbounded) .* is_upper_unbounded)
    ind_uub = findall((is_lower_unbounded) .* (.~is_upper_unbounded))
    ind_lb = findall(xl .!= -T(Inf))
    ind_ub = findall(xu .!= T(Inf))

    return (
        xl=xl,
        xu=xu,
        ind_eq=ind_eq,
        ind_ineq=ind_ineq,
        ind_lb=ind_lb,
        ind_ub=ind_ub,
        ind_llb=ind_llb,
        ind_uub=ind_uub,
    )
end

#=
    Constructors
=#

function create_callback(
    ::Type{SparseCallback},
    nlp::AbstractNLPModel{T,VT};
    fixed_variable_treatment = MakeParameter,
    equality_treatment = EnforceEquality,
) where {T,VT}
    n = get_nvar(nlp)
    m = get_ncon(nlp)
    nnzj = get_nnzj(nlp.meta)
    nnzh = get_nnzh(nlp.meta)

    x0 = get_x0(nlp)

    jac_I = similar(x0, Int, nnzj)
    jac_J = similar(x0, Int, nnzj)
    hess_I = similar(x0, Int, nnzh)
    hess_J = similar(x0, Int, nnzh)

    jac_buffer = similar(x0, nnzj)
    fill!(jac_buffer, zero(T))
    obj_scale = Ref(one(T))
    con_scale = similar(jac_buffer, m)
    fill!(con_scale, one(T))

    if nnzj > 0
        NLPModels.jac_structure!(nlp, jac_I, jac_J)
    end
    if nnzh > 0
        NLPModels.hess_structure!(nlp, hess_I, hess_J)
    end

    hess_buffer = similar(x0, nnzh)
    fill!(hess_buffer, zero(T))
    fixed_handler, nx, nnzj, nnzh = create_sparse_fixed_handler(
        fixed_variable_treatment,
        nlp,
        jac_I,
        jac_J,
        hess_I,
        hess_J,
        hess_buffer,
    )
    equality_handler = equality_treatment()

    jac_scale = similar(jac_buffer, nnzj)
    fill!(jac_scale, one(T))
    var_scale = similar(x0, nx)
    fill!(var_scale, one(T))
    hess_scale = similar(x0, nnzh)
    fill!(hess_scale, one(T))
    x_buffer = similar(x0, nx)
    fill!(x_buffer, zero(T))
    con_buffer = similar(x0, m)
    fill!(con_buffer, zero(T))
    grad_buffer = similar(x0, nx)
    fill!(grad_buffer, zero(T))

    # Get indexing
    lvar = get_lvar(nlp)
    uvar = get_uvar(nlp)
    lcon = get_lcon(nlp)
    ucon = get_ucon(nlp)

    # Get fixed variables
    ind_fixed = findall(lvar .== uvar)
    if length(ind_fixed) > 0 && fixed_variable_treatment == MakeParameter
        ind_free = findall(lvar .< uvar)
        # Remove fixed variables from problem's formulation
        lvar = lvar[ind_free]
        uvar = uvar[ind_free]
    end

    indexes = _parse_indexes(lvar, uvar, lcon, ucon, equality_treatment)

    return SparseCallback(
        nlp,
        nx,
        m,
        nnzj,
        nnzh,
        con_buffer,
        jac_buffer,
        grad_buffer,
        hess_buffer,
        jac_I,
        jac_J,
        hess_I,
        hess_J,
        obj_scale,
        get_minimize(nlp) ? one(T) : -one(T),
        con_scale,
        jac_scale,
        var_scale,
        hess_scale,
        x_buffer,
        Ref(false),
        fixed_handler,
        equality_handler,
        indexes.ind_eq,
        indexes.ind_ineq,
        ind_fixed,
        indexes.ind_lb,
        indexes.ind_ub,
        indexes.ind_llb,
        indexes.ind_uub,
    )
end

function create_callback(
    ::Type{DenseCallback},
    nlp::AbstractNLPModel{T,VT};
    fixed_variable_treatment = MakeParameter,
    equality_treatment = EnforceEquality,
) where {T,VT}
    n = get_nvar(nlp)
    m = get_ncon(nlp)

    x0 = similar(get_x0(nlp))
    con_buffer = similar(x0, m)
    fill!(con_buffer, zero(T))
    grad_buffer = similar(x0, n)
    fill!(grad_buffer, zero(T))
    obj_scale = Ref(one(T))
    con_scale = similar(x0, m)
    fill!(con_scale, one(T))

    fixed_handler = create_dense_fixed_handler(fixed_variable_treatment, nlp)
    equality_handler = equality_treatment()

    # Get indexing
    lvar = get_lvar(nlp)
    uvar = get_uvar(nlp)
    lcon = get_lcon(nlp)
    ucon = get_ucon(nlp)

    indexes = _parse_indexes(lvar, uvar, lcon, ucon, equality_treatment)

    # Get fixed variables
    ind_fixed = findall(lvar .== uvar)
    if length(ind_fixed) > 0 && fixed_variable_treatment == MakeParameter
        # Keep fixed variables but remove them from lb/ub
        ind_lb = findall((indexes.xl .!= -T(Inf)) .* (indexes.xl .!= indexes.xu))
        ind_ub = findall((indexes.xu .!= T(Inf)) .* (indexes.xl .!= indexes.xu))
    else
        ind_lb = indexes.ind_lb
        ind_ub = indexes.ind_ub
    end

    return DenseCallback(
        nlp,
        n,
        m,
        con_buffer,
        grad_buffer,
        obj_scale,
        get_minimize(nlp) ? one(T) : -one(T),
        con_scale,
        fixed_handler,
        equality_handler,
        indexes.ind_eq,
        indexes.ind_ineq,
        ind_fixed,
        ind_lb,
        ind_ub,
        indexes.ind_llb,
        indexes.ind_uub,
    )
end

#=
    Callback's initialization
=#

# Constant subtracted from each constraint. Under `RelaxEquality` every row is a
# slack row whose bounds carry the right-hand side, so the constant is zero and the
# relaxation is applied once, to the slack bounds.
function _treat_equality_initialize!(::EnforceEquality, rhs, lcon, ucon)
    rhs .= (lcon .== ucon) .* lcon
    return
end
function _treat_equality_initialize!(::RelaxEquality, rhs, lcon, ucon)
    fill!(rhs, zero(eltype(rhs)))
    return
end
# Initiate fixed variables. By default, do nothing.
function _treat_fixed_variable_initialize!(cb::AbstractCallback, x0, lvar, uvar) end
function _treat_fixed_variable_initialize!(
    cb::DenseCallback{T, VT, VI, NLP, FH},
    x0,
    lvar,
    uvar,
)  where {T, VT, VI, NLP, FH<:MakeParameter}
    x0[cb.fixed_handler.fixed] .= lvar[cb.fixed_handler.fixed]
    lvar[cb.fixed_handler.fixed] .= -T(Inf)
    uvar[cb.fixed_handler.fixed] .=  T(Inf)
    return
end

function initialize!(
    cb::AbstractCallback,
    x,
    xl,
    xu,
    y0,
    rhs,
    ind_ineq;
    tol = 1e-8,
    bound_push = 1e-2,
    bound_fac = 1e-2,
)
    x0 = variable(x)
    lvar = variable(xl)
    uvar = variable(xu)

    nlp = cb.nlp
    con_buffer = cb.con_buffer
    grad_buffer = cb.grad_buffer

    x0 .= get_x0(cb)
    y0 .= get_y0(cb)
    lvar .= get_lvar(cb)
    uvar .= get_uvar(cb)
    lcon = get_lcon(nlp)
    ucon = get_ucon(nlp)

    _treat_fixed_variable_initialize!(cb, x0, lvar, uvar)

    set_initial_bounds!(lvar, uvar, tol)
    initialize_variables!(x0, lvar, uvar, bound_push, bound_fac)

    _eval_cons_wrapper!(cb, x0, con_buffer)

    slack(xl) .= @inbounds(view(lcon, ind_ineq))
    slack(xu) .= @inbounds(view(ucon, ind_ineq))
    _treat_equality_initialize!(cb.equality_handler, rhs, lcon, ucon)
    copyto!(slack(x), @inbounds(view(con_buffer, ind_ineq)))

    set_initial_bounds!(slack(xl), slack(xu), tol)
    initialize_variables!(slack(x), slack(xl), slack(xu), bound_push, bound_fac)
    return
end

#=
    Getters
=#

n_variables(cb::AbstractCallback) = get_nvar(cb.nlp)
n_constraints(cb::AbstractCallback) = get_ncon(cb.nlp)
get_x0(cb::AbstractCallback) = get_x0(cb.nlp)
get_y0(cb::AbstractCallback) = get_y0(cb.nlp)
get_lvar(cb::AbstractCallback) = get_lvar(cb.nlp)
get_uvar(cb::AbstractCallback) = get_uvar(cb.nlp)
# Getters to unpack solution
unpack_obj(cb::AbstractCallback, obj_val) = cb.obj_sign * obj_val ./ cb.obj_scale[]
function unpack_cons!(c_full::AbstractVector, cb::AbstractCallback, c::AbstractVector)
    c_full .= c ./ cb.con_scale
end
function unpack_x!(x_full, cb::AbstractCallback, x)
    x_full .= _model_x(cb, x)
end
function unpack_z!(z_full, cb::AbstractCallback, z)
    z_full .= z .* (cb.obj_sign / cb.obj_scale[])
    _div_var_scale!(cb, z_full)
end
function unpack_y!(y_full, cb::AbstractCallback, y)
    y_full .= (y .* cb.con_scale) .* (cb.obj_sign / cb.obj_scale[])
end

# N.B.: Special getters if we use SparseCallback with a MakeParameter fixed_handler,
# as the dimension of the problem is reduced.
function n_variables(cb::SparseCallback{T, VT, VI, NLP, FH}) where {T, VT, VI, NLP, FH<:MakeParameter}
    return length(cb.fixed_handler.free)
end
function get_x0(cb::SparseCallback{T, VT, VI, NLP, FH}) where {T, VT, VI, NLP, FH<:MakeParameter}
    free = cb.fixed_handler.free
    x0 = get_x0(cb.nlp)
    return x0[free]
end
function get_lvar(cb::SparseCallback{T, VT, VI, NLP, FH}) where {T, VT, VI, NLP, FH<:MakeParameter}
    free = cb.fixed_handler.free
    xl = get_lvar(cb.nlp)
    return xl[free]
end
function get_uvar(cb::SparseCallback{T, VT, VI, NLP, FH}) where {T, VT, VI, NLP, FH<:MakeParameter}
    free = cb.fixed_handler.free
    xu = get_uvar(cb.nlp)
    return xu[free]
end
function unpack_x!(x_full, cb::SparseCallback{T, VT, VI, NLP, FH}, x) where {T, VT, VI, NLP, FH<:MakeParameter}
    _update_x!(cb.fixed_handler, _model_x(cb, x))
    x_full .= cb.fixed_handler.x_full
end
function unpack_z!(z_full, cb::SparseCallback{T, VT, VI, NLP, FH}, z) where {T, VT, VI, NLP, FH<:MakeParameter}
    free = cb.fixed_handler.free
    z_full[free] .= z .* (cb.obj_sign / cb.obj_scale[])
    _div_var_scale!(cb, view(z_full, free))
end


"""
    _scaling_point(cb, x0)

The point at which the gradient-based scaling is computed: the starting point given by the
model, with fixed variables at their fixed values, before the bound push moves it into the
interior. Ipopt computes its scaling at the same point. `x0` is only used for its type and size.
"""
function _scaling_point(cb::AbstractCallback, x0)
    xs = similar(x0)
    lvar = similar(x0)
    uvar = similar(x0)
    xs .= get_x0(cb)
    lvar .= get_lvar(cb)
    uvar .= get_uvar(cb)
    _treat_fixed_variable_initialize!(cb, xs, lvar, uvar)
    return xs
end

"""
    _eval_for_scaling!(eval!, buf, xs, x0)

Evaluate `eval!(x, buf)` at the scaling point `xs`. If the evaluation throws or returns a
nonfinite value (a function such as `log(x)` evaluated on its bound), evaluate at the
bound-pushed point `x0` instead.
"""
function _eval_for_scaling!(eval!, buf, xs, x0)
    ok = try
        eval!(xs, buf)
        all(isfinite, buf)
    catch e
        e isa InterruptException && rethrow()
        false
    end
    ok || eval!(x0, buf)
    return buf
end

function set_scaling!(
    cb::SparseCallback,
    x,
    xl,
    xu,
    y0,
    rhs,
    ind_ineq,
    nlp_scaling_max_gradient;
    method::Symbol = :gradient,
    ruiz_sweeps::Int = 10,
    ruiz_min::Real = 1e-4,
    ruiz_max::Real = 1e4,
)
    method in (:ruiz, :osqp) && return set_ruiz_scaling!(cb, x, xl, xu, y0, rhs, ind_ineq, ruiz_sweeps, ruiz_min, ruiz_max; osqp = method === :osqp)
    method === :gradient || error("nlp_scaling_method must be :gradient, :ruiz or :osqp")
    x0 = variable(x)

    nlp = cb.nlp
    obj_scale = cb.obj_scale
    con_scale = cb.con_scale
    jac_scale = cb.jac_scale
    con_buffer = cb.con_buffer
    grad_buffer = cb.grad_buffer

    # Set scaling
    xs = _scaling_point(cb, x0)
    jac = similar(con_buffer, cb.nnzj)
    _eval_for_scaling!((x, buf) -> _eval_jac_wrapper!(cb, x, buf), jac, xs, x0)
    set_con_scale_sparse!(con_scale, cb.jac_I, jac, nlp_scaling_max_gradient)
    set_jac_scale_sparse!(jac_scale, con_scale, cb.jac_I)

    _eval_for_scaling!((x, buf) -> _eval_grad_f_wrapper!(cb, x, buf), grad_buffer, xs, x0)
    set_obj_scale!(obj_scale, grad_buffer, nlp_scaling_max_gradient)

    con_scale_slk = @inbounds(view(con_scale, ind_ineq))
    y0 ./= con_scale
    rhs .*= con_scale
    slack(x) .*= con_scale_slk
    slack(xl) .*= con_scale_slk
    slack(xu) .*= con_scale_slk
    return
end

# Round a scaling factor to the nearest power of 2 within [lo, hi], so that applying and undoing
# it is exact.
_ruiz_factor(d::T, lo, hi) where {T} = T(exp2(clamp(round(log2(d)), ceil(log2(lo)), floor(log2(hi)))))

# Ruiz factors: symmetric sweeps of RuizScaler on |K|, K = [H Jᵀ; J 0], then the objective.
function _ruiz_scaling(::Type{T}, nx, m, hI, hJ, hv, jI, jJ, jv, grad, sweeps, lo, hi) where {T}
    rows = vcat(hI, hJ, jI .+ nx, jJ)
    cols = vcat(hJ, hI, jJ, jI .+ nx)
    vals = vcat(hv, hv, jv, jv)
    K = sparse(rows, cols, vals, nx + m, nx + m, max)
    d = ones(T, nx + m)
    dnew = similar(d)
    for _ in 1:sweeps
        scaling_sweep!(dnew, d, getcolptr(K), rowvals(K), nonzeros(K), Int[])
        d, dnew = dnew, d
    end
    d .= _ruiz_factor.(d, lo, hi)
    dx = d[1:nx]
    dc = d[nx+1:end]

    # objective: 1 / max(mean column norm of D_x H D_x, ‖D_x ∇f‖∞), as in OSQP
    g = grad .* dx
    colmax = zeros(T, nx)
    for k in eachindex(hv)
        v = hv[k] * dx[hI[k]] * dx[hJ[k]]
        colmax[hI[k]] = max(colmax[hI[k]], v)
        colmax[hJ[k]] = max(colmax[hJ[k]], v)
    end
    c = max(nx > 0 ? sum(colmax) / nx : zero(T), nx > 0 ? maximum(abs, g) : zero(T))
    # a tiny or infinite norm leaves the objective unscaled
    obj_scale = (c < T(lo) || !isfinite(c)) ? one(T) : _ruiz_factor(one(T) / c, lo, hi)

    return dx, dc, obj_scale
end

# OSQP's scaling (scale_data in osqp/src/scaling.c) with P = H, A = J, q = ∇f: `iters` iterations of
#   D ← D / sqrt(col ∞-norms of [P; A]),  E ← E / sqrt(row ∞-norms of A),
#   c ← c / max(mean col ∞-norm of P, ‖q‖∞),
# each norm below 1e-4 taken as 1 and above 1e4 as 1e4. No bound on the accumulated factors.
function _osqp_scaling(::Type{T}, nx, m, hI, hJ, hv, jI, jJ, jv, q, iters) where {T}
    limit(v) = v < T(1e-4) ? one(T) : min(v, T(1e4))
    P, A = copy(hv), copy(jv)
    D, E, c = ones(T, nx), ones(T, m), one(T)
    Dt, Et = zeros(T, nx), zeros(T, m)
    for _ in 1:iters
        fill!(Dt, zero(T)); fill!(Et, zero(T))
        for k in eachindex(P)
            Dt[hI[k]] = max(Dt[hI[k]], P[k]); Dt[hJ[k]] = max(Dt[hJ[k]], P[k])
        end
        for k in eachindex(A)
            Dt[jJ[k]] = max(Dt[jJ[k]], A[k]); Et[jI[k]] = max(Et[jI[k]], A[k])
        end
        Dt .= one(T) ./ sqrt.(limit.(Dt))
        Et .= one(T) ./ sqrt.(limit.(Et))
        P .*= view(Dt, hI) .* view(Dt, hJ)
        A .*= view(Et, jI) .* view(Dt, jJ)
        q .*= Dt
        D .*= Dt; E .*= Et
        fill!(Dt, zero(T))
        for k in eachindex(P)
            Dt[hI[k]] = max(Dt[hI[k]], P[k]); Dt[hJ[k]] = max(Dt[hJ[k]], P[k])
        end
        ct = limit(max(nx > 0 ? sum(Dt) / nx : zero(T), limit(nx > 0 ? maximum(q) : zero(T))))
        P ./= ct; q ./= ct; c /= ct
    end
    return D, E, c
end

"""
    set_ruiz_scaling!(cb, x, xl, xu, y0, rhs, ind_ineq, sweeps, lo = 1e-4, hi = 1e4)

Scale the variables, the constraints and the objective by Ruiz equilibration at the starting
point. With H the Hessian of the objective and J the Jacobian at the point where
[`_scaling_point`](@ref) puts it, `sweeps` sweeps of symmetric infinity-norm scaling
([`scaling_sweep!`](@ref), the sweep of `RuizScaler`) of

    K = [H  Jᵀ
         J  0 ]

give D = diag(D_x, D_c), so that D K D has rows of infinity norm close to one. The variables
are scaled by D_x (the solver works with x̃ = x ./ D_x), the constraint rows by D_c, and the
objective by 1 / max(mean column norm of D_x H D_x, ‖D_x ∇f‖∞), as in OSQP. Every factor is
clamped to [lo, hi] and rounded to a power of 2. With `osqp = true` the factors are instead
OSQP's ([`_osqp_scaling`](@ref)).
"""
function set_ruiz_scaling!(cb::SparseCallback{T}, x, xl, xu, y0, rhs, ind_ineq, sweeps, lo = 1e-4, hi = 1e4; osqp = false) where {T}
    nx, m = cb.nvar, cb.ncon
    x0 = variable(x)

    # evaluate at the scaling point with every scaling off
    cb.obj_scale[] = one(T)
    fill!(cb.con_scale, one(T))
    fill!(cb.jac_scale, one(T))
    fill!(cb.var_scale, one(T))
    fill!(cb.hess_scale, one(T))
    cb.var_scaled[] = false

    xs = _scaling_point(cb, x0)
    jac = similar(cb.con_buffer, cb.nnzj)
    _eval_for_scaling!((x, buf) -> _eval_jac_wrapper!(cb, x, buf), jac, xs, x0)
    hess = similar(cb.con_buffer, cb.nnzh)
    y = similar(cb.con_buffer, m)
    fill!(y, zero(T))
    _eval_for_scaling!((x, buf) -> _eval_lag_hess_wrapper!(cb, x, y, buf), hess, xs, x0)
    _eval_for_scaling!((x, buf) -> _eval_grad_f_wrapper!(cb, x, buf), cb.grad_buffer, xs, x0)

    hI, hJ, jI, jJ = Array(cb.hess_I), Array(cb.hess_J), Array(cb.jac_I), Array(cb.jac_J)
    hv, jv = abs.(Array(hess)), abs.(Array(jac))
    if osqp
        dx, dc, obj_scale = _osqp_scaling(T, nx, m, hI, hJ, hv, jI, jJ, jv, abs.(Array(cb.grad_buffer)), sweeps)
    else
        dx, dc, obj_scale = _ruiz_scaling(T, nx, m, hI, hJ, hv, jI, jJ, jv, Array(cb.grad_buffer), sweeps, lo, hi)
    end

    cb.obj_scale[] = obj_scale
    copyto!(cb.var_scale, dx)
    copyto!(cb.con_scale, dc)
    copyto!(cb.jac_scale, dc[jI] .* dx[jJ])
    copyto!(cb.hess_scale, dx[hI] .* dx[hJ])
    cb.var_scaled[] = true

    con_scale_slk = @view(cb.con_scale[ind_ineq])
    y0 ./= cb.con_scale
    rhs .*= cb.con_scale
    slack(x) .*= con_scale_slk
    slack(xl) .*= con_scale_slk
    slack(xu) .*= con_scale_slk
    x0 ./= cb.var_scale
    variable(xl) ./= cb.var_scale
    variable(xu) ./= cb.var_scale
    return
end

function set_scaling!(
    cb::DenseCallback,
    x,
    xl,
    xu,
    y0,
    rhs,
    ind_ineq,
    nlp_scaling_max_gradient;
    method::Symbol = :gradient,
    ruiz_sweeps::Int = 10,
    ruiz_min::Real = 1e-4,
    ruiz_max::Real = 1e4,
)
    method === :gradient || error("nlp_scaling_method = :$method needs a SparseCallback")
    x0 = variable(x)

    nlp = cb.nlp
    obj_scale = cb.obj_scale
    con_scale = cb.con_scale
    con_buffer = cb.con_buffer
    grad_buffer = cb.grad_buffer
    # N.B.: Allocate jac_buffer here as it is used only once.
    #       GC should take care of it once the scaling has been initialized.
    jac_buffer = similar(grad_buffer, cb.ncon, cb.nvar)

    # Set scaling
    xs = _scaling_point(cb, x0)
    _eval_for_scaling!((x, buf) -> NLPModels.jac_dense!(nlp, x, buf), jac_buffer, xs, x0)
    set_con_scale_dense!(con_scale, jac_buffer, nlp_scaling_max_gradient)

    _eval_for_scaling!((x, buf) -> NLPModels.grad!(nlp, x, buf), grad_buffer, xs, x0)
    set_obj_scale!(obj_scale, grad_buffer, nlp_scaling_max_gradient)

    con_scale_slk = @inbounds(view(con_scale, ind_ineq))
    y0 ./= con_scale
    rhs .*= con_scale
    slack(x) .*= con_scale_slk
    slack(xl) .*= con_scale_slk
    slack(xu) .*= con_scale_slk
    return
end

#=
    Callbacks: default
=#

# Variable scaling. With it the solver works with x̃ = x ./ var_scale, so the model is evaluated at
# x = var_scale .* x̃, its derivatives with respect to x̃ carry a factor var_scale per variable, and
# bound multipliers are unscaled by dividing by var_scale. Only SparseCallback scales variables.
is_var_scaled(cb::AbstractCallback) = false
is_var_scaled(cb::SparseCallback) = cb.var_scaled[]
function _model_x(cb::AbstractCallback, x)
    is_var_scaled(cb) || return x
    cb.x_buffer .= x .* cb.var_scale
    return cb.x_buffer
end
# v .*= var_scale (or ./=), on the variables of the callback's space
function _mul_var_scale!(cb::AbstractCallback, v)
    is_var_scaled(cb) && (v .*= cb.var_scale)
    return v
end
function _div_var_scale!(cb::AbstractCallback, v)
    is_var_scaled(cb) && (v ./= cb.var_scale)
    return v
end
function _mul_hess_scale!(cb::AbstractCallback, h)
    is_var_scaled(cb) && (h .*= cb.hess_scale)
    return h
end

function _eval_f_wrapper(cb::AbstractCallback, x::AbstractVector)
    return NLPModels.obj(cb.nlp, _model_x(cb, x)) * cb.obj_scale[]
end

function _eval_cons_wrapper!(cb::AbstractCallback, x::AbstractVector, c::AbstractVector)
    NLPModels.cons!(cb.nlp, _model_x(cb, x), c)
    c .*= cb.con_scale
    return c
end

function _eval_grad_f_wrapper!(
    cb::AbstractCallback{T},
    x::AbstractVector,
    grad::AbstractVector,
) where {T}
    NLPModels.grad!(cb.nlp, _model_x(cb, x), grad)
    grad .*= cb.obj_scale[]
    _mul_var_scale!(cb, grad)
    return grad
end

function _eval_jtprod_wrapper!(
    cb::AbstractCallback,
    x::AbstractVector,
    v::AbstractVector,
    jvt::AbstractVector,
)
    y = cb.con_buffer
    y .= v .* cb.con_scale
    NLPModels.jtprod!(cb.nlp, _model_x(cb, x), y, jvt)
    _mul_var_scale!(cb, jvt)
    return jvt
end

#=
    Callbacks: SparseCallback
=#

function _jac_sparsity_wrapper!(cb::SparseCallback, I::AbstractVector, J::AbstractVector)
    copyto!(I, cb.jac_I)
    copyto!(J, cb.jac_J)
    return
end

function _hess_sparsity_wrapper!(cb::SparseCallback, I::AbstractVector, J::AbstractVector)
    copyto!(I, cb.hess_I)
    copyto!(J, cb.hess_J)
    return
end

function _eval_jac_wrapper!(
    cb::SparseCallback{T, VT, VI, NLP, FH},
    x::AbstractVector,
    jac::AbstractVector,
) where {T, VT, VI, NLP, FH}
    NLPModels.jac_coord!(cb.nlp, _model_x(cb, x), jac)
    jac .*= cb.jac_scale
    return jac
end

function _eval_jac_wrapper!(
    cb::SparseCallback{T},
    x::AbstractVector,
    jac::AbstractMatrix,
) where {T}
    jac_buffer = view(cb.jac_buffer, 1:cb.nnzj)
    _eval_jac_wrapper!(cb, x, jac_buffer)
    fill!(jac, zero(T))
    @inbounds @simd for k in 1:length(cb.jac_I)
        i, j = cb.jac_I[k], cb.jac_J[k]
        jac[i, j] += jac_buffer[k]
    end
    return jac
end

function _eval_lag_hess_wrapper!(
    cb::SparseCallback{T, VT, VI, NLP, FH},
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(T),
) where {T, VT, VI, NLP, FH}
    cb.con_buffer .= y .* cb.con_scale
    nnzh_orig = get_nnzh(cb.nlp.meta)
    NLPModels.hess_coord!(
        cb.nlp,
        _model_x(cb, x),
        cb.con_buffer,
        view(hess, 1:nnzh_orig);
        obj_weight = obj_weight * cb.obj_scale[],
    )
    _mul_hess_scale!(cb, view(hess, 1:nnzh_orig))
    return hess
end

function _eval_lag_hess_wrapper!(
    cb::SparseCallback{T},
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractMatrix;
    obj_weight = one(T),
) where {T}
    hess_buffer = view(cb.hess_buffer, 1:cb.nnzh)
    # N.B.: the objective scaling is applied inside the vector callback
    _eval_lag_hess_wrapper!(cb, x, y, hess_buffer; obj_weight = obj_weight)
    fill!(hess, zero(T))
    @inbounds @simd for k in 1:length(cb.hess_I)
        i, j = cb.hess_I[k], cb.hess_J[k]
        hess[i, j] += hess_buffer[k]
        if i != j
            hess[j, i] += hess_buffer[k]
        end
    end
    return hess
end

#=
    Callback: DenseCallback
=#

function _eval_jac_wrapper!(
    cb::DenseCallback{T, VT, VI, NLP, FH},
    x::AbstractVector,
    jac::AbstractMatrix,
) where {T, VT, VI, NLP, FH}
    NLPModels.jac_dense!(cb.nlp, x, jac)
    jac .*= cb.con_scale
    return jac
end

function _eval_lag_hess_wrapper!(
    cb::DenseCallback{T, VT, VI, NLP, FH},
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractMatrix;
    obj_weight = one(T),
) where {T, VT, VI, NLP, FH}
    NLPModels.hess_dense!(cb.nlp, x, y, hess; obj_weight = obj_weight * cb.obj_scale[])
    return hess
end

#=
    Callback for SparseCallback+MakeParameter
=#

function _update_x!(fixed_handler::MakeParameter, x::AbstractVector)
    idx = norm(x, 2)
    if fixed_handler.hash_x[] == idx
        return
    end
    fixed_handler.hash_x[] = idx
    # Update x_full
    free = fixed_handler.free
    fixed_handler.x_full[free] .= x
    return
end

function _eval_f_wrapper(
    cb::SparseCallback{T, VT, VI, NLP, FH},
    x::AbstractVector,
) where {T, VT, VI, NLP, FH<:MakeParameter}
    _update_x!(cb.fixed_handler, _model_x(cb, x))
    x_full = cb.fixed_handler.x_full
    return NLPModels.obj(cb.nlp, x_full) * cb.obj_scale[]
end

function _eval_cons_wrapper!(
    cb::SparseCallback{T, VT, VI, NLP, FH},
    x::AbstractVector,
    c::AbstractVector,
) where {T, VT, VI, NLP, FH<:MakeParameter}
    _update_x!(cb.fixed_handler, _model_x(cb, x))
    x_full = cb.fixed_handler.x_full
    NLPModels.cons!(cb.nlp, x_full, c)
    c .*= cb.con_scale
    return c
end

function _eval_grad_f_wrapper!(
    cb::SparseCallback{T, VT, VI, NLP, FH},
    x::AbstractVector,
    grad::AbstractVector,
) where {T, VT, VI, NLP, FH<:MakeParameter}
    _update_x!(cb.fixed_handler, _model_x(cb, x))
    x_full = cb.fixed_handler.x_full
    g_full = cb.fixed_handler.g_full
    NLPModels.grad!(cb.nlp, x_full, g_full)
    grad .= g_full[cb.fixed_handler.free]
    grad .*= cb.obj_scale[]
    _mul_var_scale!(cb, grad)
    return grad
end

function _eval_jac_wrapper!(
    cb::SparseCallback{T, VT, VI, NLP, FH},
    x::AbstractVector,
    jac::AbstractVector,
) where {T, VT, VI, NLP, FH<:MakeParameter}
    _update_x!(cb.fixed_handler, _model_x(cb, x))
    nnzj_orig = get_nnzj(cb.nlp.meta)
    jac_full = cb.jac_buffer
    x_full = cb.fixed_handler.x_full
    @assert length(jac_full) == nnzj_orig
    NLPModels.jac_coord!(cb.nlp, x_full, jac_full)
    jac .= @inbounds(view(jac_full, cb.fixed_handler.ind_jac_free))
    jac .*= cb.jac_scale
    return jac
end

function _eval_jtprod_wrapper!(
    cb::SparseCallback{T, VT, VI, NLP, FH},
    x::AbstractVector,
    v::AbstractVector,
    jvt::AbstractVector,
) where {T, VT, VI, NLP, FH<:MakeParameter}
    _update_x!(cb.fixed_handler, _model_x(cb, x))
    x_full = cb.fixed_handler.x_full
    jvt_full = cb.fixed_handler.g_full
    y = cb.con_buffer
    y .= v .* cb.con_scale
    NLPModels.jtprod!(cb.nlp, x_full, y, jvt_full)
    jvt .= jvt_full[cb.fixed_handler.free]
    _mul_var_scale!(cb, jvt)
    return jvt
end

function _eval_lag_hess_wrapper!(
    cb::SparseCallback{T, VT, VI, NLP, FH},
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(T),
) where {T, VT, VI, NLP, FH<:MakeParameter}
    _update_x!(cb.fixed_handler, _model_x(cb, x))
    x_full = cb.fixed_handler.x_full
    nnzh_orig = get_nnzh(cb.nlp.meta)
    hess_full = cb.hess_buffer
    @assert length(hess_full) == nnzh_orig
    cb.con_buffer .= y .* cb.con_scale
    NLPModels.hess_coord!(
        cb.nlp,
        x_full,
        cb.con_buffer,
        hess_full;
        obj_weight = obj_weight * cb.obj_scale[],
    )
    hess .= @inbounds(view(hess_full, cb.fixed_handler.ind_hess_free))
    _mul_hess_scale!(cb, hess)
    return hess
end

#=
    Callback for DenseCallback+MakeParameter
=#

function _eval_grad_f_wrapper!(
    cb::DenseCallback{T, VT, VI, NLP, FH},
    x::AbstractVector,
    grad::AbstractVector,
) where {T, VT, VI, NLP, FH<:MakeParameter}
    NLPModels.grad!(cb.nlp, x, grad)
    grad .*= cb.obj_scale[]
    map!(
         (x, y) -> x - y,
        @inbounds(view(grad, cb.fixed_handler.fixed)),
        @inbounds(view(x, cb.fixed_handler.fixed)),
        @inbounds(view(get_lvar(cb.nlp), cb.fixed_handler.fixed))
    )
    return grad
end

function _eval_jac_wrapper!(
    cb::DenseCallback{T, VT, VI, NLP, FH},
    x::AbstractVector,
    jac::AbstractMatrix,
) where {T, VT, VI, NLP, FH<:MakeParameter}
    NLPModels.jac_dense!(cb.nlp, x, jac)
    jac .*= cb.con_scale
    return jac[:, cb.fixed_handler.fixed] .= zero(T)
end

function _eval_lag_hess_wrapper!(
    cb::DenseCallback{T, VT, VI, NLP, FH},
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractMatrix;
    obj_weight = one(T),
) where {T, VT, VI, NLP, FH<:MakeParameter}
    NLPModels.hess_dense!(cb.nlp, x, y, hess; obj_weight = obj_weight * cb.obj_scale[])
    fixed = cb.fixed_handler.fixed
    hess[:, fixed] .= zero(T)
    hess[fixed, :] .= zero(T)
    _set_diag!(hess, fixed, one(T))
    return hess
end

#=
    Compute bounds' multipliers for fixed variables

    At a KKT solution, the multipliers (y, zl, zu) returned to the user satisfy

    ∇f + ∇cᵀ y - zl + zu = 0

    with (zl, zu) >= 0 for a min problem and (zl, zu) <= 0 for a max problem.
=#

# N.B.: by default do nothing as the bounds' multipliers are computed by the algorithm
function update_z!(cb, x, y, zl, zu, jacl) end

function update_z!(cb::AbstractCallback{T, VT, FH}, x, y, zl, zu, jacl) where {T, VT, FH<:MakeParameter}
    fixed_handler = cb.fixed_handler::MakeParameter
    ind_fixed = fixed_handler.fixed
    g_full = fixed_handler.g_full
    jtv = similar(g_full) ; fill!(jtv, zero(T))
    NLPModels.grad!(cb.nlp, x, g_full)       # ∇f
    NLPModels.jtprod!(cb.nlp, x, y, jtv)     # ∇cᵀ y
    g_full .+= jtv                           # ∇f + ∇cᵀ y
    g_fixed = @inbounds(view(g_full, ind_fixed))
    zl[ind_fixed] .= cb.obj_sign .* max.(zero(T), cb.obj_sign .* g_fixed)
    zu[ind_fixed] .= cb.obj_sign .* max.(zero(T), .-cb.obj_sign .* g_fixed)
    return
end

function _set_diag!(A, inds, a)
    @inbounds @simd for i in inds
        A[i, i] = a
    end
end
