# API for getting internal members of IPM
# The idea here is that a subtype of AbstractMadNLPSolver may specialize
# some or all of these and the core functionality of the IPM in MadCore
# should function as normal.

# Fields that every solver built on MadCore has. Each field `x` has a getter `get_x(solver)`
# and a setter `set_x!(solver, val)`, which read and write the field of the same name. A
# solver with a different layout overloads the getters it needs, and a solver with more
# fields generates their accessors itself.
const SOLVER_FIELDS = (
    :nlp, :cb, :kkt, :opt, :cnt, :logger, :n, :m, :nlb, :nub, :x, :y, :zl, :zu, :xl,
    :xu, :obj_val, :f, :c, :jacl, :d, :p, :_w1, :_w2, :rhs, :ind_ineq, :ind_fixed,
    :ind_llb, :ind_uub, :x_lr, :x_ur, :xl_r, :xu_r, :zl_r, :zu_r, :dx_lr, :dx_ur,
    :inf_pr, :inf_du, :inf_compl, :mu, :del_w, :del_c, :status,
)

"""
    define_solver_accessors(mod::Module, S::Type, fields)

Define in `mod`, for each field `x` in `fields`, the getter `get_x(solver::S)` and the setter
`set_x!(solver::S, val)`, which read and write the field of the same name. A getter or setter
that `mod` imports from MadCore gets a new method.
"""
function define_solver_accessors(mod::Module, S::Type, fields)
    for attribute in fields
        getter = Symbol("get_", attribute)
        setter = Symbol("set_", attribute, "!")
        Core.eval(mod, quote
            @inline $getter(solver::$S) = getfield(solver, $(QuoteNode(attribute)))
            @inline $setter(solver::$S, rhs) = setfield!(solver, $(QuoteNode(attribute)), rhs)
        end)
    end
    return
end

define_solver_accessors(@__MODULE__, AbstractMadNLPSolver, SOLVER_FIELDS)

# Getter of a field that only some solvers have, used by `solve_refine_wrapper!`. The solver
# that has the field adds the method.
function get_iterator end

# Computed quantities
# TODO(@anton) I now think these would be useful to have, but which quantities are wrapped this way is a good question
get_inf_compl(solver::AbstractMadNLPSolver{T}, sc::T; mu=get_mu(solver)) where {T} = get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver), mu, sc)
