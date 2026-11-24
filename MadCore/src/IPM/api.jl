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

for attribute in SOLVER_FIELDS
    getter = Symbol("get_", attribute)
    setter = Symbol("set_", attribute, "!")
    @eval begin
        @inline $getter(solver::AbstractMadNLPSolver) = getfield(solver, $(QuoteNode(attribute)))
        @inline $setter(solver::AbstractMadNLPSolver, rhs) = setfield!(solver, $(QuoteNode(attribute)), rhs)
    end
end

# Getter of a field that only some solvers have, used by `solve_refine_wrapper!`. The solver
# that has the field adds the method.
function get_iterator end

# Computed quantities
# TODO(@anton) I now think these would be useful to have, but which quantities are wrapped this way is a good question
get_kkt_error(solver::AbstractMadNLPSolver) = max(get_inf_pr(solver), get_inf_du(solver), get_inf_compl(solver))
get_inf_compl(solver::AbstractMadNLPSolver{T}, sc::T; mu=get_mu(solver)) where {T} = get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver), mu, sc)
