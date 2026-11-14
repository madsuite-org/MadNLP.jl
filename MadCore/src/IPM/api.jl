# API for getting internal members of IPM
# The idea here is that a subtype of AbstractMadNLPSolver may specialize
# some or all of these and the core functionality of the IPM in MadCore
# should function as normal.

# Fields of `MadNLPSolver`. Each field `x` has a getter `get_x(solver)` and a setter
# `set_x!(solver, val)`, which read and write the field of the same name. A solver
# with a different layout overloads the getters it needs.
const SOLVER_FIELDS = (
    :nlp, :cb, :kkt, :opt, :cnt, :logger, :n, :m,
    :nlb, :nub, :x, :y, :zl, :zu, :xl, :xu,
    :obj_val, :f, :c, :jacl, :d, :p, :_w1, :_w2,
    :_w3, :_w4, :x_trial, :c_trial, :obj_val_trial, :c_slk, :rhs, :ind_ineq,
    :ind_fixed, :ind_llb, :ind_uub, :x_lr, :x_ur, :xl_r, :xu_r, :zl_r,
    :zu_r, :dx_lr, :dx_ur, :x_trial_lr, :x_trial_ur, :iterator, :inf_pr, :inf_du,
    :inf_compl, :inf_compl_mu, :mu, :tau, :alpha, :alpha_z,
    :del_w, :del_w_last, :del_c, :inertia_corrector, :intermediate_callback,
    :status, :output,
)

for attribute in SOLVER_FIELDS
    getter = Symbol("get_", attribute)
    setter = Symbol("set_", attribute, "!")
    @eval begin
        @inline $getter(solver::AbstractMadNLPSolver) = getfield(solver, $(QuoteNode(attribute)))
        @inline $setter(solver::AbstractMadNLPSolver, rhs) = setfield!(solver, $(QuoteNode(attribute)), rhs)
    end
end

# Computed quantities
# TODO(@anton) I now think these would be useful to have, but which quantities are wrapped this way is a good question
get_kkt_error(solver::AbstractMadNLPSolver) = max(get_inf_pr(solver), get_inf_du(solver), get_inf_compl(solver))
get_inf_compl(solver::AbstractMadNLPSolver{T}, sc::T; mu=get_mu(solver)) where {T} = get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver), mu, sc)
get_inf_barrier(solver::AbstractMadNLPSolver) = max(get_inf_pr(solver),get_inf_du(solver),get_inf_compl_mu(solver))
# Violation of the bounds l <= g <= u by g; zero inside them.
bound_violation(g, l, u) = max(l - g, g - u, zero(g))

"""
    get_inf_pr_orig(solver)

Primal infeasibility for the termination test. On equality rows it is |c|. On an inequality row
it is the larger of two quantities.
- The violation of the slack bounds by g(x) = c + s. Those bounds are the constraint bounds
  after scaling and the `bound_relax_factor` relaxation.
- The slack residual |c| weighted by the row's multiplier, |y| |c|.
A slack that lags g(x) inside the bounds therefore counts only as far as its multiplier makes it
matter. In particular, a slack held at its bound with a nonzero multiplier, while g(x) is inside,
still has to close the gap for the solve to stop.
"""
function get_inf_pr_orig(solver::AbstractMadNLPSolver{T}) where {T}
    # _w4 is scratch outside a linear solve; _w1 holds a backup during restore!
    buf = dual(get__w4(solver))
    ind_ineq = get_ind_ineq(solver)
    copyto!(buf, get_c(solver))
    c = _iview(get_c(solver), ind_ineq)
    y = _iview(get_y(solver), ind_ineq)
    s = slack(get_x(solver))
    _iview(buf, ind_ineq) .= max.(
        bound_violation.(c .+ s, slack(get_xl(solver)), slack(get_xu(solver))),
        abs.(y) .* abs.(c),
    )
    return norm(buf, Inf)
end
# The convergence test uses this measure in place of the slack residual; the filter and the
# barrier sub-problem keep the slack residual.
get_inf_total(solver::AbstractMadNLPSolver) = max(get_inf_pr_orig(solver), get_inf_du(solver), get_inf_compl(solver))
