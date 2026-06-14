# API for getting internal members of IPM
# The idea here is that a subtype of AbstractMadNLPSolver may specialize
# some or all of these and the core functionality of the IPM in MadNLP
# should function as normal.

# getters for all fields
for (k, attribute) in enumerate(fieldnames(MadNLPSolver))
    fname = "get_$(attribute)"
    @eval begin
        @inline function $(Symbol(fname))(solver::MadNLPSolver)
            return getfield(solver, $k)
        end
    end
end

# Necessary Mutators
for (k, (attribute, type)) in enumerate(zip(fieldnames(MadNLPSolver), fieldtypes(MadNLPSolver)))
    # These types shouldn't ever be set by value so we ignore them.
    if type <: Union{AbstractVector,
                     PrimalVector,
                     AbstractKKTVector,
                     MadNLPOptions,
                     MadNLPCounters,
                     MadNLPLogger,
                     AbstractKKTSystem,
                     AbstractNLPModel,
                     AbstractCallback,
                     AbstractIterator,
                     AbstractInertiaCorrector}
      continue
    end
    fname = "set_$(attribute)!"
    @eval begin
        @inline function $(Symbol(fname))(solver::MadNLPSolver, rhs)
            return setfield!(solver, $k, rhs)
        end
    end
end

# Computed quantities
# TODO(@anton) I now think these would be useful to have, but which quantities are wrapped this way is a good question
get_theta(solver::AbstractMadNLPSolver) = get_theta(get_c(solver))
get_varphi(solver::AbstractMadNLPSolver) = get_varphi(get_obj_val(solver), get_x_lr(solver), get_xl_r(solver), get_xu_r(solver), get_x_ur(solver), get_mu(solver))
get_kkt_error(solver::AbstractMadNLPSolver) = max(get_inf_pr(solver), get_inf_du(solver), get_inf_compl(solver))
get_inf_compl(solver::AbstractMadNLPSolver{T}, sc::T; mu=get_mu(solver)) where {T} = get_inf_compl(get_x_lr(solver),get_xl_r(solver),get_zl_r(solver),get_xu_r(solver),get_x_ur(solver),get_zu_r(solver), mu, sc)
get_inf_barrier(solver::AbstractMadNLPSolver) = max(get_inf_pr(solver),get_inf_du(solver),get_inf_compl_mu(solver))
# Primal infeasibility measured as the ORIGINAL-NLP constraint violation (Ipopt
# curr_nlp_constraint_violation), NOT the slack-equality residual ‖g(x)-s‖ that
# get_inf_pr(solver) holds. Equality rows keep |g(x)-rhs| (=|c|); inequality rows
# use the violation of g(x)=c+s against its bounds [slack(xl),slack(xu)]. Ipopt uses
# this for its convergence error (IpIpoptCalculatedQuantities.cpp:3094, with the
# slack-residual variant commented out) while the filter θ and barrier sub-problem
# error keep using the slack residual.
function get_inf_pr_orig(solver::AbstractMadNLPSolver{T}; unscaled::Bool=false) where T
    buf = dual(get__w1(solver))
    ind_ineq = get_ind_ineq(solver)
    sl = slack(get_xl(solver))
    su = slack(get_xu(solver))
    copyto!(buf, get_c(solver))
    buf[ind_ineq] .+= slack(get_x(solver))                                  # g(x) on ineq rows
    buf[ind_ineq] .= max.(zero(T), buf[ind_ineq] .- su, sl .- buf[ind_ineq]) # bound violation
    # Unscale per-row (c_scaled = con_scale * c_unscaled) to get the violation in the original NLP.
    unscaled && (buf ./= get_cb(solver).con_scale)
    return norm(buf, Inf)
end
get_inf_total(solver::AbstractMadNLPSolver) = max(get_inf_pr_orig(solver),get_inf_du(solver),get_inf_compl(solver))

# Ipopt-style convergence test: overall scaled error <= tol AND the three UNSCALED component
# tolerances (IpOptErrorConvCheck.cpp:226). dual_inf and complementarity are unscaled by dividing
# out obj_scale (the objective scaling), the constraint violation by the per-row con_scale.
function is_converged(solver::AbstractMadNLPSolver{T}) where T
    opt = get_opt(solver)
    get_inf_total(solver) <= opt.tol || return false
    obj_scale = get_cb(solver).obj_scale[]
    (get_inf_du(solver) / obj_scale <= opt.dual_inf_tol) || return false
    (get_inf_pr_orig(solver; unscaled=true) <= opt.constr_viol_tol) || return false
    (get_inf_compl(solver) <= opt.compl_inf_tol) || return false
    return true
end

# Ipopt-style acceptable-point test: same structure as is_converged but with the looser
# acceptable_* component tolerances (IpOptErrorConvCheck.cpp:328). The obj-change term is
# omitted (acceptable_obj_change_tol defaults to 1e20, i.e. vacuous).
function is_acceptable(solver::AbstractMadNLPSolver{T}) where T
    opt = get_opt(solver)
    get_inf_total(solver) <= opt.acceptable_tol || return false
    obj_scale = get_cb(solver).obj_scale[]
    (get_inf_du(solver) / obj_scale <= opt.acceptable_dual_inf_tol) || return false
    (get_inf_pr_orig(solver; unscaled=true) <= opt.acceptable_constr_viol_tol) || return false
    (get_inf_compl(solver) <= opt.acceptable_compl_inf_tol) || return false
    return true
end
