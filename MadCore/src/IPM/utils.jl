"""
    MadNLPExecutionStats{T, VT} <: AbstractExecutionStats

Store the results returned by MadNLP once the interior-point
algorithm has terminated.

"""
mutable struct MadNLPExecutionStats{T, VT} <: AbstractExecutionStats
    options::AbstractOptions
    status::Status
    solution::VT
    objective::T
    constraints::VT
    dual_feas::T
    primal_feas::T
    multipliers::VT
    multipliers_L::VT
    multipliers_U::VT
    iter::Int
    counters::MadNLPCounters
end

function MadNLPExecutionStats(solver::AbstractMadNLPSolver{T}) where {T}
    n, m = get_nvar(get_nlp(solver)), get_ncon(get_nlp(solver))
    VT = typeof(get_x0(get_nlp(solver)))
    x = similar(VT, n)
    zl = similar(VT, n)
    zu = similar(VT, n)
    c = similar(VT, m)
    unpack_cons!(c, get_cb(solver), get_c(solver))
    unpack_x!(x, get_cb(solver), variable(get_x(solver)))
    unpack_z!(zl, get_cb(solver), variable(get_zl(solver)))
    unpack_z!(zu, get_cb(solver), variable(get_zu(solver)))
    return MadNLPExecutionStats(
        get_opt(solver),
        get_status(solver),
        x,
        unpack_obj(get_cb(solver), get_obj_val(solver)),
        c,
        get_inf_du(solver),
        get_inf_pr(solver),
        copy(get_y(solver)),
        zl,
        zu,
        0,
        get_cnt(solver),
    )
end

function update!(stats::MadNLPExecutionStats, solver::AbstractMadNLPSolver)
    stats.status = get_status(solver)
    unpack_x!(stats.solution, get_cb(solver), variable(get_x(solver)))
    unpack_y!(stats.multipliers, get_cb(solver), get_y(solver))
    unpack_z!(stats.multipliers_L, get_cb(solver), variable(get_zl(solver)))
    unpack_z!(stats.multipliers_U, get_cb(solver), variable(get_zu(solver)))
    stats.objective = unpack_obj(get_cb(solver), get_obj_val(solver))
    # rhs and slacks are stored in the scaled space: add them before unscaling
    stats.constraints .= get_c(solver) .+ get_rhs(solver)
    stats.constraints[get_ind_ineq(solver)] .+= slack(get_x(solver))
    unpack_cons!(stats.constraints, get_cb(solver), stats.constraints)
    stats.dual_feas = get_inf_du(solver)
    stats.primal_feas = get_inf_pr(solver)
    update_z!(get_cb(solver), stats.solution, stats.multipliers, stats.multipliers_L, stats.multipliers_U, get_jacl(solver))
    stats.iter = get_cnt(solver).k
    return stats
end

get_counters(nlp::NLPModels.AbstractNLPModel) = nlp.counters
get_counters(nlp::NLPModels.AbstractNLSModel) = nlp.counters.counters
getStatus(result::MadNLPExecutionStats) = get_status_output(result.status, result.options)

# Exceptions
struct InvalidNumberException <: Exception
    callback::Symbol
end
struct NotEnoughDegreesOfFreedomException <: Exception end

# Utilities
has_constraints(solver) = get_m(solver) != 0

@inline function count_lu_bounds(lb::AbstractVector, ub::AbstractVector)
    num_lu_bounds = 0
    num_le_bounds = 0
    num_ue_bounds = 0
    for ii in eachindex(lb)
        if (lb[ii] !=-Inf) && (ub[ii] != Inf)
            num_lu_bounds += 1
        elseif (lb[ii] == -Inf)
            num_ue_bounds += 1
        elseif (ub[ii] == Inf)
            num_le_bounds += 1
        end
    end
    return num_lu_bounds, num_le_bounds, num_ue_bounds
end


function get_vars_info(solver::AbstractMadNLPSolver)
    nlp = get_nlp(solver)

    x_lb = get_lvar(nlp)
    x_ub = get_uvar(nlp)
    num_fixed = length(get_ind_fixed(solver))
    num_var = get_nvar(nlp) - num_fixed
    num_llb_vars = length(get_ind_llb(solver))

    num_lu_vars,_,_ = count_lu_bounds(x_lb, x_ub)
    num_lu_vars = num_lu_vars - num_fixed
    num_uub_vars = length(get_ind_uub(solver))
    return (
        n_free=num_var,
        n_fixed=num_fixed,
        n_only_lb=num_llb_vars,
        n_only_ub=num_uub_vars,
        n_bounded=num_lu_vars,
    )
end

function get_cons_info(solver::AbstractMadNLPSolver)
    nlp = get_nlp(solver)

    g_lb = get_lcon(nlp)
    g_ub = get_ucon(nlp)

    num_eq_cons = length(solver.cb.ind_eq)
    num_ineq_cons = length(solver.cb.ind_ineq)
    _,num_le_cons,num_ue_cons = count_lu_bounds(g_lb, g_ub)
    num_lu_cons = num_ineq_cons - num_le_cons - num_ue_cons

    return (
        n_eq=num_eq_cons,
        n_ineq=num_ineq_cons,
        n_only_lb=num_le_cons,
        n_only_ub=num_ue_cons,
        n_bounded=num_lu_cons,
    )
end

# Print functions -----------------------------------------------------------
function print_init(solver::AbstractMadNLPSolver)
    @notice(get_logger(solver),@sprintf("Number of nonzeros in constraint Jacobian............: %8i", get_nnzj(get_nlp(solver).meta)))
    @notice(get_logger(solver),@sprintf("Number of nonzeros in Lagrangian Hessian.............: %8i\n", get_nnzh(get_nlp(solver).meta)))
    var_info = get_vars_info(solver)
    con_info = get_cons_info(solver)

    if get_nvar(get_nlp(solver)) < con_info.n_eq
        throw(NotEnoughDegreesOfFreedomException())
    end

    @notice(get_logger(solver),@sprintf("Total number of variables............................: %8i",var_info.n_free))
    @notice(get_logger(solver),@sprintf("                     variables with only lower bounds: %8i",var_info.n_only_lb))
    @notice(get_logger(solver),@sprintf("                variables with lower and upper bounds: %8i",var_info.n_bounded))
    @notice(get_logger(solver),@sprintf("                     variables with only upper bounds: %8i",var_info.n_only_ub))
    @notice(get_logger(solver),@sprintf("Total number of equality constraints.................: %8i",con_info.n_eq))
    @notice(get_logger(solver),@sprintf("Total number of inequality constraints...............: %8i",con_info.n_ineq))
    @notice(get_logger(solver),@sprintf("        inequality constraints with only lower bounds: %8i",con_info.n_only_lb))
    @notice(get_logger(solver),@sprintf("   inequality constraints with lower and upper bounds: %8i",con_info.n_bounded))
    @notice(get_logger(solver),@sprintf("        inequality constraints with only upper bounds: %8i\n",con_info.n_only_ub))
    return
end

function print_iter(solver::AbstractMadNLPSolver; is_resto=false)
    obj_scale = get_cb(solver).obj_scale[]
    mod(get_cnt(solver).k,10)==0&& @info(get_logger(solver),@sprintf(
        "iter    objective    inf_pr   inf_du inf_compl lg(mu) lg(rg) alpha_pr ir ls"))
    if is_resto
        RR = get_RR(solver)::RobustRestorer
        inf_du = RR.inf_du_R
        inf_pr = RR.inf_pr_R
        inf_compl = RR.inf_compl_R
        mu = log10(RR.mu_R)
    else
        inf_du = get_inf_du(solver)
        inf_pr = get_inf_pr(solver)
        inf_compl = get_inf_compl(solver)
        mu = log10(get_mu(solver))
    end
    @info(get_logger(solver),@sprintf(
        "%4i%s% 10.7e %6.2e %6.2e %7.2e %5.1f  %s  %6.2e %2i %2i%s",
        get_cnt(solver).k,is_resto ? "r" : " ",get_obj_val(solver)/obj_scale,
        inf_pr, inf_du, inf_compl, mu,
        # get_cnt(solver).k == 0 ? 0. : norm(primal(get_d(solver)),Inf),
        get_del_w(solver) == 0 ? "   - " : @sprintf("%5.1f",log(10,get_del_w(solver))),
        get_alpha(solver),
        get_cnt(solver).ir,
        get_cnt(solver).l,
        get_ftype(solver),))
    return
end

function print_summary(solver::AbstractMadNLPSolver)
    # TODO inquire this from nlpmodel wrapper
    obj_scale = get_cb(solver).obj_scale[]
    get_cnt(solver).solver_time = get_cnt(solver).total_time-get_cnt(solver).linear_solver_time-get_cnt(solver).eval_function_time

    @notice(get_logger(solver),"")
    @notice(get_logger(solver),"Number of Iterations....: $(get_cnt(solver).k)\n")
    @notice(get_logger(solver),"                                   (scaled)                 (unscaled)")
    @notice(get_logger(solver),@sprintf("Objective...............:  % 1.16e   % 1.16e",get_obj_val(solver),get_obj_val(solver)/obj_scale))
    @notice(get_logger(solver),@sprintf("Dual infeasibility......:   %1.16e    %1.16e",get_inf_du(solver),get_inf_du(solver)/obj_scale))
    @notice(get_logger(solver),@sprintf("Constraint violation....:   %1.16e    %1.16e",norm(get_c(solver),Inf),get_inf_pr(solver)))
    @notice(get_logger(solver),@sprintf("Complementarity.........:   %1.16e    %1.16e",
                                get_inf_compl(solver)*obj_scale,get_inf_compl(solver)))
    @notice(get_logger(solver),@sprintf("Overall NLP error.......:   %1.16e    %1.16e\n",
                                max(get_inf_du(solver)*obj_scale,norm(get_c(solver),Inf),get_inf_compl(solver)),
                                max(get_inf_du(solver),get_inf_pr(solver),get_inf_compl(solver))))

    @notice(get_logger(solver),"Number of objective function evaluations              = $(get_cnt(solver).obj_cnt)")
    @notice(get_logger(solver),"Number of objective gradient evaluations              = $(get_cnt(solver).obj_grad_cnt)")
    @notice(get_logger(solver),"Number of constraint evaluations                      = $(get_cnt(solver).con_cnt)")
    @notice(get_logger(solver),"Number of constraint Jacobian evaluations             = $(get_cnt(solver).con_jac_cnt)")
    @notice(get_logger(solver),"Number of Lagrangian Hessian evaluations              = $(get_cnt(solver).lag_hess_cnt)")
    @notice(get_logger(solver),"Number of KKT factorizations                          = $(get_cnt(solver).factorization_cnt)")
    @notice(get_logger(solver),"Number of KKT backsolves                              = $(get_cnt(solver).backsolve_cnt)\n")
    @notice(get_logger(solver),"Total wall secs in initialization                     = $(format_time(get_cnt(solver).init_time))")
    @notice(get_logger(solver),"Total wall secs in linear solver                      = $(format_time(get_cnt(solver).linear_solver_time))")
    @notice(get_logger(solver),"Total wall secs in NLP function evaluations           = $(format_time(get_cnt(solver).eval_function_time))")
    @notice(get_logger(solver),"Total wall secs in solver (w/o init./fun./lin. alg.)  = $(format_time(get_cnt(solver).total_time - get_cnt(solver).init_time - get_cnt(solver).linear_solver_time - get_cnt(solver).eval_function_time))")
    @notice(get_logger(solver),"Total wall secs                                       = $(format_time(get_cnt(solver).total_time))\n")
end

format_time(t::Float64) = isnan(t) ? " unavailable" : @sprintf("%6.3f s", t)

function string(solver::AbstractMadNLPSolver)
    """
                Interior point solver

                number of variables......................: $(get_nvar(get_nlp(solver)))
                number of constraints....................: $(get_ncon(get_nlp(solver)))
                number of nonzeros in Lagrangian Hessian.: $(get_nnzh(get_nlp(solver).meta))
                number of nonzeros in constraint Jacobian: $(get_nnzj(get_nlp(solver).meta))
                status...................................: $(get_status(solver))
                """
end
print(io::IO,solver::AbstractMadNLPSolver) = print(io, string(solver))
show(io::IO,solver::AbstractMadNLPSolver) = print(io,solver)
