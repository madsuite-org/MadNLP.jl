
#=
    Regular filter line-search algorithm.
=#

# Moves the slacks of the trial point onto the constraints, if every original row already holds
# to `slack_reset_tol`. A one-sided row whose g(x) lies inside its bound then has no slack
# residual, so that θ measures infeasibility and not how far the slack lags g(x).
function reset_slacks!(solver::AbstractMadNLPSolver{T}) where {T}
    tol = get_opt(solver).slack_reset_tol
    xs = slack(get_x_trial(solver))
    xl = slack(get_xl(solver))
    xu = slack(get_xu(solver))
    c = get_c_trial(solver)
    cs = view(c, get_ind_ineq(solver))
    w = dual(get__w1(solver))
    ws = view(w, get_ind_ineq(solver))

    # equality rows
    w .= c
    ws .= zero(T)
    norm(w, Inf) > tol && return

    # inequality rows, at g(x) = c + s
    ws .= bound_violation.(cs .+ xs, xl, xu)
    norm(ws, Inf) > tol && return

    ws .= ifelse.(((cs .> 0) .& isinf.(xu)) .| ((cs .< 0) .& isinf.(xl)), cs, zero(T))
    xs .+= ws
    cs .-= ws
    return
end

function filter_line_search!(solver::AbstractMadNLPSolver{T}) where T
    # the reductions of the step are read together below (one wait on a GPU)
    theta = _deferred(get_theta, get_c(solver))
    varphi = _deferred(get_varphi, get_obj_val(solver), get_x_lr(solver), get_xl_r(solver), get_xu_r(solver), get_x_ur(solver), get_mu(solver))
    varphi_d = _deferred(get_varphi_d,
        primal(get_f(solver)),
        primal(get_x(solver)),
        primal(get_xl(solver)),
        primal(get_xu(solver)),
        primal(get_d(solver)),
        get_mu(solver),
    )

    alpha_max = _deferred(get_alpha_max,
        primal(get_x(solver)),
        primal(get_xl(solver)),
        primal(get_xu(solver)),
        primal(get_d(solver)),
        get_tau(solver),
    )
    alpha_z = _deferred(get_alpha_z, get_zl_r(solver), get_zu_r(solver), dual_lb(get_d(solver)), dual_ub(get_d(solver)), get_tau(solver))
    rel_search_norm = _deferred(get_rel_search_norm, primal(get_x(solver)), primal(get_d(solver)))
    theta, varphi, varphi_d, alpha_max, alpha_z, rel_search_norm =
        _values(theta, varphi, varphi_d, alpha_max, alpha_z, rel_search_norm)
    # During a watchdog, steps are accepted relative to the saved reference, as in Ipopt.
    if watchdog_active(solver)
        wd = watchdog_state(solver)
        theta, varphi, varphi_d = wd.theta_ref, wd.varphi_ref, wd.varphi_d_ref
    end
    set_alpha_z!(solver, alpha_z)
    alpha_min = get_alpha_min(theta,varphi_d,get_theta_min(solver),get_opt(solver).gamma_theta,get_opt(solver).gamma_phi,
                                get_opt(solver).alpha_min_frac,get_opt(solver).delta,get_opt(solver).s_theta,get_opt(solver).s_phi)
    get_cnt(solver).l = 1
    set_alpha!(solver, alpha_max)
    varphi_trial= zero(T)
    theta_trial = zero(T)
    small_search_norm = rel_search_norm < 10*eps(T)
    switching_condition = is_switching(varphi_d,get_alpha(solver),get_opt(solver).s_phi,get_opt(solver).delta,theta,get_opt(solver).s_theta)
    armijo_condition = false
    unsuccessful_iterate = false

    while true

        copyto!(full(get_x_trial(solver)), full(get_x(solver)))
        axpy!(get_alpha(solver), primal(get_d(solver)), primal(get_x_trial(solver)))
        set_obj_val_trial!(solver, eval_f_wrapper(solver, get_x_trial(solver)))
        eval_cons_wrapper!(solver, get_c_trial(solver), get_x_trial(solver))

        theta_trial, varphi_trial = _values(
            _deferred(get_theta, get_c_trial(solver)),
            _deferred(get_varphi, get_obj_val_trial(solver), get_x_trial_lr(solver), get_xl_r(solver), get_xu_r(solver), get_x_trial_ur(solver), get_mu(solver)),
        )
        armijo_condition = is_armijo(varphi_trial,varphi,get_opt(solver).eta_phi,get_alpha(solver),varphi_d)

        # A tiny step is accepted and flagged; the barrier update then decreases μ once. During a
        # watchdog it is not an acceptance relative to the reference.
        if small_search_norm
            get_cnt(solver).tiny_step = true
            watchdog_active(solver) && set_ftype!(solver, "w")
            break
        end

        set_ftype!(solver, get_ftype(
            get_filter(solver),theta,theta_trial,varphi,varphi_trial,switching_condition,armijo_condition,
            get_theta_min(solver),get_opt(solver).obj_max_inc,get_opt(solver).gamma_theta,get_opt(solver).gamma_phi,
            has_constraints(solver))
                   )

        if get_ftype(solver) in ("f","h")
            @trace(get_logger(solver),"Step accepted with type $(get_ftype(solver))")
            break
        end

        # During a watchdog, the full step is taken tentatively (type "w") when it is not accepted.
        # A full step that cannot be evaluated ends the watchdog: backtracking against the
        # reference from a tentative point cannot succeed, so regular! rolls back instead.
        if watchdog_active(solver) && get_cnt(solver).l == 1
            if is_valid(get_c_trial(solver)) && isfinite(varphi_trial) && isfinite(theta_trial)
                set_ftype!(solver, "w")
                break
            else
                return RESTORE
            end
        end

        if get_cnt(solver).l==1 && theta_trial>=theta
            if second_order_correction(
                solver,alpha_max,theta,varphi,theta_trial,varphi_d,switching_condition
            )
                break
            end
        end

        unsuccessful_iterate = true
        set_alpha!(solver, get_alpha(solver)/2)
        get_cnt(solver).l += 1
        if get_alpha(solver) < alpha_min
            @debug(get_logger(solver),
                    "Cannot find an acceptable step at iteration $(get_cnt(solver).k). Switching to restoration phase.")
            get_cnt(solver).k+=1
            return RESTORE
        else
            @trace(get_logger(solver),"Step rejected; proceed with the next trial step.")
            if get_alpha(solver) * norm(primal(get_d(solver))) < eps(T)*10
                if (get_cnt(solver).restoration_fail_count += 1) >= 4
                    return get_cnt(solver).acceptable_cnt >0 ?
                        SOLVED_TO_ACCEPTABLE_LEVEL : SEARCH_DIRECTION_BECOMES_TOO_SMALL
                else
                    # (experimental) while giving up directly
                    # we give MadNLP.jl second chance to explore
                    # some possibility at the current iterate

                    fill!(get_y(solver), zero(T))
                    fill!(get_zl_r(solver), one(T))
                    fill!(get_zu_r(solver), one(T))
                    empty!(get_filter(solver))
                    push!(get_filter(solver),(get_theta_max(solver),-Inf))
                    get_cnt(solver).k+=1

                    return REGULAR
                end
            end
        end
    end

    # this implements the heuristics in Section 3.2 of Ipopt paper.
    # Case I is only implemented
    if unsuccessful_iterate
        if (get_cnt(solver).unsuccessful_iterate += 1) >= 4
            if get_theta_max(solver)/10 > theta_trial
                @debug(get_logger(solver), "restarting filter")
                set_theta_max!(solver, get_theta_max(solver)/10)
                empty!(get_filter(solver))
                push!(get_filter(solver),(get_theta_max(solver),-Inf))
            end
            get_cnt(solver).unsuccessful_iterate = 0
        end
    else
        get_cnt(solver).unsuccessful_iterate = 0
    end

    # The filter is not augmented by a tentative watchdog step. A step that ends a watchdog
    # augments it as usual, with the switching and Armijo tests taken relative to the reference.
    # (Ipopt's reference-relative Armijo test uses the step length of the first watchdog step;
    # here it uses the current step length.)
    if (!switching_condition || !armijo_condition) && get_ftype(solver) != "w"
        @trace(get_logger(solver),"Augmenting filter.")
        augment_filter!(get_filter(solver), theta_trial, varphi_trial, get_opt(solver).gamma_theta, get_opt(solver).gamma_phi)
    end

    return LINESEARCH_SUCCEEDED
end

#=
    Variant of filter line-search for feasibility restoration phase.
=#

function filter_line_search_RR!(solver::AbstractMadNLPSolver{T}) where T
    RR = get_RR(solver)
    theta_R = get_theta_R(get_c(solver),RR.pp,RR.nn)
    varphi_R = get_varphi_R(RR.obj_val_R,get_x_lr(solver),get_xl_r(solver),get_xu_r(solver),get_x_ur(solver),RR.pp,RR.nn,RR.mu_R)
    varphi_d_R = get_varphi_d_R(
        RR.f_R,
        primal(get_x(solver)),
        primal(get_xl(solver)),
        primal(get_xu(solver)),
        primal(get_d(solver)),
        RR.pp,RR.nn,RR.dpp,RR.dnn,RR.mu_R,get_opt(solver).rho,
    )

    # set alpha_min
    alpha_max = get_alpha_max_R(
        primal(get_x(solver)),
        primal(get_xl(solver)),
        primal(get_xu(solver)),
        primal(get_d(solver)),
        RR.pp,RR.dpp,RR.nn,RR.dnn,RR.tau_R,
    )
    set_alpha_z!(solver, get_alpha_z_R(get_zl_r(solver),get_zu_r(solver),dual_lb(get_d(solver)),dual_ub(get_d(solver)),RR.zp,RR.dzp,RR.zn,RR.dzn,RR.tau_R))
    alpha_min = get_alpha_min(theta_R,varphi_d_R,get_theta_min(solver),get_opt(solver).gamma_theta,get_opt(solver).gamma_phi,
                                get_opt(solver).alpha_min_frac,get_opt(solver).delta,get_opt(solver).s_theta,get_opt(solver).s_phi)

    set_alpha!(solver, alpha_max)
    get_cnt(solver).l = 1
    theta_R_trial = zero(T)
    varphi_R_trial = zero(T)
    small_search_norm = get_rel_search_norm(primal(get_x(solver)), primal(get_d(solver))) < 10*eps(T)
    switching_condition = is_switching(varphi_d_R,get_alpha(solver),get_opt(solver).s_phi,get_opt(solver).delta,theta_R,get_opt(solver).s_theta)
    armijo_condition = false

    while true
        copyto!(full(get_x_trial(solver)), full(get_x(solver)))
        copyto!(RR.pp_trial,RR.pp)
        copyto!(RR.nn_trial,RR.nn)
        axpy!(get_alpha(solver),primal(get_d(solver)),primal(get_x_trial(solver)))
        axpy!(get_alpha(solver),RR.dpp,RR.pp_trial)
        axpy!(get_alpha(solver),RR.dnn,RR.nn_trial)

        RR.obj_val_R_trial = get_obj_val_R(
            RR.pp_trial,RR.nn_trial,RR.D_R,primal(get_x_trial(solver)),RR.x_ref,get_opt(solver).rho,RR.zeta)
        eval_cons_wrapper!(solver, get_c_trial(solver), get_x_trial(solver))
        theta_R_trial  = get_theta_R(get_c_trial(solver),RR.pp_trial,RR.nn_trial)
        varphi_R_trial = get_varphi_R(
            RR.obj_val_R_trial,get_x_trial_lr(solver),get_xl_r(solver),get_xu_r(solver),get_x_trial_ur(solver),RR.pp_trial,RR.nn_trial,RR.mu_R)

        armijo_condition = is_armijo(varphi_R_trial,varphi_R,get_opt(solver).eta_phi,get_alpha(solver),varphi_d_R)

        small_search_norm && break
        set_ftype!(solver, get_ftype(
            RR.filter,theta_R,theta_R_trial,varphi_R,varphi_R_trial,
            switching_condition,armijo_condition,
            get_theta_min(solver),get_opt(solver).obj_max_inc,get_opt(solver).gamma_theta,get_opt(solver).gamma_phi,
            has_constraints(solver))
                )
        get_ftype(solver) in ["f","h"] && (@trace(get_logger(solver),"Step accepted with type $(get_ftype(solver))"); break)

        set_alpha!(solver, get_alpha(solver)/2)
        get_cnt(solver).l += 1
        if get_alpha(solver) < alpha_min
            @debug(get_logger(solver),"Restoration phase cannot find an acceptable step at iteration $(get_cnt(solver).k).")
            if (get_cnt(solver).restoration_fail_count += 1) >= 4
                return RESTORATION_FAILED
            else
                # (experimental) while giving up directly
                # we give MadNLP.jl second chance to explore
                # some possibility at the current iterate

                fill!(get_y(solver), zero(T))
                fill!(get_zl_r(solver), one(T))
                fill!(get_zu_r(solver), one(T))
                empty!(get_filter(solver))
                push!(get_filter(solver),(get_theta_max(solver),-Inf))

                get_cnt(solver).k+=1
                get_cnt(solver).t+=1
                return REGULAR
            end
        else
            @trace(get_logger(solver),"Step rejected; proceed with the next trial step.")
            get_alpha(solver) < eps(T)*10 && return get_cnt(solver).acceptable_cnt >0 ?
                SOLVED_TO_ACCEPTABLE_LEVEL : SEARCH_DIRECTION_BECOMES_TOO_SMALL
        end
    end

    if !switching_condition || !armijo_condition
        @trace(get_logger(solver),"Augmenting restoration phase filter.")
        augment_filter!(RR.filter, theta_R_trial, varphi_R_trial, get_opt(solver).gamma_theta, get_opt(solver).gamma_phi)
    end

    return LINESEARCH_SUCCEEDED
end
