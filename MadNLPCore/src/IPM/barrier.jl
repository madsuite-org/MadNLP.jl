
"""
    update_barrier!(barrier::AbstractBarrierUpdate{T}, solver::AbstractMadNLPSolver{T}, sc::T) where T

Update barrier using the rule in `barrier`. Store the results in `solver.mu`.

"""
function update_barrier! end

# Weights of the bounds in the complementarity measures. A slack that relaxes an equality
# (RelaxEquality) has two bounds: with relaxed_equality_complementarity = false both get 0, with
# relaxed_equality_active_side = true the side with the larger dual gets 1 and the other 0
# (recomputed at each call); every other bound gets 1. `nothing` (all ones) with the defaults.
# The pairing is cached per solver.
const RELAXED_PAIR_WEIGHTS = WeakKeyDict{Any, Any}()
complementarity_weights(solver::AbstractMadNLPSolver) = nothing
function complementarity_weights(solver::MadNLPSolver{T}) where {T}
    opt = get_opt(solver)
    opt.relaxed_equality_complementarity && !opt.relaxed_equality_active_side && return nothing
    wl0, wu0, pl, pu = relaxed_pairs(solver)
    opt.relaxed_equality_active_side || return (wl0, wu0)
    # the active side of each relaxed pair: the bound with the larger dual
    wl = copy(wl0); wu = copy(wu0)
    zl = view(get_zl_r(solver), pl); zu = view(get_zu_r(solver), pu)
    view(wl, pl) .= ifelse.(zl .>= zu, one(T), zero(T))
    view(wu, pu) .= ifelse.(zl .>= zu, zero(T), one(T))
    return (wl, wu)
end

# The bounds of the relaxed equalities, cached per solver: static weights (0 on the pairs, 1
# elsewhere), the position of each pair in the lower and upper bound lists, and its primal index.
function relaxed_pairs(solver::MadNLPSolver{T}) where {T}
    return get!(RELAXED_PAIR_WEIGHTS, solver) do
        cb = get_cb(solver)
        ind = Array(get_ind_ineq(solver))
        nv = get_n(solver) - length(ind)
        eq = Array(cb.nlp.meta.lcon)[ind] .== Array(cb.nlp.meta.ucon)[ind]
        relaxed(i) = i > nv && eq[i - nv]
        ilb = Array(cb.ind_lb); iub = Array(cb.ind_ub)
        w(idx) = T[relaxed(i) ? zero(T) : one(T) for i in idx]
        posu = Dict(i => k for (k, i) in enumerate(iub))
        pairs = [(k, posu[i]) for (k, i) in enumerate(ilb) if relaxed(i) && haskey(posu, i)]
        wl = similar(get_zl_r(solver)); copyto!(wl, w(ilb))
        wu = similar(get_zu_r(solver)); copyto!(wu, w(iub))
        dv(v) = (o = similar(get_ind_ineq(solver), Int, length(v)); copyto!(o, v); o)
        (wl, wu, dv(first.(pairs)), dv(last.(pairs)), dv([ilb[k] for (k, _) in pairs]))
    end
end

step_bounds(solver::AbstractMadNLPSolver) = (primal(get_xl(solver)), primal(get_xu(solver)))
project_relaxed_slacks!(solver::AbstractMadNLPSolver, x) = nothing

# relaxed_equality_step_active_side: the primal bounds the step length sees, with the inactive side
# of each relaxed pair opened (the side with the smaller dual)
function step_bounds(solver::MadNLPSolver{T}) where {T}
    xl = primal(get_xl(solver)); xu = primal(get_xu(solver))
    get_opt(solver).relaxed_equality_step_active_side || return xl, xu
    _, _, pl, pu, gi = relaxed_pairs(solver)
    lower = view(get_zl_r(solver), pl) .>= view(get_zu_r(solver), pu)
    xl_e = copy(xl); xu_e = copy(xu)
    view(xl_e, gi) .= ifelse.(lower, view(xl, gi), T(-Inf))
    view(xu_e, gi) .= ifelse.(lower, T(Inf), view(xu, gi))
    return xl_e, xu_e
end

# keep the slack of each relaxed pair inside its interval on the inactive side after a step that
# only the active side limited; the active side is left to the fraction to the boundary, so the
# slack can still reach the bound it ends at
function project_relaxed_slacks!(solver::MadNLPSolver{T}, x) where {T}
    get_opt(solver).relaxed_equality_step_active_side || return
    _, _, pl, pu, gi = relaxed_pairs(solver)
    lower = view(get_zl_r(solver), pl) .>= view(get_zu_r(solver), pu)
    s = view(primal(x), gi)
    l = view(primal(get_xl(solver)), gi); u = view(primal(get_xu(solver)), gi)
    theta = T(1e-2)
    s .= ifelse.(lower, min.(s, u .- theta .* (u .- l)), max.(s, l .+ theta .* (u .- l)))
    return
end

# the weights the barrier update uses: none (every bound) with relaxed_equality_termination_only
barrier_weights(solver::AbstractMadNLPSolver) =
    get_opt(solver).relaxed_equality_termination_only ? nothing : complementarity_weights(solver)

function MadCore.get_inf_compl(solver::MadNLPSolver{T}, sc::T; mu = get_mu(solver)) where {T}
    # at μ = 0 this is the termination test; at μ > 0 the barrier update
    w = iszero(mu) ? complementarity_weights(solver) : barrier_weights(solver)
    w === nothing && return MadCore.get_inf_compl(get_x_lr(solver), get_xl_r(solver), get_zl_r(solver),
        get_xu_r(solver), get_x_ur(solver), get_zu_r(solver), mu, sc)
    wl, wu = w
    a = mapreduce((x, l, z, c) -> c * abs((x - l) * z - mu), max, get_x_lr(solver), get_xl_r(solver), get_zl_r(solver), wl; init = zero(T))
    b = mapreduce((u, x, z, c) -> c * abs((u - x) * z - mu), max, get_xu_r(solver), get_x_ur(solver), get_zu_r(solver), wu; init = zero(T))
    return max(a, b) / sc
end

function MadCore.get_average_complementarity(solver::MadNLPSolver{T}) where {T}
    w = barrier_weights(solver)
    w === nothing && return MadCore.get_average_complementarity(get_x_lr(solver), get_xl_r(solver), get_zl_r(solver),
        get_x_ur(solver), get_xu_r(solver), get_zu_r(solver))
    wl, wu = w
    n = sum(wl; init = zero(T)) + sum(wu; init = zero(T))
    n > 0 || return zero(T)
    a = mapreduce((x, l, z, c) -> c * (x - l) * z, +, get_x_lr(solver), get_xl_r(solver), get_zl_r(solver), wl; init = zero(T))
    b = mapreduce((u, x, z, c) -> c * (u - x) * z, +, get_xu_r(solver), get_x_ur(solver), get_zu_r(solver), wu; init = zero(T))
    return (a + b) / n
end

# Implement monotone update in a dedicated function as this code is reused
# throughout all the barrier updates.
function _update_monotone!(
    barrier::AbstractBarrierUpdate{T},
    solver::AbstractMadNLPSolver{T},
    sc::T
) where T
    set_inf_compl_mu!(solver, get_inf_compl(solver, sc))
    # After a tiny step, μ decreases once even if the barrier sub-problem is not solved. This is
    # a simplified form of Ipopt's tiny_step_flag, which waits for a second consecutive tiny step.
    tiny = get_cnt(solver).tiny_step
    get_cnt(solver).tiny_step = false
    while (get_mu(solver) > max((barrier.mu_min::T), get_opt(solver).tol/10)) &&
            (tiny || get_inf_barrier(solver) <= get_opt(solver).barrier_tol_factor * get_mu(solver))
        tiny = false
        mu_new = get_mu(
            get_mu(solver),
            (barrier.mu_min::T),
            (barrier.mu_linear_decrease_factor::T),
            (barrier.mu_superlinear_decrease_power::T),
            get_opt(solver).tol,
            get_opt(solver).barrier_tol_factor,
        )
        # At the floor, μ cannot decrease further
        mu_new == get_mu(solver) && break
        set_mu!(solver, mu_new)
        set_tau!(solver, get_tau(mu_new, get_opt(solver).tau_min))
        # Re-evaluate the barrier error with the new barrier parameter
        set_inf_compl_mu!(solver, get_inf_compl(solver, sc))
        empty!(get_filter(solver))
        push!(get_filter(solver), (get_theta_max(solver), -Inf))
    end
    return
end

# Implement monotone update for feasibility restoration.
# N.B: we always fall back to monotone update for restoration, no matter
# the barrier update being used in the regular phase.
function _update_monotone_RR!(
    barrier::AbstractBarrierUpdate{T},
    solver::AbstractMadNLPSolver{T},
    sc::T
) where T
    RR = get_RR(solver)
    inf_compl_mu_R = get_inf_compl_R(
        get_x_lr(solver),
        get_xl_r(solver),
        get_zl_r(solver),
        get_xu_r(solver),
        get_x_ur(solver),
        get_zu_r(solver),
        RR.pp,
        RR.zp,
        RR.nn,
        RR.zn,
        RR.mu_R,
        sc,
    )
    while RR.mu_R >= (barrier.mu_min::T) &&
        max(RR.inf_pr_R,RR.inf_du_R,inf_compl_mu_R) <= get_opt(solver).barrier_tol_factor*RR.mu_R
        mu_R = get_mu(
            RR.mu_R,
            (barrier.mu_min::T),
            (barrier.mu_linear_decrease_factor::T),
            (barrier.mu_superlinear_decrease_power::T),
            get_opt(solver).tol,
            get_opt(solver).barrier_tol_factor,
        )
        # At the floor, μ cannot decrease further
        mu_R == RR.mu_R && break
        RR.mu_R = mu_R
        inf_compl_mu_R = get_inf_compl_R(
            get_x_lr(solver),
            get_xl_r(solver),
            get_zl_r(solver),
            get_xu_r(solver),
            get_x_ur(solver),
            get_zu_r(solver),
            RR.pp,
            RR.zp,
            RR.nn,
            RR.zn,
            RR.mu_R,
            sc,
        )
        RR.tau_R= max(get_opt(solver).tau_min,1-RR.mu_R)
        RR.zeta = sqrt(RR.mu_R)
        empty!(RR.filter)
        push!(RR.filter,(get_theta_max(solver),-Inf))
    end
    return
end

function update_barrier!(barrier::MonotoneUpdate{T}, solver::AbstractMadNLPSolver{T}, sc::T) where T
    _update_monotone!(barrier, solver, sc)
end

#=
    Adaptive updates
=#

# Initial barrier parameter used when we fallback to the monotone barrier update.
function get_fixed_mu(solver::AbstractMadNLPSolver{T}, barrier::AbstractAdaptiveUpdate{T}) where T
    mu = T(0.8) * get_average_complementarity(solver)
    return clamp(mu, (barrier.mu_min::T), (barrier.mu_max::T))
end

function _check_progress(barrier::AbstractAdaptiveUpdate{T}, solver::AbstractMadNLPSolver{T}) where T
    if !barrier.globalization
        return true
    end
    kappa_1 = T(1e-5) # filter margin width
    kappa_2 = T(1.0)  # filter margin maximum width
    # Check current progress using filter line search
    theta = get_theta(get_c(solver))
    varphi = get_varphi(get_obj_val(solver), get_x_lr(solver), get_xl_r(solver), get_xu_r(solver), get_x_ur(solver), get_mu(solver))
    kkt_error = max(get_inf_pr(solver), get_inf_du(solver), get_inf_compl(solver))
    delta = kappa_1 * min(kappa_2, kkt_error)
    return is_filter_acceptable(get_filter(solver), theta + delta, varphi + delta)
end

function update_barrier!(barrier::AbstractAdaptiveUpdate{T}, solver::AbstractMadNLPSolver{T}, sc::T) where T
    old_mu = get_mu(solver)
    progress = _check_progress(barrier, solver)
    # Update state of barrier algorithm
    if !barrier.free_mode
        if progress
            @trace(get_logger(solver), "Moving adaptive barrier back to free mode.")
            barrier.free_mode = true
        else
            _update_monotone!(barrier, solver, sc)
        end
    else
        if !progress
            @trace(get_logger(solver), "Moving adaptive barrier to monotone mode.")
            barrier.free_mode = false
            # Reset barrier parameter using current average complementarity
            set_mu!(solver, get_fixed_mu(solver, barrier))
        else
            @trace(get_logger(solver), "Keeping adaptive barrier in free mode.")
        end
    end
    if barrier.free_mode
        set_mu!(solver, get_adaptive_mu(solver, barrier))
    end
    # Update tau and reset filter if barrier has been updated
    if get_mu(solver) != old_mu
        set_tau!(solver, get_tau(get_mu(solver), get_opt(solver).tau_min))
        empty!(get_filter(solver))
        push!(get_filter(solver), (get_theta_max(solver), -Inf))
    end
    return
end

# Evaluate the linear quality function described in [Nocedal2009, Eq. (4.2)]
function _evaluate_quality_function(solver::AbstractMadNLPSolver{T}, sigma, step_aff, step_cen, res_primal, res_dual) where T
    n, m = get_n(solver), get_m(solver)
    nlb, nub = get_nlb(solver), get_nub(solver)
    tau = get_tau(solver)
    d = get_d(solver) # Load buffer
    # Δ(σ) = Δ_aff + σ Δ_cen
    full(d) .= full(step_aff) .+ sigma .* full(step_cen)
    # Primal step
    alpha_pr = get_alpha_max(
        primal(get_x(solver)),
        primal(get_xl(solver)),
        primal(get_xu(solver)),
        primal(d),
        tau,
    )
    # Dual step
    alpha_du = get_alpha_z(
        get_zl_r(solver),
        get_zu_r(solver),
        dual_lb(d),
        dual_ub(d),
        tau,
    )
    w = barrier_weights(solver)
    # ||(X + αp ΔX - Xl) (Zl + αd ΔZl)||^2, without the bounds of relaxed equalities if so set
    inf_compl_lb = w === nothing ? mapreduce(
        (x, xl, dx, z, dz) -> ((x + alpha_pr * dx - xl) * (z + alpha_du * dz))^2,
        +,
        get_x_lr(solver), get_xl_r(solver), get_dx_lr(solver), get_zl_r(solver), dual_lb(d);
        init=zero(T),
    ) : mapreduce(
        (x, xl, dx, z, dz, c) -> c * ((x + alpha_pr * dx - xl) * (z + alpha_du * dz))^2,
        +,
        get_x_lr(solver), get_xl_r(solver), get_dx_lr(solver), get_zl_r(solver), dual_lb(d), w[1];
        init=zero(T),
    )

    # ||(Xu - X - αp ΔX) (Zu + αd ΔZu)||^2
    inf_compl_ub = w === nothing ? mapreduce(
        (x, xu, dx, z, dz) -> ((xu - x - alpha_pr * dx) * (z + alpha_du * dz))^2,
        +,
        get_x_ur(solver), get_xu_r(solver), get_dx_ur(solver), get_zu_r(solver), dual_ub(d);
        init=zero(T),
    ) : mapreduce(
        (x, xu, dx, z, dz, c) -> c * ((xu - x - alpha_pr * dx) * (z + alpha_du * dz))^2,
        +,
        get_x_ur(solver), get_xu_r(solver), get_dx_ur(solver), get_zu_r(solver), dual_ub(d), w[2];
        init=zero(T),
    )
    if w !== nothing
        nlb = sum(w[1]; init = zero(T))
        nub = sum(w[2]; init = zero(T))
    end
    # Primal infeasibility
    inf_pr = (m > 0) ? (one(T) - alpha_pr)^2 * res_primal^2 / m : zero(T)
    # Dual infeasibility
    inf_du = (one(T) - alpha_du)^2 * res_dual^2 / n
    # Complementarity infeasibility
    inf_compl = (nlb + nub) > 0 ? (inf_compl_lb + inf_compl_ub ) / (nlb + nub) : zero(T)

    @debug(get_logger(solver), @sprintf("sigma=%4.1e inf_pr=%4.2e inf_du=%4.2e inf_cc=%4.2e a_pr=%4.2e a_du=%4.2e", sigma, inf_pr, inf_du, inf_compl, alpha_pr, alpha_du))

    # Return quality function qL defined in Eq. (4.2)
    return inf_du + inf_pr + inf_compl
end

# Golden-section search algorithm to find a minimum of the linear quality function.
# The algorithm assumes the quality function is unimodal.
function _run_golden_search!(solver::AbstractMadNLPSolver{T}, barrier, sigma_lb, sigma_ub, step_aff, step_cen, res_primal, res_dual) where T
    gfac = (3 - sqrt(T(5))) / 2
    sigma_1, sigma_2 = sigma_lb, sigma_ub
    # Initiate Golden search
    phi_1 = _evaluate_quality_function(solver, sigma_1, step_aff, step_cen, res_primal, res_dual)
    phi_2 = _evaluate_quality_function(solver, sigma_2, step_aff, step_cen, res_primal, res_dual)
    sigma_1_in, sigma_2_in, phi_1_in, phi_2_in = sigma_1, sigma_2, phi_1, phi_2
    sigma_mid1 = sigma_lb + gfac * (sigma_ub - sigma_lb)
    sigma_mid2 = sigma_lb + (one(T) - gfac) * (sigma_ub - sigma_lb)
    phi_mid1 = _evaluate_quality_function(solver, sigma_mid1, step_aff, step_cen, res_primal, res_dual)
    phi_mid2 = _evaluate_quality_function(solver, sigma_mid2, step_aff, step_cen, res_primal, res_dual)
    # Run Golden search
    for i in 1:barrier.max_gs_iter
        if phi_mid1 > phi_mid2
            sigma_1 = sigma_mid1
            phi_1 = phi_mid1
            sigma_mid1 = sigma_mid2
            sigma_mid2 = sigma_1 + (one(T) - gfac) * (sigma_2 - sigma_1)
            phi_mid1 = phi_mid2
            phi_mid2 = _evaluate_quality_function(solver, sigma_mid2, step_aff, step_cen, res_primal, res_dual)
        else
            sigma_2 = sigma_mid2
            phi_2 = phi_mid2
            sigma_mid2 = sigma_mid1
            sigma_mid1 = sigma_1 + gfac * (sigma_2 - sigma_1)
            phi_mid2 = phi_mid1
            phi_mid1 = _evaluate_quality_function(solver, sigma_mid1, step_aff, step_cen, res_primal, res_dual)
        end
        if (sigma_2 - sigma_1 < barrier.sigma_tol * sigma_2)
            break
        end
    end
    # Compute final sigma
    sigma, phi = phi_mid1 < phi_mid2 ? (sigma_mid1, phi_mid1) : (sigma_mid2, phi_mid2)
    # Take into account that most the time the algorithm hasn't converged.
    if sigma_2 == sigma_2_in && phi_2_in < phi
        sigma = sigma_2_in
    elseif sigma_1 == sigma_1_in && phi_1_in < phi
        sigma = sigma_1_in
    end
    return sigma
end

function set_centering_aug_rhs!(solver::AbstractMadNLPSolver, kkt::AbstractKKTSystem, mu)
    px = primal(get_p(solver))
    py = dual(get_p(solver))
    pzl = dual_lb(get_p(solver))
    pzu = dual_ub(get_p(solver))
    px .= 0
    py .= 0
    pzl .= mu
    pzu .= -mu
    return
end

function get_adaptive_mu(solver::AbstractMadNLPSolver{T}, barrier::QualityFunctionUpdate{T}) where T
    # No inequality constraint: early return as barrier update is useless
    if get_nlb(solver) + get_nub(solver) == 0
        return barrier.mu_min
    end
    step_aff = get__w3(solver) # buffer 1
    step_cen = get__w4(solver) # buffer 2
    # The solves below use the last factorization of the KKT system. When the duals start at zero,
    # nothing has factorized it yet (the least-squares dual initialization would have): factorize
    # the current system first.
    if get_cnt(solver).factorization_cnt == 0
        set_aug_diagonal!(get_kkt(solver), solver)
        factorize_wrapper!(solver)
    end
    # Affine step
    set_aug_rhs!(solver, get_kkt(solver), get_c(solver), zero(T))
    # Get primal and dual infeasibility directly from the values in RHS p
    res_primal = norm(dual(get_p(solver)))
    res_dual = norm(primal(get_p(solver)))
    # Get approximate solution without iterative refinement
    copyto!(full(step_aff), full(get_p(solver)))
    solve_kkt!(get_kkt(solver), step_aff)
    # Get average complementarity
    mu = get_average_complementarity(solver)
    # Centering step
    set_centering_aug_rhs!(solver, get_kkt(solver), mu)
    # NOTE(@anton) Ipopt also applies the dual infeasibility perturbation for some reason???
    dual_inf_perturbation!(primal(get_p(solver)),get_ind_llb(solver),get_ind_uub(solver),mu,get_opt(solver).kappa_d)
    # Get (again) approximate solution without iterative refinement
    copyto!(full(step_cen), full(get_p(solver)))
    solve_kkt!(get_kkt(solver), step_cen)
    # Refine the search interval using Ipopt's heuristics
    # First, check if sigma is greater than 1.
    phi1 = _evaluate_quality_function(solver, one(T), step_aff, step_cen, res_primal, res_dual)
    sigma_1m = one(T) - T(1e-4)
    phi1m = _evaluate_quality_function(solver, sigma_1m, step_aff, step_cen, res_primal, res_dual)
    # Restrict search interval
    if phi1m > phi1
        sigma_min = one(T)
        sigma_max = min(barrier.sigma_max, barrier.mu_max / mu)
    else
        # μ_new = σ μ_avg ≥ μ / mu_decrease_max, μ the current barrier parameter
        sigma_min = max(barrier.sigma_min, barrier.mu_min / mu, get_mu(solver) / (barrier.mu_decrease_max * mu))
        sigma_max = min(max(sigma_min, sigma_1m), barrier.mu_max / mu)
    end
    # Run Golden-section search (assume the quality function is unimodal)
    sigma_opt = _run_golden_search!(solver, barrier, sigma_min, sigma_max, step_aff, step_cen, res_primal, res_dual)
    # Increment counter
    barrier.n_update += 1
    return clamp(sigma_opt * mu, barrier.mu_min, barrier.mu_max)
end

function get_adaptive_mu(solver::AbstractMadNLPSolver{T}, barrier::LOQOUpdate{T}) where T
    # No inequality constraint: early return as barrier update is useless
    if get_nlb(solver) + get_nub(solver) == 0
        return barrier.mu_min
    end
    mu = get_average_complementarity(solver) # get average complementarity.
    ncc = get_nlb(solver) + get_nub(solver)
    min_cc = get_min_complementarity(solver)
    xi = min_cc/mu
    sigma = barrier.gamma*min((1-barrier.r)*((1-xi)/xi),2)^3
    mu = clamp(sigma * mu, barrier.mu_min, barrier.mu_max)
    return mu
end

