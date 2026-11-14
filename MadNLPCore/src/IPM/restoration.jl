function WatchdogState(solver::AbstractMadNLPSolver{T}) where {T}
    VT = typeof(get_y(solver))
    vec(x) = similar(get_y(solver), length(x))
    return WatchdogState{T, VT}(
        false, 0, 0,
        vec(full(get_x(solver))), vec(get_y(solver)),
        vec(full(get_zl(solver))), vec(full(get_zu(solver))),
        vec(full(get_xl(solver))), vec(full(get_xu(solver))),
        vec(get_c(solver)),
        zero(T), zero(T), zero(T), zero(T), (zero(T), zero(T)), (zero(T), zero(T)),
        vec(full(get_d(solver))),
    )
end

# Saves the iterate and the search direction as the watchdog's reference. The filter is not
# saved: tentative steps never augment it, and the only change it can undergo during a watchdog
# is the reset by a change of μ, which must persist after a rollback.
function save_watchdog_iterate!(solver::AbstractMadNLPSolver{T}, wd::WatchdogState) where {T}
    copyto!(wd.x_ref, full(get_x(solver)))
    copyto!(wd.y_ref, get_y(solver))
    copyto!(wd.zl_ref, full(get_zl(solver)))
    copyto!(wd.zu_ref, full(get_zu(solver)))
    copyto!(wd.xl_ref, full(get_xl(solver)))
    copyto!(wd.xu_ref, full(get_xu(solver)))
    copyto!(wd.c_ref, get_c(solver))
    wd.obj_val_ref = get_obj_val(solver)
    wd.theta_ref = get_theta(get_c(solver))
    wd.varphi_ref = get_varphi(get_obj_val(solver), get_x_lr(solver), get_xl_r(solver), get_xu_r(solver), get_x_ur(solver), get_mu(solver))
    varphi0 = get_varphi(get_obj_val(solver), get_x_lr(solver), get_xl_r(solver), get_xu_r(solver), get_x_ur(solver), zero(T))
    varphi1 = get_varphi(get_obj_val(solver), get_x_lr(solver), get_xl_r(solver), get_xu_r(solver), get_x_ur(solver), one(T))
    wd.varphi_ref_affine = (varphi0, varphi1 - varphi0)
    varphi_d(mu) = get_varphi_d(
        primal(get_f(solver)), primal(get_x(solver)), primal(get_xl(solver)), primal(get_xu(solver)),
        primal(get_d(solver)), mu,
    )
    wd.varphi_d_ref = varphi_d(get_mu(solver))
    wd.varphi_d_ref_affine = (varphi_d(zero(T)), varphi_d(one(T)) - varphi_d(zero(T)))
    copyto!(wd.d_ref, full(get_d(solver)))
    return
end

# Re-evaluates the reference's barrier objective and directional derivative at a new μ.
function update_watchdog_reference!(wd::WatchdogState, mu)
    wd.varphi_ref = wd.varphi_ref_affine[1] + mu * wd.varphi_ref_affine[2]
    wd.varphi_d_ref = wd.varphi_d_ref_affine[1] + mu * wd.varphi_d_ref_affine[2]
    return
end

# Rolls back to the reference and recomputes the gradient there. The Jacobian product `jacl`
# and the KKT factorization stay at the tentative point until the next iteration evaluates them;
# a second-order correction or a restoration phase started right after a rollback sees them.
function restore_watchdog_iterate!(solver::AbstractMadNLPSolver, wd::WatchdogState)
    copyto!(full(get_x(solver)), wd.x_ref)
    copyto!(get_y(solver), wd.y_ref)
    copyto!(full(get_zl(solver)), wd.zl_ref)
    copyto!(full(get_zu(solver)), wd.zu_ref)
    copyto!(full(get_xl(solver)), wd.xl_ref)
    copyto!(full(get_xu(solver)), wd.xu_ref)
    copyto!(get_c(solver), wd.c_ref)
    set_obj_val!(solver, wd.obj_val_ref)
    copyto!(full(get_d(solver)), wd.d_ref)
    eval_grad_f_wrapper!(solver, get_f(solver), get_x(solver))
    return
end

function RobustRestorer(solver::AbstractMadNLPSolver{T}) where {T}
    gn = get_opt(solver).feasibility_fallback_method === :gauss_newton
    ngn, mgn = gn ? get_n(solver) : 0, gn ? get_m(solver) : 0

    f_R = similar(get_y(solver), get_n(solver))
    x_ref = similar(get_y(solver), get_n(solver))
    D_R = similar(get_y(solver), get_n(solver))
    pp = similar(get_y(solver), get_m(solver))
    nn = similar(get_y(solver), get_m(solver))
    pp_trial = similar(get_y(solver), get_m(solver))
    nn_trial = similar(get_y(solver), get_m(solver))

    nn = similar(get_y(solver), get_m(solver))
    zp = similar(get_y(solver), get_m(solver))
    zn = similar(get_y(solver), get_m(solver))
    dpp= similar(get_y(solver), get_m(solver))
    dnn= similar(get_y(solver), get_m(solver))
    dzp= similar(get_y(solver), get_m(solver))
    dzn= similar(get_y(solver), get_m(solver))
    pp_trial = similar(get_y(solver), get_m(solver))
    nn_trial = similar(get_y(solver), get_m(solver))

    return RobustRestorer(
        zero(T), 
        f_R, 
        x_ref, 
        zero(T), 
        D_R, 
        zero(T), 
        pp, 
        nn, 
        zp, zn, 
        dpp, dnn, dzp, dzn, 
        pp_trial, 
        nn_trial, 
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), 
        Tuple{T, T}[], 
        0,
        0,
        false,
        similar(get_y(solver), get_n(solver)),
        similar(get_y(solver), get_m(solver)),
        similar(get_y(solver), get_n(solver)),
        similar(get_y(solver), get_n(solver)),
        similar(get_y(solver), get_n(solver)),
        # Gauss-Newton workspaces, empty unless that method is selected
        similar(get_y(solver), ngn),
        similar(get_y(solver), ngn),
        similar(get_y(solver), ngn),
        similar(get_y(solver), ngn),
        similar(get_y(solver), mgn),
        similar(get_y(solver), mgn),
    )
end

function initialize_robust_restorer!(solver::AbstractMadNLPSolver{T}) where {T}
    @trace(get_logger(solver),"Initializing restoration phase variables.")
    get_RR(solver) == nothing && (set_RR!(solver, RobustRestorer(solver)))
    # Explicitly type otherwise we get type instability because compiler can't know RR != nothing
    RR::RobustRestorer = get_RR(solver)

    copyto!(RR.x_ref, full(get_x(solver)))
    RR.theta_ref = get_theta(get_c(solver))
    RR.D_R .= min.(one(T), one(T) ./ abs.(RR.x_ref))

    RR.mu_R = max(get_mu(solver), norm(get_c(solver), Inf))
    RR.tau_R= max(get_opt(solver).tau_min,1-RR.mu_R)
    RR.zeta = sqrt(RR.mu_R)

    rho = get_opt(solver).rho
    mu = RR.mu_R
    populate_RR_nn!(RR.nn, get_c(solver), mu, rho)
    RR.pp .= get_c(solver) .+ RR.nn
    RR.zp .= RR.mu_R ./ RR.pp
    RR.zn .= RR.mu_R ./ RR.nn

    RR.obj_val_R = get_obj_val_R(RR.pp,RR.nn,RR.D_R,full(get_x(solver)),RR.x_ref,get_opt(solver).rho,RR.zeta)
    fill!(RR.f_R, zero(T))
    empty!(RR.filter)
    push!(RR.filter, (get_theta_max(solver),-Inf))

    fill!(get_y(solver), zero(T))
    get_zl_r(solver) .= min.(get_opt(solver).rho, get_zl_r(solver))
    get_zu_r(solver) .= min.(get_opt(solver).rho, get_zu_r(solver))
    # fill!(get_zl_r(solver), one(T)) # Experimental
    # fill!(get_zu_r(solver), one(T)) # Experimental
    
    get_cnt(solver).t = 0

    # misc
    set_del_w!(solver, zero(T))
end

