function WatchdogState(solver::AbstractMadNLPSolver{T}) where {T}
    VT = typeof(get_y(solver))
    return WatchdogState{T,VT}(
        false, 0, 0,
        similar(get_y(solver), length(full(get_x(solver)))),  # x_ref (primals + slacks)
        similar(get_y(solver), get_m(solver)),                # y_ref
        similar(get_y(solver), length(full(get_zl(solver)))), # zl_ref (full)
        similar(get_y(solver), length(full(get_zu(solver)))), # zu_ref (full)
        similar(get_y(solver), length(full(get_xl(solver)))), # xl_ref (full)
        similar(get_y(solver), length(full(get_xu(solver)))), # xu_ref (full)
        similar(get_y(solver), get_m(solver)),                # c_ref
        zero(T),
        similar(get_y(solver), length(full(get_d(solver)))),  # d_ref
        Tuple{T,T}[],
    )
end

# Save the full primal-dual iterate + search direction as the watchdog reference.
function save_watchdog_iterate!(solver::AbstractMadNLPSolver, wd::WatchdogState)
    copyto!(wd.x_ref,  full(get_x(solver)))
    copyto!(wd.y_ref,  get_y(solver))
    copyto!(wd.zl_ref, full(get_zl(solver)))
    copyto!(wd.zu_ref, full(get_zu(solver)))
    copyto!(wd.xl_ref, full(get_xl(solver)))
    copyto!(wd.xu_ref, full(get_xu(solver)))
    copyto!(wd.c_ref,  get_c(solver))
    wd.obj_val_ref = get_obj_val(solver)
    copyto!(wd.d_ref, full(get_d(solver)))
    empty!(wd.filter); append!(wd.filter, get_filter(solver))
    return
end

# Roll back to the saved reference iterate (bit-identical) and recompute the gradient.
function restore_watchdog_iterate!(solver::AbstractMadNLPSolver, wd::WatchdogState)
    copyto!(full(get_x(solver)),  wd.x_ref)
    copyto!(get_y(solver),  wd.y_ref)
    copyto!(full(get_zl(solver)), wd.zl_ref)
    copyto!(full(get_zu(solver)), wd.zu_ref)
    copyto!(full(get_xl(solver)), wd.xl_ref)
    copyto!(full(get_xu(solver)), wd.xu_ref)
    copyto!(get_c(solver),  wd.c_ref)
    set_obj_val!(solver, wd.obj_val_ref)
    copyto!(full(get_d(solver)), wd.d_ref)
    empty!(get_filter(solver)); append!(get_filter(solver), wd.filter)
    eval_grad_f_wrapper!(solver, get_f(solver), get_x(solver))
    return
end

function RobustRestorer(solver::AbstractMadNLPSolver{T}) where {T}

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

