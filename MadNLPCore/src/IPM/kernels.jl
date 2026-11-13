# Filter line search and restoration kernels

# Fields used only by the filter line search and the restoration phase.
const FILTER_FIELDS = (:theta_min, :theta_max, :ftype, :filter, :RR, :watchdog)

for attribute in FILTER_FIELDS
    getter = Symbol("get_", attribute)
    setter = Symbol("set_", attribute, "!")
    @eval begin
        @inline $getter(solver::AbstractMadNLPSolver) = getfield(solver, $(QuoteNode(attribute)))
        @inline $setter(solver::AbstractMadNLPSolver, rhs) = setfield!(solver, $(QuoteNode(attribute)), rhs)
    end
end

# The watchdog state, or `nothing` for a solver type without a `watchdog` field, which then
# runs without the watchdog.
watchdog_state(solver::AbstractMadNLPSolver) =
    hasfield(typeof(solver), :watchdog) ? get_watchdog(solver) : nothing
watchdog_active(solver::AbstractMadNLPSolver) =
    get_opt(solver).watchdog && (wd = watchdog_state(solver)) !== nothing && wd.in_watchdog

get_theta(solver::AbstractMadNLPSolver) = get_theta(get_c(solver))
get_varphi(solver::AbstractMadNLPSolver) = get_varphi(get_obj_val(solver), get_x_lr(solver), get_xl_r(solver), get_xu_r(solver), get_x_ur(solver), get_mu(solver))

function print_iter(solver::AbstractMadNLPSolver; is_resto=false)
    # the residuals below cost reductions on the device; skip them when nothing is printed
    logger = get_logger(solver)
    get_level(logger) <= INFO || (get_file(logger) !== nothing && get_file_level(logger) <= INFO) || return
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
        inf_pr = get_inf_pr_orig(solver)
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

# Robust restoration
function set_aug_RR!(kkt::AbstractKKTSystem, solver::AbstractMadNLPSolver, RR::RobustRestorer)
    x = full(get_x(solver))
    xl = full(get_xl(solver))
    xu = full(get_xu(solver))
    zl = full(get_zl(solver))
    zu = full(get_zu(solver))
    kkt.reg .= get_opt(solver).default_primal_regularization .+ RR.zeta .* RR.D_R .^ 2
    kkt.du_diag .= .- get_opt(solver).default_dual_regularization .- RR.pp ./ RR.zp .- RR.nn ./ RR.zn
    copyto!(kkt.l_lower, get_zl_r(solver))
    copyto!(kkt.u_lower, get_zu_r(solver))
    kkt.l_diag .= get_xl_r(solver) .- get_x_lr(solver)
    kkt.u_diag .= get_x_ur(solver) .- get_xu_r(solver)

    _set_aug_diagonal!(kkt)
    return
end

function set_aug_RR!(kkt::ScaledSparseKKTSystem, solver::AbstractMadNLPSolver, RR::RobustRestorer)
    x = full(get_x(solver))
    xl = full(get_xl(solver))
    xu = full(get_xu(solver))
    zl = full(get_zl(solver))
    zu = full(get_zu(solver))
    kkt.reg .= get_opt(solver).default_primal_regularization .+ RR.zeta .* RR.D_R .^ 2
    kkt.du_diag .= .- get_opt(solver).default_dual_regularization .- RR.pp ./ RR.zp .- RR.nn ./ RR.zn
    copyto!(kkt.l_lower, get_zl_r(solver))
    copyto!(kkt.u_lower, get_zu_r(solver))
    kkt.l_diag .= get_x_lr(solver) .- get_xl_r(solver)
    kkt.u_diag .= get_xu_r(solver) .- get_x_ur(solver)

    _set_aug_diagonal!(kkt)
    return
end

function set_f_RR!(solver::AbstractMadNLPSolver, RR::RobustRestorer)
    x = full(get_x(solver))
    RR.f_R .= RR.zeta .* RR.D_R .^ 2 .* (x .- RR.x_ref)
    return
end

# Set RHS RR
function set_aug_rhs_RR!(
    solver::AbstractMadNLPSolver, kkt::AbstractKKTSystem, RR::RobustRestorer, rho,
)
    x = full(get_x(solver))
    xl = full(get_xl(solver))
    xu = full(get_xu(solver))
    zl = full(get_zl(solver))
    zu = full(get_zu(solver))

    px = primal(get_p(solver))
    py = dual(get_p(solver))
    pzl = dual_lb(get_p(solver))
    pzu = dual_ub(get_p(solver))

    mu = RR.mu_R

    px .= .- RR.f_R .+ zl .- zu .- get_jacl(solver)
    py .= .- get_c(solver) .+ RR.pp .- RR.nn .+
        (mu .- (rho .- get_y(solver)) .* RR.pp) ./ RR.zp .-
        (mu .- (rho .+ get_y(solver)) .* RR.nn) ./ RR.zn

    pzl .= (get_xl_r(solver) .- get_x_lr(solver)) .* get_zl_r(solver) .+ mu
    pzu .= (get_xu_r(solver) .- get_x_ur(solver)) .* get_zu_r(solver) .- mu

    return
end

# Set ifr
function set_aug_rhs_ifr!(solver::AbstractMadNLPSolver{T}, kkt::AbstractKKTSystem, p0::AbstractKKTVector) where T
    fill!(primal(p0), zero(T))
    fill!(dual_lb(p0), zero(T))
    fill!(dual_ub(p0), zero(T))
    wy = dual(p0)
    wy .= .- get_c(solver)
    return
end

function set_g_ifr!(solver::AbstractMadNLPSolver, g::AbstractArray)
    f = full(get_f(solver))
    x = full(get_x(solver))
    xl = full(get_xl(solver))
    xu = full(get_xu(solver))
    g .= f .- get_mu(solver) ./ (x .- xl) .+ get_mu(solver) ./ (xu .- x) .+ get_jacl(solver)
end

# Finish RR
function finish_aug_solve_RR!(dpp, dnn, dzp, dzn, l, dl, pp, nn, zp, zn, mu_R, rho)
    dzp .= rho .- l .- dl .- zp
    dzn .= rho .+ l .+ dl .- zn
    dpp .= .- pp .+ mu_R ./zp .- (pp./zp) .* dzp
    dnn .= .- nn .+ mu_R ./zn .- (nn./zn) .* dzn
    return
end

function _get_varphi(x1::T, x2::T, mu::T) where T
    x = x1 - x2
    if x < 0
        return T(Inf)
    else
        return -mu * log(x)
    end
end

function get_varphi(obj_val, x_lr, xl_r, xu_r, x_ur, mu)
    varphi = obj_val
    for ii in eachindex(x_lr)
        varphi += _get_varphi(x_lr[ii],xl_r[ii],mu)
    end
    for ii in eachindex(xu_r)
        varphi += _get_varphi(xu_r[ii],x_ur[ii],mu)
    end
    return varphi
end

function get_varphi_d(
    f::AbstractVector{T},
    x::AbstractVector{T},
    xl::AbstractVector{T},
    xu::AbstractVector{T},
    dx::AbstractVector{T},
    mu,
) where T
    varphi_d = zero(T)
    for ii in eachindex(f)
        varphi_d += (f[ii] - mu/(x[ii]-xl[ii]) + mu/(xu[ii]-x[ii])) * dx[ii]
    end
    return varphi_d
end

function get_obj_val_R(
    p::AbstractVector{T},
    n::AbstractVector{T},
    D_R::AbstractVector{T},
    x::AbstractVector{T},
    x_ref::AbstractVector{T},
    rho,
    zeta,
) where T
    obj_val_R = zero(T)
    for ii in eachindex(p)
        obj_val_R += rho*(p[ii]+n[ii])
    end
    for ii in eachindex(x)
        obj_val_R += zeta/2*D_R[ii]^2*(x[ii]-x_ref[ii])^2
    end
    return obj_val_R
end

@inline get_theta(c) = norm(c, 1)

function get_theta_R(
    c::AbstractVector{T},
    p::AbstractVector{T},
    n::AbstractVector{T},
) where T
    theta_R = zero(T)
    for ii in eachindex(c)
        theta_R += abs(c[ii]-p[ii]+n[ii])
    end
    return theta_R
end

function get_inf_pr_R(
    c::AbstractVector{T},
    p::AbstractVector{T},
    n::AbstractVector{T},
) where T
    inf_pr_R = zero(T)
    for ii in eachindex(c)
        inf_pr_R = max(inf_pr_R, abs(c[ii]-p[ii]+n[ii]))
    end
    return inf_pr_R
end

function get_inf_du_R(
    f_R::AbstractVector{T},
    l::AbstractVector{T},
    zl::AbstractVector{T},
    zu::AbstractVector{T},
    jacl::AbstractVector{T},
    zp::AbstractVector{T},
    zn::AbstractVector{T},
    rho,
    sd,
) where T
    inf_du_R = zero(T)
    for ii in eachindex(f_R)
        inf_du_R = max(inf_du_R, abs(f_R[ii]-zl[ii]+zu[ii]+jacl[ii]))
    end
    for ii in eachindex(l)
        inf_du_R = max(inf_du_R, abs(rho-l[ii]-zp[ii]), abs(rho+l[ii]-zn[ii]))
    end
    return inf_du_R / sd
end

function get_inf_compl_R(
    x_lr::SubVector{T, VT, VI},
    xl_r,
    zl_r,
    xu_r,
    x_ur,
    zu_r,
    pp,
    zp,
    nn,
    zn,
    mu_R,
    sc
) where {T, VT <: AbstractVector{T}, VI}
    inf_compl_R = zero(T)
    for ii in eachindex(x_lr)
        inf_compl_R = max(inf_compl_R, abs((x_lr[ii]-xl_r[ii])*zl_r[ii]-mu_R))
    end
    for ii in eachindex(xu_r)
        inf_compl_R = max(inf_compl_R, abs((xu_r[ii]-x_ur[ii])*zu_r[ii]-mu_R))
    end
    for ii in eachindex(pp)
        inf_compl_R = max(inf_compl_R, abs(pp[ii]*zp[ii]-mu_R))
    end
    for ii in eachindex(nn)
        inf_compl_R = max(inf_compl_R, abs(nn[ii]*zn[ii]-mu_R))
    end
    return inf_compl_R / sc
end

function get_alpha_max_R(
    x::AbstractVector{T},
    xl::AbstractVector{T},
    xu::AbstractVector{T},
    dx::AbstractVector{T},
    pp::AbstractVector{T},
    dpp::AbstractVector{T},
    nn::AbstractVector{T},
    dnn::AbstractVector{T},
    tau_R,
) where T
    alpha_max_R = one(T)
    for ii in eachindex(x)
        candidate = if dx[ii] < 0
            (-x[ii]+xl[ii])*tau_R/dx[ii]
        elseif dx[ii] > 0
            (-x[ii]+xu[ii])*tau_R/dx[ii]
        else
            T(Inf)
        end
        alpha_max_R = min(alpha_max_R, candidate)
    end
    for ii in eachindex(pp)
        alpha_max_R = min(alpha_max_R, dpp[ii] < 0 ? -pp[ii]*tau_R/dpp[ii] : T(Inf))
    end
    for ii in eachindex(nn)
        alpha_max_R = min(alpha_max_R, dnn[ii] < 0 ? -nn[ii]*tau_R/dnn[ii] : T(Inf))
    end
    return alpha_max_R
end

function get_alpha_z_R(
    zl_r::SubVector{T, VT, VI},
    zu_r,
    dzl,
    dzu,
    zp,
    dzp,
    zn,
    dzn,
    tau_R,
) where {T, VT <: AbstractVector{T}, VI}
    alpha_z_R = one(T)
    for ii in eachindex(dzl)
        alpha_z_R = min(alpha_z_R, dzl[ii] < 0.0 ? -zl_r[ii]*tau_R/dzl[ii] : T(Inf))
    end
    for ii in eachindex(dzu)
        alpha_z_R = min(alpha_z_R, dzu[ii] < 0.0 ? -zu_r[ii]*tau_R/dzu[ii] : T(Inf))
    end
    for ii in eachindex(dzp)
        alpha_z_R = min(alpha_z_R, dzp[ii] < 0.0 ? -zp[ii]*tau_R/dzp[ii] : T(Inf))
    end
    for ii in eachindex(dzn)
        alpha_z_R = min(alpha_z_R, dzn[ii] < 0.0 ? -zn[ii]*tau_R/dzn[ii] : T(Inf))
    end
    return alpha_z_R
end

function get_varphi_R(
    obj_val,
    x_lr::SubVector{T, VT, VI},
    xl_r,
    xu_r,
    x_ur,
    pp,
    nn,
    mu_R,
)  where {T, VT <: AbstractVector{T}, VI}
    varphi_R = obj_val
    for ii in eachindex(x_lr)
        d = x_lr[ii] - xl_r[ii]
        varphi_R -= d < 0.0 ? T(Inf) : mu_R * log(d)
    end
    for ii in eachindex(xu_r)
        d = xu_r[ii] - x_ur[ii]
        varphi_R -= d < 0.0 ? T(Inf) : mu_R * log(d)
    end
    for ii in eachindex(pp)
        varphi_R -= pp[ii] < 0.0 ? T(Inf) : mu_R*log(pp[ii])
    end
    for ii in eachindex(nn)
        varphi_R -= nn[ii] < 0.0 ? T(Inf) : mu_R*log(nn[ii])
    end
    return varphi_R
end

function get_F(
    c::AbstractVector{T},
    f,
    zl,
    zu,
    jacl,
    x_lr,
    xl_r,
    zl_r,
    xu_r,
    x_ur,
    zu_r,
    mu,
) where T
    # NOTE: Does not allocate
    F1 = mapreduce(
        abs,
        +,
        c;
        init = zero(T)
    )

    F2 = zero(T)
    for ii in eachindex(f)
        F2 += abs(f[ii]-zl[ii]+zu[ii]+jacl[ii])
    end

    F3 = zero(T)
    for ii in eachindex(x_lr)
        F3 += (x_lr[ii] >= xl_r[ii] && zl_r[ii] >= 0) ? abs((x_lr[ii]-xl_r[ii])*zl_r[ii]-mu) : T(Inf)
    end

    F4 = zero(T)
    for ii in eachindex(xu_r)
        F4 += (xu_r[ii] >= x_ur[ii] && zu_r[ii] >= 0) ? abs((xu_r[ii]-x_ur[ii])*zu_r[ii]-mu) : T(Inf)
    end

    return F1 + F2 + F3 + F4
end

function get_varphi_d_R(
    f_R::AbstractVector{T},
    x::AbstractVector{T},
    xl::AbstractVector{T},
    xu::AbstractVector{T},
    dx::AbstractVector{T},
    pp::AbstractVector{T},
    nn::AbstractVector{T},
    dpp::AbstractVector{T},
    dnn::AbstractVector{T},
    mu_R,
    rho,
) where T
    varphi_d = zero(T)
    for ii in eachindex(f_R)
        varphi_d += (f_R[ii] - mu_R/(x[ii]-xl[ii]) + mu_R/(xu[ii]-x[ii])) * dx[ii]
    end
    for ii in eachindex(pp)
        varphi_d += (rho - mu_R/pp[ii]) * dpp[ii]
    end
    for ii in eachindex(nn)
        varphi_d += (rho - mu_R/nn[ii]) * dnn[ii]
    end
    return varphi_d
end

function get_rel_search_norm(x::AbstractVector{T}, dx::AbstractVector{T}) where T
    rel_search_norm = zero(T)
    for ii in eachindex(x)
        rel_search_norm = max(rel_search_norm, abs(dx[ii]) / (one(T) + abs(x[ii])))
    end
    return rel_search_norm
end

function get_alpha_min(
    theta,
    varphi_d,
    theta_min,
    gamma_theta,
    gamma_phi,
    alpha_min_frac,
    del,
    s_theta,
    s_phi,
)
    if varphi_d<0
        if theta<=theta_min
            return alpha_min_frac*min(
                gamma_theta,gamma_phi*theta/(-varphi_d),
                del*theta^s_theta/(-varphi_d)^s_phi,
            )
        else
            return alpha_min_frac*min(
                gamma_theta,
                -gamma_phi*theta/varphi_d,
            )
        end
    else
        return alpha_min_frac*gamma_theta
    end
end

function is_switching(varphi_d, alpha, s_phi, del, theta, s_theta)
    return (varphi_d < 0) && (alpha*(-varphi_d)^s_phi > del*theta^s_theta)
end

function is_armijo(varphi_trial, varphi, eta_phi, alpha, varphi_d)
    return (varphi_trial <= varphi + eta_phi*alpha*varphi_d)
end

function is_sufficient_progress(theta_trial::T, theta, gamma_theta, varphi_trial, varphi, gamma_phi, has_constraints) where T
    (has_constraints && ((theta_trial<=(1-gamma_theta)*theta+10*eps(T)*abs(theta))) || ((varphi_trial<=varphi-gamma_phi*theta +10*eps(T)*abs(varphi))))
end

function augment_filter!(filter, theta, varphi, gamma_theta, gamma_phi)
    return push!(filter, ((1 - gamma_theta) * theta, varphi - gamma_phi * theta))
end

function is_filter_acceptable(filter, theta, varphi)
    !isnan(theta) || return false
    !isinf(theta) || return false
    !isnan(varphi) || return false
    !isinf(varphi) || return false

    for (theta_F,varphi_F) in filter
        theta <= theta_F || varphi <= varphi_F || return false
    end
    return true
end

function is_barr_obj_rapid_increase(varphi::T, varphi_trial, obj_max_inc) where T
    return (varphi_trial >= varphi) && (log10(varphi_trial-varphi) > obj_max_inc + max(one(T), log10(abs(varphi))))
end

function get_ftype(filter,theta,theta_trial,varphi,varphi_trial,switching_condition,armijo_condition,
                   theta_min,obj_max_inc,gamma_theta,gamma_phi,has_constraints)
    is_filter_acceptable(filter,theta_trial,varphi_trial) || return " "
    !is_barr_obj_rapid_increase(varphi,varphi_trial,obj_max_inc) || return " "

    if theta <= theta_min && switching_condition
        armijo_condition && return "f"
    else
        is_sufficient_progress(
            theta_trial,theta,gamma_theta,varphi_trial,varphi,gamma_phi,has_constraints) && return "h"
    end

    return " "
end

function populate_RR_nn!(nn, c, mu, rho)
    for ii in eachindex(nn)
        nn[ii] = (mu - rho*c[ii])/(2*rho)+ sqrt(((mu-rho*c[ii])/(2*rho))^2 + mu*c[ii]/(2*rho))
    end
end
