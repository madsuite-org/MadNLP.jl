
# KKT system updates -------------------------------------------------------
# Set diagonal
set_aug_diagonal!(kkt::AbstractKKTSystem{T}, solver::AbstractMadNLPSolver{T}) where T = _set_aug_diagonal!(kkt, solver)

function _set_aug_diagonal!(kkt::AbstractKKTSystem{T}, solver::AbstractMadNLPSolver{T}) where T
    fill!(kkt.reg, get_opt(solver).default_primal_regularization)
    fill!(kkt.du_diag, -get_opt(solver).default_dual_regularization)
    kkt.l_diag .= get_xl_r(solver) .- get_x_lr(solver)   # (Xˡ - X)
    kkt.u_diag .= get_x_ur(solver) .- get_xu_r(solver)   # (X - Xᵘ)
    copyto!(kkt.l_lower, get_zl_r(solver))
    copyto!(kkt.u_lower, get_zu_r(solver))

    _set_aug_diagonal!(kkt)
    return
end

# the border diagonal of the condensed system, Σd - 1/Σs on the kept rows, tends to zero
# when a slack reaches its bound (the constraint becomes active); the perturbation δ = border_delta μ^border_delta_exp
# keeps the static factorization away from that pivot (mul! stays exact, so refinement
# removes the bias)
function set_aug_diagonal!(kkt::SparseCondensedKKTSystem{T}, solver::AbstractMadNLPSolver{T}) where {T}
    _set_aug_diagonal!(kkt, solver)
    b = kkt.border
    b.delta[1] = b.delta_base * get_mu(solver)^b.delta_exp
    return
end

function _set_aug_diagonal!(kkt::AbstractKKTSystem)
    copyto!(kkt.pr_diag, kkt.reg)
    @views kkt.pr_diag[kkt.ind_lb] .-= kkt.l_lower ./ kkt.l_diag
    @views kkt.pr_diag[kkt.ind_ub] .-= kkt.u_lower ./ kkt.u_diag
    return
end

function _set_aug_diagonal!(kkt::AbstractUnreducedKKTSystem)
    copyto!(kkt.pr_diag, kkt.reg)
    kkt.l_lower_aug .= sqrt.(kkt.l_lower)
    kkt.u_lower_aug .= sqrt.(kkt.u_lower)
    return
end

function set_aug_diagonal!(kkt::ScaledSparseKKTSystem{T}, solver::AbstractMadNLPSolver{T}) where T
    fill!(kkt.reg, get_opt(solver).default_primal_regularization)
    fill!(kkt.du_diag, -get_opt(solver).default_dual_regularization)
    # Ensure l_diag and u_diag have only non negative entries
    kkt.l_diag .= get_x_lr(solver) .- get_xl_r(solver)   # (X - Xˡ)
    kkt.u_diag .= get_xu_r(solver) .- get_x_ur(solver)   # (Xᵘ - X)
    copyto!(kkt.l_lower, get_zl_r(solver))
    copyto!(kkt.u_lower, get_zu_r(solver))
    _set_aug_diagonal!(kkt)
end

function _set_aug_diagonal!(kkt::ScaledSparseKKTSystem{T}) where T
    xlzu = kkt.buffer1
    xuzl = kkt.buffer2
    fill!(xlzu, zero(T))
    fill!(xuzl, zero(T))
    @views begin
        xlzu[kkt.ind_ub] .= kkt.u_lower    # zᵘ
        xlzu[kkt.ind_lb] .*= kkt.l_diag    # (X - Xˡ) zᵘ

        xuzl[kkt.ind_lb] .= kkt.l_lower    # zˡ
        xuzl[kkt.ind_ub] .*= kkt.u_diag    # (Xᵘ - X) zˡ

        kkt.pr_diag .= xlzu .+ xuzl

        fill!(kkt.scaling_factor, one(T))
        kkt.scaling_factor[kkt.ind_lb] .*= sqrt.(kkt.l_diag)
        kkt.scaling_factor[kkt.ind_ub] .*= sqrt.(kkt.u_diag)
    end
    # Scale regularization by scaling factor.
    kkt.pr_diag .+= kkt.reg .* kkt.scaling_factor.^2
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

# Set RHS
function set_aug_rhs!(solver::AbstractMadNLPSolver, kkt::AbstractKKTSystem, c::AbstractVector, mu)
    px = primal(get_p(solver))
    x = primal(get_x(solver))
    f = primal(get_f(solver))
    xl = primal(get_xl(solver))
    xu = primal(get_xu(solver))
    zl = full(get_zl(solver))
    zu = full(get_zu(solver))
    py = dual(get_p(solver))
    pzl = dual_lb(get_p(solver))
    pzu = dual_ub(get_p(solver))

    px .= .-f .+ zl .- zu .- get_jacl(solver)
    py .= .-c
    pzl .= (get_xl_r(solver) .- get_x_lr(solver)) .* get_zl_r(solver) .+ mu
    pzu .= (get_xu_r(solver) .- get_x_ur(solver)) .* get_zu_r(solver) .- mu
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

# solving KKT system
@inbounds function _kktmul!(
    w::AbstractKKTVector,
    x::AbstractKKTVector,
    reg,
    du_diag,
    l_lower,
    u_lower,
    l_diag,
    u_diag,
    alpha,
    beta,
)
    primal(w) .+= alpha .* reg .* primal(x)
    dual(w) .+= alpha .* du_diag .* dual(x)
    w.xp_lr .-= alpha .* dual_lb(x)
    w.xp_ur .+= alpha .* dual_ub(x)
    dual_lb(w) .= beta .* dual_lb(w) .+ alpha .* (x.xp_lr .* l_lower .- dual_lb(x) .* l_diag)
    dual_ub(w) .= beta .* dual_ub(w) .+ alpha .* (x.xp_ur .* u_lower .+ dual_ub(x) .* u_diag)
    return
end

@inbounds function reduce_rhs!(
    xp_lr, wl, l_diag,
    xp_ur, wu, u_diag,
)
    xp_lr .-= wl ./ l_diag
    xp_ur .-= wu ./ u_diag
    return
end
function reduce_rhs!(kkt::AbstractKKTSystem, d::AbstractKKTVector)
    reduce_rhs!(
        d.xp_lr, dual_lb(d), kkt.l_diag,
        d.xp_ur, dual_ub(d), kkt.u_diag,
    )
end

# Finish
function finish_aug_solve!(kkt::AbstractKKTSystem, d::AbstractKKTVector)
    dlb = dual_lb(d)
    dub = dual_ub(d)
    dlb .= (.-dlb .+ kkt.l_lower .* d.xp_lr) ./ kkt.l_diag
    dub .= (  dub .- kkt.u_lower .* d.xp_ur) ./ kkt.u_diag
    return
end

function set_initial_bounds!(xl::AbstractVector{T}, xu::AbstractVector{T}, tol) where T
    # If `tol` is set to zero, keep the bounds unchanged.
    if tol > zero(T)
        map!(
            x->x - max(one(T), abs(x)) * tol,
            xl, xl
        )
        map!(
            x->x + max(one(T), abs(x)) * tol,
            xu, xu
        )
    end
end

function set_initial_rhs!(solver::AbstractMadNLPSolver{T}, kkt::AbstractKKTSystem) where T
    f = primal(get_f(solver))
    zl = primal(get_zl(solver))
    zu = primal(get_zu(solver))
    px = primal(get_p(solver))
    px .= .-f .+ zl .- zu
    fill!(dual(get_p(solver)), zero(T))
    fill!(dual_lb(get_p(solver)), zero(T))
    fill!(dual_ub(get_p(solver)), zero(T))
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

# Kernel functions ---------------------------------------------------------
is_valid(val::R) where R <: Real = !(isnan(val) || isinf(val))
is_valid(vec::AbstractArray) = isempty(vec) ? true : mapreduce(is_valid, &, vec)

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

@inline get_inf_pr(c::AbstractVector) = norm(c, Inf)

function get_inf_du(f, zl, zu, jacl, sd)
    inf_du = zero(eltype(f))
    for ii in eachindex(f)
        inf_du = max(inf_du, abs(f[ii]-zl[ii]+zu[ii]+jacl[ii]))
    end
    return inf_du / sd
end

function get_inf_compl(x_lr, xl_r, zl_r, xu_r, x_ur, zu_r, mu, sc)
    inf_compl = zero(eltype(x_lr))
    for ii in eachindex(x_lr)
        inf_compl = max(inf_compl, abs((x_lr[ii]-xl_r[ii])*zl_r[ii]-mu))
    end
    for ii in eachindex(xu_r)
        inf_compl = max(inf_compl, abs((xu_r[ii]-x_ur[ii])*zu_r[ii]-mu))
    end

    return inf_compl / sc
end

function get_average_complementarity(x_lr::AbstractVector{T}, xl_r::AbstractVector{T}, zl_r::AbstractVector{T},
                                     x_ur::AbstractVector{T}, xu_r::AbstractVector{T}, zu_r::AbstractVector{T}) where T
    n_lb, n_ub = length(x_lr), length(x_ur)
    # If the problem is unconstrained, average complementarity is 0
    if n_lb + n_ub == 0
        return zero(T)
    end
    cc_lb = dot(x_lr, zl_r) - dot(xl_r, zl_r)
    cc_ub = dot(xu_r, zu_r) - dot(x_ur, zu_r)
    return (cc_lb + cc_ub) / (n_lb + n_ub)
end
function get_average_complementarity(solver::AbstractMadNLPSolver)
    return get_average_complementarity(
        get_x_lr(solver), get_xl_r(solver), get_zl_r(solver),
        get_x_ur(solver), get_xu_r(solver), get_zu_r(solver),
    )
end

function get_min_complementarity(x_lr::AbstractVector{T}, xl_r::AbstractVector{T}, zl_r::AbstractVector{T},
                                 x_ur::AbstractVector{T}, xu_r::AbstractVector{T}, zu_r::AbstractVector{T}) where T
    min_comp = T(Inf)
    for ii in eachindex(x_lr)
        min_comp = min(min_comp, (x_lr[ii]-xl_r[ii])*zl_r[ii])
    end
    for ii in eachindex(xu_r)
        min_comp = min(min_comp, (xu_r[ii]-x_ur[ii])*zu_r[ii])
    end
    return min_comp
end

function get_min_complementarity(solver::AbstractMadNLPSolver)
    return get_min_complementarity(
        get_x_lr(solver), get_xl_r(solver), get_zl_r(solver),
        get_x_ur(solver), get_xu_r(solver), get_zu_r(solver),
    )
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

function get_alpha_max(
    x::AbstractVector{T},
    xl::AbstractVector{T},
    xu::AbstractVector{T},
    dx::AbstractVector{T},
    tau,
) where T
    alpha_max = one(T)
    for ii in eachindex(x)
        alpha_max = min(alpha_max,
                        dx[ii] < 0 ? (-x[ii]+xl[ii])*tau/dx[ii] : T(Inf),
                        dx[ii] > 0 ? (-x[ii]+xu[ii])*tau/dx[ii] : T(Inf),
                        )
    end
    return alpha_max
end

function get_alpha_z(
    zl_r::AbstractVector{T},
    zu_r::AbstractVector{T},
    dzl::AbstractVector{T},
    dzu::AbstractVector{T},
    tau,
)  where T
    alpha_z = one(T)
    for ii in eachindex(zl_r)
        alpha_z = min(alpha_z, dzl[ii] < 0 ? (-zl_r[ii])*tau/dzl[ii] : T(Inf))
    end
    for ii in eachindex(zu_r)
        alpha_z = min(alpha_z, dzu[ii] < 0 ? (-zu_r[ii])*tau/dzu[ii] : T(Inf))
    end
    return alpha_z
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

function _initialize_variables!(x::T, xl, xu, bound_push, bound_fac) where T
    if xl!=-T(Inf) && xu!=T(Inf)
        return min(
            xu-min(bound_push*max(1,abs(xu)), bound_fac*(xu-xl)),
            max(xl+min(bound_push*max(1,abs(xl)),bound_fac*(xu-xl)),x),
        )
    elseif xl!=-T(Inf) && xu==T(Inf)
        return max(xl+bound_push*max(1,abs(xl)), x)
    elseif xl==-T(Inf) && xu!=T(Inf)
        return min(xu-bound_push*max(1,abs(xu)), x)
    end
    return x
end

function initialize_variables!(x, xl, xu, bound_push, bound_fac)
    map!((x,l,u) -> _initialize_variables!(x,l,u, bound_push, bound_fac), x, x, xl, xu)
end

function adjust_boundary!(
    x_lr::AbstractVector{T},
    xl_r::AbstractVector{T},
    x_ur::AbstractVector{T},
    xu_r::AbstractVector{T},
    mu,
) where T
    c1 = eps(T)*mu
    c2 = eps(T)^(3/4)
    map!(
        (x_lr, xl_r) -> (x_lr-xl_r < c1) ? (xl_r - c2*max(1,abs(x_lr))) : xl_r,
        xl_r, x_lr, xl_r
    )
    map!(
        (xu_r, x_ur) -> (xu_r-x_ur < c1) ? (xu_r + c2*max(1,abs(x_ur))) : xu_r,
        xu_r, xu_r, x_ur
    )
end

function get_rel_search_norm(x::AbstractVector{T}, dx::AbstractVector{T}) where T
    rel_search_norm = zero(T)
    for ii in eachindex(x)
        rel_search_norm = max(rel_search_norm, abs(dx[ii]) / (one(T) + abs(x[ii])))
    end
    return rel_search_norm
end

# Utility functions
function get_sd(l, zl_r, zu_r, s_max)
    return max(
        s_max,
        (norm(l, 1)+norm(zl_r, 1)+norm(zu_r, 1)) / max(1, (length(l)+length(zl_r)+length(zu_r))),
    ) / s_max
end
function get_sc(zl_r, zu_r, s_max)
    return max(
        s_max,
        (norm(zl_r,1)+norm(zu_r,1)) / max(1,length(zl_r)+length(zu_r)),
    ) / s_max
end

function get_mu(
    mu::T,
    mu_min,
    mu_linear_decrease_factor,
    mu_superlinear_decrease_power,
    tol,
) where {T}
    # Warning: `a * tol` should be strictly less than 100 * mu_min, see issue #242
    a = min(T(99.0) * mu_min / tol, T(0.01))
    return max(
        mu_min,
        a * tol,
        min(mu_linear_decrease_factor*mu, mu^mu_superlinear_decrease_power),
    )
end

@inline get_tau(mu, tau_min) = max(tau_min, 1-mu)

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

function augment_filter!(filter, theta, varphi, gamma_theta)
    push!(filter, ((1-gamma_theta)*theta, varphi-gamma_theta*theta))
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

function reset_bound_dual!(
    z::AbstractVector{T},
    x::AbstractVector{T},
    mu,
    kappa_sigma,
) where T
    map!(
        (z, x) -> max(min(z, (kappa_sigma*mu)/x), (mu/kappa_sigma)/x),
        z, z, x
    )
    return
end

function reset_bound_dual!(
    z::AbstractVector{T},
    x1::AbstractVector{T},
    x2::AbstractVector{T},
    mu,
    kappa_sigma,
) where T
    map!(
        (z,x1,x2) -> max(min(z, (kappa_sigma*mu)/(x1-x2)), (mu/kappa_sigma)/(x1-x2)),
        z,z,x1,x2
    )
    return
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

function dual_inf_perturbation!(px, ind_llb, ind_uub, mu, kappa_d)
    @views begin
        px[ind_llb] .-= mu*kappa_d
        px[ind_uub] .+= mu*kappa_d
    end
end

function populate_RR_nn!(nn, c, mu, rho)
    for ii in eachindex(nn)
        nn[ii] = (mu - rho*c[ii])/(2*rho)+ sqrt(((mu-rho*c[ii])/(2*rho))^2 + mu*c[ii]/(2*rho))
    end
end
