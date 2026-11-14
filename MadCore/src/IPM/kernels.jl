
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
    _iview(kkt.pr_diag, kkt.ind_lb) .-= kkt.l_lower ./ kkt.l_diag
    _iview(kkt.pr_diag, kkt.ind_ub) .-= kkt.u_lower ./ kkt.u_diag
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
    _assign2!(
        primal(w), (w, r, x) -> w + alpha * r * x, (primal(w), reg, primal(x)),
        dual(w), (w, r, x) -> w + alpha * r * x, (dual(w), du_diag, dual(x)),
    )
    # two kernels: a variable bounded on both sides is in both xp_lr and xp_ur
    _assign!(w.xp_lr, (w, d) -> w - alpha * d, w.xp_lr, dual_lb(x))
    _assign!(w.xp_ur, (w, d) -> w + alpha * d, w.xp_ur, dual_ub(x))
    _assign2!(
        dual_lb(w), (w, x, l, d, ld) -> beta * w + alpha * (x * l - d * ld), (dual_lb(w), x.xp_lr, l_lower, dual_lb(x), l_diag),
        dual_ub(w), (w, x, u, d, ud) -> beta * w + alpha * (x * u + d * ud), (dual_ub(w), x.xp_ur, u_lower, dual_ub(x), u_diag),
    )
    return
end

@inbounds function reduce_rhs!(
    xp_lr, wl, l_diag,
    xp_ur, wu, u_diag,
)
    # two kernels: a variable bounded on both sides is in both xp_lr and xp_ur
    _assign!(xp_lr, (x, w, l) -> x - w / l, xp_lr, wl, l_diag)
    _assign!(xp_ur, (x, w, u) -> x - w / u, xp_ur, wu, u_diag)
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
    _assign2!(
        dlb, (dl, l, x, ld) -> (-dl + l * x) / ld, (dlb, kkt.l_lower, d.xp_lr, kkt.l_diag),
        dub, (du, u, x, ud) -> (du - u * x) / ud, (dub, kkt.u_lower, d.xp_ur, kkt.u_diag),
    )
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

# Kernel functions ---------------------------------------------------------
is_valid(val::R) where R <: Real = !(isnan(val) || isinf(val))
is_valid(vec::AbstractArray) = isempty(vec) ? true : mapreduce(is_valid, &, vec)

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
    c2 = eps(T)^T(3 / 4)
    map!(
        (x_lr, xl_r) -> (x_lr-xl_r < c1) ? (xl_r - c2*max(1,abs(x_lr))) : xl_r,
        xl_r, x_lr, xl_r
    )
    map!(
        (xu_r, x_ur) -> (xu_r-x_ur < c1) ? (xu_r + c2*max(1,abs(x_ur))) : xu_r,
        xu_r, xu_r, x_ur
    )
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
    barrier_tol_factor,
) where {T}
    # As in Ipopt, μ does not go below min(tol, 1e-4) / (barrier_tol_factor + 1), the value
    # of the default mu_min.
    return max(
        mu_min,
        min(tol, T(1.0e-4)) / (barrier_tol_factor + one(T)),
        min(mu_linear_decrease_factor*mu, mu^mu_superlinear_decrease_power),
    )
end

@inline get_tau(mu, tau_min) = max(tau_min, 1-mu)

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


function dual_inf_perturbation!(px, ind_llb, ind_uub, mu, kappa_d)
    @views begin
        px[ind_llb] .-= mu*kappa_d
        px[ind_uub] .+= mu*kappa_d
    end
end
