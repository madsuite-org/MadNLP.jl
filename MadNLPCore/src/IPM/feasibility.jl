# Feasibility fallback: projected gradient descent on the constraint violation, used when the
# restoration phase fails. It needs constraints and the transposed Jacobian product only.

# Distance kept from a finite bound, so that the barrier stays defined at the result.
function _fallback_margin(l::T, u::T) where {T}
    l == u && return zero(T)
    isfinite(l) || isfinite(u) || return zero(T)
    a = isfinite(l) ? (isfinite(u) ? min(abs(l), abs(u)) : abs(l)) : abs(u)
    return min(T(1e-8) * max(one(T), a), (u - l) / 4)
end
_fallback_lo(l, u) = l == u ? l : l + _fallback_margin(l, u)
_fallback_hi(l, u) = l == u ? u : u - _fallback_margin(l, u)

"""
    feasibility_fallback!(solver, theta_ref) -> Bool

Minimize `½‖c(x)‖²` over the bounds by projected gradient descent, from the current iterate,
with Barzilai-Borwein steps and a nonmonotone Armijo test over the last 10 values. Stop once
`‖c‖∞` falls by the factor `feasibility_fallback_reduction`, or after
`feasibility_fallback_max_iter` iterations. If the filter accepts the result and the
violation is below `required_infeasibility_reduction * theta_ref`, set the multipliers and
return `true`; otherwise restore the starting point and return `false`.
"""
function feasibility_fallback!(solver::AbstractMadNLPSolver{T}, theta_ref) where {T}
    opt = get_opt(solver)
    RR::RobustRestorer = get_RR(solver)
    kkt = get_kkt(solver)
    x, c = get_x(solver), get_c(solver)
    xt, ct = get_x_trial(solver), get_c_trial(solver)
    l, u = full(get_xl(solver)), full(get_xu(solver))
    g, gp, xp = RR.g_fb, RR.g_prev_fb, RR.x_prev_fb
    copyto!(RR.x_entry, full(x))

    tol = max(T(opt.tol), opt.feasibility_fallback_reduction * norm(c, Inf))
    phi = dot(c, c) / 2
    hist = fill(phi, 10)
    tau = one(T)
    have_prev = false
    iter = 0
    while iter < opt.feasibility_fallback_max_iter && norm(c, Inf) > tol
        iter += 1
        eval_jac_wrapper!(solver, kkt, x)
        jtprod!(g, kkt, c)
        # projected gradient, for a stationarity test
        full(xt) .= full(x) .- clamp.(full(x) .- g, _fallback_lo.(l, u), _fallback_hi.(l, u))
        norm(full(xt), Inf) <= 1e-12 * max(one(T), norm(c, Inf)) && break
        if have_prev
            xp .= full(x) .- xp
            gp .= g .- gp
            sy = dot(xp, gp)
            tau = sy > 0 ? clamp(dot(xp, xp) / sy, T(1e-10), T(1e10)) : 2tau
        else
            tau = one(T) / max(norm(g, Inf), one(T))
        end
        copyto!(xp, full(x))
        copyto!(gp, g)
        have_prev = true
        ref = maximum(hist)
        alpha = one(T)
        accepted = false
        for _ in 1:30
            full(xt) .= clamp.(full(x) .- alpha .* tau .* g, _fallback_lo.(l, u), _fallback_hi.(l, u))
            eval_cons_wrapper!(solver, ct, xt)
            phi_t = dot(ct, ct) / 2
            if isfinite(phi_t) && phi_t <= ref - T(1e-4) * dot(g, xp .- full(xt))
                accepted = true
                phi = phi_t
                break
            end
            alpha /= 2
        end
        accepted || break
        copyto!(full(x), full(xt))
        copyto!(c, ct)
        hist[mod1(iter, 10)] = phi
    end

    theta = get_theta(c)
    set_obj_val!(solver, eval_f_wrapper(solver, x))
    mu = get_mu(solver)
    varphi = get_varphi(get_obj_val(solver), get_x_lr(solver), get_xl_r(solver), get_xu_r(solver), get_x_ur(solver), mu)
    ok = is_filter_acceptable(get_filter(solver), theta, varphi) &&
        theta <= opt.required_infeasibility_reduction * theta_ref
    @debug(get_logger(solver), "Feasibility fallback: $iter iterations, theta $theta, accepted $ok.")
    if !ok
        copyto!(full(x), RR.x_entry)
        eval_cons_wrapper!(solver, c, x)
        set_obj_val!(solver, eval_f_wrapper(solver, x))
        return false
    end
    eval_grad_f_wrapper!(solver, get_f(solver), x)
    eval_jac_wrapper!(solver, kkt, x)
    get_zl_r(solver) .= mu ./ (get_x_lr(solver) .- get_xl_r(solver))
    get_zu_r(solver) .= mu ./ (get_xu_r(solver) .- get_x_ur(solver))
    return_to_regular!(solver)
    return true
end
