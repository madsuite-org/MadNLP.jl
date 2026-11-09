# Feasibility fallback: projected gradient descent on the constraint violation, used when the
# restoration phase fails. It needs constraints and the transposed Jacobian product only.

# Distance kept from a finite bound, so that the barrier stays defined at the result.
function _fallback_margin(l::T, u::T) where {T}
    l == u && return zero(T)
    isfinite(l) || isfinite(u) || return zero(T)
    a = isfinite(l) ? (isfinite(u) ? min(abs(l), abs(u)) : abs(l)) : abs(u)
    return min(T(1.0e-8) * max(one(T), a), (u - l) / 4)
end
_fallback_lo(l, u) = l == u ? l : l + _fallback_margin(l, u)
_fallback_hi(l, u) = l == u ? u : u - _fallback_margin(l, u)

# Largest number of rescues in one solve.
const MAX_RESCUES = 3

"""
    feasibility_rescue!(solver, status; from_entry) -> Status

Run the feasibility fallback after a failure exit `status`: from the point where the failed
restoration phase started if `from_entry`, otherwise from the current point. Return `REGULAR`
if the regular phase resumes from the fallback's point. Otherwise put the solver back at the
point of the failure exit, where its gradient, multipliers and error measures still are, and
return `status`, or the limit or stop that ended the fallback. A solve gets at most
`MAX_RESCUES` rescues.
"""
function feasibility_rescue!(solver::AbstractMadNLPSolver{T}, status; from_entry::Bool) where {T}
    get_RR(solver) === nothing && set_RR!(solver, RobustRestorer(solver))
    RR::RobustRestorer = get_RR(solver)
    RR.n_rescues < MAX_RESCUES || return status
    RR.n_rescues += 1
    x, c = get_x(solver), get_c(solver)
    copyto!(RR.x_last, full(x))
    copyto!(RR.c_last, c)
    obj_last = get_obj_val(solver)

    if from_entry
        copyto!(full(x), RR.x_ref)
        eval_cons_wrapper!(solver, c, x)
        theta_ref = RR.theta_ref
    else
        theta_ref = get_theta(c)
    end
    result = feasibility_fallback!(solver, theta_ref)
    result === REGULAR && return REGULAR

    copyto!(full(x), RR.x_last)
    copyto!(c, RR.c_last)
    set_obj_val!(solver, obj_last)
    # the Jacobian was last evaluated at a point of the fallback
    get_opt(solver).jacobian_constant || eval_jac_wrapper!(solver, get_kkt(solver), x)
    return result === nothing ? status : result
end

"""
    feasibility_fallback!(solver, theta_ref) -> Union{Status, Nothing}

Minimize `½‖c(x)‖²` over the bounds by projected gradient descent, from the current iterate,
with Barzilai-Borwein steps and a nonmonotone Armijo test over the last 10 values. Stop once
`‖c‖∞` falls by the factor `feasibility_fallback_reduction`, or after
`feasibility_fallback_max_iter` iterations. If the filter accepts the result and the
violation is below `required_infeasibility_reduction * theta_ref`, set the multipliers,
return to the regular phase and return `REGULAR`; otherwise return `nothing`. The wall time
limit and the intermediate callback are checked at every iteration, and their status is
returned when they end the fallback.
"""
function feasibility_fallback!(solver::AbstractMadNLPSolver{T}, theta_ref) where {T}
    opt = get_opt(solver)
    RR::RobustRestorer = get_RR(solver)
    kkt = get_kkt(solver)
    x, c = get_x(solver), get_c(solver)
    xt, ct = get_x_trial(solver), get_c_trial(solver)
    l, u = full(get_xl(solver)), full(get_xu(solver))
    g, gp, xp = RR.g_fb, RR.g_prev_fb, RR.x_prev_fb

    tol = max(T(opt.tol), opt.feasibility_fallback_reduction * norm(c, Inf))
    phi = dot(c, c) / 2
    hist = fill(phi, 10)
    tau = one(T)
    lambda = T(1.0e-8)
    have_prev = false
    iter = 0
    max_iter = opt.feasibility_fallback_method === :gauss_newton ? min(opt.feasibility_fallback_max_iter, 500) : opt.feasibility_fallback_max_iter
    while iter < max_iter && norm(c, Inf) > tol
        time() - get_cnt(solver).start_time >= opt.max_wall_time && return MAXIMUM_WALLTIME_EXCEEDED
        !(get_intermediate_callback(solver)(solver, UserCallbackRobust())::Bool) && return USER_REQUESTED_STOP
        iter += 1
        opt.jacobian_constant || eval_jac_wrapper!(solver, kkt, x)
        jtprod!(g, kkt, c)
        # projected gradient, for a stationarity test
        full(xt) .= full(x) .- clamp.(full(x) .- g, _fallback_lo.(l, u), _fallback_hi.(l, u))
        norm(full(xt), Inf) <= 1.0e-12 * max(one(T), norm(c, Inf)) && break
        if opt.feasibility_fallback_method === :gauss_newton
            # Gauss-Newton step over the variables not held at a bound by the gradient.
            free = RR.w_fb
            free .= ifelse.(
                ((full(x) .<= _fallback_lo.(l, u)) .& (g .> 0)) .| ((full(x) .>= _fallback_hi.(l, u)) .& (g .< 0)),
                zero(T), one(T),
            )
            gauss_newton_step!(solver, RR.d_fb, g, free, lambda)
            dg = dot(g, RR.d_fb)
            copyto!(xp, full(x))
            alpha = one(T)
            accepted = false
            for _ in 1:30
                full(xt) .= clamp.(full(x) .+ alpha .* RR.d_fb, _fallback_lo.(l, u), _fallback_hi.(l, u))
                eval_cons_wrapper!(solver, ct, xt)
                phi_t = dot(ct, ct) / 2
                if isfinite(phi_t) && phi_t < phi && phi_t <= phi + T(1.0e-4) * alpha * min(dg, zero(T))
                    accepted = true
                    phi = phi_t
                    break
                end
                alpha /= 2
            end
            if accepted
                lambda = alpha == 1 ? max(lambda / 3, T(1.0e-12)) : lambda
            else
                lambda *= 100
                lambda > 1.0e12 && break
                continue
            end
        else
        if have_prev
            xp .= full(x) .- xp
            gp .= g .- gp
            sy = dot(xp, gp)
            tau = sy > 0 ? clamp(dot(xp, xp) / sy, T(1.0e-10), T(1.0e10)) : 2tau
        else
            tau = one(T) / max(norm(g, Inf), one(T))
        end
        copyto!(xp, full(x))
        copyto!(gp, g)
        have_prev = true
        gx = dot(g, xp)
        ref = maximum(hist)
        alpha = one(T)
        accepted = false
        for _ in 1:30
            full(xt) .= clamp.(full(x) .- alpha .* tau .* g, _fallback_lo.(l, u), _fallback_hi.(l, u))
            eval_cons_wrapper!(solver, ct, xt)
            phi_t = dot(ct, ct) / 2
            if isfinite(phi_t) && phi_t <= ref - T(1.0e-4) * (gx - dot(g, full(xt)))
                accepted = true
                phi = phi_t
                break
            end
            alpha /= 2
        end
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
    ok || return nothing
    eval_grad_f_wrapper!(solver, get_f(solver), x)
    opt.jacobian_constant || eval_jac_wrapper!(solver, kkt, x)
    get_zl_r(solver) .= mu ./ (get_x_lr(solver) .- get_xl_r(solver))
    get_zu_r(solver) .= mu ./ (get_xu_r(solver) .- get_x_ur(solver))
    return_to_regular!(solver)
    return REGULAR
end

"""
    gauss_newton_step!(solver, d, g, free, lambda)

Approximately solve `min ‖J D d + c‖² + lambda ‖d‖²` by CGLS, where `D` is the diagonal 0/1
mask `free` and `g = J'c`. Products `J v` are forward differences of the constraints.
"""
function gauss_newton_step!(solver::AbstractMadNLPSolver{T}, d, g, free, lambda; maxcg = 500, cgtol = 1.0e-2) where {T}
    RR::RobustRestorer = get_RR(solver)
    kkt = get_kkt(solver)
    x, c = get_x(solver), get_c(solver)
    xt, ct = get_x_trial(solver), get_c_trial(solver)
    s, p, q, r = RR.s_fb, RR.p_fb, RR.q_fb, RR.r_fb
    fill!(d, zero(T))
    r .= .-c
    s .= .-g .* free
    copyto!(p, s)
    gamma = gamma0 = dot(s, s)
    gamma0 > 0 || return d
    for _ in 1:maxcg
        # q = J (free .* p) by a forward difference
        h = sqrt(eps(T)) * (one(T) + norm(full(x), Inf)) / max(norm(p, Inf), eps(T))
        full(xt) .= full(x) .+ h .* free .* p
        eval_cons_wrapper!(solver, ct, xt)
        q .= (ct .- c) ./ h
        delta = dot(q, q) + lambda * dot(p, p)
        delta > 0 || break
        a = gamma / delta
        d .+= a .* p
        r .-= a .* q
        jtprod!(s, kkt, r)
        s .= s .* free .- lambda .* d
        gamma_new = dot(s, s)
        gamma_new <= cgtol^2 * gamma0 && break
        p .= s .+ (gamma_new / gamma) .* p
        gamma = gamma_new
    end
    return d
end
