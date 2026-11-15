# NOTE: All of these could probably be made more efficient via custom kernels or nonallocating mapreduce.
#       For now we just use the standard mapreduce functionality which may allocate.

function get_inf_du(f::VT, zl::VT, zu::VT, jacl::VT, sd) where {VT <: AbstractGPUVectorOrSubVector}
    return mapreduce((f,zl,zu,jacl) -> abs(f-zl+zu+jacl), max, f, zl, zu, jacl; init = zero(eltype(f))) / sd
end


function get_inf_compl(x_lr::VT, xl_r::VT, zl_r::VT, xu_r::VT, x_ur::VT, zu_r::VT, mu, sc) where {VT <: AbstractGPUVectorOrSubVector}
    return max(
        mapreduce(
            (x_lr, xl_r, zl_r) -> abs((x_lr-xl_r)*zl_r-mu),
            max,
            x_lr, xl_r, zl_r;
            init = zero(eltype(x_lr))
        ),
        mapreduce(
            (xu_r, x_ur, zu_r) -> abs((xu_r-x_ur)*zu_r-mu),
            max,
            xu_r, x_ur, zu_r;
            init = zero(eltype(x_lr))
        )
    ) / sc
end

function get_min_complementarity(
    x_lr::AbstractGPUVectorOrSubVector{T},
    xl_r::AbstractGPUVectorOrSubVector{T},
    zl_r::AbstractGPUVectorOrSubVector{T},
    x_ur::AbstractGPUVectorOrSubVector{T},
    xu_r::AbstractGPUVectorOrSubVector{T},
    zu_r::AbstractGPUVectorOrSubVector{T}
) where T
    cc_lb = mapreduce((x_l, xl, zl) -> (x_l-xl)*zl, min, x_lr, xl_r, zl_r, init=T(Inf))
    cc_ub = mapreduce((x_u, xu, zu) -> (xu-x_u)*zu, min, x_ur, xu_r, zu_r, init=T(Inf))
    return min(cc_lb,cc_ub)
end

# The generic method takes `dot`s of these views, which scalar-index views through an index array
# on some GPU backends.
function get_average_complementarity(
    x_lr::AbstractGPUVectorOrSubVector{T},
    xl_r::AbstractGPUVectorOrSubVector{T},
    zl_r::AbstractGPUVectorOrSubVector{T},
    x_ur::AbstractGPUVectorOrSubVector{T},
    xu_r::AbstractGPUVectorOrSubVector{T},
    zu_r::AbstractGPUVectorOrSubVector{T}
) where T
    n_lb, n_ub = length(x_lr), length(x_ur)
    n_lb + n_ub == 0 && return zero(T)
    cc_lb = mapreduce((x_l, xl, zl) -> (x_l-xl)*zl, +, x_lr, xl_r, zl_r, init=zero(T))
    cc_ub = mapreduce((x_u, xu, zu) -> (xu-x_u)*zu, +, x_ur, xu_r, zu_r, init=zero(T))
    return (cc_lb + cc_ub) / (n_lb + n_ub)
end

function get_alpha_max(
    x::AbstractGPUVectorOrSubVector{T},
    xl::AbstractGPUVectorOrSubVector{T},
    xu::AbstractGPUVectorOrSubVector{T},
    dx::AbstractGPUVectorOrSubVector{T},
    tau,
) where T
    return min(
        mapreduce(
            (x, xl, dx) -> dx < 0 ? (-x+xl)*tau/dx : T(Inf),
            min,

            x, xl, dx,
            init = one(T)
        ),
        mapreduce(
            (x, xu, dx) -> dx > 0 ? (-x+xu)*tau/dx : T(Inf),
            min,
            x, xu, dx,
            init = one(T)
        )
    )
end

function get_alpha_z(
    zl_r::AbstractGPUVectorOrSubVector{T},
    zu_r::AbstractGPUVectorOrSubVector{T},
    dzl::AbstractGPUVectorOrSubVector{T},
    dzu::AbstractGPUVectorOrSubVector{T},
    tau,
)  where T
    return min(
        mapreduce(
            (zl_r, dzl) -> dzl < 0 ? (-zl_r)*tau/dzl : T(Inf),
            min,
            zl_r, dzl,
            init = one(T)
        ),
        mapreduce(
            (zu_r, dzu) -> dzu < 0 ? (-zu_r)*tau/dzu : T(Inf),
            min,
            zu_r, dzu,
            init = one(T)
        )
    )
end

# The scalars of each phase of the interior-point iteration (see `MadCore.get_convergence_values`)
# on device arrays. Each phase computes all of them in one `mapreduce` over the indices of its
# vectors, laid end to end; an index contributes to the reductions over its own vector only, and
# the others get their initial values. AcceleratedKernels reduces the index range on the device,
# so the host waits for the device once per phase.

function MadCore.get_convergence_values(y::AbstractGPUVectorOrSubVector{T}, c, x_lr, xl_r, zl_r, xu_r, x_ur, zu_r, f, zl, zu, jacl, s_max) where {T}
    m, nlb, nub, n = length(y), length(zl_r), length(zu_r), length(f)
    o = zero(T)
    # (‖y‖₁, ‖zl‖₁, ‖zu‖₁, primal infeasibility, unscaled dual infeasibility, lower and upper complementarity)
    l1_y, l1_zl, l1_zu, inf_pr, du, compl_lb, compl_ub = AcceleratedKernels.mapreduce(
        function (i)
            return @inbounds if i <= m
                (abs(y[i]), o, o, abs(c[i]), o, o, o)
            elseif i <= m + nlb
                j = i - m
                (o, abs(zl_r[j]), o, o, o, abs((x_lr[j] - xl_r[j]) * zl_r[j]), o)
            elseif i <= m + nlb + nub
                j = i - m - nlb
                (o, o, abs(zu_r[j]), o, o, o, abs((xu_r[j] - x_ur[j]) * zu_r[j]))
            else
                j = i - m - nlb - nub
                (o, o, o, o, abs(f[j] - zl[j] + zu[j] + jacl[j]), o, o)
            end
        end,
        (a, b) -> (a[1] + b[1], a[2] + b[2], a[3] + b[3], max(a[4], b[4]), max(a[5], b[5]), max(a[6], b[6]), max(a[7], b[7])),
        1:(m + nlb + nub + n);
        init = (o, o, o, o, o, o, o), backend = get_backend(y),
    )
    sd = max(s_max, (l1_y + l1_zl + l1_zu) / max(1, m + nlb + nub)) / s_max
    sc = max(s_max, (l1_zl + l1_zu) / max(1, nlb + nub)) / s_max
    return sd, sc, inf_pr, du / sd, max(compl_lb, compl_ub) / sc
end

function MadCore.get_line_search_values(obj_val, c::AbstractGPUVectorOrSubVector{T}, x_lr, xl_r, zl_r, dzl, xu_r, x_ur, zu_r, dzu, f, x, xl, xu, dx, mu, tau) where {T}
    m, nlb, nub, n = length(c), length(zl_r), length(zu_r), length(x)
    o, l, inf = zero(T), one(T), T(Inf)
    # (θ, barrier terms of the lower and upper bounds, directional derivative of the barrier
    # function, step to the lower and upper bounds of x, of zl and of zu, relative step norm)
    theta, varphi_lb, varphi_ub, varphi_d, amax_lb, amax_ub, az_lb, az_ub, rel_search_norm = AcceleratedKernels.mapreduce(
        function (i)
            return @inbounds if i <= m
                (abs(c[i]), o, o, o, l, l, l, l, o)
            elseif i <= m + nlb
                j = i - m
                (o, _get_varphi(x_lr[j], xl_r[j], mu), o, o, l, l, dzl[j] < 0 ? -zl_r[j] * tau / dzl[j] : inf, l, o)
            elseif i <= m + nlb + nub
                j = i - m - nlb
                (o, o, _get_varphi(xu_r[j], x_ur[j], mu), o, l, l, l, dzu[j] < 0 ? -zu_r[j] * tau / dzu[j] : inf, o)
            else
                j = i - m - nlb - nub
                (
                    o, o, o, (f[j] - mu / (x[j] - xl[j]) + mu / (xu[j] - x[j])) * dx[j],
                    dx[j] < 0 ? (xl[j] - x[j]) * tau / dx[j] : inf,
                    dx[j] > 0 ? (xu[j] - x[j]) * tau / dx[j] : inf,
                    l, l, abs(dx[j]) / (l + abs(x[j])),
                )
            end
        end,
        (a, b) -> (
            a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4],
            min(a[5], b[5]), min(a[6], b[6]), min(a[7], b[7]), min(a[8], b[8]), max(a[9], b[9]),
        ),
        1:(m + nlb + nub + n);
        init = (o, o, o, o, l, l, l, l, o), backend = get_backend(c),
    )
    return theta, obj_val + varphi_lb + varphi_ub, varphi_d, min(amax_lb, amax_ub), min(az_lb, az_ub), rel_search_norm
end

function MadCore.get_trial_values(obj_val, c::AbstractGPUVectorOrSubVector{T}, x_lr, xl_r, xu_r, x_ur, mu) where {T}
    m, nlb, nub = length(c), length(x_lr), length(x_ur)
    o = zero(T)
    # (θ, barrier terms of the lower and upper bounds)
    theta, varphi_lb, varphi_ub = AcceleratedKernels.mapreduce(
        function (i)
            return @inbounds if i <= m
                (abs(c[i]), o, o)
            elseif i <= m + nlb
                (o, _get_varphi(x_lr[i - m], xl_r[i - m], mu), o)
            else
                j = i - m - nlb
                (o, o, _get_varphi(xu_r[j], x_ur[j], mu))
            end
        end,
        (a, b) -> (a[1] + b[1], a[2] + b[2], a[3] + b[3]),
        1:(m + nlb + nub);
        init = (o, o, o), backend = get_backend(c),
    )
    return theta, obj_val + varphi_lb + varphi_ub
end

function MadCore.get_quality_steps(x::AbstractGPUVectorOrSubVector{T}, xl, xu, dx, zl_r, zu_r, dzl, dzu, tau) where {T}
    n, nlb, nub = length(x), length(zl_r), length(zu_r)
    l, inf = one(T), T(Inf)
    # (αp, αd) as in `get_alpha_max` and `get_alpha_z`
    return AcceleratedKernels.mapreduce(
        function (i)
            return @inbounds if i <= n
                (min(dx[i] < 0 ? (xl[i] - x[i]) * tau / dx[i] : inf, dx[i] > 0 ? (xu[i] - x[i]) * tau / dx[i] : inf), l)
            elseif i <= n + nlb
                j = i - n
                (l, dzl[j] < 0 ? -zl_r[j] * tau / dzl[j] : inf)
            else
                j = i - n - nlb
                (l, dzu[j] < 0 ? -zu_r[j] * tau / dzu[j] : inf)
            end
        end,
        (a, b) -> (min(a[1], b[1]), min(a[2], b[2])),
        1:(n + nlb + nub);
        init = (l, l), backend = get_backend(x),
    )
end

function MadCore.get_quality_compl(x_lr::AbstractGPUVectorOrSubVector{T}, xl_r, dx_lr, zl_r, dzl, x_ur, xu_r, dx_ur, zu_r, dzu, alpha_pr, alpha_du) where {T}
    nlb, nub = length(x_lr), length(x_ur)
    o = zero(T)
    return AcceleratedKernels.mapreduce(
        function (i)
            return @inbounds if i <= nlb
                (((x_lr[i] + alpha_pr * dx_lr[i] - xl_r[i]) * (zl_r[i] + alpha_du * dzl[i]))^2, o)
            else
                j = i - nlb
                (o, ((xu_r[j] - x_ur[j] - alpha_pr * dx_ur[j]) * (zu_r[j] + alpha_du * dzu[j]))^2)
            end
        end,
        (a, b) -> (a[1] + b[1], a[2] + b[2]),
        1:(nlb + nub);
        init = (o, o), backend = get_backend(x_lr),
    )
end
