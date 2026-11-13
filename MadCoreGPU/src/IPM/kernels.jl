# The reductions of the iteration on device arrays. Each has a deferred form (see
# `MadCore._deferred`), so that a caller can wait for several at once; the plain form waits
# for its own value.

const GV = AbstractGPUVectorOrSubVector

_deferred(::typeof(get_inf_du), f::VT, zl::VT, zu::VT, jacl::VT, sd) where {VT <: GV} =
    _lazy(/, _reduce((f, zl, zu, jacl) -> abs(f - zl + zu + jacl), max, zero(eltype(f)), f, zl, zu, jacl), sd)
get_inf_du(f::VT, zl::VT, zu::VT, jacl::VT, sd) where {VT <: GV} = _value1(_deferred(get_inf_du, f, zl, zu, jacl, sd))

function _deferred(::typeof(get_inf_compl), x_lr::VT, xl_r::VT, zl_r::VT, xu_r::VT, x_ur::VT, zu_r::VT, mu, sc) where {VT <: GV}
    T = eltype(x_lr)
    lb = _reduce((x_lr, xl_r, zl_r) -> abs((x_lr - xl_r) * zl_r - mu), max, zero(T), x_lr, xl_r, zl_r)
    ub = _reduce((xu_r, x_ur, zu_r) -> abs((xu_r - x_ur) * zu_r - mu), max, zero(T), xu_r, x_ur, zu_r)
    return _lazy((lb, ub, sc) -> max(lb, ub) / sc, lb, ub, sc)
end
get_inf_compl(x_lr::VT, xl_r::VT, zl_r::VT, xu_r::VT, x_ur::VT, zu_r::VT, mu, sc) where {VT <: GV} =
    _value1(_deferred(get_inf_compl, x_lr, xl_r, zl_r, xu_r, x_ur, zu_r, mu, sc))

_deferred(::typeof(get_inf_pr), c::GV) = _reduce(abs, max, zero(eltype(c)), c)
get_inf_pr(c::GV) = _value1(_deferred(get_inf_pr, c))

function _deferred(::typeof(get_sd), l::GV, zl_r::GV, zu_r::GV, s_max)
    T = eltype(l)
    n = max(1, length(l) + length(zl_r) + length(zu_r))
    return _lazy((a, b, c) -> max(s_max, (a + b + c) / n) / s_max,
        _reduce(abs, +, zero(T), l), _reduce(abs, +, zero(T), zl_r), _reduce(abs, +, zero(T), zu_r))
end
get_sd(l::GV, zl_r::GV, zu_r::GV, s_max) = _value1(_deferred(get_sd, l, zl_r, zu_r, s_max))

function _deferred(::typeof(get_sc), zl_r::GV, zu_r::GV, s_max)
    T = eltype(zl_r)
    n = max(1, length(zl_r) + length(zu_r))
    return _lazy((a, b) -> max(s_max, (a + b) / n) / s_max, _reduce(abs, +, zero(T), zl_r), _reduce(abs, +, zero(T), zu_r))
end
get_sc(zl_r::GV, zu_r::GV, s_max) = _value1(_deferred(get_sc, zl_r, zu_r, s_max))

function _deferred(::typeof(get_min_complementarity), x_lr::GV{T}, xl_r::GV{T}, zl_r::GV{T}, x_ur::GV{T}, xu_r::GV{T}, zu_r::GV{T}) where {T}
    lb = _reduce((x_l, xl, zl) -> (x_l - xl) * zl, min, T(Inf), x_lr, xl_r, zl_r)
    ub = _reduce((x_u, xu, zu) -> (xu - x_u) * zu, min, T(Inf), x_ur, xu_r, zu_r)
    return _lazy(min, lb, ub)
end
get_min_complementarity(x_lr::GV{T}, xl_r::GV{T}, zl_r::GV{T}, x_ur::GV{T}, xu_r::GV{T}, zu_r::GV{T}) where {T} =
    _value1(_deferred(get_min_complementarity, x_lr, xl_r, zl_r, x_ur, xu_r, zu_r))

# The generic method takes `dot`s of these views, which scalar-index views through an index array
# on some GPU backends.
function _deferred(::typeof(get_average_complementarity), x_lr::GV{T}, xl_r::GV{T}, zl_r::GV{T}, x_ur::GV{T}, xu_r::GV{T}, zu_r::GV{T}) where {T}
    n_lb, n_ub = length(x_lr), length(x_ur)
    n_lb + n_ub == 0 && return zero(T)
    lb = _reduce((x_l, xl, zl) -> (x_l - xl) * zl, +, zero(T), x_lr, xl_r, zl_r)
    ub = _reduce((x_u, xu, zu) -> (xu - x_u) * zu, +, zero(T), x_ur, xu_r, zu_r)
    return _lazy((lb, ub) -> (lb + ub) / (n_lb + n_ub), lb, ub)
end
get_average_complementarity(x_lr::GV{T}, xl_r::GV{T}, zl_r::GV{T}, x_ur::GV{T}, xu_r::GV{T}, zu_r::GV{T}) where {T} =
    _value1(_deferred(get_average_complementarity, x_lr, xl_r, zl_r, x_ur, xu_r, zu_r))

function _deferred(::typeof(get_alpha_max), x::GV{T}, xl::GV{T}, xu::GV{T}, dx::GV{T}, tau) where {T}
    lb = _reduce((x, xl, dx) -> dx < 0 ? (-x + xl) * tau / dx : T(Inf), min, one(T), x, xl, dx)
    ub = _reduce((x, xu, dx) -> dx > 0 ? (-x + xu) * tau / dx : T(Inf), min, one(T), x, xu, dx)
    return _lazy(min, lb, ub)
end
get_alpha_max(x::GV{T}, xl::GV{T}, xu::GV{T}, dx::GV{T}, tau) where {T} = _value1(_deferred(get_alpha_max, x, xl, xu, dx, tau))

function _deferred(::typeof(get_alpha_z), zl_r::GV{T}, zu_r::GV{T}, dzl::GV{T}, dzu::GV{T}, tau) where {T}
    lb = _reduce((zl_r, dzl) -> dzl < 0 ? (-zl_r) * tau / dzl : T(Inf), min, one(T), zl_r, dzl)
    ub = _reduce((zu_r, dzu) -> dzu < 0 ? (-zu_r) * tau / dzu : T(Inf), min, one(T), zu_r, dzu)
    return _lazy(min, lb, ub)
end
get_alpha_z(zl_r::GV{T}, zu_r::GV{T}, dzl::GV{T}, dzu::GV{T}, tau) where {T} = _value1(_deferred(get_alpha_z, zl_r, zu_r, dzl, dzu, tau))
