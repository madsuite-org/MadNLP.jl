# The line search's reductions on device arrays, with deferred forms (see `MadCore._deferred`).

const GV = AbstractGPUVectorOrSubVector

_deferred(::typeof(get_theta), c::GV) = _reduce(abs, +, zero(eltype(c)), c)
get_theta(c::GV) = _value1(_deferred(get_theta, c))

function _deferred(::typeof(get_varphi), obj_val, x_lr::VT, xl_r::VT, xu_r::VT, x_ur::VT, mu) where {T, VT <: GV{T}}
    lb = _reduce((x1, x2) -> _get_varphi(x1, x2, mu), +, zero(T), x_lr, xl_r)
    ub = _reduce((x1, x2) -> _get_varphi(x1, x2, mu), +, zero(T), xu_r, x_ur)
    return _lazy((obj_val, lb, ub) -> obj_val + lb + ub, obj_val, lb, ub)
end
get_varphi(obj_val, x_lr::VT, xl_r::VT, xu_r::VT, x_ur::VT, mu) where {T, VT <: GV{T}} =
    _value1(_deferred(get_varphi, obj_val, x_lr, xl_r, xu_r, x_ur, mu))

_deferred(::typeof(get_varphi_d), f::GV{T}, x::GV{T}, xl::GV{T}, xu::GV{T}, dx::GV{T}, mu) where {T} =
    _reduce((f, x, xl, xu, dx) -> (f - mu / (x - xl) + mu / (xu - x)) * dx, +, zero(T), f, x, xl, xu, dx)
get_varphi_d(f::GV{T}, x::GV{T}, xl::GV{T}, xu::GV{T}, dx::GV{T}, mu) where {T} =
    _value1(_deferred(get_varphi_d, f, x, xl, xu, dx, mu))

_deferred(::typeof(get_rel_search_norm), x::GV{T}, dx::GV{T}) where {T} =
    _reduce((x, dx) -> abs(dx) / (one(T) + abs(x)), max, zero(T), x, dx)
get_rel_search_norm(x::GV{T}, dx::GV{T}) where {T} = _value1(_deferred(get_rel_search_norm, x, dx))
