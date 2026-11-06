
"""
    AbstractKKTVector{T, VT}

Supertype for KKT's right-hand-side vectors ``(x, s, y, z, ν, w)``.
"""
abstract type AbstractKKTVector{T, VT} end

"""
    full(X::AbstractKKTVector)

Return all the values stored inside the KKT vector `X`.
"""
function full end

Base.length(rhs::AbstractKKTVector) = length(full(rhs))

"""
    number_primal(X::AbstractKKTVector)

Get total number of primal values ``(x, s)`` in KKT vector `X`.
"""
number_primal(rhs::AbstractKKTVector) = length(primal(rhs))

"""
    number_dual(X::AbstractKKTVector)

Get total number of dual values ``(y, z)`` in KKT vector `X`.
"""
number_dual(rhs::AbstractKKTVector) = length(dual(rhs))

"""
    primal(X::AbstractKKTVector)

Return the primal values ``(x, s)`` stored in the KKT vector `X`.
"""
function primal end

"""
    dual(X::AbstractKKTVector)

Return the dual values ``(y, z)`` stored in the KKT vector `X`.
"""
function dual end

"""
    primal_dual(X::AbstractKKTVector)

Return both the primal and the dual values ``(x, s, y, z)`` stored in the KKT vector `X`.
"""
function primal_dual end

"""
    dual_lb(X::AbstractKKTVector)

Return the dual values ``ν`` associated to the lower-bound
stored in the KKT vector `X`.
"""
function dual_lb end

"""
    dual_ub(X::AbstractKKTVector)

Return the dual values ``w`` associated to the upper-bound
stored in the KKT vector `X`.
"""
function dual_ub end

function Base.fill!(rhs::AbstractKKTVector{T}, val::T) where T
    fill!(full(rhs), val)
end

# Overload basic BLAS operations.
norm(X::AbstractKKTVector, p::Real=2.0) = norm(full(X), p)
dot(X::AbstractKKTVector, Y::AbstractKKTVector) = dot(full(X), full(Y))
function axpy!(a::Number, X::AbstractKKTVector, Y::AbstractKKTVector)
    axpy!(a, full(X), full(Y))
end

#=
    UnreducedKKTVector
=#

"""
    UnreducedKKTVector{T, VT<:AbstractVector{T}} <: AbstractKKTVector{T, VT}

Full KKT vector ``(x, s, y, z, ν, w)``, associated to a [`AbstractUnreducedKKTSystem`](@ref).

"""
struct UnreducedKKTVector{T, VT<:AbstractVector{T}, VI} <: AbstractKKTVector{T, VT}
    values::VT
    x::VT  # unsafe view
    xp::VT # unsafe view
    xp_lr::SubVector{T, VT, VI}
    xp_ur::SubVector{T, VT, VI}
    xl::VT # unsafe view
    xzl::VT # unsafe view
    xzu::VT # unsafe view
end

function UnreducedKKTVector(
    ::Type{VT}, n::Int, m::Int, nlb::Int, nub::Int, ind_lb, ind_ub
) where {T, VT <: AbstractVector{T}}
    values = VT(undef,n+m+nlb+nub)
    fill!(values, zero(T))
    # Wrap directly array x to avoid dealing with views
    x = _madnlp_unsafe_wrap(values, n + m) # Primal-Dual
    xp = _madnlp_unsafe_wrap(values, n) # Primal
    xl = _madnlp_unsafe_wrap(values, m, n+1) # Dual
    xzl = _madnlp_unsafe_wrap(values, nlb, n + m + 1) # Lower bound
    xzu = _madnlp_unsafe_wrap(values, nub, n + m + nlb + 1) # Upper bound

    xp_lr = view(xp, ind_lb)
    xp_ur = view(xp, ind_ub)

    return UnreducedKKTVector(values, x, xp, xp_lr, xp_ur, xl, xzl, xzu)
end

function UnreducedKKTVector(kkt::AbstractKKTSystem{T, VT}) where {T, VT}
    return UnreducedKKTVector(
        VT,
        length(kkt.pr_diag),
        length(kkt.du_diag),
        length(kkt.l_diag),
        length(kkt.u_diag),
        kkt.ind_lb,
        kkt.ind_ub,
    )
end

function Base.copy(rhs::UnreducedKKTVector{T, VT}) where {T, VT}
    new_rhs = UnreducedKKTVector(
        VT,
        length(rhs.xp),
        length(rhs.xl),
        length(rhs.xzl),
        length(rhs.xzu),
        rhs.xp_lr.indices[1],
        rhs.xp_ur.indices[1],
    )
    copyto!(full(new_rhs), full(rhs))
    return new_rhs
end

full(rhs::UnreducedKKTVector) = rhs.values
primal(rhs::UnreducedKKTVector) = rhs.xp
dual(rhs::UnreducedKKTVector) = rhs.xl
primal_dual(rhs::UnreducedKKTVector) = rhs.x
dual_lb(rhs::UnreducedKKTVector) = rhs.xzl
dual_ub(rhs::UnreducedKKTVector) = rhs.xzu


"""
    PrimalVector{T, VT<:AbstractVector{T}} <: AbstractKKTVector{T, VT}

Primal vector ``(x, s)``.

"""
struct PrimalVector{T, VT<:AbstractVector{T}, VI} <: AbstractKKTVector{T, VT}
    values::VT
    values_lr::SubVector{T, VT, VI}
    values_ur::SubVector{T, VT, VI}
    x::VT  # unsafe view
    s::VT # unsafe view
end

function PrimalVector(::Type{VT}, nx::Int, ns::Int, ind_lb, ind_ub) where {T, VT <: AbstractVector{T}}
    values = VT(undef, nx+ns)
    fill!(values, zero(T))
    x = _madnlp_unsafe_wrap(values, nx)
    s = _madnlp_unsafe_wrap(values, ns, nx+1)
    values_lr = view(values, ind_lb)
    values_ur = view(values, ind_ub)

    return PrimalVector(
        values, values_lr, values_ur, x, s,
    )
end

full(rhs::PrimalVector) = rhs.values
primal(rhs::PrimalVector) = rhs.values
variable(rhs::PrimalVector) = rhs.x
slack(rhs::PrimalVector) = rhs.s

