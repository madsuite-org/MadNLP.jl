@kwdef mutable struct SymCGOptions <: AbstractOptions
    symcg_max_iter::Int = 10
    symcg_max_cycles::Int = 5
    symcg_tol::Float64
    symcg_acceptable_tol::Float64
end

"""
    SymCGIterator

Iterative refinement of the unreduced KKT system by preconditioned conjugate gradients on its
symmetrized form, with the option `iterator = SymCGIterator`.

With `Z` the bound multipliers and `S` the distances to the bounds, the unreduced system is
made symmetric by scaling the bound-multiplier columns by `Z^½` and their rows by `∓Z^-½`:

    [W + δ   Jᵀ   -Zₗ^½   Zᵤ^½]
    [J       D     0       0  ]
    [-Zₗ^½   0    -Sₗ      0  ]
    [Zᵤ^½    0     0      -Sᵤ ]

The preconditioner is the factorized KKT system scaled the same way. Both are indefinite, so the
plain CG recurrence is used, without a check on the sign of the preconditioned residual. Each
cycle of at most `symcg_max_iter` steps starts from the true residual of the solution so far.

A bound whose multiplier is zero (the least-squares estimate of the constraint multipliers sets
them to zero in the KKT system) has a row holding only its diagonal: it is solved directly and
left out of the CG system.
"""
struct SymCGIterator{T, KKT <: AbstractKKTSystem{T}, KV <: AbstractKKTVector{T}, VT <: AbstractVector{T}} <: AbstractIterator{T}
    kkt::KKT
    u::KV
    v::KV
    c::VT    # column scaling
    r::VT    # row scaling
    ci::VT   # inverse of the column scaling, zero on rows left out
    ri::VT   # inverse of the row scaling, zero on rows left out
    xt::VT
    rt::VT
    zt::VT
    pt::VT
    qt::VT
    opt::SymCGOptions
    cnt::MadNLPCounters
    logger::MadNLPLogger
end

function SymCGIterator(
    kkt::AbstractKKTSystem{T};
    opt = default_options(SymCGIterator, sqrt(eps(T))),
    logger = MadNLPLogger(),
    cnt = MadNLPCounters(),
) where {T}
    u = UnreducedKKTVector(kkt)
    v = UnreducedKKTVector(kkt)
    w() = similar(full(u))
    return SymCGIterator(kkt, u, v, w(), w(), w(), w(), w(), w(), w(), w(), w(), opt, cnt, logger)
end

default_options(::Type{SymCGIterator}, tol) =
    SymCGOptions(symcg_tol = tol^(5 / 4), symcg_acceptable_tol = tol^(5 / 8))

_inv_or_zero(s) = ifelse(iszero(s), zero(s), inv(s))

function _set_symcg_scaling!(it::SymCGIterator{T}) where {T}
    kkt = it.kkt
    np = length(kkt.pr_diag) + length(kkt.du_diag)
    nlb = length(kkt.l_lower)
    il = (np + 1):(np + nlb)
    iu = (np + nlb + 1):(np + nlb + length(kkt.u_lower))
    fill!(it.c, one(T))
    fill!(it.r, one(T))
    view(it.c, il) .= sqrt.(kkt.l_lower)
    view(it.c, iu) .= sqrt.(kkt.u_lower)
    view(it.r, il) .= .-_inv_or_zero.(view(it.c, il))
    view(it.r, iu) .= _inv_or_zero.(view(it.c, iu))
    it.ci .= _inv_or_zero.(it.c)
    it.ri .= _inv_or_zero.(it.r)
    return
end

# y = R K C p
function _symcg_op!(y, it::SymCGIterator{T}, p) where {T}
    full(it.u) .= it.c .* p
    mul!(it.v, it.kkt, it.u, one(T), zero(T))
    y .= it.r .* full(it.v)
    return y
end

# z = C⁻¹ P⁻¹ R⁻¹ r
function _symcg_prec!(z, it::SymCGIterator, r)
    full(it.u) .= it.ri .* r
    solve_kkt!(it.kkt, it.u)
    z .= it.ci .* full(it.u)
    return z
end

function solve_refine!(x::VT, it::SymCGIterator{T}, b::VT, w::VT) where {T, VT <: AbstractKKTVector{T}}
    @debug(it.logger, "Iterative solver initiated")
    kkt = it.kkt
    norm_b = norm(full(b), Inf)
    fill!(full(x), zero(T))
    it.cnt.ir = 0
    norm_b == zero(T) && return true
    _set_symcg_scaling!(it)

    # the bounds with zero multipliers: -l_diag dzₗ = bₗ, u_diag dzᵤ = bᵤ
    dual_lb(x) .= ifelse.(iszero.(kkt.l_lower), .-dual_lb(b) ./ kkt.l_diag, zero(T))
    dual_ub(x) .= ifelse.(iszero.(kkt.u_lower), dual_ub(b) ./ kkt.u_diag, zero(T))
    copyto!(full(w), full(b))
    mul!(w, kkt, x, -one(T), one(T))

    xt, rt, zt, pt, qt = it.xt, it.rt, it.zt, it.pt, it.qt
    residual_ratio = T(Inf)
    for _ in 1:it.opt.symcg_max_cycles
        rt .= it.r .* full(w)
        fill!(xt, zero(T))
        norm_r0 = norm(rt, Inf)
        _symcg_prec!(zt, it, rt)
        pt .= zt
        γ = dot(rt, zt)
        for _ in 1:it.opt.symcg_max_iter
            _symcg_op!(qt, it, pt)
            pq = dot(pt, qt)
            (iszero(pq) || !isfinite(pq) || !isfinite(γ)) && break
            α = γ / pq
            xt .+= α .* pt
            rt .-= α .* qt
            it.cnt.ir += 1
            norm(rt, Inf) <= it.opt.symcg_tol * norm_r0 && break
            _symcg_prec!(zt, it, rt)
            γ_next = dot(rt, zt)
            pt .= zt .+ (γ_next / γ) .* pt
            γ = γ_next
        end
        full(x) .+= it.c .* xt

        # residual ratio of the refined solution, as in RichardsonIterator
        copyto!(full(w), full(b))
        mul!(w, kkt, x, -one(T), one(T))
        residual_ratio = norm(full(w), Inf) / (min(norm(full(x), Inf), 1.0e6 * norm_b) + norm_b)
        (residual_ratio < it.opt.symcg_tol || !isfinite(residual_ratio)) && break
    end
    @debug(
        it.logger,
        @sprintf(
            "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
            it.cnt.ir, residual_ratio,
        ),
    )
    return residual_ratio < it.opt.symcg_acceptable_tol
end
