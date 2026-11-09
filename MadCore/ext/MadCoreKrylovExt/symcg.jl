@kwdef mutable struct SymCGOptions <: AbstractOptions
    symcg_max_iter::Int = 10
    symcg_max_cycles::Int = 5
    symcg_tol::Float64
    symcg_acceptable_tol::Float64
end

# The diagonal scalings that make the unreduced KKT system symmetric: K̃ = R K C, and the
# preconditioner C⁻¹ P⁻¹ R⁻¹. A bound with a zero multiplier gets zero in all four, which
# leaves it out of the CG system.
struct SymCGScaling{VT}
    c::VT
    r::VT
    ci::VT
    ri::VT
end

# K̃ = R K C, acting on full vectors
struct SymCGOperator{T, KKT <: AbstractKKTSystem{T}, KV <: AbstractKKTVector{T}, S <: SymCGScaling}
    kkt::KKT
    u::KV
    v::KV
    s::S
end
Base.size(A::SymCGOperator) = (length(A.s.c), length(A.s.c))
Base.size(A::SymCGOperator, i::Integer) = size(A)[i]
Base.eltype(::SymCGOperator{T}) where {T} = T
function mul!(y::AbstractVector, A::SymCGOperator{T}, x::AbstractVector) where {T}
    full(A.u) .= A.s.c .* x
    mul!(A.v, A.kkt, A.u, one(T), zero(T))
    y .= A.s.r .* full(A.v)
    return y
end

# C⁻¹ P⁻¹ R⁻¹, with P the factorized KKT system
struct SymCGPreconditioner{T, KKT <: AbstractKKTSystem{T}, KV <: AbstractKKTVector{T}, S <: SymCGScaling}
    kkt::KKT
    u::KV
    s::S
end
function ldiv!(y::AbstractVector, P::SymCGPreconditioner, x::AbstractVector)
    full(P.u) .= P.s.ri .* x
    solve_kkt!(P.kkt, P.u)
    y .= P.s.ci .* full(P.u)
    return y
end

struct SymCGState{T, KKT <: AbstractKKTSystem{T}, VT, Op, Prec, W} <: AbstractIterator{T}
    kkt::KKT
    s::SymCGScaling{VT}
    rhs::VT
    x_prev::VT
    op::Op
    prec::Prec
    workspace::W
    opt::SymCGOptions
    cnt::MadNLPCounters
    logger::MadNLPLogger
end

function SymCGState(
        kkt::AbstractKKTSystem{T};
        opt = default_options(SymCGState, sqrt(eps(T))),
        logger = MadNLPLogger(),
        cnt = MadNLPCounters(),
    ) where {T}
    u = UnreducedKKTVector(kkt)
    VT = typeof(full(u))
    n = length(full(u))
    s = SymCGScaling(similar(full(u)), similar(full(u)), similar(full(u)), similar(full(u)))
    op = SymCGOperator(kkt, u, UnreducedKKTVector(kkt), s)
    prec = SymCGPreconditioner(kkt, UnreducedKKTVector(kkt), s)
    workspace = Krylov.CgWorkspace(n, n, VT)
    return SymCGState(kkt, s, similar(full(u)), similar(full(u)), op, prec, workspace, opt, cnt, logger)
end

default_options(::Type{SymCGState}, tol) =
    SymCGOptions(symcg_tol = tol^(5 / 4), symcg_acceptable_tol = tol^(5 / 8))

_inv_or_zero(a) = ifelse(iszero(a), zero(a), inv(a))

_residual_ratio(w, x, norm_b::T) where {T} =
    norm(full(w), Inf) / (min(norm(full(x), Inf), T(1.0e6) * norm_b) + norm_b)

function set_scaling!(it::SymCGState{T}) where {T}
    kkt, s = it.kkt, it.s
    np = length(kkt.pr_diag) + length(kkt.du_diag)
    nlb = length(kkt.l_lower)
    il = (np + 1):(np + nlb)
    iu = (np + nlb + 1):length(s.c)
    fill!(view(s.c, 1:np), one(T))
    fill!(view(s.r, 1:np), one(T))
    view(s.c, il) .= sqrt.(kkt.l_lower)
    view(s.c, iu) .= sqrt.(kkt.u_lower)
    view(s.r, il) .= .-_inv_or_zero.(view(s.c, il))
    view(s.r, iu) .= _inv_or_zero.(view(s.c, iu))
    s.ci .= _inv_or_zero.(s.c)
    s.ri .= _inv_or_zero.(s.r)
    return
end

function solve_refine!(x::VT, it::SymCGState{T}, b::VT, w::VT) where {T, VT <: AbstractKKTVector{T}}
    @debug(it.logger, "Iterative solver initiated")
    kkt = it.kkt
    norm_b = norm(full(b), Inf)
    fill!(full(x), zero(T))
    it.cnt.ir = 0
    norm_b == zero(T) && return true
    set_scaling!(it)

    # the bounds with zero multipliers: -l_diag dzₗ = bₗ, u_diag dzᵤ = bᵤ
    dual_lb(x) .= ifelse.(iszero.(kkt.l_lower), .-dual_lb(b) ./ kkt.l_diag, zero(T))
    dual_ub(x) .= ifelse.(iszero.(kkt.u_lower), dual_ub(b) ./ kkt.u_diag, zero(T))
    copyto!(full(w), full(b))
    mul!(w, kkt, x, -one(T), one(T))
    residual_ratio = _residual_ratio(w, x, norm_b)

    for _ in 1:it.opt.symcg_max_cycles
        (residual_ratio < it.opt.symcg_tol || !isfinite(residual_ratio)) && break
        copyto!(it.x_prev, full(x))
        it.rhs .= it.s.r .* full(w)
        Krylov.cg!(
            it.workspace, it.op, it.rhs;
            M = it.prec, ldiv = true, indefinite = true,
            atol = zero(T), rtol = T(it.opt.symcg_tol),
            itmax = it.opt.symcg_max_iter,
        )
        full(x) .+= it.s.c .* Krylov.solution(it.workspace)
        it.cnt.ir += Krylov.iteration_count(it.workspace)

        # Residual ratio of the refined solution, as in RichardsonIterator.
        copyto!(full(w), full(b))
        mul!(w, kkt, x, -one(T), one(T))
        ratio = _residual_ratio(w, x, norm_b)
        # On an indefinite system the residual of CG is not monotone, and a step close to
        # breakdown can leave the cycle worse than it started: keep the better iterate.
        if !(ratio < residual_ratio)
            copyto!(full(x), it.x_prev)
            copyto!(full(w), full(b))
            mul!(w, kkt, x, -one(T), one(T))
            break
        end
        residual_ratio = ratio
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
