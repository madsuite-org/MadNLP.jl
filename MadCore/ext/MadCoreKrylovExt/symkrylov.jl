# GMRES on the unreduced KKT system made symmetric by diagonal scalings (SymKrylovIterator).

# The diagonal scalings that make the unreduced KKT system symmetric: K̃ = R K C, and the
# preconditioner C⁻¹ P⁻¹ R⁻¹. A bound with a zero multiplier gets zero in all four, which
# leaves it out of the Krylov system.
struct SymScaling{VT}
    c::VT
    r::VT
    ci::VT
    ri::VT
end

# K̃ = R K C, acting on full vectors
struct SymOperator{T, KKT <: AbstractKKTSystem{T}, KV <: AbstractKKTVector{T}, S <: SymScaling}
    kkt::KKT
    u::KV
    v::KV
    s::S
end
Base.size(A::SymOperator) = (length(A.s.c), length(A.s.c))
Base.size(A::SymOperator, i::Integer) = size(A)[i]
Base.eltype(::SymOperator{T}) where {T} = T
function mul!(y::AbstractVector, A::SymOperator{T}, x::AbstractVector) where {T}
    full(A.u) .= A.s.c .* x
    mul!(A.v, A.kkt, A.u, one(T), zero(T))
    y .= A.s.r .* full(A.v)
    return y
end

# C⁻¹ P⁻¹ R⁻¹, with P the factorized KKT system
struct SymPreconditioner{T, KKT <: AbstractKKTSystem{T}, KV <: AbstractKKTVector{T}, S <: SymScaling}
    kkt::KKT
    u::KV
    s::S
end
function ldiv!(y::AbstractVector, P::SymPreconditioner, x::AbstractVector)
    full(P.u) .= P.s.ri .* x
    solve_kkt!(P.kkt, P.u)
    y .= P.s.ci .* full(P.u)
    return y
end

struct SymKrylovState{T, KKT <: AbstractKKTSystem{T}, VT, Op, Prec, W} <: AbstractIterator{T}
    kkt::KKT
    s::SymScaling{VT}
    rhs::VT
    op::Op
    prec::Prec
    workspace::W
    opt::KrylovOptions
    cnt::MadNLPCounters
    logger::MadNLPLogger
end

function SymKrylovState(
        kkt::AbstractKKTSystem{T};
        opt = default_options(KrylovState, sqrt(eps(T))),
        logger = MadNLPLogger(),
        cnt = MadNLPCounters(),
    ) where {T}
    u = UnreducedKKTVector(kkt)
    VT = typeof(full(u))
    n = length(full(u))
    s = SymScaling(similar(full(u)), similar(full(u)), similar(full(u)), similar(full(u)))
    op = SymOperator(kkt, u, UnreducedKKTVector(kkt), s)
    prec = SymPreconditioner(kkt, UnreducedKKTVector(kkt), s)
    workspace = Krylov.GmresWorkspace(n, n, VT; memory = opt.krylov_memory)
    return SymKrylovState(kkt, s, similar(full(u)), op, prec, workspace, opt, cnt, logger)
end

_inv_or_zero(a) = ifelse(iszero(a), zero(a), inv(a))

function set_scaling!(it::SymKrylovState{T}) where {T}
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

# As the GMRES iterator, on the symmetric system: cycles of at most `krylov_max_iter` steps,
# each from the true residual of the solution so far.
function solve_refine!(x::VT, it::SymKrylovState{T}, b::VT, w::VT) where {T, VT <: AbstractKKTVector{T}}
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

    residual_ratio = T(Inf)
    for _ in 1:it.opt.krylov_max_cycles
        it.rhs .= it.s.r .* full(w)
        Krylov.gmres!(
            it.workspace, it.op, it.rhs;
            N = it.prec, ldiv = true,
            atol = zero(T), rtol = T(it.opt.krylov_tol),
            itmax = it.opt.krylov_max_iter,
        )
        full(x) .+= it.s.c .* Krylov.solution(it.workspace)
        it.cnt.ir += Krylov.iteration_count(it.workspace)

        # Residual ratio of the refined solution, as in RichardsonIterator.
        copyto!(full(w), full(b))
        mul!(w, kkt, x, -one(T), one(T))
        residual_ratio = norm(full(w), Inf) / (min(norm(full(x), Inf), T(1.0e6) * norm_b) + norm_b)
        (residual_ratio < it.opt.krylov_tol || !isfinite(residual_ratio)) && break
    end
    @debug(
        it.logger,
        @sprintf(
            "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
            it.cnt.ir, residual_ratio,
        ),
    )
    return residual_ratio < it.opt.krylov_acceptable_tol
end
