@kwdef mutable struct KrylovOptions <: AbstractOptions
    krylov_memory::Int = 20
    krylov_max_iter::Int = 10
    krylov_max_cycles::Int = 5
    krylov_tol::Float64
    krylov_acceptable_tol::Float64
    # prototype: the target system with the dual regularization τ = ratio · δ_current on the equality
    # rows (condensed KKT only; the factorization stays at δ); 0 is the unshifted system
    krylov_dual_reg_ratio::Float64 = 0.0
    # prototype: start from the direct solve's step (x0 = P⁻¹ b) and run GMRES on its residual
    krylov_warm_start::Bool = false
    # prototype: Richardson first, up to this many steps with the condensed solve, stopping at the
    # tolerance, on a stall (ratio above krylov_stall times the previous) or on a step that raises the
    # residual (undone); GMRES only after that; 0: off
    krylov_richardson_max::Int = 0
    krylov_stall::Float64 = 0.5
end

# The KKT system acting on the full vector of an `UnreducedKKTVector`.
mutable struct KKTOperator{T, KKT <: AbstractKKTSystem{T}, KV <: AbstractKKTVector{T}}
    kkt::KKT
    x::KV
    y::KV
    tau::T
end
KKTOperator(kkt::AbstractKKTSystem{T}, x, y) where {T} = KKTOperator(kkt, x, y, zero(T))
function _sub_tau!(y, kkt::MadCore.SparseCondensedKKTSystem, tau, x)
    o = size(kkt.hess_com, 1) + length(kkt.ind_ineq)
    view(y, o .+ kkt.ind_eq) .-= tau .* view(x, o .+ kkt.ind_eq)
    return y
end
_sub_tau!(y, kkt, tau, x) = y
_eq_shift(kkt::MadCore.SparseCondensedKKTSystem) = kkt.eq_shift.current
_eq_shift(kkt) = 0.0
Base.size(A::KKTOperator) = (length(full(A.x)), length(full(A.x)))
Base.size(A::KKTOperator, i::Integer) = size(A)[i]
Base.eltype(::KKTOperator{T}) where {T} = T
function mul!(y::AbstractVector, A::KKTOperator{T}, x::AbstractVector) where {T}
    copyto!(full(A.x), x)
    mul!(A.y, A.kkt, A.x, one(T), zero(T))
    copyto!(y, full(A.y))
    iszero(A.tau) || _sub_tau!(y, A.kkt, A.tau, x)
    return y
end

# The factorized KKT system, used as a right preconditioner.
struct KKTPreconditioner{T, KKT <: AbstractKKTSystem{T}, KV <: AbstractKKTVector{T}}
    kkt::KKT
    x::KV
end
function ldiv!(y::AbstractVector, P::KKTPreconditioner, x::AbstractVector)
    copyto!(full(P.x), x)
    solve_kkt!(P.kkt, P.x)
    copyto!(y, full(P.x))
    return y
end

struct KrylovState{T, KKT <: AbstractKKTSystem{T}, Op, Prec, W} <: AbstractIterator{T}
    kkt::KKT
    op::Op
    prec::Prec
    workspace::W
    opt::KrylovOptions
    cnt::MadNLPCounters
    logger::MadNLPLogger
end

function KrylovState(
    kkt::AbstractKKTSystem{T};
    opt = default_options(KrylovState, sqrt(eps(T))),
    logger = MadNLPLogger(),
    cnt = MadNLPCounters(),
) where {T}
    op = KKTOperator(kkt, UnreducedKKTVector(kkt), UnreducedKKTVector(kkt))
    prec = KKTPreconditioner(kkt, UnreducedKKTVector(kkt))
    n = length(full(op.x))
    workspace = Krylov.GmresWorkspace(n, n, typeof(full(op.x)); memory = opt.krylov_memory)
    return KrylovState(kkt, op, prec, workspace, opt, cnt, logger)
end

default_options(::Type{KrylovState}, tol) =
    KrylovOptions(krylov_tol = tol^(5 / 4), krylov_acceptable_tol = tol^(5 / 8))

# GMRES cycles of at most `krylov_max_iter` steps, each on the true residual of the solution so
# far: GMRES's own residual estimate can fall far below the true residual when the
# factorization is inexact (a shifted or statically pivoted LDLᵀ), and only a restart from the
# true residual corrects the solution beyond that point.
function solve_refine!(x::VT, it::KrylovState{T}, b::VT, w::VT; max_iter_factor::Integer = 1) where {T, VT <: AbstractKKTVector{T}}
    @debug(it.logger, "Iterative solver initiated")
    norm_b = norm(full(b), Inf)
    fill!(full(x), zero(T))
    norm_b == zero(T) && return true

    copyto!(full(w), full(b))
    it.cnt.ir = 0
    residual_ratio = T(Inf)
    it.op.tau = T(it.opt.krylov_dual_reg_ratio) * T(_eq_shift(it.kkt))
    if it.opt.krylov_warm_start
        copyto!(full(x), full(b))
        solve_kkt!(it.kkt, x)
        mul!(full(w), it.op, full(x))
        full(w) .= full(b) .- full(w)
        it.cnt.ir += 1
    end
    if it.opt.krylov_richardson_max > 0
        steps = 0
        for _ in 1:it.opt.krylov_richardson_max
            copyto!(full(it.prec.x), full(w))
            solve_kkt!(it.kkt, it.prec.x)          # the step P⁻¹ r; it.prec.x keeps it
            full(x) .+= full(it.prec.x)
            steps += 1
            mul!(full(w), it.op, full(x))
            full(w) .= full(b) .- full(w)
            new = norm(full(w), Inf) / (min(norm(full(x), Inf), 1.0e6 * norm_b) + norm_b)
            if !(new < residual_ratio)
                full(x) .-= full(it.prec.x)
                mul!(full(w), it.op, full(x))
                full(w) .= full(b) .- full(w)
                break
            end
            stalled = new > it.opt.krylov_stall * residual_ratio
            residual_ratio = new
            (residual_ratio < it.opt.krylov_tol || stalled) && break
        end
        it.cnt.ir += steps
    end
    for _ in 1:it.opt.krylov_max_cycles
        residual_ratio < it.opt.krylov_tol && break
        Krylov.gmres!(
            it.workspace, it.op, full(w);
            N = it.prec, ldiv = true,
            atol = zero(T), rtol = T(it.opt.krylov_tol),
            itmax = max_iter_factor * it.opt.krylov_max_iter,
        )
        full(x) .+= Krylov.solution(it.workspace)
        it.cnt.ir += Krylov.iteration_count(it.workspace)

        # Residual ratio of the refined solution, as in RichardsonIterator.
        if iszero(it.op.tau)
            copyto!(full(w), full(b))
            mul!(w, it.kkt, x, -one(T), one(T))
        else
            mul!(full(w), it.op, full(x))
            full(w) .= full(b) .- full(w)
        end
        residual_ratio = norm(full(w), Inf) / (min(norm(full(x), Inf), 1.0e6 * norm_b) + norm_b)
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
