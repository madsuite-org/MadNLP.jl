@kwdef mutable struct KrylovOptions <: AbstractOptions
    krylov_memory::Int = 20
    krylov_max_iter::Int = 10
    krylov_tol::Float64
    krylov_acceptable_tol::Float64
end

# The KKT system acting on the full vector of an `UnreducedKKTVector`.
struct KKTOperator{T, KKT <: AbstractKKTSystem{T}, KV <: AbstractKKTVector{T}}
    kkt::KKT
    x::KV
    y::KV
end
Base.size(A::KKTOperator) = (length(full(A.x)), length(full(A.x)))
Base.size(A::KKTOperator, i::Integer) = size(A)[i]
Base.eltype(::KKTOperator{T}) where {T} = T
function mul!(y::AbstractVector, A::KKTOperator{T}, x::AbstractVector) where {T}
    copyto!(full(A.x), x)
    mul!(A.y, A.kkt, A.x, one(T), zero(T))
    copyto!(y, full(A.y))
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

function solve_refine!(x::VT, it::KrylovState{T}, b::VT, w::VT) where {T, VT <: AbstractKKTVector{T}}
    @debug(it.logger, "Iterative solver initiated")
    norm_b = norm(full(b), Inf)
    fill!(full(x), zero(T))
    norm_b == zero(T) && return true

    Krylov.gmres!(
        it.workspace, it.op, full(b);
        N = it.prec, ldiv = true,
        atol = zero(T), rtol = T(it.opt.krylov_tol),
        itmax = it.opt.krylov_max_iter,
    )
    copyto!(full(x), Krylov.solution(it.workspace))
    it.cnt.ir = Krylov.iteration_count(it.workspace)

    # Residual ratio of the refined solution, as in RichardsonIterator.
    copyto!(full(w), full(b))
    mul!(w, it.kkt, x, -one(T), one(T))
    residual_ratio = norm(full(w), Inf) / (min(norm(full(x), Inf), 1e6 * norm_b) + norm_b)
    @debug(
        it.logger,
        @sprintf(
            "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
            it.cnt.ir, residual_ratio,
        ),
    )
    return residual_ratio < it.opt.krylov_acceptable_tol
end
