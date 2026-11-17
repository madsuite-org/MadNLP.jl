
# Prototype: MINRES on the reduced KKT system of SparseCondensedKKTSystem with a symmetrizing
# preconditioner. The reduced system (bounds condensed into pr_diag) in the unknowns (x, s, y),
#
#     [ A    0    Jᵀ ] [x]   [r_x]        A = H + Σx + reg,  Σs the slack diagonal,
#     [ 0    Σs  -Iₛᵀ] [s] = [r_s]        Iₛ the slack rows,  Σd = du_diag (≤ 0),
#     [ J   -Iₛ   Σd ] [y]   [r_y]
#
# is symmetric indefinite. With B = [0; J] and G the (s, y) block of the SHIFTED system (the
# equality rows get Σd - δ), the shifted matrix is U diag(K, G) Uᵀ with K = A - Bᵀ G⁻¹ B =
# A + Jᵀ D J, the condensed matrix, and U = [I BᵀG⁻¹; 0 I]. P = U diag(K, |G|) Uᵀ is symmetric
# positive definite (K is, |G| is), so MINRES applies to P⁻¹ M, and P⁻¹ M_shifted has the
# eigenvalues ±1: the unshifted M differs from it by δ on the equality rows only.
# P⁻¹ r = U⁻ᵀ diag(K⁻¹, |G|⁻¹) U⁻¹ r costs one triangular solve with the factorization and two
# products with J.

@kwdef mutable struct MinresOptions <: AbstractOptions
    minres_max_iter::Int = 20
    minres_max_cycles::Int = 5
    minres_tol::Float64
    minres_acceptable_tol::Float64
    minres_conlim::Float64 = Inf   # Krylov's limit on the estimated condition number (1e8 stops early on these systems)
end

# The reduced KKT operator on a vector [x; s; y]: the full product of the KKT system on a
# vector with zero bound-multiplier blocks (so that the symmetric Hessian product is the one
# every backend implements), with the primal diagonal pr_diag (bounds condensed) in place of reg.
struct ReducedKKTOperator{T, KKT <: MadCore.SparseCondensedKKTSystem{T}, KV}
    kkt::KKT
    n::Int
    ns::Int
    m::Int
    xk::KV
    yk::KV
end
Base.size(A::ReducedKKTOperator) = (A.n + A.ns + A.m, A.n + A.ns + A.m)
Base.size(A::ReducedKKTOperator, i::Integer) = size(A)[i]
Base.eltype(::ReducedKKTOperator{T}) where {T} = T
function mul!(w::AbstractVector, A::ReducedKKTOperator{T}, v::AbstractVector) where {T}
    kkt, N = A.kkt, A.n + A.ns + A.m
    fx, fy = full(A.xk), full(A.yk)
    copyto!(view(fx, 1:N), v)
    fill!(view(fx, (N + 1):length(fx)), zero(T))
    mul!(A.yk, kkt, A.xk, one(T), zero(T))
    copyto!(w, view(fy, 1:N))
    np = A.n + A.ns
    view(w, 1:np) .+= (view(kkt.pr_diag, 1:np) .- view(kkt.reg, 1:np)) .* view(v, 1:np)
    return w
end

# P⁻¹ with the pieces of the current factorization.
struct SymmetrizingPreconditioner{T, KKT <: MadCore.SparseCondensedKKTSystem{T}, VT}
    kkt::KKT
    n::Int
    ns::Int
    m::Int
    t::VT      # n
    g::VT      # m, J u_x
end
function ldiv!(w::AbstractVector, P::SymmetrizingPreconditioner{T}, r::AbstractVector) where {T}
    kkt, n, ns, m = P.kkt, P.n, P.ns, P.m
    rx = view(r, 1:n); rs = view(r, (n + 1):(n + ns)); ry = view(r, (n + ns + 1):(n + ns + m))
    wx = view(w, 1:n); ws = view(w, (n + 1):(n + ns)); wy = view(w, (n + ns + 1):(n + ns + m))
    Σs = view(kkt.pr_diag, (n + 1):(n + ns))
    Σd = kkt.du_diag
    D = kkt.diag_buffer                    # 1 / (δ - Σd) on the equality rows, Σs / (1 - Σd Σs) on the others
    δ = kkt.eq_shift.current
    # G⁻¹ r on the dual rows: equality rows -D r_y; inequality rows (r_s + Σs r_y) / det, det = Σs Σd - 1
    wy .= .-D .* ry                        # right on the equality rows
    view(wy, kkt.ind_ineq) .= (rs .+ Σs .* view(ry, kkt.ind_ineq)) ./ (Σs .* view(Σd, kkt.ind_ineq) .- one(T))
    # U⁻¹ r: t_x = r_x - Jᵀ (G⁻¹ r)_y;  then K⁻¹ t_x
    P.t .= rx
    mul!(P.t, MadCore.jac_op(kkt, 'T'), wy, -one(T), one(T))
    MadCore.solve_linear_system!(kkt.linear_solver, P.t)
    wx .= P.t
    # |G|⁻¹ (r_s, r_y): equality rows (δ - Σd)⁻¹ r_y = D r_y; inequality rows the 2x2 [a -1; -1 c],
    # a = Σs, c = Σd, |G|⁻¹ = Q |Λ|⁻¹ Qᵀ with λ± = (a + c) / 2 ± sqrt(((a - c) / 2)² + 1), q+ ∝ (-1, λ+ - a)
    wy .= D .* ry
    a = Σs; c = view(Σd, kkt.ind_ineq); ryi = view(ry, kkt.ind_ineq)
    # with d = (a - c) / 2 and h = sqrt(d² + 1): λ+ = (a + c) / 2 + h, λ- = (a c - 1) / λ+ (the product
    # of the eigenvalues, no cancellation when a is large and c near zero), λ+ - a = 1 / (h + d),
    # λ- - a = -(d + h); q± ∝ (-1, λ± - a)
    d = (a .- c) ./ 2
    h = sqrt.(d .^ 2 .+ one(T))
    lp = (a .+ c) ./ 2 .+ h
    lm = (a .* c .- one(T)) ./ lp
    lpa = one(T) ./ (h .+ d)
    lma = .-(d .+ h)
    np2 = one(T) .+ lpa .^ 2
    nm2 = one(T) .+ lma .^ 2
    cp = (.-rs .+ lpa .* ryi) ./ np2 ./ abs.(lp)     # (q+ᵀ r) / (|λ+| |q+|²), with q+ unnormalized
    cm = (.-rs .+ lma .* ryi) ./ nm2 ./ abs.(lm)
    ws .= .-cp .- cm
    view(wy, kkt.ind_ineq) .= cp .* lpa .+ cm .* lma
    # U⁻ᵀ: (s, y) -= G⁻¹ (0, J u_x)
    mul!(P.g, MadCore.jac_op(kkt, 'N'), wx)
    view(wy, kkt.ind_eq) .+= view(D, kkt.ind_eq) .* view(P.g, kkt.ind_eq)       # y -= (-D g)
    gi = view(P.g, kkt.ind_ineq)
    det = a .* c .- one(T)
    ws .-= gi ./ det
    view(wy, kkt.ind_ineq) .-= a .* gi ./ det
    return w
end

struct MinresState{T, KKT <: AbstractKKTSystem{T}, Op, Prec, W, VT} <: AbstractIterator{T}
    kkt::KKT
    op::Op
    prec::Prec
    workspace::W
    rhs::VT
    opt::MinresOptions
    cnt::MadNLPCounters
    logger::MadNLPLogger
end

function MinresState(
    kkt::MadCore.SparseCondensedKKTSystem{T};
    opt = default_options(MinresState, sqrt(eps(T))),
    logger = MadNLPLogger(),
    cnt = MadNLPCounters(),
) where {T}
    n = size(kkt.hess_com, 1); ns = length(kkt.ind_ineq); m = size(kkt.jt_csc, 2)
    VT = typeof(kkt.pr_diag)
    op = ReducedKKTOperator(kkt, n, ns, m, UnreducedKKTVector(kkt), UnreducedKKTVector(kkt))
    prec = SymmetrizingPreconditioner(kkt, n, ns, m, VT(undef, n), VT(undef, m))
    N = n + ns + m
    workspace = Krylov.MinresWorkspace(N, N, VT)
    return MinresState(kkt, op, prec, workspace, VT(undef, N), opt, cnt, logger)
end
MinresState(kkt::AbstractKKTSystem; kwargs...) =
    error("MinresIterator is a prototype for SparseCondensedKKTSystem (CondensedPreset) only")

default_options(::Type{MinresState}, tol) =
    MinresOptions(minres_tol = tol^(5 / 4), minres_acceptable_tol = tol^(5 / 8))

# MINRES cycles on the reduced system, each restarted from the true residual of the refined
# solution; the bound multipliers are recovered from the primal step at the end, as the
# condensed solve does, and the acceptance test is on the full system, as in RichardsonIterator.
function solve_refine!(x::VT, it::MinresState{T}, b::VT, w::VT; max_iter_factor::Integer = 1) where {T, VT <: AbstractKKTVector{T}}
    @debug(it.logger, "Iterative solver initiated")
    kkt = it.kkt
    N = length(it.rhs)
    norm_b = norm(full(b), Inf)
    fill!(full(x), zero(T))
    it.cnt.ir = 0
    norm_b == zero(T) && return true

    # the reduced right-hand side: the bound rows folded into the primal block
    copyto!(full(x), full(b))
    MadCore.reduce_rhs!(kkt, x)
    copyto!(it.rhs, view(full(x), 1:N))
    fill!(view(full(x), 1:N), zero(T))
    residual_ratio = T(Inf)
    for _ in 1:it.opt.minres_max_cycles
        Krylov.minres!(
            it.workspace, it.op, it.rhs;
            M = it.prec, ldiv = true,
            atol = zero(T), rtol = T(it.opt.minres_tol), conlim = T(it.opt.minres_conlim),
            itmax = max_iter_factor * it.opt.minres_max_iter, history = true,
        )
        it.cnt.ir += Krylov.iteration_count(it.workspace)
        sol = Krylov.solution(it.workspace)
        if !all(isfinite, sol)
            @debug(it.logger, "minres cycle: non-finite solution, discarded")
            break
        end
        view(full(x), 1:N) .+= sol
        st = it.workspace.stats
        @debug(it.logger, @sprintf("minres cycle: %s, %d iterations, residuals %s", st.status, Krylov.iteration_count(it.workspace), join([@sprintf("%.1e", r) for r in st.residuals], " ")))
        # the residual of the reduced system, for the next cycle
        mul!(it.rhs, it.op, view(full(x), 1:N))
        copyto!(full(w), full(b)); MadCore.reduce_rhs!(kkt, w)
        it.rhs .= view(full(w), 1:N) .- it.rhs
        residual_ratio = norm(it.rhs, Inf) / (min(norm(view(full(x), 1:N), Inf), 1.0e6 * norm_b) + norm_b)
        @debug(it.logger, @sprintf("minres cycle residual ratio %.2e (|b| %.2e, |x| %.2e)", residual_ratio, norm_b, norm(view(full(x), 1:N), Inf)))
        (residual_ratio < it.opt.minres_tol || !isfinite(residual_ratio)) && break
    end
    # the bound multipliers from the primal step (finish_aug_solve! reads the bound rows of the
    # right-hand side, still in x from the copy above)
    MadCore.finish_aug_solve!(kkt, x)
    @debug(
        it.logger,
        @sprintf(
            "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
            it.cnt.ir, residual_ratio,
        ),
    )
    return residual_ratio < it.opt.minres_acceptable_tol
end
