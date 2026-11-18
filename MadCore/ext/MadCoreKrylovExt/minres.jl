# MINRES hybrid refinement of SparseCondensedKKTSystem.
#
# Each solve first takes Richardson steps with the condensed solve, as RichardsonIterator does, and
# switches to MINRES only when they stall. MINRES runs on the full KKT system in (x, s, y, zl, zu),
# symmetrized by scaling the bound rows by z^(1/2):
#
#     dzl = -zl^(1/2) v_l,  dzu = zu^(1/2) v_u,  bound right-hand sides r_zl / zl^(1/2), r_zu / zu^(1/2),
#
# which turns the bound rows into  zl^(1/2) p_lb - (x - xˡ) v_l  and  zu^(1/2) p_ub - (xᵘ - x) v_u,
# symmetric with the primal rows. The preconditioner is symmetric positive definite: with the bound
# rows eliminated first (1x1 blocks G_b = -(x - xˡ), -(xᵘ - x)), the (s, y) block G of the SHIFTED
# system and K the condensed matrix, P = U_b U diag(K, |G|, |G_b|) Uᵀ U_bᵀ, under which the shifted
# system has the eigenvalues ±1. One application of P⁻¹ costs one triangular solve with the condensed
# factorization and two products with J.
#
# Richardson refines toward the unshifted system, as RichardsonIterator does. MINRES refines toward the
# system whose equality rows carry the dual regularization τ = minres_dual_reg_ratio δ (δ the current
# equality shift; the factorization is at δ), toward which it converges where the dual Schur complement
# is nearly singular. Both phases stop on the residual of the unsymmetrized system, the test
# RichardsonIterator uses.

@kwdef mutable struct MinresOptions <: AbstractOptions
    minres_tol::Float64
    minres_acceptable_tol::Float64
    # Richardson steps before MINRES: at most this many, ended at the tolerance, by a step that does not
    # bring the residual ratio below minres_stall times the previous one (a stall, which hands over to
    # MINRES), or by a step that raises it (undone, also handing over)
    minres_richardson_max::Int = 10
    minres_stall::Float64 = 0.5
    # MINRES: cycles of at most minres_max_iter iterations, each restarted from the true residual.
    # Krylov's own stopping tests are disabled (its ‖A‖ estimate is inflated by the 1/δ scaling of P⁻¹,
    # so they fire after one iteration); every iteration is tested on the unsymmetrized residual.
    minres_max_iter::Int = 50
    minres_max_cycles::Int = 4
    # τ / δ of the target system: 0 is the unshifted system, toward which MINRES does not converge where
    # the dual Schur complement is nearly singular; 1 is the shifted system the factorization solves
    minres_dual_reg_ratio::Float64 = 0.1
end

# The symmetrized KKT operator on [x; s; y; v_l; v_u]: the product of the KKT system on a vector with
# zero bound blocks, the dual regularization τ on the equality rows, and the scaled bound couplings.
mutable struct SymmetrizedKKTOperator{T, KKT <: MadCore.SparseCondensedKKTSystem{T}, KV, VT}
    kkt::KKT
    n::Int     # primal
    ns::Int    # slack
    m::Int     # dual
    xk::KV
    yk::KV
    sl::VT     # zl^(1/2)
    su::VT     # zu^(1/2)
    tau::T
end
_n3(A::SymmetrizedKKTOperator) = A.n + A.ns + A.m
Base.size(A::SymmetrizedKKTOperator) = (_n3(A) + length(A.sl) + length(A.su), _n3(A) + length(A.sl) + length(A.su))
Base.size(A::SymmetrizedKKTOperator, i::Integer) = size(A)[i]
Base.eltype(::SymmetrizedKKTOperator{T}) where {T} = T
function mul!(w::AbstractVector, A::SymmetrizedKKTOperator{T}, v::AbstractVector) where {T}
    kkt = A.kkt
    N3 = _n3(A); np = A.n + A.ns; nlb = length(A.sl); nub = length(A.su)
    vp = view(v, 1:np); vl = view(v, (N3 + 1):(N3 + nlb)); vu = view(v, (N3 + nlb + 1):(N3 + nlb + nub))
    fx, fy = full(A.xk), full(A.yk)
    copyto!(view(fx, 1:N3), view(v, 1:N3))
    fill!(view(fx, (N3 + 1):length(fx)), zero(T))
    mul!(A.yk, kkt, A.xk, one(T), zero(T))
    copyto!(view(w, 1:N3), view(fy, 1:N3))
    if !iszero(A.tau)
        wy = view(w, (np + 1):N3); vy = view(v, (np + 1):N3)
        view(wy, kkt.ind_eq) .-= A.tau .* view(vy, kkt.ind_eq)
    end
    wp = view(w, 1:np)
    view(wp, kkt.ind_lb) .+= A.sl .* vl
    view(wp, kkt.ind_ub) .+= A.su .* vu
    view(w, (N3 + 1):(N3 + nlb)) .= A.sl .* view(vp, kkt.ind_lb) .+ kkt.l_diag .* vl              # l_diag = xˡ - x
    view(w, (N3 + nlb + 1):(N3 + nlb + nub)) .= A.su .* view(vp, kkt.ind_ub) .+ kkt.u_diag .* vu  # u_diag = x - xᵘ
    return w
end
function _scalings!(A::SymmetrizedKKTOperator)
    A.sl .= sqrt.(A.kkt.l_lower)
    A.su .= sqrt.(A.kkt.u_lower)
    return A
end

# P⁻¹ = U_b⁻ᵀ U⁻ᵀ diag(K⁻¹, |G|⁻¹, |G_b|⁻¹) U⁻¹ U_b⁻¹ with the pieces of the current factorization.
struct SymmetrizingPreconditioner{T, Op <: SymmetrizedKKTOperator{T}, VT}
    op::Op
    r3::VT     # N3
    t::VT      # n
    g::VT      # m, J u_x
end
function ldiv!(w::AbstractVector, P::SymmetrizingPreconditioner{T}, r::AbstractVector) where {T}
    A = P.op; kkt = A.kkt
    n, ns, m = A.n, A.ns, A.m
    N3 = _n3(A); np = n + ns; nlb = length(A.sl); nub = length(A.su)
    rl = view(r, (N3 + 1):(N3 + nlb)); ru = view(r, (N3 + nlb + 1):(N3 + nlb + nub))
    # U_b⁻¹: r_p - B_bᵀ G_b⁻¹ r_b, with B_b = z^(1/2) and G_b = l_diag, u_diag
    copyto!(P.r3, view(r, 1:N3))
    view(P.r3, kkt.ind_lb) .-= A.sl .* rl ./ kkt.l_diag
    view(P.r3, kkt.ind_ub) .-= A.su .* ru ./ kkt.u_diag
    rx = view(P.r3, 1:n); rs = view(P.r3, (n + 1):np); ry = view(P.r3, (np + 1):N3)
    wx = view(w, 1:n); ws = view(w, (n + 1):np); wy = view(w, (np + 1):N3)
    Σs = view(kkt.pr_diag, (n + 1):np)
    Σd = kkt.du_diag
    D = kkt.diag_buffer                    # 1 / (δ - Σd) on the equality rows
    # G⁻¹ r on the dual rows: equality rows -D r_y; inequality rows (r_s + Σs r_y) / (Σs Σd - 1)
    wy .= .-D .* ry
    view(wy, kkt.ind_ineq) .= (rs .+ Σs .* view(ry, kkt.ind_ineq)) ./ (Σs .* view(Σd, kkt.ind_ineq) .- one(T))
    # U⁻¹: t_x = r_x - Jᵀ (G⁻¹ r)_y, then K⁻¹ t_x
    P.t .= rx
    mul!(P.t, MadCore.jac_op(kkt, 'T'), wy, -one(T), one(T))
    MadCore.solve_linear_system!(kkt.linear_solver, P.t)
    wx .= P.t
    # |G|⁻¹ on (r_s, r_y): equality rows D r_y; inequality rows the 2x2 [a -1; -1 c] (a = Σs, c = Σd),
    # |G|⁻¹ = Q |Λ|⁻¹ Qᵀ. With d = (a - c) / 2, h = sqrt(d² + 1): λ+ = (a + c) / 2 + h, λ- = (a c - 1) / λ+
    # (no cancellation when a is large and c near zero), and q± ∝ (-1, λ± - a) with λ+ - a = 1 / (h + d),
    # λ- - a = -(d + h).
    wy .= D .* ry
    a = Σs; c = view(Σd, kkt.ind_ineq); ryi = view(ry, kkt.ind_ineq)
    d = (a .- c) ./ 2
    h = sqrt.(d .^ 2 .+ one(T))
    lp = (a .+ c) ./ 2 .+ h
    lm = (a .* c .- one(T)) ./ lp
    lpa = one(T) ./ (h .+ d)
    lma = .-(d .+ h)
    cp = (.-rs .+ lpa .* ryi) ./ (one(T) .+ lpa .^ 2) ./ abs.(lp)
    cm = (.-rs .+ lma .* ryi) ./ (one(T) .+ lma .^ 2) ./ abs.(lm)
    ws .= .-cp .- cm
    view(wy, kkt.ind_ineq) .= cp .* lpa .+ cm .* lma
    # U⁻ᵀ: (s, y) -= G⁻¹ (0, J u_x)
    mul!(P.g, MadCore.jac_op(kkt, 'N'), wx)
    view(wy, kkt.ind_eq) .+= view(D, kkt.ind_eq) .* view(P.g, kkt.ind_eq)
    gi = view(P.g, kkt.ind_ineq)
    det = a .* c .- one(T)
    ws .-= gi ./ det
    view(wy, kkt.ind_ineq) .-= a .* gi ./ det
    # U_b⁻ᵀ and |G_b|⁻¹: (r_b + z^(1/2) w_p) / |G_b|
    wp = view(w, 1:np)
    view(w, (N3 + 1):(N3 + nlb)) .= (rl .+ A.sl .* view(wp, kkt.ind_lb)) ./ abs.(kkt.l_diag)
    view(w, (N3 + nlb + 1):(N3 + nlb + nub)) .= (ru .+ A.su .* view(wp, kkt.ind_ub)) ./ abs.(kkt.u_diag)
    return w
end

struct MinresState{T, KKT <: AbstractKKTSystem{T}, Op, Prec, W, VT} <: AbstractIterator{T}
    kkt::KKT
    op::Op
    prec::Prec
    workspace::W
    rhs::VT    # the residual of the current solution (symmetrized)
    r::VT      # work: unsymmetrized residual
    s::VT      # work: unsymmetrized solution
    xbest::VT  # the last Richardson step, then the best MINRES iterate of a cycle
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
    nlb = length(kkt.l_diag); nub = length(kkt.u_diag)
    VT = typeof(kkt.pr_diag)
    N3 = n + ns + m; N = N3 + nlb + nub
    op = SymmetrizedKKTOperator(kkt, n, ns, m, UnreducedKKTVector(kkt), UnreducedKKTVector(kkt), VT(undef, nlb), VT(undef, nub), zero(T))
    prec = SymmetrizingPreconditioner(op, VT(undef, N3), VT(undef, n), VT(undef, m))
    workspace = Krylov.MinresWorkspace(N, N, VT)
    return MinresState(kkt, op, prec, workspace, VT(undef, N), VT(undef, N), VT(undef, N), VT(undef, N), opt, cnt, logger)
end
# another KKT system (the fallback of CondensedPreset to ShiftedPreset) gets Richardson refinement
MinresState(kkt::AbstractKKTSystem; opt = default_options(MinresState, sqrt(eps(Float64))), logger = MadNLPLogger(), cnt = MadNLPCounters()) =
    RichardsonIterator(kkt; opt = RichardsonOptions(richardson_tol = opt.minres_tol, richardson_acceptable_tol = opt.minres_acceptable_tol), logger, cnt)

default_options(::Type{MinresState}, tol) =
    MinresOptions(minres_tol = tol^(5 / 4), minres_acceptable_tol = tol^(5 / 8))

# the unsymmetrized vector from the symmetrized one, up to the sign of dzl (irrelevant to the norms
# taken of it): the bound rows times z^(1/2)
function _unsymmetrize!(v, A::SymmetrizedKKTOperator)
    N3 = _n3(A); nlb = length(A.sl)
    view(v, (N3 + 1):(N3 + nlb)) .*= A.sl
    view(v, (N3 + nlb + 1):length(v)) .*= A.su
    return v
end
# the residual ratio of RichardsonIterator, on the unsymmetrized residual r and solution x
function _ratio!(it::MinresState{T}, r, x, norm_b) where {T}
    copyto!(it.r, r); _unsymmetrize!(it.r, it.op)
    copyto!(it.s, x); _unsymmetrize!(it.s, it.op)
    return norm(it.r, Inf) / (min(norm(it.s, Inf), T(1.0e6) * norm_b) + norm_b)
end
# it.rhs = b - A x and its ratio
function _residual!(it::MinresState, x, w, norm_b)
    mul!(it.rhs, it.op, x)
    it.rhs .= w .- it.rhs
    return _ratio!(it, it.rhs, x, norm_b)
end

# Krylov's callback: the residual ratio of x + the cycle's iterate (one operator product per
# iteration), the best iterate kept in it.xbest, the cycle stopped at the tolerance
mutable struct TrueResidualStop{IT, XV, T}
    it::IT
    x::XV
    norm_b::T
    best::T
end
function (c::TrueResidualStop{IT, XV, T})(workspace) where {IT, XV, T}
    it = c.it
    xk = Krylov.solution(workspace)
    mul!(it.r, it.op, xk)
    it.r .= it.rhs .- it.r
    _unsymmetrize!(it.r, it.op)
    it.s .= c.x .+ xk
    _unsymmetrize!(it.s, it.op)
    ratio = norm(it.r, Inf) / (min(norm(it.s, Inf), T(1.0e6) * c.norm_b) + c.norm_b)
    if ratio < c.best
        c.best = ratio
        copyto!(it.xbest, xk)
    end
    return ratio < it.opt.minres_tol
end

function solve_refine!(x::VT, it::MinresState{T}, b::VT, w::VT; max_iter_factor::Integer = 1) where {T, VT <: AbstractKKTVector{T}}
    @debug(it.logger, "Iterative solver initiated")
    kkt = it.kkt; A = it.op
    N3 = _n3(A); nlb = length(A.sl); N = length(it.rhs)
    lb = (N3 + 1):(N3 + nlb); ub = (N3 + nlb + 1):N
    norm_b = norm(full(b), Inf)
    fill!(full(x), zero(T))
    it.cnt.ir = 0
    norm_b == zero(T) && return true

    _scalings!(A)
    A.tau = zero(T)                                   # Richardson: the unshifted system
    f(r, z) = iszero(z) ? r : r / z
    xs = full(x)
    ws = full(w)
    # the symmetrized right-hand side, kept in w
    copyto!(ws, full(b))
    view(ws, lb) .= f.(view(ws, lb), A.sl)
    view(ws, ub) .= f.(view(ws, ub), A.su)

    # Richardson: a step is the condensed solve of the unsymmetrized residual, symmetrized
    y = A.yk; fy = full(y)
    ratio = T(Inf)
    steps = 0
    for j in 1:it.opt.minres_richardson_max
        if j == 1
            copyto!(fy, full(b))
        else
            copyto!(fy, it.rhs); _unsymmetrize!(fy, A)
        end
        MadCore.solve_kkt!(kkt, y)
        view(fy, lb) .= .-f.(view(fy, lb), A.sl)   # dzl = -zl^(1/2) v_l
        view(fy, ub) .= f.(view(fy, ub), A.su)     # dzu = zu^(1/2) v_u
        copyto!(it.xbest, fy)                      # the operator product below overwrites fy
        xs .+= it.xbest
        steps += 1
        new = _residual!(it, xs, ws, norm_b)
        if !(new < ratio)
            xs .-= it.xbest
            _residual!(it, xs, ws, norm_b)
            break
        end
        stalled = new > it.opt.minres_stall * ratio
        ratio = new
        (ratio < it.opt.minres_tol || stalled) && break
    end
    it.cnt.ir += steps
    @debug(it.logger, @sprintf("richardson: %d steps, residual ratio %.2e", steps, ratio))

    # MINRES cycles from the residual of the Richardson solution, toward the regularized target
    if !(ratio < it.opt.minres_tol)
        A.tau = T(it.opt.minres_dual_reg_ratio) * T(kkt.eq_shift.current)
        ratio = _residual!(it, xs, ws, norm_b)
    end
    cycle = 0
    while cycle < it.opt.minres_max_cycles && !(ratio < it.opt.minres_tol)
        cycle += 1
        stop = TrueResidualStop(it, xs, norm_b, T(Inf))
        try
            Krylov.minres!(
                it.workspace, A, it.rhs;
                M = it.prec, ldiv = true,
                atol = zero(T), rtol = zero(T), etol = zero(T), conlim = T(Inf),
                itmax = max_iter_factor * it.opt.minres_max_iter, callback = stop,
            )
            it.cnt.ir += Krylov.iteration_count(it.workspace)
        catch e
            # Krylov throws when the P-inner product of a residual comes out non-positive (rounding at
            # machine level, or a bound diagonal at zero): the cycle ends with its best iterate so far
            e isa ErrorException || rethrow()
            @debug(it.logger, "minres cycle: " * e.msg)
        end
        (isfinite(stop.best) && all(isfinite, it.xbest)) || break
        xs .+= it.xbest
        new = _residual!(it, xs, ws, norm_b)
        if !(new < ratio)
            xs .-= it.xbest
            _residual!(it, xs, ws, norm_b)
            @debug(it.logger, "minres cycle undone: the residual did not decrease")
            break
        end
        ratio = new
        @debug(it.logger, @sprintf("minres cycle %d: residual ratio %.2e", cycle, ratio))
    end

    # back to MadNLP's bound multipliers
    view(xs, lb) .*= .-A.sl
    view(xs, ub) .*= A.su
    @debug(it.logger, @sprintf("Iterative solver terminated with %4i refinement steps and residual = %6.2e", it.cnt.ir, ratio))
    return ratio < it.opt.minres_acceptable_tol
end
