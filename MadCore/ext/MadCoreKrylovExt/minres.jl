
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
    minres_max_iter::Int = 10      # per cycle; Krylov's own stopping tests are disabled (its ‖A‖ estimate is
    minres_max_cycles::Int = 10    # inflated by the 1/δ scaling of P⁻¹, so they fire after one iteration)
    minres_tol::Float64
    minres_acceptable_tol::Float64
    minres_conlim::Float64 = Inf   # Krylov's limit on the estimated condition number (1e8 stops early on these systems)
    # the dual regularization τ of the equality rows of the system refined toward: 0 is the unshifted
    # system, δ the shifted one; the factorization stays at δ
    minres_dual_reg::Float64 = 0.0
    # the same as a fraction of the CURRENT equality shift (after a retry the factorization is at 100 δ
    # or more, and a fixed τ far below it leaves a band too wide for MINRES): τ = max(minres_dual_reg,
    # minres_dual_reg_ratio δ_current); 0.1 is about what Richardson's truncation amounts to
    minres_dual_reg_ratio::Float64 = 0.0
    # adaptive dual regularization from MINRES's own estimate of the condition number κ of P⁻¹M (its
    # largest eigenvalue is about 1, so 1/κ is the smallest): when κ exceeds minres_cond_max, the solve
    # is redone toward the target regularized by τ = δ (1 / minres_cond_max - 1 / κ), which lifts the
    # smallest eigenvalue to 1 / minres_cond_max
    minres_adaptive_reg::Bool = true
    minres_cond_max::Float64 = 1e3
    minres_max_restarts::Int = 2
    # the regularization switched to on detection: a value, or (negative) the equality shift δ itself,
    # which makes the target the shifted system (the condensed solve, refined only for rounding)
    minres_adaptive_tau::Float64 = -1.0
    # MINRES-QLP instead of MINRES: the minimum-norm solution where the system is nearly singular
    minres_qlp::Bool = false
    # the symmetrized system in (x, s, y, zl, zu), the bound rows scaled by z^(1/2) (diagonals -(x - xˡ)
    # and -(xᵘ - x)), instead of the reduced (x, s, y) system with Z/X folded into the primal diagonal
    minres_symmetrized::Bool = true
    # GMRES instead of MINRES on the same (symmetrized) system, P⁻¹ as right preconditioner: full
    # orthogonalization, the true residual minimized, tolerant of an inexact preconditioner
    minres_gmres::Bool = false
    minres_gmres_memory::Int = 20
    # the Krylov method on the symmetrized system: :minres, :minres_qlp (minimum-norm solution on a
    # nearly singular system), :minares (minimizes the A-residual), :symmlq (the Galerkin iterate,
    # transferred to the CG point when it exists), :gmres (as minres_gmres), and the positive-definite
    # methods :cg_lanczos (stops on negative curvature), :cr and :car, which the ±1 spectrum of P⁻¹M
    # leaves without a guarantee; minres_qlp and minres_gmres take precedence
    minres_method::Symbol = :minres
    # stop on the residual of the UNSYMMETRIZED system (the bound rows times zl^(1/2), zu^(1/2), the
    # bound multipliers in MadNLP's scaling), checked at every Krylov iteration through Krylov's
    # callback, the cycle keeping its best iterate; false: the symmetrized residual after each cycle
    minres_raw_stop::Bool = false
    # Richardson steps with the condensed solve before the Krylov cycles (1: MINRES starts from the
    # direct solve's step instead of from zero)
    minres_warm_start::Int = 0
    # Richardson first: up to this many steps with the condensed solve, each checked on the
    # residual (unsymmetrized with minres_raw_stop); MINRES runs only when they stall (a step that
    # does not bring the ratio below minres_stall times the previous one) or run out; 0: off
    minres_richardson_max::Int = 0
    minres_stall::Float64 = 0.5
end

_method(opt::MinresOptions) = opt.minres_gmres ? :gmres : opt.minres_qlp ? :minres_qlp : opt.minres_method

function _workspace(method::Symbol, N, VT, opt::MinresOptions)
    method === :minres && return Krylov.MinresWorkspace(N, N, VT)
    method === :minres_qlp && return Krylov.MinresQlpWorkspace(N, N, VT)
    method === :minares && return Krylov.MinaresWorkspace(N, N, VT)
    method === :symmlq && return Krylov.SymmlqWorkspace(N, N, VT)
    method === :cg_lanczos && return Krylov.CgLanczosWorkspace(N, N, VT)
    method === :cr && return Krylov.CrWorkspace(N, N, VT)
    method === :car && return Krylov.CarWorkspace(N, N, VT)
    method === :gmres && return Krylov.GmresWorkspace(N, N, VT; memory = opt.minres_gmres_memory)
    error("minres_method = $method: expected one of :minres, :minres_qlp, :minares, :symmlq, :cg_lanczos, :cr, :car, :gmres")
end

# The reduced KKT operator on a vector [x; s; y]: the full product of the KKT system on a
# vector with zero bound-multiplier blocks (so that the symmetric Hessian product is the one
# every backend implements), with the primal diagonal pr_diag (bounds condensed) in place of reg.
mutable struct ReducedKKTOperator{T, KKT <: MadCore.SparseCondensedKKTSystem{T}, KV}
    kkt::KKT
    n::Int
    ns::Int
    m::Int
    xk::KV
    yk::KV
    tau::T     # dual regularization of the equality rows of the target system
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
    if !iszero(A.tau)
        wy = view(w, (np + 1):N); vy = view(v, (np + 1):N)
        view(wy, kkt.ind_eq) .-= A.tau .* view(vy, kkt.ind_eq)
    end
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

# The symmetrized full system on [x; s; y; v_l; v_u] with dzl = -zl^(1/2) v_l and dzu = zu^(1/2) v_u:
#   p-rows:   A' p + Jᵀ y (+ the slack couplings) + zl^(1/2) v_l (on ind_lb) + zu^(1/2) v_u (on ind_ub)
#   v_l-rows: zl^(1/2) p_lb - (x - xˡ) v_l,   v_u-rows: zu^(1/2) p_ub - (xᵘ - x) v_u,
# A' = H + reg without the Z/X terms, which the elimination of v_l and v_u produces (pr_diag). The
# right-hand side has r_zl / zl^(1/2) and r_zu / zu^(1/2) on the bound rows (MadNLP's bound rows are
# zl dx + (x - xˡ) dzl = r_zl and zu dx - (xᵘ - x) dzu = r_zu).
struct SymmetrizedKKTOperator{T, Op <: ReducedKKTOperator{T}, VT}
    op::Op
    sl::VT     # zl^(1/2)
    su::VT     # zu^(1/2)
    w3::VT
end
Base.size(A::SymmetrizedKKTOperator) = (size(A.op, 1) + length(A.sl) + length(A.su), size(A.op, 1) + length(A.sl) + length(A.su))
Base.size(A::SymmetrizedKKTOperator, i::Integer) = size(A)[i]
Base.eltype(::SymmetrizedKKTOperator{T}) where {T} = T
function _sym_scalings!(A::SymmetrizedKKTOperator)
    kkt = A.op.kkt
    A.sl .= sqrt.(kkt.l_lower)
    A.su .= sqrt.(kkt.u_lower)
    return A
end
function mul!(w::AbstractVector, A::SymmetrizedKKTOperator{T}, v::AbstractVector) where {T}
    kkt = A.op.kkt
    N3 = size(A.op, 1); np = A.op.n + A.op.ns; nlb = length(A.sl); nub = length(A.su)
    v3 = view(v, 1:N3); vl = view(v, (N3 + 1):(N3 + nlb)); vu = view(v, (N3 + nlb + 1):(N3 + nlb + nub))
    mul!(A.w3, A.op, v3)
    copyto!(view(w, 1:N3), A.w3)
    wp = view(w, 1:np); vp = view(v, 1:np)
    wp .-= (view(kkt.pr_diag, 1:np) .- view(kkt.reg, 1:np)) .* vp        # A' instead of A' + Z/X
    view(wp, kkt.ind_lb) .+= A.sl .* vl
    view(wp, kkt.ind_ub) .+= A.su .* vu
    view(w, (N3 + 1):(N3 + nlb)) .= A.sl .* view(vp, kkt.ind_lb) .+ kkt.l_diag .* vl        # l_diag = xˡ - x
    view(w, (N3 + nlb + 1):(N3 + nlb + nub)) .= A.su .* view(vp, kkt.ind_ub) .+ kkt.u_diag .* vu   # u_diag = x - xᵘ
    return w
end

# P on the full system: the bounds eliminated first (1x1 blocks G_b = l_diag, u_diag, negative), then
# the reduced preconditioner on what remains: P = U_b U diag(K, |G|, |G_b|) Uᵀ U_bᵀ.
struct SymmetrizedPreconditioner{T, Prec <: SymmetrizingPreconditioner{T}, Op <: SymmetrizedKKTOperator{T}, VT}
    prec::Prec
    op::Op
    r3::VT
    w3::VT
end
function ldiv!(w::AbstractVector, P::SymmetrizedPreconditioner{T}, r::AbstractVector) where {T}
    kkt = P.prec.kkt; A = P.op
    N3 = size(A.op, 1); np = A.op.n + A.op.ns; nlb = length(A.sl); nub = length(A.su)
    rl = view(r, (N3 + 1):(N3 + nlb)); ru = view(r, (N3 + nlb + 1):(N3 + nlb + nub))
    # U_b⁻¹ r: r_p - B_bᵀ G_b⁻¹ r_b, with B_b = z^(1/2) and G_b = l_diag (= -(x - xˡ)), u_diag
    copyto!(P.r3, view(r, 1:N3))
    rp = view(P.r3, 1:np)
    view(rp, kkt.ind_lb) .-= A.sl .* rl ./ kkt.l_diag
    view(rp, kkt.ind_ub) .-= A.su .* ru ./ kkt.u_diag
    ldiv!(P.w3, P.prec, P.r3)
    copyto!(view(w, 1:N3), P.w3)
    wp = view(w, 1:np)
    # |G_b|⁻¹ r_b - G_b⁻¹ B_b w_p: (r_b + z^(1/2) w_p) / |G_b|
    view(w, (N3 + 1):(N3 + nlb)) .= (rl .+ A.sl .* view(wp, kkt.ind_lb)) ./ abs.(kkt.l_diag)
    view(w, (N3 + nlb + 1):(N3 + nlb + nub)) .= (ru .+ A.su .* view(wp, kkt.ind_ub)) ./ abs.(kkt.u_diag)
    return w
end

struct MinresState{T, KKT <: AbstractKKTSystem{T}, Op, Prec, W, VT} <: AbstractIterator{T}
    kkt::KKT
    op::Op
    prec::Prec
    workspace::W
    rhs::VT
    z::VT      # P⁻¹ r, for the Rayleigh quotient of the residual direction
    q::VT      # M z
    xbest::VT  # the best iterate of a cycle by the unsymmetrized residual (minres_raw_stop)
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
    op = ReducedKKTOperator(kkt, n, ns, m, UnreducedKKTVector(kkt), UnreducedKKTVector(kkt), T(opt.minres_dual_reg))
    prec = SymmetrizingPreconditioner(kkt, n, ns, m, VT(undef, n), VT(undef, m))
    N = n + ns + m
    if opt.minres_symmetrized
        nlb = length(kkt.l_diag); nub = length(kkt.u_diag)
        sop = SymmetrizedKKTOperator(op, VT(undef, nlb), VT(undef, nub), VT(undef, N))
        sprec = SymmetrizedPreconditioner(prec, sop, VT(undef, N), VT(undef, N))
        N5 = N + nlb + nub
        workspace = _workspace(_method(opt), N5, VT, opt)
        return MinresState(kkt, sop, sprec, workspace, VT(undef, N5), VT(undef, N5), VT(undef, N5), VT(undef, N5), opt, cnt, logger)
    end
    workspace = opt.minres_qlp ? Krylov.MinresQlpWorkspace(N, N, VT) : Krylov.MinresWorkspace(N, N, VT)
    return MinresState(kkt, op, prec, workspace, VT(undef, N), VT(undef, N), VT(undef, N), VT(undef, N), opt, cnt, logger)
end
# another KKT system (the fallback of CondensedPreset to ShiftedPreset) gets Richardson refinement
MinresState(kkt::AbstractKKTSystem; opt = default_options(MinresState, sqrt(eps(Float64))), logger = MadNLPLogger(), cnt = MadNLPCounters()) =
    RichardsonIterator(kkt; opt = RichardsonOptions(richardson_tol = opt.minres_tol, richardson_acceptable_tol = opt.minres_acceptable_tol), logger, cnt)

default_options(::Type{MinresState}, tol) =
    MinresOptions(minres_tol = tol^(5 / 4), minres_acceptable_tol = tol^(5 / 8))

# MINRES cycles on the reduced system, each restarted from the true residual of the refined
# solution; the bound multipliers are recovered from the primal step at the end, as the
# condensed solve does, and the acceptance test is on the full system, as in RichardsonIterator.
_tau(op::ReducedKKTOperator) = op.tau
_tau(op::SymmetrizedKKTOperator) = op.op.tau
_target!(op::ReducedKKTOperator, tau) = (op.tau = tau; op)
_target!(op::SymmetrizedKKTOperator, tau) = (op.op.tau = tau; op)

function solve_refine!(x::VT, it::MinresState{T}, b::VT, w::VT; max_iter_factor::Integer = 1) where {T, VT <: AbstractKKTVector{T}}
    return it.opt.minres_symmetrized ? _refine_symmetrized!(x, it, b, w; max_iter_factor) : _refine_reduced!(x, it, b, w; max_iter_factor)
end

# one MINRES cycle on the workspace; Krylov throws when the P-inner product of a residual comes out
# non-positive (rounding on a residual at machine level, or a bound diagonal at zero), which ends the
# refinement with what it has rather than the solve
_no_stop(workspace) = false
function _minres_cycle!(it::MinresState{T}, max_iter_factor; callback = _no_stop) where {T}
    try
        _minres_cycle_unguarded!(it, max_iter_factor; callback)
    catch e
        # Krylov throws an ErrorException when the P-inner product of a residual comes out non-positive;
        # the positive-definite methods (cr, car) reach a square root of a negative curvature
        e isa Union{ErrorException, DomainError} || rethrow()
        @debug(it.logger, "minres cycle: " * sprint(showerror, e))
        return nothing
    end
end
function _minres_cycle_unguarded!(it::MinresState{T}, max_iter_factor; callback = _no_stop) where {T}
    method = _method(it.opt)
    ws, A, b, P = it.workspace, it.op, it.rhs, it.prec
    z = zero(T)
    itmax = max_iter_factor * it.opt.minres_max_iter
    conlim = T(it.opt.minres_conlim)
    # Krylov's own stopping tests are disabled (atol, rtol, etol, Artol at zero): the cycle runs its
    # itmax iterations and the loop outside judges it by the true residual
    if method === :gmres
        Krylov.gmres!(ws, A, b; N = P, ldiv = true, atol = z, rtol = z, itmax, history = true)
    elseif method === :minres_qlp
        Krylov.minres_qlp!(ws, A, b; M = P, ldiv = true, atol = z, rtol = z, itmax, history = true, callback)
    elseif method === :minares
        Krylov.minares!(ws, A, b; M = P, ldiv = true, atol = z, rtol = z, Artol = z, itmax, history = true, callback)
    elseif method === :symmlq
        Krylov.symmlq!(ws, A, b; M = P, ldiv = true, atol = z, rtol = z, etol = z, conlim, itmax, history = true, callback)
    elseif method === :cg_lanczos
        Krylov.cg_lanczos!(ws, A, b; M = P, ldiv = true, check_curvature = true, atol = z, rtol = z, itmax, history = true, callback)
    elseif method === :cr
        Krylov.cr!(ws, A, b; M = P, ldiv = true, atol = z, rtol = z, itmax, history = true, callback)
    elseif method === :car
        Krylov.car!(ws, A, b; M = P, ldiv = true, atol = z, rtol = z, itmax, history = true, callback)
    else
        Krylov.minres!(ws, A, b; M = P, ldiv = true, atol = z, rtol = z, etol = z, conlim, itmax, history = true, callback)
    end
    it.cnt.ir += Krylov.iteration_count(it.workspace)
    st = it.workspace.stats
    @debug(it.logger, @sprintf("minres cycle: %s, %d iterations, residuals %s", st.status, Krylov.iteration_count(it.workspace), join([@sprintf("%.1e", r) for r in st.residuals], " ")))
    return st
end

# the unsymmetrized residual (or step, for its norm) from the symmetrized one: the bound rows times
# zl^(1/2), zu^(1/2)
function _to_raw!(r, A::SymmetrizedKKTOperator, N3)
    nlb = length(A.sl)
    view(r, (N3 + 1):(N3 + nlb)) .*= A.sl
    view(r, (N3 + nlb + 1):length(r)) .*= A.su
    return r
end
function _raw_ratio!(it, N3, rs, xs, norm_b)
    copyto!(it.z, rs); _to_raw!(it.z, it.op, N3)
    copyto!(it.q, xs); _to_raw!(it.q, it.op, N3)
    return norm(it.z, Inf) / (min(norm(it.q, Inf), 1.0e6 * norm_b) + norm_b)
end
# Krylov's callback: the unsymmetrized residual ratio of the accumulated solution plus the cycle's
# current iterate (one product with the operator per iteration); keeps the best iterate and stops
# the cycle once the ratio is below the tolerance
mutable struct RawStop{IT, XV, T}
    it::IT
    x::XV       # the accumulated solution (symmetrized coordinates)
    N3::Int
    norm_b::T
    tol::T
    best::T
end
function (c::RawStop)(workspace)
    it = c.it
    xk = Krylov.solution(workspace)
    mul!(it.z, it.op, xk)
    it.z .= it.rhs .- it.z
    _to_raw!(it.z, it.op, c.N3)
    it.q .= c.x .+ xk
    _to_raw!(it.q, it.op, c.N3)
    ratio = norm(it.z, Inf) / (min(norm(it.q, Inf), 1.0e6 * c.norm_b) + c.norm_b)
    if ratio < c.best
        c.best = ratio
        copyto!(it.xbest, xk)
    end
    return ratio < c.tol
end
# k Richardson steps with the condensed solve on the target system, from zero, accumulated in the
# symmetrized coordinates (it.rhs holds the scaled right-hand side)
function _warm_start!(x, it::MinresState{T}, b, N3, k) where {T}
    A = it.op; y = A.op.yk; fy = full(y)
    nlb = length(A.sl); lb = (N3 + 1):(N3 + nlb); ub = (N3 + nlb + 1):length(fy)
    f(r, z) = iszero(z) ? r : r / z
    for j in 1:k
        if j == 1
            copyto!(fy, full(b))
        else
            mul!(it.z, A, full(x))      # clobbers A.op.xk and A.op.yk
            it.z .= it.rhs .- it.z
            _to_raw!(it.z, A, N3)
            copyto!(fy, it.z)
        end
        MadCore.solve_kkt!(it.kkt, y)
        view(fy, lb) .= .-f.(view(fy, lb), A.sl)   # dzl = -zl^(1/2) v_l
        view(fy, ub) .= f.(view(fy, ub), A.su)     # dzu = zu^(1/2) v_u
        full(x) .+= fy
    end
    it.cnt.ir += k
    return x
end

# Richardson steps with the condensed solve from zero, accumulated in the symmetrized coordinates,
# stopping at the tolerance, on a stall, or on a step that increases the residual (undone); leaves
# the residual of the result in it.rhs and returns its ratio
function _richardson_first!(x, it::MinresState{T}, b, w, N3, kmax, stall, norm_b, raw) where {T}
    A = it.op; y = A.op.yk; fy = full(y)
    nlb = length(A.sl); lb = (N3 + 1):(N3 + nlb); ub = (N3 + nlb + 1):length(fy)
    f(r, z) = iszero(z) ? r : r / z
    ratio = T(Inf)
    steps = 0
    for j in 1:kmax
        if j == 1
            copyto!(fy, full(b))
        else
            copyto!(fy, it.rhs); _to_raw!(fy, A, N3)
        end
        MadCore.solve_kkt!(it.kkt, y)
        view(fy, lb) .= .-f.(view(fy, lb), A.sl)   # dzl = -zl^(1/2) v_l
        view(fy, ub) .= f.(view(fy, ub), A.su)     # dzu = zu^(1/2) v_u
        copyto!(it.xbest, fy)                      # the step (the product below clobbers fy)
        full(x) .+= it.xbest
        steps += 1
        mul!(it.rhs, A, full(x))
        it.rhs .= full(w) .- it.rhs
        new = raw ? _raw_ratio!(it, N3, it.rhs, full(x), norm_b) :
            norm(it.rhs, Inf) / (min(norm(full(x), Inf), T(1.0e6) * norm_b) + norm_b)
        if !(new < ratio)
            full(x) .-= it.xbest
            mul!(it.rhs, A, full(x))
            it.rhs .= full(w) .- it.rhs
            break
        end
        stalled = new > stall * ratio
        ratio = new
        (ratio < it.opt.minres_tol || stalled) && break
    end
    it.cnt.ir += steps
    @debug(it.logger, @sprintf("richardson first: %d steps, residual ratio %.2e", steps, ratio))
    return ratio
end

# MINRES on the symmetrized full system: x and the right-hand side scaled on the bound rows
# (r_zl / zl^(1/2), r_zu / zu^(1/2); dzl = -zl^(1/2) v_l, dzu = zu^(1/2) v_u), no reduction and
# no recovery of the bound multipliers afterwards.
function _refine_symmetrized!(x::VT, it::MinresState{T}, b::VT, w::VT; max_iter_factor::Integer = 1) where {T, VT <: AbstractKKTVector{T}}
    @debug(it.logger, "Iterative solver initiated (symmetrized)")
    kkt = it.kkt
    N5 = length(it.rhs)
    A = it.op
    nlb = length(A.sl); nub = length(A.su); N3 = N5 - nlb - nub
    norm_b = norm(full(b), Inf)
    fill!(full(x), zero(T))
    it.cnt.ir = 0
    norm_b == zero(T) && return true

    _sym_scalings!(A)
    f(r, z) = iszero(z) ? r : r / z
    # the scaled right-hand side, kept in w for the restarts
    copyto!(full(w), full(b))
    view(full(w), (N3 + 1):(N3 + nlb)) .= f.(view(full(w), (N3 + 1):(N3 + nlb)), A.sl)
    view(full(w), (N3 + nlb + 1):N5) .= f.(view(full(w), (N3 + nlb + 1):N5), A.su)
    copyto!(it.rhs, full(w))
    _target!(A, max(T(it.opt.minres_dual_reg), T(it.opt.minres_dual_reg_ratio) * T(kkt.eq_shift.current)))
    restarts = 0
    residual_ratio = T(Inf)
    raw = it.opt.minres_raw_stop
    if it.opt.minres_richardson_max > 0
        residual_ratio = _richardson_first!(x, it, b, w, N3, it.opt.minres_richardson_max, T(it.opt.minres_stall), norm_b, raw)
    elseif it.opt.minres_warm_start > 0
        _warm_start!(x, it, b, N3, it.opt.minres_warm_start)
        mul!(it.rhs, A, full(x))
        it.rhs .= full(w) .- it.rhs
        residual_ratio = raw ? _raw_ratio!(it, N3, it.rhs, full(x), norm_b) :
            norm(it.rhs, Inf) / (min(norm(full(x), Inf), 1.0e6 * norm_b) + norm_b)
        @debug(it.logger, @sprintf("minres warm start: %d Richardson steps, residual ratio %.2e", it.opt.minres_warm_start, residual_ratio))
    end
    cycle = 0
    while cycle < it.opt.minres_max_cycles && !(residual_ratio < it.opt.minres_tol)
        cycle += 1
        stop = raw ? RawStop(it, full(x), N3, norm_b, T(it.opt.minres_tol), T(Inf)) : nothing
        st = raw ? _minres_cycle!(it, max_iter_factor; callback = stop) : _minres_cycle!(it, max_iter_factor)
        st === nothing && (stop === nothing || !isfinite(stop.best)) && break
        acond = st !== nothing && hasproperty(st, :Acond) && !isempty(st.Acond) ? T(last(st.Acond)) : zero(T)
        sol = raw && isfinite(stop.best) ? it.xbest : Krylov.solution(it.workspace)
        if !all(isfinite, sol)
            @debug(it.logger, "minres cycle: non-finite solution, discarded")
            break
        end
        full(x) .+= sol
        previous_ratio = residual_ratio
        mul!(it.rhs, A, full(x))
        it.rhs .= full(w) .- it.rhs
        residual_ratio = raw ? _raw_ratio!(it, N3, it.rhs, full(x), norm_b) :
            norm(it.rhs, Inf) / (min(norm(full(x), Inf), 1.0e6 * norm_b) + norm_b)
        lambda = T(NaN)
        if residual_ratio >= it.opt.minres_tol && it.opt.minres_adaptive_reg && restarts < it.opt.minres_max_restarts
            ldiv!(it.z, it.prec, it.rhs)
            mul!(it.q, A, it.z)
            lambda = abs(dot(it.z, it.q)) / dot(it.z, it.rhs)
            if isfinite(lambda) && lambda < one(T) / T(it.opt.minres_cond_max)
                tau = it.opt.minres_adaptive_tau < 0 ? T(kkt.eq_shift.current) : T(it.opt.minres_adaptive_tau)
                if tau > _tau(A)
                    _target!(A, tau)
                    restarts += 1
                    @debug(it.logger, @sprintf("minres: smallest eigenvalue about %.1e (κ %.1e), restarting toward the target with dual regularization %.1e", lambda, acond, tau))
                    fill!(full(x), zero(T))
                    copyto!(it.rhs, full(w))
                    residual_ratio = T(Inf)
                    cycle = 0
                    continue
                end
            end
        end
        @debug(it.logger, @sprintf("minres cycle residual ratio %.2e (|b| %.2e, |x| %.2e, κ %.1e, λmin %.1e, τ %.1e)", residual_ratio, norm_b, norm(full(x), Inf), acond, lambda, _tau(A)))
        if !(residual_ratio < previous_ratio)
            full(x) .-= sol
            residual_ratio = previous_ratio
            @debug(it.logger, "minres cycle undone: the true residual did not decrease")
            break
        end
        residual_ratio < it.opt.minres_tol && break
    end
    # back to MadNLP's bound multipliers
    view(full(x), (N3 + 1):(N3 + nlb)) .*= .-A.sl
    view(full(x), (N3 + nlb + 1):N5) .*= A.su
    @debug(
        it.logger,
        @sprintf(
            "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
            it.cnt.ir, residual_ratio,
        ),
    )
    return residual_ratio < it.opt.minres_acceptable_tol
end

function _refine_reduced!(x::VT, it::MinresState{T}, b::VT, w::VT; max_iter_factor::Integer = 1) where {T, VT <: AbstractKKTVector{T}}
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
    _target!(it.op, max(T(it.opt.minres_dual_reg), T(it.opt.minres_dual_reg_ratio) * T(kkt.eq_shift.current)))
    restarts = 0
    residual_ratio = T(Inf)
    cycle = 0
    while cycle < it.opt.minres_max_cycles
        cycle += 1
        if it.opt.minres_qlp
            Krylov.minres_qlp!(
                it.workspace, it.op, it.rhs;
                M = it.prec, ldiv = true,
                atol = zero(T), rtol = zero(T),
                itmax = max_iter_factor * it.opt.minres_max_iter, history = true,
            )
        else
            Krylov.minres!(
                it.workspace, it.op, it.rhs;
                M = it.prec, ldiv = true,
                atol = zero(T), rtol = zero(T), etol = zero(T), conlim = T(it.opt.minres_conlim),
                itmax = max_iter_factor * it.opt.minres_max_iter, history = true,
            )
        end
        it.cnt.ir += Krylov.iteration_count(it.workspace)
        st = it.workspace.stats
        @debug(it.logger, @sprintf("minres cycle: %s, %d iterations, residuals %s", st.status, Krylov.iteration_count(it.workspace), join([@sprintf("%.1e", r) for r in st.residuals], " ")))
        acond = hasproperty(st, :Acond) && !isempty(st.Acond) ? T(last(st.Acond)) : zero(T)
        sol = Krylov.solution(it.workspace)
        if !all(isfinite, sol)
            @debug(it.logger, "minres cycle: non-finite solution, discarded")
            break
        end
        view(full(x), 1:N) .+= sol
        previous_ratio = residual_ratio
        # the residual of the reduced system, for the next cycle
        mul!(it.rhs, it.op, view(full(x), 1:N))
        copyto!(full(w), full(b)); MadCore.reduce_rhs!(kkt, w)
        it.rhs .= view(full(w), 1:N) .- it.rhs
        residual_ratio = norm(it.rhs, Inf) / (min(norm(view(full(x), 1:N), Inf), 1.0e6 * norm_b) + norm_b)
        # the residual left by a cycle lies in the eigendirections of P⁻¹M of smallest magnitude; its
        # Rayleigh quotient in the P-inner product, (zᵀ M z) / (zᵀ r) with z = P⁻¹ r, estimates that
        # smallest eigenvalue (the largest is about 1). Below 1 / minres_cond_max the target is nearly
        # singular there: it is regularized by τ = δ (1 / minres_cond_max - λ) and the solve starts over.
        lambda = T(NaN)
        if residual_ratio >= it.opt.minres_tol && it.opt.minres_adaptive_reg && restarts < it.opt.minres_max_restarts
            ldiv!(it.z, it.prec, it.rhs)
            mul!(it.q, it.op, it.z)
            lambda = abs(dot(it.z, it.q)) / dot(it.z, it.rhs)
            if isfinite(lambda) && lambda < one(T) / T(it.opt.minres_cond_max)
                tau = it.opt.minres_adaptive_tau < 0 ? T(kkt.eq_shift.current) : T(it.opt.minres_adaptive_tau)
                if tau > _tau(it.op)
                    _target!(it.op, tau)
                    restarts += 1
                    @debug(it.logger, @sprintf("minres: smallest eigenvalue about %.1e (κ %.1e), restarting toward the target with dual regularization %.1e", lambda, acond, tau))
                    fill!(view(full(x), 1:N), zero(T))
                    copyto!(full(w), full(b)); MadCore.reduce_rhs!(kkt, w)
                    copyto!(it.rhs, view(full(w), 1:N))
                    residual_ratio = T(Inf)
                    cycle = 0
                    continue
                end
            end
        end
        @debug(it.logger, @sprintf("minres cycle residual ratio %.2e (|b| %.2e, |x| %.2e, κ %.1e, λmin %.1e, τ %.1e)", residual_ratio, norm_b, norm(view(full(x), 1:N), Inf), acond, lambda, _tau(it.op)))
        if !(residual_ratio < previous_ratio)
            # a cycle that did not reduce the true residual is undone (its iterates can be large where
            # the system is nearly singular) and the refinement stops with what it had
            view(full(x), 1:N) .-= sol
            residual_ratio = previous_ratio
            @debug(it.logger, "minres cycle undone: the true residual did not decrease")
            break
        end
        residual_ratio < it.opt.minres_tol && break
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
