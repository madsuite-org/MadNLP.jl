using LinearAlgebra

# Small inline models used by the regression tests below.

# min x1 + 2 x2
# s.t. a (x1 + x2)  = a
#      a (x1 - x2) <= a / 2
#      x >= 0
# With a large `a`, MadNLP scales the constraints.
struct ScaledLPModel{T} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    counters::NLPModels.Counters
    a::T
end
function ScaledLPModel(; a = 1000.0)
    meta = NLPModels.NLPModelMeta(
        2;
        ncon = 2, nnzj = 4, nnzh = 0,
        x0 = [0.3, 0.3], lvar = [0.0, 0.0], uvar = [Inf, Inf],
        lcon = [a, -Inf], ucon = [a, a / 2],
    )
    return ScaledLPModel(meta, NLPModels.Counters(), a)
end
NLPModels.obj(::ScaledLPModel, x::AbstractVector) = x[1] + 2 * x[2]
NLPModels.grad!(::ScaledLPModel, x::AbstractVector, g::AbstractVector) = (g .= (1.0, 2.0); g)
function NLPModels.cons!(nlp::ScaledLPModel, x::AbstractVector, c::AbstractVector)
    c[1] = nlp.a * (x[1] + x[2])
    c[2] = nlp.a * (x[1] - x[2])
    return c
end
function NLPModels.jac_structure!(::ScaledLPModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
    I .= (1, 1, 2, 2)
    J .= (1, 2, 1, 2)
    return I, J
end
function NLPModels.jac_coord!(nlp::ScaledLPModel, x::AbstractVector, J::AbstractVector)
    J .= (nlp.a, nlp.a, nlp.a, -nlp.a)
    return J
end
NLPModels.hess_structure!(::ScaledLPModel, I::AbstractVector{T}, J::AbstractVector{T}) where T = (I, J)
NLPModels.hess_coord!(::ScaledLPModel, x::AbstractVector, y::AbstractVector, H::AbstractVector; obj_weight = 1.0) = H

# min c (x - 1)^2 / 2, unconstrained and without bounds.
# With a large `c`, MadNLP scales the objective.
struct QuadModel{T} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    counters::NLPModels.Counters
    c::T
end
function QuadModel(; T = Float64, c = 1.0, x0 = zeros(T, 1))
    meta = NLPModels.NLPModelMeta(1; ncon = 0, nnzj = 0, nnzh = 1, x0 = x0)
    return QuadModel(meta, NLPModels.Counters(), T(c))
end
NLPModels.obj(nlp::QuadModel, x::AbstractVector) = nlp.c * (x[1] - 1)^2 / 2
NLPModels.grad!(nlp::QuadModel, x::AbstractVector, g::AbstractVector) = (g[1] = nlp.c * (x[1] - 1); g)
NLPModels.cons!(::QuadModel, x::AbstractVector, c::AbstractVector) = c
NLPModels.jac_structure!(::QuadModel, I::AbstractVector{T}, J::AbstractVector{T}) where T = (I, J)
NLPModels.jac_coord!(::QuadModel, x::AbstractVector, J::AbstractVector) = J
NLPModels.hess_structure!(::QuadModel, I::AbstractVector{T}, J::AbstractVector{T}) where T = (I .= 1; J .= 1; (I, J))
function NLPModels.hess_coord!(nlp::QuadModel, x::AbstractVector, y::AbstractVector, H::AbstractVector; obj_weight = one(eltype(x)))
    H[1] = obj_weight * nlp.c
    return H
end

@testset "Unscaled constraints in MadNLPExecutionStats" begin
    nlp = ScaledLPModel()
    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR)
    stats = MadNLP.solve!(solver)
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
    @test all(solver.cb.con_scale .!= 1.0)
    @test stats.constraints ≈ NLPModels.cons(nlp, stats.solution)
    @test stats.constraints[1] ≈ nlp.a
end

@testset "Hessian scaled once with SparseCallback + dense KKT" begin
    nlp = QuadModel(; c = 1000.0, x0 = [3.0])
    hessians = map((
        (MadNLP.SparseKKTSystem, MadNLP.MumpsSolver),
        (MadNLP.DenseKKTSystem, MadNLP.LapackCPUSolver),
        (MadNLP.DenseCondensedKKTSystem, MadNLP.LapackCPUSolver),
    )) do (kkt_system, linear_solver)
        solver = MadNLPSolver(
            nlp;
            callback = MadNLP.SparseCallback,
            kkt_system = kkt_system,
            linear_solver = linear_solver,
            print_level = MadNLP.ERROR,
        )
        MadNLP.initialize!(solver)
        @test solver.cb.obj_scale[] < 1.0
        return MadNLP.get_hessian(solver.kkt)[1] / solver.cb.obj_scale[]
    end
    @test all(h -> h ≈ nlp.c, hessians)
end

@testset "Resolve with Float32" begin
    nlp = MadNLPTests.HS15Model(; T = Float32)
    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR)
    @test MadNLP.solve!(solver).status == MadNLP.SOLVE_SUCCEEDED
    @test MadNLP.solve!(solver).status == MadNLP.SOLVE_SUCCEEDED
end

@testset "get_F accounts for upper bounds" begin
    T = Float64
    empty = T[]
    zero1 = zeros(T, 1)
    # One variable with only an upper bound: (xu - x) * zu = (3 - 1) * 2
    F = MadNLP.get_F(empty, zero1, zero1, zero1, zero1, empty, empty, empty, [3.0], [1.0], [2.0], zero(T))
    @test F ≈ 4.0
end

@testset "Quality function: primal residual" begin
    nlp = MadNLPTests.HS15Model()
    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR)
    MadNLP.initialize!(solver)
    step_aff = copy(solver._w3)
    step_cen = copy(solver._w4)
    fill!(MadNLP.full(step_aff), 0.0)
    fill!(MadNLP.full(step_cen), 0.0)
    # Blocking primal step (alpha_pr < 1), no dual step (alpha_du = 1):
    # only the primal residual contributes to the quality function.
    MadNLP.primal(step_aff)[1] = 1e3
    q(res_primal, res_dual) = MadNLP._evaluate_quality_function(solver, 1.0, step_aff, step_cen, res_primal, res_dual)
    q0 = q(0.0, 0.0)
    @test q(1.0, 0.0) > q0
    @test q(0.0, 1.0) ≈ q0
end

@testset "Golden-section search finds the minimum of the quality function" begin
    nlp = MadNLPTests.HS15Model(; x0 = [1.0, 1.0])
    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR)
    MadNLP.initialize!(solver)
    MadNLP.set_aug_diagonal!(solver.kkt, solver)
    MadNLP.factorize_wrapper!(solver)
    # Build the affine and centering steps as in `get_adaptive_mu`
    step_aff, step_cen = solver._w3, solver._w4
    MadNLP.set_aug_rhs!(solver, solver.kkt, solver.c, 0.0)
    res_primal = norm(MadNLP.dual(solver.p))
    res_dual = norm(MadNLP.primal(solver.p))
    copyto!(MadNLP.full(step_aff), MadNLP.full(solver.p))
    MadNLP.solve_kkt!(solver.kkt, step_aff)
    mu = MadNLP.get_average_complementarity(solver)
    MadNLP.set_centering_aug_rhs!(solver, solver.kkt, mu)
    copyto!(MadNLP.full(step_cen), MadNLP.full(solver.p))
    MadNLP.solve_kkt!(solver.kkt, step_cen)

    q(sigma) = MadNLP._evaluate_quality_function(solver, sigma, step_aff, step_cen, res_primal, res_dual)
    barrier = MadNLP.QualityFunctionUpdate(; max_gs_iter = 100, sigma_tol = 1e-10)
    sigma_lb, sigma_ub = 1e-6, 1.0
    sigma_gs = MadNLP._run_golden_search!(solver, barrier, sigma_lb, sigma_ub, step_aff, step_cen, res_primal, res_dual)
    grid = range(sigma_lb, sigma_ub; length = 10_001)
    q_grid = minimum(q, grid)
    @test q(sigma_gs) <= q_grid + 1e-8 * abs(q_grid)
end

@testset "Monotone barrier update uses the new barrier parameter" begin
    nlp = MadNLPTests.HS15Model()
    # Huge barrier_tol_factor forces several decreases of mu in a single update.
    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR, tau_min = 0.0, barrier_tol_factor = 1e20)
    MadNLP.initialize!(solver)
    mu_init = solver.mu
    sc = 1.0
    MadNLP.update_barrier!(solver.opt.barrier, solver, sc)
    @test solver.mu < mu_init
    @test solver.tau == MadNLP.get_tau(solver.mu, solver.opt.tau_min)
    @test solver.inf_compl_mu == MadNLP.get_inf_compl(solver, sc)
end

@kwdef struct FilterLengthCallback <: MadNLP.AbstractUserCallback
    lengths::Vector{Int} = Int[]
end
function (cb::FilterLengthCallback)(solver::MadNLP.AbstractMadNLPSolver, mode)
    push!(cb.lengths, length(solver.filter))
    return true
end

@testset "Switching condition uses the current constraint violation" begin
    # For an unconstrained problem theta = 0, so the switching condition holds
    # and every step is accepted by the Armijo condition: the filter keeps
    # only its initial entry.
    cb = FilterLengthCallback()
    stats = madnlp(QuadModel(); print_level = MadNLP.ERROR, intermediate_callback = cb)
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
    @test all(==(1), cb.lengths)
end

@testset "DenseCondensedKKTSystem expanded multiplication" begin
    nlp = MadNLPTests.HS15Model()
    cb = MadNLP.create_callback(MadNLP.DenseCallback, nlp)
    kkt = MadNLP.create_kkt_system(MadNLP.DenseCondensedKKTSystem, cb, MadNLP.LapackCPUSolver)
    n, m = size(kkt.jac, 2), size(kkt.jac, 1)
    ns = kkt.n_ineq
    H = [2.0 0.5; 0.5 3.0]
    copyto!(kkt.hess, H)
    copyto!(kkt.jac, [1.0 2.0; 3.0 4.0])
    kkt.pr_diag .= 1:n+ns
    kkt.du_diag .= -0.1
    # Assemble the reduced (K2) KKT matrix explicitly
    A_s = zeros(m, ns)
    for (j, i) in enumerate(kkt.ind_ineq)
        A_s[i, j] = -1.0
    end
    K = [
        H+Diagonal(kkt.pr_diag[1:n]) zeros(n, ns)                   kkt.jac'
        zeros(ns, n)                 Diagonal(kkt.pr_diag[n+1:end]) A_s'
        kkt.jac                      A_s                            Diagonal(kkt.du_diag)
    ]
    x = collect(1.0:n+ns+m)
    y = zeros(n + ns + m)
    MadNLP._mul_expanded!(y, kkt, x)
    @test y ≈ K * x
end
