using Krylov, LinearAlgebra

@testset "SymCGIterator" begin
    test_madnlp(
        "SymCGIterator",
        () -> MadNLP.Optimizer(iterator = SymCGIterator, print_level = MadNLP.ERROR),
        [],
    )
    test_madnlp(
        "Ruiz scaling, dual shift, CHOLMOD LDL, SymCG",
        () -> MadNLP.Optimizer(
            kkt_system = MadNLP.SparseKKTSystem,
            linear_solver = MadNLP.CHOLMODSolver,
            cholmod_algorithm = MadNLP.LDL,
            cholmod_dbound = 1.0e-10,
            kkt_scaler = MadNLP.RuizScaler(),
            dual_shift = 1.0e-10,
            iterator = SymCGIterator,
            print_level = MadNLP.ERROR,
        ),
        [],
    )

    # With an inexact factorization (a large dual shift) and two CG steps per cycle, the solution
    # reaches symcg_tol only through restarts from the true residual. (With one step per cycle,
    # a restarted CG step is a Richardson step with the Galerkin step length, which on this
    # indefinite system contracts too slowly to reach symcg_tol in a few dozen cycles.)
    MC = MadNLP.MadCore
    solver = MadNLP.MadNLPSolver(
        MadNLPTests.HS15Model();
        callback = MadNLP.SparseCallback,
        kkt_system = MadNLP.SparseKKTSystem,
        linear_solver = MadNLP.CHOLMODSolver,
        cholmod_algorithm = MadNLP.LDL,
        dual_shift = 1.0e-4,
        iterator = SymCGIterator,
        symcg_max_iter = 2,
        symcg_max_cycles = 20,
        max_iter = 3,
        print_level = MadNLP.ERROR,
    )
    MadNLP.solve!(solver)
    MadNLP.factorize_wrapper!(solver)
    it = solver.iterator
    b = MC.UnreducedKKTVector(solver.kkt)
    x = MC.UnreducedKKTVector(solver.kkt)
    w = MC.UnreducedKKTVector(solver.kkt)
    MC.full(b) .= 1.0:length(MC.full(b))
    @test MC.solve_refine!(x, it, b, w)
    copyto!(MC.full(w), MC.full(b))
    mul!(w, solver.kkt, x, -1.0, 1.0)
    residual_ratio = norm(MC.full(w), Inf) / (norm(MC.full(x), Inf) + norm(MC.full(b), Inf))
    @test it.cnt.ir > 2
    @test residual_ratio < it.opt.symcg_tol

    # The scaled operator is symmetric, and so is the scaled preconditioner, up to the rounding
    # of the factorization.
    @test all(>(0), solver.kkt.l_lower) && length(solver.kkt.l_lower) > 0
    E = Base.get_extension(MC, :MadCoreKrylovExt)
    E.set_scaling!(it)
    N = length(MC.full(b))
    p, q, Kp, Kq = randn(N), randn(N), zeros(N), zeros(N)
    mul!(Kp, it.op, p); mul!(Kq, it.op, q)
    @test dot(q, Kp) ≈ dot(p, Kq) rtol = 1.0e-12
    ldiv!(Kp, it.prec, p); ldiv!(Kq, it.prec, q)
    @test dot(q, Kp) ≈ dot(p, Kq) rtol = 1.0e-6

    # The system of the least-squares estimate of the constraint multipliers, which
    # `initialize!` sets up with zero bound multipliers: a bound row holds only its diagonal,
    # and is solved directly.
    MC.initialize!(solver.kkt)
    @test all(iszero, solver.kkt.l_lower) && all(iszero, solver.kkt.u_lower)
    MadNLP.factorize_wrapper!(solver)
    @test MC.solve_refine!(x, it, b, w)
    copyto!(MC.full(w), MC.full(b))
    mul!(w, solver.kkt, x, -1.0, 1.0)
    @test all(isfinite, MC.full(x))
    @test norm(MC.full(w), Inf) / (norm(MC.full(x), Inf) + norm(MC.full(b), Inf)) < it.opt.symcg_acceptable_tol
end
