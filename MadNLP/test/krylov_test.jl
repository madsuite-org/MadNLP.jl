using Krylov, LinearAlgebra, SparseArrays

@testset "KrylovIterator" begin
    test_madnlp(
        "KrylovIterator",
        () -> MadNLP.Optimizer(iterator = KrylovIterator, print_level = MadNLP.ERROR),
        [],
    )

    # With an inexact factorization (a large dual shift) and one GMRES step per cycle, the
    # solution reaches krylov_tol only through restarts from the true residual.
    MC = MadNLP.MadCore
    solver = MadNLP.MadNLPSolver(
        MadNLPTests.HS15Model();
        callback = MadNLP.SparseCallback,
        kkt_system = MadNLP.SparseKKTSystem,
        linear_solver = MadNLP.CHOLMODSolver,
        cholmod_algorithm = MadNLP.LDL,
        dual_shift = 1.0e-4,
        iterator = KrylovIterator,
        krylov_max_iter = 1,
        krylov_max_cycles = 20,
        max_iter = 3,
        print_level = MadNLP.ERROR,
    )
    MadNLP.solve!(solver)
    MadNLP.factorize_wrapper!(solver)
    b = MC.UnreducedKKTVector(solver.kkt)
    x = MC.UnreducedKKTVector(solver.kkt)
    w = MC.UnreducedKKTVector(solver.kkt)
    MC.full(b) .= 1.0:length(MC.full(b))
    @test MC.solve_refine!(x, solver.iterator, b, w)
    copyto!(MC.full(w), MC.full(b))
    mul!(w, solver.kkt, x, -1.0, 1.0)
    residual_ratio = norm(MC.full(w), Inf) / (norm(MC.full(x), Inf) + norm(MC.full(b), Inf))
    @test solver.iterator.cnt.ir > 1
    @test residual_ratio < solver.iterator.opt.krylov_tol
end

@testset "Scaled and shifted factorization" begin
    test_madnlp(
        "Ruiz scaling, dual shift, CHOLMOD LDL, GMRES",
        () -> MadNLP.Optimizer(
            kkt_system = MadNLP.SparseKKTSystem,
            linear_solver = MadNLP.CHOLMODSolver,
            cholmod_algorithm = MadNLP.LDL,
            cholmod_dbound = 1.0e-10,
            kkt_scaler = MadNLP.RuizScaler(),
            dual_shift = 1.0e-10,
            iterator = KrylovIterator,
            print_level = MadNLP.ERROR,
        ),
        [],
    )
    for scaler in (MadNLP.JacobiScaler(), MadNLP.MaxNormScaler(), MadNLP.RuizScaler())
        test_madnlp(
            "$(nameof(typeof(scaler))) with MUMPS",
            () -> MadNLP.Optimizer(kkt_system = MadNLP.SparseKKTSystem, kkt_scaler = scaler, print_level = MadNLP.ERROR),
            [],
        )
    end

    # Lower triangle of [H J'; J 0], every diagonal entry stored, as a sparse KKT system assembles it.
    # The constraint Jacobian has a repeated row.
    n, m = 4, 3
    H = Diagonal([2.0, 3.0, 1.0, 4.0])
    J = [1.0 1.0 0.0 0.0; 0.0 1.0 1.0 1.0; 1.0 1.0 0.0 0.0]
    I, Jc, V = findnz(sparse(LowerTriangular([Matrix(H) J'; J zeros(m, m)])))
    append!(I, (n + 1):(n + m)); append!(Jc, (n + 1):(n + m)); append!(V, zeros(m))
    A = sparse(I, Jc, V, n + m, n + m)
    K = Symmetric(A, :L)

    # A scaling alone does not change the solution of a nonsingular system.
    A2 = copy(A); A2[n + 3, n + 3] = -1.0
    for scaler in (MadNLP.JacobiScaler(), MadNLP.MaxNormScaler(), MadNLP.RuizScaler())
        opt = MadNLP.default_options(MadNLP.CHOLMODSolver)
        opt.cholmod_algorithm = MadNLP.LDL
        S = MadNLP.create_kkt_linear_solver(MadNLP.CHOLMODSolver, A2, n; opt, kkt_scaler = scaler)
        MadNLP.factorize!(S)
        b = collect(1.0:(n + m))
        @test MadNLP.solve_linear_system!(S, copy(b)) ≈ Symmetric(A2, :L) \ b
    end

    # The dual shift and the pivot bound keep the inertia of the rank-deficient matrix free of zeros.
    opt = MadNLP.default_options(MadNLP.CHOLMODSolver)
    opt.cholmod_algorithm = MadNLP.LDL
    opt.cholmod_dbound = 1.0e-10
    S = MadNLP.create_kkt_linear_solver(MadNLP.CHOLMODSolver, A, n; opt, kkt_scaler = MadNLP.RuizScaler(), dual_shift = 1.0e-10)
    MadNLP.factorize!(S)
    @test MadNLP.inertia(S) == (n, 0, m)
end
