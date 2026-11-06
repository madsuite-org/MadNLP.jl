using LinearAlgebra

@testset "[KKT vector] $KKTVector" for KKTVector in [
    MadNLP.UnreducedKKTVector,
]
    T = Float64
    VT = Vector{Float64}
    n, m = 10, 20
    nlb, nub = 5, 6

    ind_lb = [2,3,4]
    ind_ub = [4,5,6]
    rhs = KKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    @test length(rhs) == length(MadNLP.full(rhs))
    @test MadNLP.number_primal(rhs) == length(MadNLP.primal(rhs)) == n
    @test MadNLP.number_dual(rhs) == length(MadNLP.dual(rhs)) == m
    @test norm(rhs) == 0

    fill!(rhs, one(T))
    @test norm(rhs) == sqrt(length(rhs))

    # Test copy
    copy_rhs = copy(rhs)
    @test MadNLP.full(rhs) == MadNLP.full(copy_rhs)
end

@testset "[KKT system] $(KKTSystem)" for (KKTSystem, Callback) in [
    (MadNLP.SparseKKTSystem, MadNLP.SparseCallback),
    (MadNLP.SparseUnreducedKKTSystem, MadNLP.SparseCallback),
    (MadNLP.SparseCondensedKKTSystem, MadNLP.SparseCallback),
    (MadNLP.ScaledSparseKKTSystem, MadNLP.SparseCallback),
    (MadNLP.DenseKKTSystem, MadNLP.DenseCallback),
    (MadNLP.DenseCondensedKKTSystem, MadNLP.DenseCallback),
]
    linear_solver = MadNLP.LapackCPUSolver

    nlp = MadNLPTests.HS15Model()
    cb = MadNLP.create_callback(
        Callback, nlp,
    )

    kkt = MadNLP.create_kkt_system(
        KKTSystem,
        cb,
        linear_solver;
    )
    MadNLPTests.test_kkt_system(kkt, cb)
end

@testset "[KKT system] $(KKTSystem)+LBFGS" for (KKTSystem, Callback) in [
    (MadNLP.SparseKKTSystem, MadNLP.SparseCallback),
    (MadNLP.SparseUnreducedKKTSystem, MadNLP.SparseCallback),
    (MadNLP.SparseCondensedKKTSystem, MadNLP.SparseCallback),
    (MadNLP.ScaledSparseKKTSystem, MadNLP.SparseCallback),
]
    linear_solver = MadNLP.LapackCPUSolver
    nlp = MadNLPTests.HS15Model()
    cb = MadNLP.create_callback(Callback, nlp)
    # Define options for LBFGS
    p = 20
    qn_options = MadNLP.QuasiNewtonOptions(; max_history=p)

    kkt = MadNLP.create_kkt_system(
        KKTSystem,
        cb,
        linear_solver;
        hessian_approximation=MadNLP.CompactLBFGS,
        qn_options=qn_options,
    )
    @test isa(kkt.quasi_newton, MadNLP.CompactLBFGS)
    # Test options are correctly passed to MadNLP
    @test kkt.quasi_newton.max_mem == p
end

