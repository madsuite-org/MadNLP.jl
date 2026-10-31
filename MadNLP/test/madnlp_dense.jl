using Test
using NLPModels
using LinearAlgebra
using MadNLPTests
using SparseArrays
using Random

function _compare_dense_with_sparse(
    kkt_system, n, m, ind_fixed, ind_eq;
    inertia=MadNLP.InertiaBased,
)

    for (T,tol,atol) in [(Float32,Float32(1e-4),Float32(1e1)), (Float64,1e-8,1e-6)]

        sparse_options = Dict{Symbol, Any}(
            :kkt_system=>MadNLP.SparseKKTSystem,
            :callback=>MadNLP.SparseCallback,
            :inertia_correction_method=>inertia,
            :linear_solver=>MadNLP.LapackCPUSolver,
            :print_level=>MadNLP.ERROR,
            :dual_initialized=>true,
            :tol=>tol
        )
        dense_options = Dict{Symbol, Any}(
            :kkt_system=>kkt_system,
            :callback=>MadNLP.DenseCallback,
            :inertia_correction_method=>inertia,
            :linear_solver=>MadNLP.LapackCPUSolver,
            :print_level=>MadNLP.ERROR,
            :dual_initialized=>true,
            :tol=>tol
        )

        # N.B.: depending on the variables we fix, the problem may not satisfy
        # LICQ, implying the multipliers are non-unique at the solution.
        nlp = MadNLPTests.DenseDummyQP(zeros(T,n), m=m, fixed_variables=ind_fixed, equality_cons=ind_eq)

        solver = MadNLPSolver(nlp; sparse_options...)
        solverd = MadNLPSolver(nlp; dense_options...)

        result_sparse = MadNLP.solve!(solver)
        result_dense = MadNLP.solve!(solverd)

        # Check that dense formulation matches exactly sparse formulation
        @test result_dense.status == MadNLP.SOLVE_SUCCEEDED
        @test result_sparse.counters.k == result_dense.counters.k
        @test result_sparse.objective ≈ result_dense.objective atol=atol
        @test result_sparse.solution ≈ result_dense.solution atol=atol
        ind_free = setdiff(1:n, ind_fixed)
        n_free = length(ind_free)
        @test solver.kkt.jac_com[:, 1:n_free] == solverd.kkt.jac[:, ind_free]
    end
end

@testset "MadNLP: API $(kkt)" for kkt in [
    MadNLP.DenseKKTSystem,
    MadNLP.DenseCondensedKKTSystem,
    ]

    n = 10 # number of variables
    @testset "Unconstrained" begin
        dense_options = Dict{Symbol, Any}(
            :kkt_system=>kkt,
            :linear_solver=>MadNLP.LapackCPUSolver,
        )
        m = 0
        nlp = MadNLPTests.DenseDummyQP(zeros(n); m=m)
        solverd = MadNLPSolver(nlp; dense_options...)

        kkt = solverd.kkt
        @test isempty(kkt.jac)
        @test solverd.kkt.linear_solver.A === kkt.aug_com
        @test size(kkt.hess) == (n, n)
        @test length(kkt.pr_diag) == n
        @test length(kkt.du_diag) == m

        # Test that using a sparse solver is forbidden in dense mode
        dense_options_error = Dict{Symbol, Any}(
            :kkt_system=>kkt,
            :linear_solver=>MadNLP.UmfpackSolver,
        )
        @test_throws Exception MadNLPSolver(nlp; dense_options_error...)
    end
    @testset "Constrained" begin
        dense_options = Dict{Symbol, Any}(
            :kkt_system=>MadNLP.DenseKKTSystem,
            :linear_solver=>MadNLP.LapackCPUSolver,
        )
        m = 5
        nlp = MadNLPTests.DenseDummyQP(zeros(n); m=m)
        solverd = MadNLPSolver(nlp; dense_options...)
        ns = length(solverd.ind_ineq)

        kkt = solverd.kkt
        @test isa(kkt, MadNLP.DenseKKTSystem)
        @test size(kkt.jac) == (m, n)
        @test solverd.kkt.linear_solver.A === kkt.aug_com
        @test size(kkt.hess) == (n, n)
        @test length(kkt.pr_diag) == n + ns
        @test length(kkt.du_diag) == m
    end
end


@testset "MadNLP: option kkt_system=$(kkt)" for kkt in [MadNLP.DenseKKTSystem, MadNLP.DenseCondensedKKTSystem]
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        _compare_dense_with_sparse(kkt, n, m, Int[], Int[])
        _compare_dense_with_sparse(kkt, n, m, Int[], Int[]; inertia=MadNLP.InertiaFree)
    end
    # Test with non-trivial equality constraints.
    @testset "Equality constraints" begin
        n, m = 20, 15
        _compare_dense_with_sparse(kkt, n, m, Int[], Int[1, 8])
        _compare_dense_with_sparse(kkt, n, m, Int[], Int[1, 8]; inertia=MadNLP.InertiaFree)
    end
    # @testset "Fixed variables" begin
    #     n, m = 10, 5
    #     _compare_dense_with_sparse(kkt, n, m, Int[1, 2], Int[])
    #     _compare_dense_with_sparse(kkt, n, m, Int[1, 2], Int[]; inertia=MadNLP.InertiaFree)
    # end
end

# Now we do not support custom KKT constructor
# @testset "MadNLP: custom KKT constructor" begin
#     solver = MadNLPSolver(nlp; kkt_system = MadNLP.DenseKKTSystem, linear_solver=LapackCPUSolver)
#     @test isa(solver.kkt, KKT)
# end

@testset "MadNLP: restart (PR #113)" begin
    n, m = 10, 6
    nlp = MadNLPTests.DenseDummyQP(zeros(n); m=m)
    sparse_options = Dict{Symbol, Any}(
        :kkt_system=>MadNLP.SparseKKTSystem,
        :callback=>MadNLP.SparseCallback,
        :print_level=>MadNLP.ERROR,
    )

    solver = MadNLPSolver(nlp; sparse_options...)
    MadNLP.solve!(solver)

    # Restart (should hit MadNLP.reinitialize function)
    res = MadNLP.solve!(solver)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
end

