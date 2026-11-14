using Metal
using MadNLPTests

# Metal has no dense factorizations: LapackCPUSolver factorizes on the host, and the BLAS
# kernels of the dense KKT systems run on the host, in place for shared storage and on copies
# for private storage. Metal has no Float64.

function _compare_metal_with_cpu(KKTSystem, n, m, ind_fixed, alg; storage = Metal.PrivateStorage)
    madnlp_options = Dict{Symbol, Any}(
        :callback => MadNLP.DenseCallback,
        :kkt_system => KKTSystem,
        :linear_solver => MadNLP.LapackCPUSolver,
        :lapack_algorithm => alg,
        :print_level => MadNLP.ERROR,
        :tol => 1e-4,
    )
    nlph = MadNLPTests.DenseDummyQP(zeros(Float32, n); m = m, fixed_variables = ind_fixed)
    nlpd = MadNLPTests.DenseDummyQP(Metal.zeros(Float32, n; storage); m = m, fixed_variables = MtlArray(ind_fixed))
    h_solver = MadNLPSolver(nlph; madnlp_options...)
    results_cpu = MadNLP.solve!(h_solver)
    d_solver = MadNLPSolver(nlpd; madnlp_options...)
    results_gpu = MadNLP.solve!(d_solver)
    @test isa(d_solver.kkt, KKTSystem{Float32})
    @test Metal.storagemode(d_solver.kkt.aug_com) == storage
    @test results_gpu.status == results_cpu.status
    @test results_gpu.objective ≈ results_cpu.objective rtol = 1.0f-3
    @test Array(results_gpu.solution) ≈ results_cpu.solution atol = 1.0f-2
end

@testset "Metal dense -- ($(kkt_system), $(alg))" for kkt_system in [
        MadNLP.DenseKKTSystem,
        MadNLP.DenseCondensedKKTSystem,
    ], alg in [MadNLP.BUNCHKAUFMAN, MadNLP.LU, MadNLP.QR]
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        _compare_metal_with_cpu(kkt_system, n, m, Int[], alg)
    end
    @testset "Fixed variables" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        _compare_metal_with_cpu(kkt_system, n, m, Int[1, 2], alg)
    end
    @testset "Shared storage: ($n, $m)" for (n, m) in [(10, 5), (50, 10)]
        _compare_metal_with_cpu(kkt_system, n, m, Int[1, 2], alg; storage = Metal.SharedStorage)
    end
end

@testset "Metal dense -- default solver" begin
    nlp = MadNLPTests.DenseDummyQP(Metal.zeros(Float32, 20); m = 5)
    solver = MadNLPSolver(nlp; callback = MadNLP.DenseCallback, kkt_system = MadNLP.DenseCondensedKKTSystem, print_level = MadNLP.ERROR)
    @test solver.kkt.linear_solver.wrapped isa MadNLP.LapackCPUSolver
    @test MadNLP.solve!(solver).status == MadNLP.SOLVE_SUCCEEDED
end

@testset "Metal dense -- $QN + $KKT" for QN in [
        MadNLP.BFGS,
        MadNLP.DampedBFGS,
    ], KKT in [
        MadNLP.DenseKKTSystem,
        MadNLP.DenseCondensedKKTSystem,
    ]
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        opts = (; callback = MadNLP.DenseCallback, print_level = MadNLP.ERROR, kkt_system = KKT, tol = 1e-4)
        results_ref = madnlp(MadNLPTests.DenseDummyQP(zeros(Float32, n); m = m); opts..., hessian_approximation = QN)
        solver_qn = MadNLPSolver(MadNLPTests.DenseDummyQP(Metal.zeros(Float32, n); m = m); opts..., hessian_approximation = QN)
        results_qn = MadNLP.solve!(solver_qn)
        @test results_qn.status == results_ref.status
        @test results_qn.objective ≈ results_ref.objective rtol = 1.0f-3
        @test solver_qn.cnt.lag_hess_cnt == 0
    end
end
