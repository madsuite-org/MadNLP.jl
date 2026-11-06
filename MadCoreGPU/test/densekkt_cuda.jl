using CUDA
using MadNLPTests

function _compare_gpu_with_cpu(KKTSystem, n, m, ind_fixed)
    for (T,tol,atol) in [
        (Float32,1e-4,1e1),
        (Float64,1e-8,1e-6)
        ]
        madnlp_options = Dict{Symbol, Any}(
            :callback=>MadNLP.DenseCallback,
            :kkt_system=>KKTSystem,
            :linear_solver => MadCoreGPU.LapackCUDASolver,
            :print_level=>MadNLP.ERROR,
            :tol=>tol
        )

        # Host evaluator
        nlph = MadNLPTests.DenseDummyQP(zeros(T,n); m=m, fixed_variables=ind_fixed)
        # Device evaluator
        nlpd = MadNLPTests.DenseDummyQP(CUDA.zeros(T,n); m=m, fixed_variables=CuArray(ind_fixed))

        # Solve on CPU
        h_solver = MadNLPSolver(nlph; madnlp_options...)
        results_cpu = MadNLP.solve!(h_solver)

        # Solve on GPU
        d_solver = MadNLPSolver(nlpd; madnlp_options...)
        results_gpu = MadNLP.solve!(d_solver)

        @test isa(d_solver.kkt, KKTSystem{T})
        # Check that both results match exactly
        @test h_solver.cnt.k == d_solver.cnt.k
        @test results_cpu.objective ≈ results_gpu.objective
        @test results_cpu.solution ≈ Array(results_gpu.solution) atol=atol
        @test results_cpu.multipliers ≈ Array(results_gpu.multipliers) atol=atol
    end
end

@testset "LapackCUDASolver -- ($(kkt_system))" for kkt_system in [
        MadNLP.DenseKKTSystem,
        MadNLP.DenseCondensedKKTSystem,
    ]
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        _compare_gpu_with_cpu(kkt_system, n, m, Int[])
    end
    @testset "Fixed variables" for (n,m) in [(10, 0), (10, 5), (50, 10)]
        _compare_gpu_with_cpu(kkt_system, n, m, Int[1, 2])
    end
end

@testset "MadNLP -- LapackCUDASolver: $QN + $KKT" for QN in [
    MadNLP.BFGS,
    MadNLP.DampedBFGS,
], KKT in [
    MadNLP.DenseKKTSystem,
    MadNLP.DenseCondensedKKTSystem,
]
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        nlp = MadNLPTests.DenseDummyQP(zeros(Float64, n); m=m)
        solver_exact = MadNLPSolver(
            nlp;
            callback=MadNLP.DenseCallback,
            print_level=MadNLP.ERROR,
            kkt_system=KKT,
            linear_solver = MadCoreGPU.LapackCUDASolver,
        )
        results_ref = MadNLP.solve!(solver_exact)

        nlp = MadNLPTests.DenseDummyQP(CUDA.zeros(Float64, n); m=m)
        solver_qn = MadNLPSolver(
            nlp;
            callback=MadNLP.DenseCallback,
            print_level=MadNLP.ERROR,
            kkt_system=KKT,
            hessian_approximation=QN,
            linear_solver = MadCoreGPU.LapackCUDASolver,
        )
        results_qn = MadNLP.solve!(solver_qn)

        @test results_qn.status == MadNLP.SOLVE_SUCCEEDED
        @test results_qn.objective ≈ results_ref.objective atol=1e-6
        @test Array(results_qn.solution) ≈ Array(results_ref.solution) atol=1e-6
        @test solver_qn.cnt.lag_hess_cnt == 0
    end
end
