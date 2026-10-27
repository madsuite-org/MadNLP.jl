using CUDA
using MadNLPTests

@testset "MadNLP -- $SOLVER: $QN + $KKT" for QN in [
    MadNLP.CompactLBFGS,
], KKT in [
    MadNLP.SparseCondensedKKTSystem,
], SOLVER in [
    MadNLPGPU.CUDSSSolver,
    # MadNLPGPU.LapackCUDASolver,
]
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        if SOLVER == MadNLPGPU.LapackCUDASolver
            nlp = MadNLPTests.DenseDummyQP(zeros(Float64, n); m=m)
            solver_exact = MadNLPSolver(
                nlp;
                callback=MadNLP.SparseCallback,
                print_level=MadNLP.ERROR,
                kkt_system=KKT,
                linear_solver=SOLVER,
            )
            results_ref = MadNLP.solve!(solver_exact)
        end

        nlp = MadNLPTests.DenseDummyQP(CUDA.zeros(Float64, n); m=m)
        solver_qn = MadNLPSolver(
            nlp;
            callback=MadNLP.SparseCallback,
            print_level=MadNLP.ERROR,
            kkt_system=KKT,
            hessian_approximation=QN,
            linear_solver=SOLVER,
        )
        results_qn = MadNLP.solve!(solver_qn)

        @test results_qn.status == MadNLP.SOLVE_SUCCEEDED
        @test results_qn.objective ≈ results_ref.objective atol=1e-6
        @test Array(results_qn.solution) ≈ Array(results_ref.solution) atol=1e-6
        @test solver_qn.cnt.lag_hess_cnt == 0
    end
end

