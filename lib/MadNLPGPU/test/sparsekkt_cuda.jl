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

# dense Jacobian rows kept out of the condensation as a border, factored by cuDSS's LDLᵀ:
# every row of DenseDummyQP is dense, so with the thresholds lowered all m rows are kept
@testset "MadNLP -- CUDSSSolver: SparseCondensedKKTSystem with a border" begin
    n, m = 50, 10
    border = Dict{Symbol,Any}(:dense_row_abs => 8, :dense_row_rel => 0.1, :dense_row_fill_min => 0.0)
    nlp_ref = MadNLPTests.DenseDummyQP(zeros(Float64, n); m=m)
    results_ref = MadNLP.madnlp(nlp_ref; kkt_system=MadNLP.SparseKKTSystem, print_level=MadNLP.ERROR)

    nlp = MadNLPTests.DenseDummyQP(CUDA.zeros(Float64, n); m=m)
    solver = MadNLPSolver(
        nlp;
        callback=MadNLP.SparseCallback,
        print_level=MadNLP.ERROR,
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        linear_solver=MadNLPGPU.CUDSSSolver,
        kkt_options=border,
    )
    kkt = MadNLP.get_kkt(solver)
    @test length(kkt.ind_D) == m
    @test size(kkt.aug_com, 1) == n + m
    results = MadNLP.solve!(solver)
    @test results.status == MadNLP.SOLVE_SUCCEEDED
    @test results.objective ≈ results_ref.objective atol=1e-6
    @test Array(results.solution) ≈ Array(results_ref.solution) atol=1e-6
end
