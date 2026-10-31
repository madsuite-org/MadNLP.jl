using Test
using LinearAlgebra
using CUDA
using CUDSS
using MadNLP
using cuMadNLP, MadCoreGPU
using MadNLPTests

@testset "GPUSchurComplementKKTSystem" begin

    @testset "Basic convergence — quadratic with coupling" begin
        ns, nv, nd, nc = 3, 1, 1, 1
        θ = [4.0, 6.0, 8.0]

        nlp = build_twostage_qp(
            CUDA.zeros(Float64, ns * nv + nd);
            ns, nv, nd, nc,
            hess_v = fill(2.0, nv, ns), hess_d = fill(2.0, nd),
            g_v = reshape(-2 .* θ, nv, ns), g_d = [-2.0],
            A_v = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
            lcon = zeros(nc, ns), ucon = zeros(nc, ns),
            lvar_v = fill(-100.0, nv, ns), uvar_v = fill(100.0, nv, ns),
            lvar_d = fill(-100.0, nd), uvar_d = fill(100.0, nd),
        )

        result = madnlp(
            nlp;
            callback = MadNLP.SparseCallback,
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCUDASolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "Match CPU reference" begin
        ns, nv, nd, nc = 2, 2, 1, 1
        θ = [1.0 3.0; 2.0 4.0]

        common = (
            ns = ns, nv = nv, nd = nd, nc = nc,
            hess_v = fill(2.0, nv, ns), hess_d = fill(2.0, nd),
            g_v = -2 .* θ, g_d = [0.0],
            A_v = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
            lcon = zeros(nc, ns), ucon = zeros(nc, ns),
            lvar_v = fill(-50.0, nv, ns), uvar_v = fill(50.0, nv, ns),
            lvar_d = fill(-50.0, nd), uvar_d = fill(50.0, nd),
        )

        nlp_cpu = build_twostage_qp(zeros(Float64, ns * nv + nd); common...)
        nlp_gpu = build_twostage_qp(CUDA.zeros(Float64, ns * nv + nd); common...)

        ref = madnlp(
            nlp_cpu;
            callback = MadNLP.SparseCallback,
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCPUSolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        gpu_result = madnlp(
            nlp_gpu;
            callback = MadNLP.SparseCallback,
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCUDASolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )

        @test ref.status == MadNLP.SOLVE_SUCCEEDED
        @test gpu_result.status == MadNLP.SOLVE_SUCCEEDED
        @test isapprox(gpu_result.objective, ref.objective; atol = 1.0e-6)
        @test isapprox(Array(gpu_result.solution), Array(ref.solution); atol = 1.0e-4)
    end

    @testset "Multiple recourse vars and design vars" begin
        ns, nv, nd, nc = 2, 2, 2, 1
        θ = [1.0, 2.0]
        H_v = zeros(nv, ns)
        for k in 1:ns, j in 1:nv
            H_v[j, k] = 2θ[k]
        end

        nlp = build_twostage_qp(
            CUDA.zeros(Float64, ns * nv + nd);
            ns, nv, nd, nc,
            hess_v = H_v, hess_d = fill(2.0, nd),
            g_v = zeros(nv, ns), g_d = zeros(nd),
            A_v = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
            lcon = ones(nc, ns), ucon = ones(nc, ns),
            lvar_v = fill(-50.0, nv, ns), uvar_v = fill(50.0, nv, ns),
            lvar_d = fill(-50.0, nd), uvar_d = fill(50.0, nd),
        )

        result = madnlp(
            nlp;
            callback = MadNLP.SparseCallback,
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCUDASolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "cuDSS IR + multi-RHS invariant tripwire" begin
        # The GPU Schur path re-analyzes the scenario cuDSS handle with a
        # (blk × nd) × ns multi-RHS descriptor, and then does two solves
        # per IPM iteration: a (blk × 1) × ns single-RHS solve (forward
        # elimination) and a (blk × nd) × ns multi-RHS solve. Correctness
        # relies on "analyze with larger RHS, solve with smaller RHS". The
        # config most likely to surface a regression in that invariant is
        # iterative refinement, which in some sparse solvers keeps per-
        # column state sized from the analysis shape. Enabling cuDSS IR on
        # the scenario solver exercises that code path end-to-end; a future
        # cuDSS change that violated the invariant should trip this test.
        ns, nv, nd, nc = 2, 2, 2, 1
        θ = [1.0, 2.0]
        H_v = zeros(nv, ns)
        for k in 1:ns, j in 1:nv
            H_v[j, k] = 2θ[k]
        end

        scenario_opt = MadNLP.default_options(CUDSSSolver)
        scenario_opt.cudss_ir = 2

        nlp = build_twostage_qp(
            CUDA.zeros(Float64, ns * nv + nd);
            ns, nv, nd, nc,
            hess_v = H_v, hess_d = fill(2.0, nd),
            g_v = zeros(nv, ns), g_d = zeros(nd),
            A_v = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
            lcon = ones(nc, ns), ucon = ones(nc, ns),
            lvar_v = fill(-50.0, nv, ns), uvar_v = fill(50.0, nv, ns),
            lvar_d = fill(-50.0, nd), uvar_d = fill(50.0, nd),
        )

        opts = schur_opts(; ns, nv, nd, nc)
        opts[:schur_scenario_opt_linear_solver] = scenario_opt

        result = madnlp(
            nlp;
            callback = MadNLP.SparseCallback,
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCUDASolver,
            kkt_options = opts,
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "Known solution with inactive constraints" begin
        ns, nv, nd, nc = 2, 1, 1, 1
        θ = [3.0, 7.0]

        nlp = build_twostage_qp(
            CUDA.zeros(Float64, ns * nv + nd);
            ns, nv, nd, nc,
            hess_v = fill(2.0, nv, ns), hess_d = fill(2.0, nd),
            g_v = reshape(-2 .* θ, nv, ns), g_d = [-10.0],
            A_v = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
            lcon = fill(-100.0, nc, ns), ucon = fill(100.0, nc, ns),
            lvar_v = fill(-100.0, nv, ns), uvar_v = fill(100.0, nv, ns),
            lvar_d = fill(-100.0, nd), uvar_d = fill(100.0, nd),
        )

        result = madnlp(
            nlp;
            callback = MadNLP.SparseCallback,
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCUDASolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
        sol = Array(result.solution)
        @test isapprox(sol[1], 3.0; atol = 1.0e-3)
        @test isapprox(sol[2], 7.0; atol = 1.0e-3)
        @test isapprox(sol[3], 5.0; atol = 1.0e-3)
    end
end
