using Test
using LinearAlgebra
using SparseArrays
using CUDA
using CUDSS
using MadNLP
using MadNLPGPU
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
            linear_solver = CUDSSSolver,
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
            linear_solver = MadNLP.MumpsSolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        # The equality constraints (lcon == ucon == 0) are relaxed into slacks, whose
        # barrier weight σ_s ~ 1/μ² blows up near convergence and enters each condensed
        # per-scenario block as σ_s·JᵀJ — a rank-1 term that drives cond → 1e16. cuDSS
        # iterative refinement cannot refine a numerically singular system and raises
        # CUDSS_STATUS_IR_FAILED, so pin the batched scenario solver to ir=0 (best-effort
        # direct solve); MadNLP's outer Richardson refinement handles the residual.
        # (With the CUDSSSolver wrappers, leaving ir>0 here now surfaces as a clean
        # ERROR_IN_STEP_COMPUTATION instead of a crash — see the failure testset below —
        # but this test pins ir=0 to assert SOLVE_SUCCEEDED.)
        gpu_opts = schur_opts(; ns, nv, nd, nc)
        scenario_opt = MadNLP.default_options(CUDSSSolver)
        scenario_opt.cudss_ir = 0
        gpu_opts[:schur_scenario_opt_linear_solver] = scenario_opt
        gpu_result = madnlp(
            nlp_gpu;
            callback = MadNLP.SparseCallback,
            kkt_system = SchurComplementKKTSystem,
            linear_solver = CUDSSSolver,
            kkt_options = gpu_opts,
            print_level = MadNLP.ERROR,
        )

        @test ref.status == MadNLP.SOLVE_SUCCEEDED
        @test gpu_result.status == MadNLP.SOLVE_SUCCEEDED
        @test isapprox(gpu_result.objective, ref.objective; atol = 1.0e-6)
        @test isapprox(Array(gpu_result.solution), Array(ref.solution); atol = 1.0e-4)
    end

    @testset "cuDSS IR failure → ERROR_IN_STEP_COMPUTATION (not a crash)" begin
        # Same near-singular relaxed-equality QP as "Match CPU reference", but left at the
        # default scenario IR (= SCHUR_DEFAULT_CUDSS_IR > 0). cuDSS's iterative refinement
        # raises CUDSS_STATUS_IR_FAILED on the numerically singular per-scenario block near
        # convergence. The CUDSSSolver wrappers translate that raw CUDSSError into a MadNLP
        # Solve/FactorizationException, which the IPM maps to ERROR_IN_STEP_COMPUTATION (-3).
        # `rethrow_error = false` lets us read the status instead of re-raising. This pins the
        # fix: an *untranslated* CUDSSError would fall through to INTERNAL_ERROR (-6) here.
        ns, nv, nd, nc = 2, 2, 1, 1
        θ = [1.0 3.0; 2.0 4.0]

        nlp_gpu = build_twostage_qp(
            CUDA.zeros(Float64, ns * nv + nd);
            ns, nv, nd, nc,
            hess_v = fill(2.0, nv, ns), hess_d = fill(2.0, nd),
            g_v = -2 .* θ, g_d = [0.0],
            A_v = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
            lcon = zeros(nc, ns), ucon = zeros(nc, ns),
            lvar_v = fill(-50.0, nv, ns), uvar_v = fill(50.0, nv, ns),
            lvar_d = fill(-50.0, nd), uvar_d = fill(50.0, nd),
        )

        # Note: default scenario/complement IR (not pinned to 0) — this is what triggers
        # the cuDSS IR failure on the singular block.
        result = madnlp(
            nlp_gpu;
            callback = MadNLP.SparseCallback,
            kkt_system = SchurComplementKKTSystem,
            linear_solver = CUDSSSolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
            rethrow_error = false,
        )
        @test result.status == MadNLP.ERROR_IN_STEP_COMPUTATION
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
            linear_solver = CUDSSSolver,
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
            linear_solver = CUDSSSolver,
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
            linear_solver = CUDSSSolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
        sol = Array(result.solution)
        @test isapprox(sol[1], 3.0; atol = 1.0e-3)
        @test isapprox(sol[2], 7.0; atol = 1.0e-3)
        @test isapprox(sol[3], 5.0; atol = 1.0e-3)
    end

    @testset "Design-only constraints — match CPU reference" begin
        # Non-contiguous layout + design-only equality and inequality constraints.
        for (ns, nv, nd) in ((2, 2, 2), (3, 2, 3))
            n = ns * nv + nd
            qp_cpu, _, _, kkt_opts =
                build_twostage_qp_general(zeros(Float64, n); ns, nv, nd, permute = true)
            qp_gpu, _, _, _ =
                build_twostage_qp_general(CUDA.zeros(Float64, n); ns, nv, nd, permute = true)

            # Pin both cuDSS solvers to ir=0: relaxed-equality slacks drive the condensed
            # systems numerically singular near convergence (σ_s·JᵀJ, σ_s ~ 1/μ²), where
            # cuDSS iterative refinement raises CUDSS_STATUS_IR_FAILED. Here the design-only
            # equality lands in the first-stage Schur complement, so the complement solver
            # is the one that goes singular (the scenario blocks can too). The best-effort
            # direct solve plus MadNLP's outer Richardson refinement suffice.
            scenario_opt = MadNLP.default_options(CUDSSSolver)
            scenario_opt.cudss_ir = 0
            complement_opt = MadNLP.default_options(CUDSSSolver)
            complement_opt.cudss_ir = 0
            kkt_opts[:schur_scenario_opt_linear_solver] = scenario_opt
            kkt_opts[:schur_opt_linear_solver] = complement_opt

            ref = madnlp(qp_cpu; linear_solver = LapackCPUSolver, print_level = MadNLP.ERROR)
            gpu_result = madnlp(
                qp_gpu;
                callback = MadNLP.SparseCallback,
                kkt_system = SchurComplementKKTSystem,
                linear_solver = CUDSSSolver,
                kkt_options = kkt_opts,
                print_level = MadNLP.ERROR,
            )

            @test ref.status == MadNLP.SOLVE_SUCCEEDED
            @test gpu_result.status == MadNLP.SOLVE_SUCCEEDED
            @test isapprox(gpu_result.objective, ref.objective; atol = 1.0e-6)
            @test isapprox(Array(gpu_result.solution), Array(ref.solution); atol = 1.0e-4)
        end
    end

    # Assemble one KKT matrix (jac+hess at x0, pr_diag=1, du_diag=-1e-8) and return it.
    # RelaxEquality (the Schur default): all constraints condense, so a coupling
    # equality becomes a coupling inequality whose J'ΣJ splits across A_kk / C_dk / S_dd.
    function _assembled_kkt(qp, solver, kkt_opts)
        cb = MadNLP.create_callback(
            MadNLP.SparseCallback, qp; equality_treatment = MadNLP.RelaxEquality,
        )
        kkt = MadNLP.create_kkt_system(MadNLP.SchurComplementKKTSystem, cb, solver; kkt_opts...)
        x0 = cb.nlp.meta.x0
        y0 = cb.nlp.meta.y0
        MadNLP._eval_jac_wrapper!(cb, x0, MadNLP.get_jacobian(kkt))
        MadNLP.compress_jacobian!(kkt)
        MadNLP._eval_lag_hess_wrapper!(cb, x0, y0, MadNLP.get_hessian(kkt))
        MadNLP.compress_hessian!(kkt)
        fill!(kkt.pr_diag, 1.0)
        fill!(kkt.du_diag, -1.0e-8)
        MadNLP.build_kkt!(kkt)
        return kkt
    end

    @testset "Sparse Schur (cuDSS) — GPU schur_csc matches CPU schur_csc" begin
        # Both the CPU and GPU first-stage Schur complements are sparse lower-triangular
        # CSCs (same symbolic pattern, size nd × nd, SPD). Densified + symmetrized they
        # must agree. Directly catches triangle / sign / nzpos / double-count bugs in the
        # sparse assembly, including the relaxed coupling-equality → C_dk cross-term.
        for (ns, nv, nd) in ((2, 2, 2), (3, 2, 3), (4, 1, 2))
            n = ns * nv + nd
            qp_cpu, _, _, kkt_opts =
                build_twostage_qp_general(zeros(Float64, n); ns, nv, nd, permute = true)
            qp_gpu, _, _, _ =
                build_twostage_qp_general(CUDA.zeros(Float64, n); ns, nv, nd, permute = true)

            kkt_cpu = _assembled_kkt(qp_cpu, MadNLP.MumpsSolver, kkt_opts)
            kkt_gpu = _assembled_kkt(qp_gpu, CUDSSSolver, kkt_opts)

            A_dense = Array(Symmetric(Matrix(kkt_cpu.schur_csc), :L))
            S = kkt_gpu.schur_csc
            nd_aug = size(S, 1)
            S_lower = SparseMatrixCSC(
                nd_aug, nd_aug, Array(S.colPtr), Array(S.rowVal), Array(S.nzVal),
            )
            S_full = Array(Symmetric(Matrix(S_lower), :L))

            @test kkt_gpu.m <= nd
            @test isapprox(S_full, A_dense; atol = 1.0e-7, rtol = 1.0e-7)
        end
    end

    @testset "Sparse Schur (cuDSS) — reduced coupling buffers sized to m ≤ nd" begin
        # The reduced C_dk / tmp / Schur-block buffers are width m (the coupled-design
        # count), not nd — the memory/compute win. m ≤ nd always; for SCOPF m ≪ nd.
        ns, nv, nd = 3, 2, 3
        n = ns * nv + nd
        qp_gpu, _, _, kkt_opts =
            build_twostage_qp_general(CUDA.zeros(Float64, n); ns, nv, nd, permute = true)
        kkt = _assembled_kkt(qp_gpu, CUDSSSolver, kkt_opts)
        @test 1 <= kkt.m <= nd
        @test size(kkt.C_dk_batched, 2) == kkt.m
        @test size(kkt.tmp_blk_nd_batched, 2) == kkt.m
        @test size(kkt.schur_block_batched, 1) == kkt.m
        @test size(kkt.schur_block_batched, 2) == kkt.m
    end
end
