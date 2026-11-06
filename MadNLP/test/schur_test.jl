using Test
using LinearAlgebra
using MadNLP
using MadNLPTests

@testset "SchurComplementKKTSystem" begin

    @testset "Basic convergence — quadratic with coupling" begin
        # min sum_k (v_k - θ_k)^2 + (d - 1)^2
        # s.t. v_k + d = 0 for k = 1..ns
        ns, nv, nd, nc = 3, 1, 1, 1
        θ = [4.0, 6.0, 8.0]

        qp = build_twostage_qp(;
            ns, nv, nd, nc,
            hess_v = fill(2.0, nv, ns),
            hess_d = fill(2.0, nd),
            g_v = reshape(-2 .* θ, nv, ns),
            g_d = [-2.0],
            A_v = fill(1.0, nc, nv, ns),
            A_d = fill(1.0, nc, nd, ns),
            lcon = zeros(nc, ns), ucon = zeros(nc, ns),
            lvar_v = fill(-100.0, nv, ns), uvar_v = fill(100.0, nv, ns),
            lvar_d = fill(-100.0, nd), uvar_d = fill(100.0, nd),
        )

        result = madnlp(
            qp;
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCPUSolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
        # Analytic optimum: d* = (1 - sum θ) / (ns + 1), v_k* = -d*
        d_star = (1.0 - sum(θ)) / (ns + 1)
        @test isapprox(result.solution[end], d_star; atol = 1.0e-3)
        @test all(isapprox(result.solution[k], -d_star; atol = 1.0e-3) for k in 1:ns)
    end

    @testset "Match SparseKKT reference" begin
        # min sum_{k,j} (v_{k,j} - θ_{k,j})^2 + d^2
        # s.t. v_{k,1} + v_{k,2} + d = 0
        ns, nv, nd, nc = 2, 2, 1, 1
        θ = [1.0 3.0; 2.0 4.0]   # θ[j, k]

        function mk()
            build_twostage_qp(;
                ns, nv, nd, nc,
                hess_v = fill(2.0, nv, ns),
                hess_d = fill(2.0, nd),
                g_v = -2 .* θ,
                g_d = [0.0],
                A_v = fill(1.0, nc, nv, ns),
                A_d = fill(1.0, nc, nd, ns),
                lcon = zeros(nc, ns), ucon = zeros(nc, ns),
                lvar_v = fill(-50.0, nv, ns), uvar_v = fill(50.0, nv, ns),
                lvar_d = fill(-50.0, nd), uvar_d = fill(50.0, nd),
            )
        end

        ref = madnlp(mk(); linear_solver = LapackCPUSolver, print_level = MadNLP.ERROR)
        schur = madnlp(
            mk();
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCPUSolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )

        @test ref.status == MadNLP.SOLVE_SUCCEEDED
        @test schur.status == MadNLP.SOLVE_SUCCEEDED
        @test isapprox(schur.objective, ref.objective; atol = 1.0e-6)
        @test isapprox(schur.solution, ref.solution; atol = 1.0e-4)
    end

    @testset "Multiple recourse vars and design vars" begin
        # min sum_{k,j} θ_k * v_{k,j}^2 + sum_j d_j^2
        # s.t. v_{k,1} + v_{k,2} + d_1 + d_2 = 1
        ns, nv, nd, nc = 2, 2, 2, 1
        θ = [1.0, 2.0]

        H_v = zeros(nv, ns)
        for k in 1:ns, j in 1:nv
            H_v[j, k] = 2θ[k]
        end

        qp = build_twostage_qp(;
            ns, nv, nd, nc,
            hess_v = H_v, hess_d = fill(2.0, nd),
            g_v = zeros(nv, ns), g_d = zeros(nd),
            A_v = fill(1.0, nc, nv, ns),
            A_d = fill(1.0, nc, nd, ns),
            lcon = ones(nc, ns), ucon = ones(nc, ns),
            lvar_v = fill(-50.0, nv, ns), uvar_v = fill(50.0, nv, ns),
            lvar_d = fill(-50.0, nd), uvar_d = fill(50.0, nd),
        )

        result = madnlp(
            qp;
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCPUSolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
    end

    @testset "Known solution with inactive constraints" begin
        # min sum_k (v_k - θ_k)^2 + (d - 5)^2
        # s.t. -100 <= v_k + d <= 100  (inactive at optimum)
        # Solution: v_k* = θ_k, d* = 5
        ns, nv, nd, nc = 2, 1, 1, 1
        θ = [3.0, 7.0]

        qp = build_twostage_qp(;
            ns, nv, nd, nc,
            hess_v = fill(2.0, nv, ns), hess_d = fill(2.0, nd),
            g_v = reshape(-2 .* θ, nv, ns), g_d = [-10.0],
            A_v = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
            lcon = fill(-100.0, nc, ns), ucon = fill(100.0, nc, ns),
            lvar_v = fill(-100.0, nv, ns), uvar_v = fill(100.0, nv, ns),
            lvar_d = fill(-100.0, nd), uvar_d = fill(100.0, nd),
        )

        result = madnlp(
            qp;
            kkt_system = SchurComplementKKTSystem,
            linear_solver = LapackCPUSolver,
            kkt_options = schur_opts(; ns, nv, nd, nc),
            print_level = MadNLP.ERROR,
        )
        @test result.status == MadNLP.SOLVE_SUCCEEDED
        @test isapprox(result.solution[1], 3.0; atol = 1.0e-3)
        @test isapprox(result.solution[2], 7.0; atol = 1.0e-3)
        @test isapprox(result.solution[3], 5.0; atol = 1.0e-3)
    end

    @testset "Autodetect dims via tags" begin
        # Fake `cb` exposing the tag interface that _resolve_schur_dims expects.
        mkcb(tags) = (; nlp = (; tags))

        # Happy case: ns=2, nv=1, nd=1, nc=1 → var_scen [1, 2, 0], con_scen [1, 2]
        ok = (; ns = 2, var_scenario = [1, 2, 0], con_scenario = [1, 2])
        @test MadNLP._resolve_schur_dims(mkcb(ok), 3, 2, 0, 0, 0, 0) == (2, 1, 1, 1)

        # Out-of-range variable tag
        bad_tag = (; ns = 2, var_scenario = [1, 5, 0], con_scenario = [1, 2])
        @test_throws ErrorException MadNLP._resolve_schur_dims(mkcb(bad_tag), 3, 2, 0, 0, 0, 0)

        # Non-uniform per-scenario variable count: scenario 2 has 2 vars, scenario 1 has 1.
        nu_var = (; ns = 2, var_scenario = [1, 2, 2, 0], con_scenario = [1, 2])
        @test_throws ErrorException MadNLP._resolve_schur_dims(mkcb(nu_var), 4, 2, 0, 0, 0, 0)

        # Non-uniform per-scenario constraint count.
        nu_con = (; ns = 2, var_scenario = [1, 2, 0], con_scenario = [1, 2, 2])
        @test_throws ErrorException MadNLP._resolve_schur_dims(mkcb(nu_con), 3, 3, 0, 0, 0, 0)

        # Design-only constraint (con tag 0) is rejected.
        d_only = (; ns = 2, var_scenario = [1, 2, 0], con_scenario = [1, 2, 0])
        @test_throws ErrorException MadNLP._resolve_schur_dims(mkcb(d_only), 3, 3, 0, 0, 0, 0)
    end

    @testset "Layout validation" begin
        # ns=2, nv=1, nd=1, nc=1: vars [v1, v2, d], cons [c1, c2]
        ns, nv, nd, nc = 2, 1, 1, 1
        n, m = ns*nv + nd, ns*nc

        # Happy case as a baseline: diagonal Hessian, each constraint touches its own
        # scenario var and the design var, all equality.
        hess_I_ok = Int32[1, 2, 3]; hess_J_ok = Int32[1, 2, 3]
        jac_I_ok  = Int32[1, 1, 2, 2]; jac_J_ok = Int32[1, 3, 2, 3]
        ind_eq    = Int32[1, 2]
        ind_ineq  = Int32[]
        @test MadNLP._build_schur_symbolic(
            Float64, n, m, ns, nv, nd, nc,
            hess_I_ok, hess_J_ok, jac_I_ok, jac_J_ok, ind_eq, ind_ineq,
        ) isa NamedTuple

        # Cross-scenario Hessian coupling: entry at (row=2, col=1) couples scenarios 1 and 2.
        hess_I_bad = Int32[1, 2, 3, 2]
        hess_J_bad = Int32[1, 2, 3, 1]
        @test_throws ErrorException MadNLP._build_schur_symbolic(
            Float64, n, m, ns, nv, nd, nc,
            hess_I_bad, hess_J_bad, jac_I_ok, jac_J_ok, ind_eq, ind_ineq,
        )

        # Cross-scenario Jacobian coupling: c_1 reaches v_2 (scenario 2's variable).
        jac_I_bad = Int32[1, 1, 1, 2, 2]
        jac_J_bad = Int32[1, 2, 3, 2, 3]
        @test_throws ErrorException MadNLP._build_schur_symbolic(
            Float64, n, m, ns, nv, nd, nc,
            hess_I_ok, hess_J_ok, jac_I_bad, jac_J_bad, ind_eq, ind_ineq,
        )

        # Non-uniform eq/ineq counts: c_1 is equality, c_2 is inequality.
        @test_throws ErrorException MadNLP._build_schur_symbolic(
            Float64, n, m, ns, nv, nd, nc,
            hess_I_ok, hess_J_ok, jac_I_ok, jac_J_ok, Int32[1], Int32[2],
        )

        # Non-uniform sparsity: scenario 2 has an extra off-diagonal Hessian entry
        # at local (2,1) that scenario 1 doesn't, but the entry wouldn't exist in
        # this 1-var-per-scenario layout. Use nv=2 instead.
        ns2, nv2, nd2, nc2 = 2, 2, 1, 1
        n2, m2 = ns2*nv2 + nd2, ns2*nc2
        # vars: [v1_1, v1_2, v2_1, v2_2, d], cons: [c1, c2]
        # Scenario 1 has Hessian entries at (1,1),(2,2) only.
        # Scenario 2 has Hessian entries at (3,3),(4,4) AND off-diag (4,3).
        hess_I_nu = Int32[1, 2, 3, 4, 4, 5]
        hess_J_nu = Int32[1, 2, 3, 4, 3, 5]
        jac_I_nu  = Int32[1, 1, 1, 2, 2, 2]
        jac_J_nu  = Int32[1, 2, 5, 3, 4, 5]
        @test_throws ErrorException MadNLP._build_schur_symbolic(
            Float64, n2, m2, ns2, nv2, nd2, nc2,
            hess_I_nu, hess_J_nu, jac_I_nu, jac_J_nu, Int32[1, 2], Int32[],
        )
    end
end
