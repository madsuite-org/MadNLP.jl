# Tests of the SparseDirectSolver.jl binding (`SDSSolver`, MadCoreGPUSDSExt) and of the option presets
# for GPU models, generic in the array type `AT`. MadCoreGPU's tests call them with the GPU array of the
# CI leg; oneMadNLP and mtlMadNLP, which have no CI leg on a GPU, include this file by path. The
# factorizations themselves are tested by SparseDirectSolver's own suite.

import MadCore, MadCoreGPU, MadNLPCore

# SparseDirectSolver requires Julia 1.13, MadCoreGPU supports older versions: there these tests are
# skipped. From 1.13 on a missing SparseDirectSolver is an error rather than a silent skip.
const SDS_TESTS = VERSION >= v"1.13"
@static if SDS_TESTS
    import SparseDirectSolver
end

sds_extension() = Base.get_extension(MadCoreGPU, :MadCoreGPUSDSExt)

sds_pipeline(M) = occursin("fused", MadCore.introduce(M)) ? "fused" : "stock"

# The lower triangle of the dense symmetric `K` as the KKT systems pass it, in CSC on `AT`.
function sds_csc(AT, K)
    idx = [(i, j) for j in axes(K, 2) for i in j:size(K, 1) if !iszero(K[i, j])]
    A = MadCore.sparse(Int32.(first.(idx)), Int32.(last.(idx)), [K[i, j] for (i, j) in idx], size(K)...)
    return MadCore._get_sparse_csc(size(A), AT(A.colptr), AT(A.rowval), AT(A.nzval))
end

# An SDSSolver on `sds_csc(AT, K)` with the options `kwargs`.
function sds_create(AT, K, n_primal; kwargs...)
    csc = sds_csc(AT, K)
    opt = MadCore.default_options(MadCoreGPU.SDSSolver)
    for (k, v) in kwargs
        setproperty!(opt, k, v)
    end
    MadCore.set_block_structure!(opt, n_primal, size(K, 1) - n_primal)
    return csc, MadCore.create_linear_solver(MadCoreGPU.SDSSolver, csc; opt)
end

# The pipeline SparseDirectSolver chooses on the backend of `AT` without pivoting.
sds_default_pipeline(AT, T) = sds_pipeline(last(sds_create(AT, T[2 1; 1 3], 2)))

function test_sds_extension()
    return @testset "SDSSolver extension" begin
        @test MadCore.default_options(MadCoreGPU.SDSSolver) isa sds_extension().SDSSolverOptions
        @test MadCore.input_type(MadCoreGPU.SDSSolver) === :csc
    end
end

# The option presets of models in `AT{T}` beyond the linear solver. Needs no SparseDirectSolver.
function test_gpu_presets(AT, T)
    return @testset "Option presets for GPU models, $T" begin
        nlp = MadNLPCore.SparseWrapperModel(AT, MadNLPCore.HS15Model(; T))
        @test MadNLPCore.preset_perturbation(Float64) == 1.0e-10
        opt = MadNLPCore.MadNLPOptions{T}(nlp)
        @test opt.preset === MadNLPCore.ShiftedPreset
        @test opt.kkt_system === MadNLPCore.SparseKKTSystem
        @test opt.kkt_scaler isa MadCore.RuizScaler
        @test opt.dual_shift === MadNLPCore.preset_perturbation(T)
        @test opt.iterator === MadCore.KrylovIterator
        @test opt.barrier isa MadNLPCore.QualityFunctionUpdate{T}
        @test opt.barrier.sigma_max == 2.0 && opt.barrier.mu_min ≈ 1.0e-3 * opt.tol
        @test opt.slack_reset
        # the condensed system keeps its own preset
        opt = MadNLPCore.MadNLPOptions{T}(nlp; kkt_system = MadNLPCore.SparseCondensedKKTSystem)
        @test opt.kkt_scaler isa MadCore.NoScaler && iszero(opt.dual_shift)
        @test opt.barrier isa MadNLPCore.MonotoneUpdate && !opt.slack_reset
        opt = MadNLPCore.MadNLPOptions{T}(nlp; preset = MadNLPCore.AugmentedPreset)
        @test opt.kkt_system === MadNLPCore.SparseKKTSystem
        @test opt.kkt_scaler isa MadCore.NoScaler && iszero(opt.dual_shift)
        @test opt.barrier isa MadNLPCore.MonotoneUpdate && !opt.slack_reset
        opt = MadNLPCore.MadNLPOptions{T}(nlp; preset = MadNLPCore.CondensedPreset)
        @test opt.kkt_system === MadNLPCore.SparseCondensedKKTSystem && opt.barrier isa MadNLPCore.MonotoneUpdate
    end
end

# SDSSolver under the option presets of models in `AT{T}`, on HS15.
function test_sds_presets(AT, T)
    SDSSolver = MadCoreGPU.SDSSolver
    return @testset "SDSSolver under the option presets, $T" begin
        nlp = MadNLPCore.SparseWrapperModel(AT, MadNLPCore.HS15Model(; T))
        default_pipeline = sds_default_pipeline(AT, T)
        for (preset, kkt_system, pivoting, ir, pipeline) in (
                (MadNLPCore.ShiftedPreset, MadNLPCore.SparseKKTSystem, :none, 2, default_pipeline),
                (MadNLPCore.CondensedPreset, MadNLPCore.SparseCondensedKKTSystem, :none, 2, default_pipeline),
                (MadNLPCore.AugmentedPreset, MadNLPCore.SparseKKTSystem, :bk, 0, "stock"),
            )
            lopt = MadCore.default_options(nlp, preset, kkt_system, SDSSolver)
            @test lopt.sds_factorization === :ldl && lopt.sds_pivoting === pivoting && lopt.sds_ir == ir
            @test lopt.sds_matching == "algo6" && lopt.sds_deferred_analysis
            @test lopt.sds_pivot_epsilon == (pivoting === :none ? Float64(MadNLPCore.preset_perturbation(T)) : nothing)
            solver = MadNLPCore.MadNLPSolver(nlp; preset, linear_solver = SDSSolver, print_level = MadNLPCore.ERROR)
            @test occursin("(ldl, $pipeline pipeline)", MadCore.introduce(solver.kkt.linear_solver))
            stats = MadNLPCore.solve!(solver)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
            # HS15 has two local minima; the condensed system reaches the second from the origin
            minima = preset === MadNLPCore.CondensedPreset ? (360.38,) : (306.5, 360.38)
            @test any(m -> isapprox(stats.objective, T(m); rtol = 1.0e-2), minima)
        end
        # Cholesky on the condensed system: the preset's LDLᵀ options are ignored
        stats = MadNLPCore.madnlp(
            nlp; preset = MadNLPCore.CondensedPreset, linear_solver = SDSSolver, print_level = MadNLPCore.ERROR,
            sds_factorization = :cholesky,
        )
        @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
    end
end

# SDSSolver against `solver` (CUDSSSolver on NVIDIA GPUs): the same KKT systems, so the same path.
function test_sds_against(AT, solver)
    return @testset "SDSSolver against $solver, $name, $preset" for (name, model) in (
                ("HS15", () -> MadNLPCore.SparseWrapperModel(AT, MadNLPCore.HS15Model())),
                ("chain 1000", () -> MadNLPCore.PrecompileChainModel(1000, AT{Float64}(undef, 0))),
            ),
            preset in (MadNLPCore.ShiftedPreset, MadNLPCore.CondensedPreset, MadNLPCore.AugmentedPreset)

        ref = MadNLPCore.madnlp(model(); preset, linear_solver = solver, print_level = MadNLPCore.ERROR)
        sds = MadNLPCore.madnlp(model(); preset, linear_solver = MadCoreGPU.SDSSolver, print_level = MadNLPCore.ERROR)
        @test ref.status == MadNLPCore.SOLVE_SUCCEEDED
        @test sds.status == MadNLPCore.SOLVE_SUCCEEDED
        # the iteration count of chain 1000 under ShiftedPreset varies from run to run with either solver
        (name == "chain 1000" && preset === MadNLPCore.ShiftedPreset) || @test abs(sds.iter - ref.iter) <= 2
        @test isapprox(sds.objective, ref.objective; rtol = 1.0e-6)
    end
end

# The linear solver alone, on the lower triangle in CSC (`AT`).
function test_sds_linear_solver(AT, T)
    SDS = sds_extension().SparseDirectSolver
    return @testset "SDSSolver linear solver, $T" begin
        create(K, n_primal; kwargs...) = sds_create(AT, K, n_primal; kwargs...)
        function solves(M, K)
            b = T.(1:size(K, 1))
            # a copy, which `AT = Array` would not make: the solve overwrites its argument
            x = Array(MadCore.solve_linear_system!(M, AT(copy(b))))
            return isapprox(x, K \ b; rtol = sqrt(eps(T)))
        end
        spd = T[1 0.1; 0.1 2]
        # [H Jᵀ; J -D] with H positive definite, J of full rank and D = 0.1 I: inertia (3, 0, 2)
        kkt = T[4 1 0 1 0; 1 3 0 1 1; 0 0 2 0 1; 1 1 0 -0.1 0; 0 1 1 0 -0.1]
        default_pipeline = sds_default_pipeline(AT, T)
        @testset "$factorization, $pivoting" for (factorization, pivoting, K, n_primal, inertia) in (
                (:cholesky, :none, spd, 2, (2, 0, 0)),
                (:ldl, :none, kkt, 3, (3, 0, 2)),
                (:ldl, :bk, kkt, 3, (3, 0, 2)),
            )
            # Bunch-Kaufman pivoting stays on the stock pipeline whatever `sds_fused` says
            csc, M = create(K, n_primal; sds_factorization = factorization, sds_pivoting = pivoting, sds_fused = pivoting === :bk ? true : nothing)
            expected = pivoting === :bk ? "stock" : default_pipeline
            @test MadCore.introduce(M) == "SparseDirectSolver.jl ($factorization, $expected pipeline)"
            MadCore.factorize!(M)
            @test MadCore.is_factorized(M) && MadCore.is_inertia(M)
            @test MadCore.inertia(M) == inertia
            @test solves(M, K)
            # new values in the same pattern
            K2 = copy(K)
            for i in axes(K, 1)
                K2[i, i] += i <= n_primal ? 1 : -1
            end
            copyto!(MadCore.nzval(csc), MadCore.nzval(sds_csc(AT, K2)))
            MadCore.factorize!(M)
            @test MadCore.is_factorized(M)
            @test MadCore.inertia(M) == inertia
            @test solves(M, K2)
        end
        @testset "failed factorization" begin
            _, M = create(kkt, 3; sds_factorization = :cholesky)
            MadCore.factorize!(M)
            @test !MadCore.is_factorized(M)
            @test MadCore.inertia(M) == (3, 1, 1)
        end
        @testset "options reach the solver" begin
            # the options of ShiftedPreset; the analysis waits for the first factorize!
            _, M = create(
                kkt, 3; sds_factorization = :ldl, sds_matching = "algo6", sds_deferred_analysis = true,
                sds_pivot_epsilon = 1.0e-9, sds_ir = 2, sds_reordering = "algo3",
                sds_tuning = SDS.SDSTuning(; nd_nseps = 2),
            )
            @test_throws SDS.FactorizationError SDS.inertia(M.inner)
            MadCore.factorize!(M)
            @test MadCore.inertia(M) == (3, 0, 2)
            @test solves(M, kkt)
            @test SDS.getparam(M.inner, "matching_alg") == "algo6"
            @test SDS.getparam(M.inner, "pivot_epsilon") == 1.0e-9
            @test SDS.getparam(M.inner, "pivot_sign") == Int8[1, 1, 1, -1, -1]
            @test SDS.getparam(M.inner, "ir_n_steps") == 2
            @test SDS.getparam(M.inner, "reordering_alg") == "algo3"
            @test SDS.getparam(M.inner, "nd_nseps") == 2
            _, M = create(kkt, 3; sds_factorization = :ldl, sds_pivoting = :bk, sds_pivot_threshold = 0.3, sds_user_perm = [5, 4, 3, 2, 1])
            @test SDS.getparam(M.inner, "pivot_threshold") == 0.3
            @test SDS.getparam(M.inner, "user_perm") == [5, 4, 3, 2, 1]
            @test sds_pipeline(last(create(kkt, 3; sds_factorization = :ldl, sds_fused = false))) == "stock"
            default_pipeline == "fused" && @test sds_pipeline(last(create(kkt, 3; sds_factorization = :ldl, sds_fused = true))) == "fused"
        end
        @test_throws SDS.InvalidValueError create(spd, 2; sds_pivoting = :bk)
        @test_throws SDS.InvalidValueError create(kkt, 3; sds_factorization = :ldl, sds_ir = -1)
    end
end
