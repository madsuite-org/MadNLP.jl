using Test, cuMadNLP, MadNLPCore, MadCore, MadCoreGPU, CUDA

@testset "cuMadNLP" begin
    @testset "Exports and extensions" begin
        @test CUDSSSolver === MadCoreGPU.CUDSSSolver
        @test LapackCUDASolver === MadCoreGPU.LapackCUDASolver
        @test madnlp === MadNLPCore.madnlp
        @test :CUDABackend in names(cuMadNLP)
        @test cuMadNLP.CUDABackend === CUDA.CUDABackend
        # cuMadNLP loads CUDA and CUDSS, so the solvers of MadCoreGPU are available.
        @test MadCore.default_options(CUDSSSolver) isa MadCore.AbstractOptions
        @test MadCore.default_options(LapackCUDASolver) isa MadCore.AbstractOptions
        # the GPU option preset of MadNLPCoreGPUExt, not the generic one, applies to CuVector models
        preset = which(MadNLPCore.MadNLPOptions{Float64}, Tuple{MadNLPCore.AbstractNLPModel{Float64, CuVector{Float64}}})
        @test preset.module === Base.get_extension(MadNLPCore, :MadNLPCoreGPUExt)
    end
    if CUDA.functional()
        @testset "Option presets for CuVector models" begin
            nlp = MadNLPCore.SparseWrapperModel(CuVector, MadNLPCore.HS15Model())
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp)
            @test opt.preset === MadNLPCore.ShiftedPreset
            @test opt.linear_solver === CUDSSSolver
            @test opt.kkt_system === MadNLPCore.SparseKKTSystem
            @test opt.kkt_scaler isa MadCore.RuizScaler
            @test opt.dual_shift == 1.0e-10
            @test opt.iterator === MadCore.KrylovIterator
            @test opt.barrier isa MadNLPCore.QualityFunctionUpdate{Float64}
            @test opt.barrier.sigma_max == 2.0 && opt.barrier.mu_min ≈ 1.0e-3 * opt.tol
            @test opt.slack_reset
            lopt = MadCore.default_options(nlp, opt.preset, opt.kkt_system, opt.linear_solver)
            @test lopt.cudss_algorithm == MadCore.LDL && lopt.cudss_pivot_epsilon == 1.0e-10
            @test lopt.cudss_matching && lopt.cudss_deferred_analysis
            stats = madnlp(nlp; print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
            # the shifted options carry over to the condensed system, with the equalities enforced
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp; kkt_system = MadNLPCore.SparseCondensedKKTSystem)
            @test opt.kkt_scaler isa MadCore.RuizScaler && opt.dual_shift == 1.0e-10
            @test opt.barrier isa MadNLPCore.QualityFunctionUpdate{Float64} && opt.slack_reset
            @test opt.equality_treatment === MadCore.EnforceEquality && opt.kkt_options[:condense_max_nnz] == 20
            @test opt.bound_relax_factor == 1.0e-8
            # AugmentedPreset: the augmented system with cuDSS and the monotone rule
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp; preset = MadNLPCore.AugmentedPreset)
            @test opt.kkt_system === MadNLPCore.SparseKKTSystem && opt.linear_solver === CUDSSSolver
            @test opt.kkt_scaler isa MadCore.NoScaler && iszero(opt.dual_shift)
            @test opt.barrier isa MadNLPCore.MonotoneUpdate && !opt.slack_reset
            lopt = MadCore.default_options(nlp, opt.preset, opt.kkt_system, opt.linear_solver)
            @test lopt.cudss_pivot_type == 'B' && lopt.cudss_matching && lopt.cudss_reanalyze_on_improve
            @test !lopt.cudss_refactorization
            stats = madnlp(nlp; preset = MadNLPCore.AugmentedPreset, print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
            # CondensedPreset: the condensed system, cuDSS LDLᵀ without a pivot bound and without matching
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp; preset = MadNLPCore.CondensedPreset)
            @test opt.kkt_system === MadNLPCore.SparseCondensedKKTSystem && opt.barrier isa MadNLPCore.QualityFunctionUpdate
            lopt = MadCore.default_options(nlp, opt.preset, opt.kkt_system, opt.linear_solver)
            @test lopt.cudss_algorithm == MadCore.LDL && lopt.cudss_pivot_epsilon == 0.0 && !lopt.cudss_matching
            stats = madnlp(nlp; preset = MadNLPCore.CondensedPreset, print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
            stats = madnlp(nlp; kkt_system = MadNLPCore.SparseCondensedKKTSystem, print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
        @testset "get_F accounts for upper bounds on the GPU" begin
            # Same case as the CPU test in MadNLP/test/regression_test.jl:
            # one variable with only an upper bound, (xu - x) * zu = (3 - 1) * 2.
            empty = CuVector{Float64}(undef, 0)
            zero1 = CUDA.zeros(Float64, 1)
            F = MadNLPCore.get_F(empty, zero1, zero1, zero1, zero1, empty, empty, empty, CuVector([3.0]), CuVector([1.0]), CuVector([2.0]), 0.0)
            @test F ≈ 4.0
        end
    end
end
