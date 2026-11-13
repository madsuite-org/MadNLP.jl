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
        # the presets beyond the linear solver are tested in MadCoreGPU/test/sds_test.jl
        @testset "Option presets for CuVector models: cuDSS" begin
            nlp = MadNLPCore.SparseWrapperModel(CuVector, MadNLPCore.HS15Model())
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp)
            @test opt.linear_solver === CUDSSSolver
            lopt = MadCore.default_options(nlp, opt.preset, opt.kkt_system, opt.linear_solver)
            @test lopt.cudss_algorithm == MadCore.LDL && lopt.cudss_pivot_epsilon == 1.0e-10
            @test lopt.cudss_matching && lopt.cudss_deferred_analysis
            stats = madnlp(nlp; print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
            # AugmentedPreset: the augmented system with cuDSS
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp; preset = MadNLPCore.AugmentedPreset)
            @test opt.linear_solver === CUDSSSolver
            lopt = MadCore.default_options(nlp, opt.preset, opt.kkt_system, opt.linear_solver)
            @test lopt.cudss_pivot_type == 'B' && lopt.cudss_matching && lopt.cudss_reanalyze_on_improve
            @test !lopt.cudss_refactorization
            stats = madnlp(nlp; preset = MadNLPCore.AugmentedPreset, print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
            # CondensedPreset: the condensed system with the same cuDSS options as ShiftedPreset
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp; preset = MadNLPCore.CondensedPreset)
            lopt = MadCore.default_options(nlp, opt.preset, opt.kkt_system, opt.linear_solver)
            @test lopt.cudss_algorithm == MadCore.LDL && lopt.cudss_pivot_epsilon == 1.0e-10 && lopt.cudss_matching
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
