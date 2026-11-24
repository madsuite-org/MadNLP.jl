using Test, rocMadNLP, MadNLPCore, MadCore, AMDGPU

@testset "rocMadNLP" begin
    @testset "Exports and extensions" begin
        @test LapackROCmSolver === MadCore.LapackROCmSolver
        @test SDSSolver === MadCore.SDSSolver
        @test madnlp === MadNLPCore.madnlp
        @test :ROCBackend in names(rocMadNLP)
        @test rocMadNLP.ROCBackend === AMDGPU.ROCBackend
        # rocMadNLP loads AMDGPU, so the rocSOLVER solvers are available.
        @test MadCore.default_options(LapackROCmSolver) isa MadCore.AbstractOptions
        SDSExt = Base.get_extension(MadCore, :MadCoreSDSExt)
        @test MadCore.default_options(SDSSolver) isa SDSExt.SDSSolverOptions
        @test MadCore.input_type(SDSSolver) === :csc
        # the GPU defaults of MadNLPCoreGPUExt (preset and shifted-preset solver) apply to ROCVector models
        GPUExt = Base.get_extension(MadNLPCore, :MadNLPCoreGPUExt)
        @test which(MadNLPCore.default_preset, Tuple{MadNLPCore.AbstractNLPModel{Float64, ROCVector{Float64}}}).module === GPUExt
        @test which(MadNLPCore.shifted_sparse_solver, Tuple{MadNLPCore.AbstractNLPModel{Float64, ROCVector{Float64}}}).module === GPUExt
    end
    if AMDGPU.functional()
        @testset "Option presets for ROCVector models" begin
            nlp = MadCore.SparseWrapperModel(ROCVector, MadNLPCore.HS15Model())
            # the default, ShiftedPreset: the augmented system with SparseDirectSolver's LDLᵀ with diagonal pivoting
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp)
            @test opt.preset === MadNLPCore.ShiftedPreset
            @test opt.linear_solver === SDSSolver
            @test opt.kkt_system === MadNLPCore.SparseKKTSystem
            @test opt.iterator === MadCore.KrylovIterator
            @test opt.barrier isa MadNLPCore.QualityFunctionUpdate{Float64}
            @test opt.barrier.sigma_max == 2.0 && opt.barrier.mu_min ≈ 1.0e-3 * opt.tol
            @test opt.slack_reset
            lopt = MadCore.default_options(nlp, opt.preset, opt.kkt_system, opt.linear_solver)
            @test lopt.sds_factorization === :ldl && lopt.sds_pivoting === :diagonal && lopt.sds_pivot_epsilon == 1.0e-10 == MadNLPCore.preset_perturbation(Float64)
            @test lopt.sds_matching == "algo6" && lopt.sds_deferred_analysis && lopt.sds_ir == 2
            stats = madnlp(nlp; print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
            # CondensedPreset: the condensed system, SparseDirectSolver's Cholesky, the quality-function barrier
            co = MadNLPCore.MadNLPOptions{Float64}(nlp; preset = MadNLPCore.CondensedPreset)
            colopt = MadCore.default_options(nlp, co.preset, co.kkt_system, co.linear_solver)
            @test colopt.sds_factorization === :cholesky && colopt.sds_pivoting === :none && colopt.sds_matching == "none"
            @test co.kkt_system === MadNLPCore.SparseCondensedKKTSystem && co.barrier isa MadNLPCore.QualityFunctionUpdate{Float64}
            stats = madnlp(nlp; preset = MadNLPCore.CondensedPreset, print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
            # AugmentedPreset: LDLᵀ with Bunch-Kaufman pivoting and matching, on the stock pipeline
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp; preset = MadNLPCore.AugmentedPreset)
            lopt = MadCore.default_options(nlp, opt.preset, opt.kkt_system, opt.linear_solver)
            @test lopt.sds_factorization === :ldl && lopt.sds_pivoting === :bk
            @test lopt.sds_matching == "algo6" && lopt.sds_deferred_analysis && lopt.sds_ir == 0
            @test lopt.sds_reanalyze_on_improve
            stats = madnlp(nlp; preset = MadNLPCore.AugmentedPreset, print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
        # on the condensed system, which a Cholesky factorization needs
        @testset "SDSSolver: $(factorization), $(pivoting)" for (factorization, pivoting) in (
                (:cholesky, :none), (:ldl, :none), (:ldl, :bk),
            )
            nlp = MadCore.SparseWrapperModel(ROCVector, MadNLPCore.HS15Model())
            stats = madnlp(
                nlp; preset = MadNLPCore.CondensedPreset, print_level = MadNLPCore.ERROR,
                sds_factorization = factorization, sds_pivoting = pivoting,
            )
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
        @testset "SDSSolver with Ruiz scaling" begin
            nlp = MadCore.SparseWrapperModel(ROCVector, MadNLPCore.HS15Model())
            solver = MadNLPCore.MadNLPSolver(nlp; print_level = MadNLPCore.ERROR, kkt_scaler = MadCore.RuizScaler())
            @test solver.kkt.linear_solver.wrapped isa MadCore.ScaledShiftedSolver
            stats = MadNLPCore.solve!(solver)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
    end
end
