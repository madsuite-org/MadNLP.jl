using Test, rocMadNLP, MadNLPCore, MadCore, MadCoreGPU, AMDGPU

@testset "rocMadNLP" begin
    @testset "Exports and extensions" begin
        @test LapackROCmSolver === MadCoreGPU.LapackROCmSolver
        @test SDSSolver === MadCoreGPU.SDSSolver
        @test madnlp === MadNLPCore.madnlp
        @test :ROCBackend in names(rocMadNLP)
        @test rocMadNLP.ROCBackend === AMDGPU.ROCBackend
        # rocMadNLP loads AMDGPU, so the rocSOLVER solvers are available.
        @test MadCore.default_options(LapackROCmSolver) isa MadCore.AbstractOptions
        SDSExt = Base.get_extension(MadCoreGPU, :MadCoreGPUSDSExt)
        @test MadCore.default_options(SDSSolver) isa SDSExt.SDSSolverOptions
        @test MadCore.input_type(SDSSolver) === :csc
        # the GPU option preset of MadNLPCoreGPUExt, not the generic one, applies to ROCVector models
        preset = which(MadNLPCore.MadNLPOptions{Float64}, Tuple{MadNLPCore.AbstractNLPModel{Float64, ROCVector{Float64}}})
        @test preset.module === Base.get_extension(MadNLPCore, :MadNLPCoreGPUExt)
    end
    if AMDGPU.functional()
        @testset "Option presets for ROCVector models" begin
            nlp = MadNLPCore.SparseWrapperModel(ROCVector, MadNLPCore.HS15Model())
            # the default, ShiftedPreset: the augmented system with SparseDirectSolver's static LDLᵀ
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp)
            @test opt.preset === MadNLPCore.ShiftedPreset
            @test opt.linear_solver === SDSSolver
            @test opt.kkt_system === MadNLPCore.SparseKKTSystem
            @test opt.iterator === MadCore.KrylovIterator
            @test opt.barrier isa MadNLPCore.QualityFunctionUpdate{Float64}
            @test opt.barrier.sigma_max == 2.0 && opt.barrier.mu_min ≈ 1.0e-3 * opt.tol
            @test opt.slack_reset
            lopt = MadCore.default_options(nlp, opt.preset, opt.kkt_system, opt.linear_solver)
            @test lopt.sds_factorization === :ldl && lopt.sds_pivoting === :none && lopt.sds_pivot_epsilon == 1.0e-10 == MadNLPCore.preset_perturbation(Float64)
            stats = madnlp(nlp; print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
            # CondensedPreset: the condensed system with the same SparseDirectSolver options
            co = MadNLPCore.MadNLPOptions{Float64}(nlp; preset = MadNLPCore.CondensedPreset)
            @test co.kkt_system === MadNLPCore.SparseCondensedKKTSystem && co.barrier isa MadNLPCore.MonotoneUpdate
            stats = madnlp(nlp; preset = MadNLPCore.CondensedPreset, print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
        # on the condensed system, which a Cholesky factorization needs
        @testset "SDSSolver: $(factorization), $(pivoting)" for (factorization, pivoting) in (
                (:cholesky, :none), (:ldl, :none), (:ldl, :bk),
            )
            nlp = MadNLPCore.SparseWrapperModel(ROCVector, MadNLPCore.HS15Model())
            stats = madnlp(
                nlp; preset = MadNLPCore.CondensedPreset, print_level = MadNLPCore.ERROR,
                sds_factorization = factorization, sds_pivoting = pivoting,
            )
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
        @testset "SDSSolver with Ruiz scaling" begin
            nlp = MadNLPCore.SparseWrapperModel(ROCVector, MadNLPCore.HS15Model())
            solver = MadNLPCore.MadNLPSolver(nlp; print_level = MadNLPCore.ERROR, kkt_scaler = MadCore.RuizScaler())
            @test solver.kkt.linear_solver.wrapped isa MadCore.ScaledShiftedSolver
            stats = MadNLPCore.solve!(solver)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
    end
end
