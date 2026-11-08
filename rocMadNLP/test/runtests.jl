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
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp)
            @test opt.linear_solver === SDSSolver
            @test opt.kkt_system === MadNLPCore.SparseCondensedKKTSystem
            @test opt.barrier isa MadNLPCore.QualityFunctionUpdate{Float64}
            @test opt.barrier.sigma_max == 2.0 && opt.barrier.mu_min == 1.0e-11
            @test opt.slack_reset
            stats = madnlp(nlp; print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
        @testset "SDSSolver: $(factorization), $(pivoting)" for (factorization, pivoting) in (
                (:cholesky, :none), (:ldl, :none), (:ldl, :bk),
            )
            nlp = MadNLPCore.SparseWrapperModel(ROCVector, MadNLPCore.HS15Model())
            stats = madnlp(
                nlp; print_level = MadNLPCore.ERROR,
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
