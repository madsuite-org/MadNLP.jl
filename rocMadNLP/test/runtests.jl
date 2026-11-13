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
        # the GPU option preset of MadNLPCoreGPUExt, not the generic one, applies to ROCVector models
        preset = which(MadNLPCore.MadNLPOptions{Float64}, Tuple{MadNLPCore.AbstractNLPModel{Float64, ROCVector{Float64}}})
        @test preset.module === Base.get_extension(MadNLPCore, :MadNLPCoreGPUExt)
    end
    # the presets beyond the linear solver and SDSSolver are tested in MadCoreGPU/test/sds_test.jl
    if AMDGPU.functional()
        @testset "Default linear solver of ROCVector models" begin
            nlp = MadNLPCore.SparseWrapperModel(ROCVector, MadNLPCore.HS15Model())
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp)
            @test opt.linear_solver === SDSSolver
            solver = MadNLPCore.MadNLPSolver(nlp; print_level = MadNLPCore.ERROR)
            @test MadNLPCore.solve!(solver).status == MadNLPCore.SOLVE_SUCCEEDED
            @test occursin("SparseDirectSolver.jl (ldl,", MadCore.introduce(solver.kkt.linear_solver))
        end
    end
end
