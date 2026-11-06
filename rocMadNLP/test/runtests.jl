using Test, rocMadNLP, MadNLPCore, MadCore, MadCoreGPU, AMDGPU

@testset "rocMadNLP" begin
    @testset "Exports and extensions" begin
        @test LapackROCmSolver === MadCoreGPU.LapackROCmSolver
        @test madnlp === MadNLPCore.madnlp
        # rocMadNLP loads AMDGPU, so the rocSOLVER solvers are available.
        @test MadCore.default_options(LapackROCmSolver) isa MadCore.AbstractOptions
        @test hasmethod(MadNLPCore.MadNLPOptions{Float64}, Tuple{MadNLPCore.AbstractNLPModel{Float64, ROCVector{Float64}}})
    end
    if AMDGPU.functional()
        @testset "Option presets for ROCVector models" begin
            nlp = MadNLPCore.SparseWrapperModel(ROCVector, MadNLPCore.HS15Model())
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp)
            @test opt.linear_solver === LapackROCmSolver
            @test opt.kkt_system === MadNLPCore.SparseCondensedKKTSystem
            stats = madnlp(nlp; print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
    end
end
