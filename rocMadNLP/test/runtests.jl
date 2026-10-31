using Test, rocMadNLP, MadNLP, MadCore, MadCoreGPU, AMDGPU

@testset "rocMadNLP" begin
    @testset "Exports and extensions" begin
        @test LapackROCmSolver === MadCoreGPU.LapackROCmSolver
        @test madnlp === MadNLP.madnlp
        # rocMadNLP loads AMDGPU, so the rocSOLVER solvers are available.
        @test MadCore.default_options(LapackROCmSolver) isa MadCore.AbstractOptions
        @test hasmethod(MadNLP.MadNLPOptions{Float64}, Tuple{MadNLP.AbstractNLPModel{Float64, ROCVector{Float64}}})
    end
    if AMDGPU.functional()
        @testset "Option presets for ROCVector models" begin
            nlp = MadNLP.SparseWrapperModel(ROCVector, MadNLP.HS15Model())
            opt = MadNLP.MadNLPOptions{Float64}(nlp)
            @test opt.linear_solver === LapackROCmSolver
            @test opt.kkt_system === MadNLP.SparseCondensedKKTSystem
            stats = madnlp(nlp; print_level = MadNLP.ERROR)
            @test stats.status == MadNLP.SOLVE_SUCCEEDED
        end
    end
end
