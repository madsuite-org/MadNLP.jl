using Test, MadNLPGPU, MadCoreGPU, cuMadNLP

@testset "MadNLPGPU" begin
    # The deprecated MadNLPGPU keeps the names of MadNLPGPU resolving.
    @test MadNLPGPU.CUDSSSolver === MadCoreGPU.CUDSSSolver
    @test MadNLPGPU.LapackCUDASolver === MadCoreGPU.LapackCUDASolver
    @test MadNLPGPU.LapackROCmSolver === MadCoreGPU.LapackROCmSolver
    @test isdefined(MadNLPGPU, :EventTimer)
    @test MadNLPGPU.madnlp === cuMadNLP.madnlp
end
