using Test, MadNLPGPU, MadCore, cuMadNLP

@testset "MadNLPGPU" begin
    # The deprecated MadNLPGPU keeps the names of MadNLPGPU resolving.
    @test MadNLPGPU.CUDSSSolver === MadCore.CUDSSSolver
    @test MadNLPGPU.LapackCUDASolver === MadCore.LapackCUDASolver
    @test MadNLPGPU.LapackROCmSolver === MadCore.LapackROCmSolver
    @test isdefined(MadNLPGPU, :EventTimer)
    @test MadNLPGPU.madnlp === cuMadNLP.madnlp
end
