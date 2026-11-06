# Tests that run without a GPU.
import MadCore

@testset "Marker types" begin
    for M in (MadCoreGPU.CUDSSSolver, MadCoreGPU.LapackCUDASolver, MadCoreGPU.LapackROCmSolver)
        @test MadCore.backend_package(M) isa String
    end
    if !HAS_CUDA
        # Without CUDA and CUDSS, the extension implementing CUDSSSolver is not loaded.
        @test_throws MethodError MadCore.default_options(MadCoreGPU.CUDSSSolver)
    end
end
