# Tests that run without a GPU.
import MadCore

@testset "Marker types" begin
    for M in (MadCoreGPU.CUDSSSolver, MadCoreGPU.LapackCUDASolver, MadCoreGPU.LapackROCmSolver, MadCoreGPU.LapackOneMKLSolver, MadCoreGPU.SDSSolver)
        @test MadCore.backend_package(M) isa String
    end
    if !HAS_CUDA
        # Without CUDA and CUDSS, the extension implementing CUDSSSolver is not loaded.
        @test_throws MethodError MadCore.default_options(MadCoreGPU.CUDSSSolver)
    end
end

# SDSSolver accepts a host SparseMatrixCSC, on SparseDirectSolver's stock pipeline.
@testset "SDSSolver on the CPU" begin
    MadNLPTests.test_linear_solver(MadCoreGPU.SDSSolver, Float32)
    MadNLPTests.test_linear_solver(MadCoreGPU.SDSSolver, Float64)
    for (name, optimizer_constructor, exclude) in [
            [
                "SDS-CPU",
                () -> MadNLP.Optimizer(linear_solver = MadCoreGPU.SDSSolver, print_level = MadNLP.ERROR),
                [],
            ],
            [
                "SDS-CPU-BUNCHKAUFMAN",
                () -> MadNLP.Optimizer(
                    linear_solver = MadCoreGPU.SDSSolver,
                    print_level = MadNLP.ERROR,
                    sds_pivoting = :bk,
                    sds_matching = "algo6",
                    sds_deferred_analysis = true,
                    sds_reanalyze_on_improve = true,
                ),
                [],
            ],
        ]
        test_madnlp(name, optimizer_constructor, exclude)
    end
end
