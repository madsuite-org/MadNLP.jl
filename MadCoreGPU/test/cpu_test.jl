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
                "SDS-CPU-DIAGONAL",
                () -> MadNLP.Optimizer(
                    linear_solver = MadCoreGPU.SDSSolver,
                    print_level = MadNLP.ERROR,
                    sds_pivoting = :diagonal,
                    sds_matching = "algo6",
                    sds_deferred_analysis = true,
                ),
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
    # diagonal pivoting runs on the stock pipeline only; without deferred analysis the constructor
    # factorizes with it
    A = MadCore.sparse(Int32[1, 2, 2], Int32[1, 1, 2], [1.0, 0.1, 2.0], 2, 2)
    opt(; kwargs...) = (o = MadCore.default_options(MadCoreGPU.SDSSolver); foreach(((k, v),) -> setproperty!(o, k, v), pairs(kwargs)); o)
    SDS = Base.get_extension(MadCoreGPU, :MadCoreGPUSDSExt).SparseDirectSolver
    @test_throws SDS.InvalidValueError MadCore.create_linear_solver(MadCoreGPU.SDSSolver, A; opt = opt(sds_pivoting = :diagonal, sds_fused = true))
    @test_throws SDS.InvalidValueError MadCore.create_linear_solver(MadCoreGPU.SDSSolver, A; opt = opt(sds_pivoting = :diagonal, sds_factorization = :cholesky))
    M = MadCore.create_linear_solver(MadCoreGPU.SDSSolver, A; opt = opt(sds_pivoting = :diagonal))
    @test SDS.getparam(M.inner.inner, "pivot_type") == 'D'
    @test MadCore.is_factorized(M) && MadCore.inertia(M) == (2, 0, 0)
end
