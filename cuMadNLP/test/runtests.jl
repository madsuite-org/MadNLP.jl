using Test, cuMadNLP, MadNLPCore, MadCore, MadCoreGPU, CUDA

@testset "cuMadNLP" begin
    @testset "Exports and extensions" begin
        @test CUDSSSolver === MadCoreGPU.CUDSSSolver
        @test LapackCUDASolver === MadCoreGPU.LapackCUDASolver
        @test madnlp === MadNLPCore.madnlp
        @test :CUDABackend in names(cuMadNLP)
        @test cuMadNLP.CUDABackend === CUDA.CUDABackend
        # cuMadNLP loads CUDA and CUDSS, so the solvers of MadCoreGPU are available.
        @test MadCore.default_options(CUDSSSolver) isa MadCore.AbstractOptions
        @test MadCore.default_options(LapackCUDASolver) isa MadCore.AbstractOptions
        @test hasmethod(MadNLPCore.MadNLPOptions{Float64}, Tuple{MadNLPCore.AbstractNLPModel{Float64, CuVector{Float64}}})
    end
    if CUDA.functional()
        @testset "Option presets for CuVector models" begin
            nlp = MadNLPCore.SparseWrapperModel(CuVector, MadNLPCore.HS15Model())
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp)
            @test opt.linear_solver === CUDSSSolver
            @test opt.kkt_system === MadNLPCore.SparseCondensedKKTSystem
            stats = madnlp(nlp; print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
        @testset "get_F accounts for upper bounds on the GPU" begin
            # Same case as the CPU test in MadNLP/test/regression_test.jl:
            # one variable with only an upper bound, (xu - x) * zu = (3 - 1) * 2.
            empty = CuVector{Float64}(undef, 0)
            zero1 = CUDA.zeros(Float64, 1)
            F = MadNLPCore.get_F(empty, zero1, zero1, zero1, zero1, empty, empty, empty, CuVector([3.0]), CuVector([1.0]), CuVector([2.0]), 0.0)
            @test F ≈ 4.0
        end
    end
end
