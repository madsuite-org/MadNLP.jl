using Test, oneMadNLP, MadNLPCore, MadCore, MadCoreGPU, oneAPI

# MadCoreGPU's tests of SDSSolver and the GPU presets, by path: this package has no CI leg on a GPU
include(joinpath(@__DIR__, "..", "..", "MadCoreGPU", "test", "sds_test.jl"))

@testset "oneMadNLP" begin
    @testset "Exports and extensions" begin
        @test LapackOneMKLSolver === MadCoreGPU.LapackOneMKLSolver
        @test SDSSolver === MadCoreGPU.SDSSolver
        @test madnlp === MadNLPCore.madnlp
        @test oneMadNLP.oneAPIBackend === oneAPI.oneAPIBackend
        @test MadCore.default_options(LapackOneMKLSolver) isa MadCore.AbstractOptions
        # the GPU option preset of MadNLPCoreGPUExt, not the generic one, applies to oneVector models
        preset = which(MadNLPCore.MadNLPOptions{Float64}, Tuple{MadNLPCore.AbstractNLPModel{Float64, oneVector{Float64}}})
        @test preset.module === Base.get_extension(MadNLPCore, :MadNLPCoreGPUExt)
    end
    test_sds_extension()
    if oneAPI.functional()
        @testset "oneMKL Cholesky reports failure" begin
            for (A, expected) in (
                    (Float32[2 0; 0 3], (2, 0, 0)),
                    (Float32[1 0; 0 -1], (0, 2, 0)),
                )
                opt = MadCore.LapackOptions(MadCore.CHOLESKY)
                M = MadCore.create_linear_solver(LapackOneMKLSolver, oneArray(A); opt)
                MadCore.factorize!(M)
                @test MadCore.inertia(M) == expected
            end
        end
        # Arc GPUs have no native Float64, so the solves run in Float32.
        T = oneL0.module_properties(oneAPI.device()).fp64flags == 0 ? Float32 : Float64
        @testset "Default linear solver of oneVector models, $T" begin
            nlp = MadNLPCore.SparseWrapperModel(oneVector, MadNLPCore.HS15Model(; T))
            @test MadNLPCore.MadNLPOptions{T}(nlp).linear_solver === SDSSolver
        end
        test_gpu_presets(oneArray, T)
        test_sds_presets(oneArray, T)
        test_sds_linear_solver(oneArray, T)
    end
end
