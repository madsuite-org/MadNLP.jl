using Test, mtlMadNLP, MadNLPCore, MadCore, MadCoreGPU, Metal, LinearAlgebra, SparseArrays

# MadCoreGPU's tests of SDSSolver and the GPU presets, by path: this package has no CI leg on a GPU
include(joinpath(@__DIR__, "..", "..", "MadCoreGPU", "test", "sds_test.jl"))

@testset "mtlMadNLP" begin
    @testset "Exports" begin
        @test SDSSolver === MadCoreGPU.SDSSolver
        @test madnlp === MadNLPCore.madnlp
        @test mtlMadNLP.MetalBackend === Metal.MetalBackend
        # the GPU option preset of MadNLPCoreGPUExt, not the generic one, applies to MtlVector models
        preset = which(MadNLPCore.MadNLPOptions{Float32}, Tuple{MadNLPCore.AbstractNLPModel{Float32, MtlVector{Float32}}})
        @test preset.module === Base.get_extension(MadNLPCore, :MadNLPCoreGPUExt)
    end
    test_sds_extension()
    if Metal.functional()
        @testset "Default linear solver of MtlVector models" begin
            nlp = MadNLPCore.SparseWrapperModel(MtlVector, MadNLPCore.HS15Model(; T = Float32))
            @test MadNLPCore.MadNLPOptions{Float32}(nlp).linear_solver === SDSSolver
        end
        test_gpu_presets(MtlArray, Float32)
        test_sds_presets(MtlArray, Float32)
        test_sds_linear_solver(MtlArray, Float32)
        @testset "MtlSparseMatrixCSC products" begin
            A = sprand(Float32, 40, 25, 0.2)
            dA = MadCore._get_sparse_csc(size(A), MtlVector(Int32.(A.colptr)), MtlVector(Int32.(A.rowval)), MtlVector(A.nzval))
            @test SparseMatrixCSC(dA) == A
            x, y = rand(Float32, 25), rand(Float32, 40)
            dy = MtlVector(y)
            mul!(dy, dA, MtlVector(x), 2.0f0, 3.0f0)
            @test Array(dy) ≈ 2 .* (A * x) .+ 3 .* y
            mul!(dy, dA, MtlVector(x), 2.0f0, 0.0f0)
            @test Array(dy) ≈ 2 .* (A * x)
            dx = MtlVector(x)
            mul!(dx, dA', MtlVector(y), 2.0f0, 3.0f0)
            @test Array(dx) ≈ 2 .* (A' * y) .+ 3 .* x
            mul!(dx, dA', MtlVector(y), 1.0f0, 0.0f0)
            @test Array(dx) ≈ A' * y
        end
        @testset "Host sort of tuples" begin
            v = [(3, 1), (1, 2), (2, 3)]
            dv = MtlVector(v)
            MadCore._sort_tuples!(dv)
            @test Array(dv) == sort(v)
            @test Array(MadCore._sortperm_tuples(MtlVector(v))) == sortperm(v)
            dv = MtlVector(v)
            MadCore._permute_tuples!(dv, MtlVector([3, 1, 2]))
            @test Array(dv) == v[[3, 1, 2]]
        end
        include("border.jl")
    end
end
