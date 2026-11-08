using Test, oneMadNLP, MadNLPCore, MadCore, MadCoreGPU, oneAPI

@testset "oneMadNLP" begin
    @testset "Exports and extensions" begin
        @test LapackOneMKLSolver === MadCoreGPU.LapackOneMKLSolver
        @test SDSSolver === MadCoreGPU.SDSSolver
        @test madnlp === MadNLPCore.madnlp
        @test oneMadNLP.oneAPIBackend === oneAPI.oneAPIBackend
        @test MadCore.default_options(LapackOneMKLSolver) isa MadCore.AbstractOptions
        @test MadCore.default_options(SDSSolver) isa MadCore.AbstractOptions
        @test MadCore.input_type(SDSSolver) === :csc
        @test hasmethod(MadNLPCore.MadNLPOptions{Float64}, Tuple{MadNLPCore.AbstractNLPModel{Float64, oneVector{Float64}}})
    end
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
        @testset "Option presets for oneVector models, $T" begin
            nlp = MadNLPCore.SparseWrapperModel(oneVector, MadNLPCore.HS15Model(; T))
            opt = MadNLPCore.MadNLPOptions{T}(nlp)
            @test opt.linear_solver === SDSSolver
            @test opt.kkt_system === MadNLPCore.SparseCondensedKKTSystem
            stats = madnlp(nlp; print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
    end
end
