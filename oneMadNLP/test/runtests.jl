using Test, oneMadNLP, MadNLPCore, MadCore, MadCoreGPU, oneAPI

@testset "oneMadNLP" begin
    @testset "Exports and extensions" begin
        @test LapackOneMKLSolver === MadCoreGPU.LapackOneMKLSolver
        @test SDSSolver <: MadCore.AbstractLinearSolver
        @test madnlp === MadNLPCore.madnlp
        @test oneMadNLP.oneAPIBackend === oneAPI.oneAPIBackend
        @test MadCore.default_options(LapackOneMKLSolver) isa MadCore.AbstractOptions
        @test MadCore.default_options(SDSSolver) isa SDSSolverOptions
        @test MadCore.input_type(SDSSolver) === :csc
        @test hasmethod(MadNLPCore.MadNLPOptions{Float64}, Tuple{MadNLPCore.AbstractNLPModel{Float64, oneVector{Float64}}})
    end
    if oneAPI.functional()
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
