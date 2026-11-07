using Test, mtlMadNLP, MadNLPCore, MadCore, MadCoreGPU, Metal

@testset "mtlMadNLP" begin
    @testset "Exports" begin
        @test SDSSolver <: MadCore.AbstractLinearSolver
        @test madnlp === MadNLPCore.madnlp
        @test mtlMadNLP.MetalBackend === Metal.MetalBackend
        @test MadCore.default_options(SDSSolver) isa SDSSolverOptions
        @test MadCore.input_type(SDSSolver) === :csc
        @test hasmethod(MadNLPCore.MadNLPOptions{Float32}, Tuple{MadNLPCore.AbstractNLPModel{Float32, MtlVector{Float32}}})
    end
    if Metal.functional()
        @testset "Option presets for MtlVector models" begin
            nlp = MadNLPCore.SparseWrapperModel(MtlVector, MadNLPCore.HS15Model(; T = Float32))
            opt = MadNLPCore.MadNLPOptions{Float32}(nlp)
            @test opt.linear_solver === SDSSolver
            @test opt.kkt_system === MadNLPCore.SparseCondensedKKTSystem
            stats = madnlp(nlp; print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
    end
end
