using Test, mtlMadNLP, MadNLPCore, MadCore, MadCoreGPU, Metal, LinearAlgebra, SparseArrays

@testset "mtlMadNLP" begin
    @testset "Exports" begin
        @test SDSSolver === MadCoreGPU.SDSSolver
        @test madnlp === MadNLPCore.madnlp
        @test mtlMadNLP.MetalBackend === Metal.MetalBackend
        SDSExt = Base.get_extension(MadCoreGPU, :MadCoreGPUSDSExt)
        @test MadCore.default_options(SDSSolver) isa SDSExt.SDSSolverOptions
        @test MadCore.input_type(SDSSolver) === :csc
        # the GPU option preset of MadNLPCoreGPUExt, not the generic one, applies to MtlVector models
        preset = which(MadNLPCore.MadNLPOptions{Float32}, Tuple{MadNLPCore.AbstractNLPModel{Float32, MtlVector{Float32}}})
        @test preset.module === Base.get_extension(MadNLPCore, :MadNLPCoreGPUExt)
    end
    if Metal.functional()
        @testset "Option presets for MtlVector models" begin
            nlp = MadCore.SparseWrapperModel(MtlVector, MadNLPCore.HS15Model(; T = Float32))
            # the default, ShiftedPreset: the augmented system with SparseDirectSolver's static LDLᵀ
            opt = MadNLPCore.MadNLPOptions{Float32}(nlp)
            @test opt.preset === MadNLPCore.ShiftedPreset
            @test opt.linear_solver === SDSSolver
            @test opt.kkt_system === MadNLPCore.SparseKKTSystem
            @test opt.dual_shift === MadNLPCore.preset_perturbation(Float32)
            lopt = MadCore.default_options(nlp, opt.preset, opt.kkt_system, opt.linear_solver)
            @test lopt.sds_factorization === :ldl && lopt.sds_pivoting === :none
            @test lopt.sds_pivot_epsilon == Float64(MadNLPCore.preset_perturbation(Float32))
            @test lopt.sds_matching == "algo6" && lopt.sds_deferred_analysis && lopt.sds_ir == 2
            stats = madnlp(nlp; print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
            # CondensedPreset
            co = MadNLPCore.MadNLPOptions{Float32}(nlp; preset = MadNLPCore.CondensedPreset)
            @test co.kkt_system === MadNLPCore.SparseCondensedKKTSystem
            stats = madnlp(nlp; preset = MadNLPCore.CondensedPreset, print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
            # The local minimum reached from the origin. Float32 and the relaxed bounds of
            # the condensed KKT system shift the result slightly.
            @test stats.objective ≈ 360.38f0 rtol = 1.0f-2
            @test Array(stats.solution) ≈ Float32[-0.7921, -1.2624] rtol = 1.0f-2
        end
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
