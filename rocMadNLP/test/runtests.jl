using Test, rocMadNLP, MadNLPCore, MadCore, MadCoreGPU, AMDGPU

@testset "rocMadNLP" begin
    @testset "Exports and extensions" begin
        @test LapackROCmSolver === MadCoreGPU.LapackROCmSolver
        @test SDSSolver <: MadCore.AbstractLinearSolver
        @test madnlp === MadNLPCore.madnlp
        @test :ROCBackend in names(rocMadNLP)
        @test rocMadNLP.ROCBackend === AMDGPU.ROCBackend
        # rocMadNLP loads AMDGPU, so the rocSOLVER solvers are available.
        @test MadCore.default_options(LapackROCmSolver) isa MadCore.AbstractOptions
        @test MadCore.default_options(SDSSolver) isa SDSSolverOptions
        @test MadCore.input_type(SDSSolver) === :csc
        @test hasmethod(MadNLPCore.MadNLPOptions{Float64}, Tuple{MadNLPCore.AbstractNLPModel{Float64, ROCVector{Float64}}})
    end
    if AMDGPU.functional()
        @testset "Option presets for ROCVector models" begin
            nlp = MadNLPCore.SparseWrapperModel(ROCVector, MadNLPCore.HS15Model())
            opt = MadNLPCore.MadNLPOptions{Float64}(nlp)
            @test opt.linear_solver === SDSSolver
            @test opt.kkt_system === MadNLPCore.SparseCondensedKKTSystem
            stats = madnlp(nlp; print_level = MadNLPCore.ERROR)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
        @testset "SDSSolver: $(factorization), $(pivoting)" for (factorization, pivoting) in (
                (:cholesky, :none), (:ldl, :none), (:ldl, :bk),
            )
            nlp = MadNLPCore.SparseWrapperModel(ROCVector, MadNLPCore.HS15Model())
            stats = madnlp(
                nlp; print_level = MadNLPCore.ERROR,
                sds_factorization = factorization, sds_pivoting = pivoting,
            )
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
        @testset "SDSSolver on the augmented system, Ruiz scaling and dual shift" begin
            nlp = MadNLPCore.SparseWrapperModel(ROCVector, MadNLPCore.HS15Model())
            solver = MadNLPCore.MadNLPSolver(
                nlp; print_level = MadNLPCore.ERROR, kkt_system = MadNLPCore.SparseKKTSystem,
                sds_factorization = :ldl, sds_pivoting = :bk,
                kkt_scaler = MadCore.RuizScaler(), dual_shift = 1.0e-10,
            )
            @test solver.kkt.linear_solver.wrapped isa MadCore.ScaledShiftedSolver
            stats = MadNLPCore.solve!(solver)
            @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        end
    end
end
