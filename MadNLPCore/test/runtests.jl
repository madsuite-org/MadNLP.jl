using Test, MadNLPCore, NLPModels

include("hs15.jl")

# A package that keeps MadCore as a weak dependency exports a plain type and defines,
# in its extension, the methods of the linear solver interface for it.
struct MyLDL end

@testset "MadNLPCore" begin
    nlp = HS15Model()

    @testset "Solver fields match the MadCore getters" begin
        @test fieldnames(MadNLPSolver) == MadNLPCore.MadCore.SOLVER_FIELDS
    end

    @testset "Default sparse solver is DummySolver" begin
        # Only MadNLP sets a default sparse linear solver (MUMPS).
        @test MadNLPCore.default_sparse_solver(nlp) === MadNLPCore.DummySolver
        @test MadNLPOptions{Float64}(nlp).linear_solver === MadNLPCore.DummySolver
        @test_throws "No sparse linear solver is selected" MadNLPSolver(nlp; print_level = MadNLPCore.ERROR)
    end

    @testset "Solve with LDLFactorizations" begin
        using LDLFactorizations
        stats = madnlp(nlp; linear_solver = LDLSolver, print_level = MadNLPCore.ERROR)
        @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        @test stats.solution ≈ [-0.79212, -1.26243] atol = 1e-4
        @test stats.objective ≈ 360.37976 atol = 1e-4
    end

    @testset "Solve with MUMPS" begin
        # MUMPS needs an LP64 BLAS, which its extension forwards without MadNLP.
        using MUMPS_seq_jll, OpenBLAS32_jll
        stats = madnlp(nlp; linear_solver = MumpsSolver, print_level = MadNLPCore.ERROR)
        @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
    end

    @testset "Any type selects a linear solver" begin
        @eval begin
            MadNLPCore.create_linear_solver(::Type{MyLDL}, A; kwargs...) =
                MadNLPCore.create_linear_solver(LDLSolver, A; kwargs...)
            MadNLPCore.default_options(::Type{MyLDL}) = MadNLPCore.default_options(LDLSolver)
            MadNLPCore.input_type(::Type{MyLDL}) = MadNLPCore.input_type(LDLSolver)
            MadNLPCore.is_supported(::Type{MyLDL}, ::Type{T}) where {T <: AbstractFloat} =
                MadNLPCore.is_supported(LDLSolver, T)
        end
        stats = madnlp(nlp; linear_solver = MyLDL, print_level = MadNLPCore.ERROR)
        @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
    end

    @testset "Solve with LAPACK" begin
        stats = madnlp(
            nlp;
            kkt_system = MadNLPCore.DenseKKTSystem,
            linear_solver = LapackCPUSolver,
            print_level = MadNLPCore.ERROR,
        )
        @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
    end
end
