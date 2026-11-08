using Test, MadNLPCore, NLPModels

include("hs15.jl")
include("wachbieg.jl")

# A package that keeps MadCore as a weak dependency exports a plain type and defines,
# in its extension, the methods of the linear solver interface for it.
struct MyLDL end

@testset "MadNLPCore" begin
    nlp = HS15Model()

    @testset "Solver fields match the MadCore getters" begin
        @test Set(fieldnames(MadNLPSolver)) == Set((MadNLPCore.MadCore.SOLVER_FIELDS..., MadNLPCore.FILTER_FIELDS...))
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

    @testset "Feasibility fallback" begin
        using LDLFactorizations
        wb = WachBiegModel()
        # The restoration phase ends at an infeasible stationary point of the violation.
        stats = madnlp(wb; linear_solver = LDLSolver, print_level = MadNLPCore.ERROR)
        @test stats.status == MadNLPCore.INFEASIBLE_PROBLEM_DETECTED
        stats = madnlp(wb; linear_solver = LDLSolver, print_level = MadNLPCore.ERROR, feasibility_fallback = true)
        @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        @test stats.solution ≈ [1.0, 0.0, 0.5] atol = 1.0e-6
        # The option leaves a solve that never enters restoration unchanged.
        a = madnlp(nlp; linear_solver = LDLSolver, print_level = MadNLPCore.ERROR)
        b = madnlp(nlp; linear_solver = LDLSolver, print_level = MadNLPCore.ERROR, feasibility_fallback = true)
        @test a.iter == b.iter && a.objective == b.objective
    end

    @testset "Regular-phase stall retry" begin
        wb = WachBiegModel()
        opts = (linear_solver = LDLSolver, print_level = MadNLPCore.ERROR)
        stats = madnlp(wb; opts..., feasibility_fallback = true, regular_stall_iter = 1)
        @test stats.status == MadNLPCore.SOLVE_SUCCEEDED
        @test stats.solution ≈ [1.0, 0.0, 0.5] atol = 1.0e-6
        # A solve that never enters restoration is unchanged.
        a = madnlp(nlp; opts...)
        b = madnlp(nlp; opts..., feasibility_fallback = true, regular_stall_iter = 1)
        @test a.iter == b.iter && a.objective == b.objective
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
