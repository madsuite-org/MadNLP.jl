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
        # The restoration phase detects a local infeasibility; the fallback then reaches the solution.
        off = madnlp(wb; linear_solver = LDLSolver, print_level = MadNLPCore.ERROR, feasibility_fallback = false)
        @test off.status == MadNLPCore.INFEASIBLE_PROBLEM_DETECTED
        on = madnlp(wb; linear_solver = LDLSolver, print_level = MadNLPCore.ERROR)
        @test on.status == MadNLPCore.SOLVE_SUCCEEDED
        @test on.solution ≈ [1.0, 0.0, 0.5] atol = 1.0e-6
        gn = madnlp(wb; linear_solver = LDLSolver, print_level = MadNLPCore.ERROR, feasibility_fallback_method = :gauss_newton)
        @test gn.status == MadNLPCore.SOLVE_SUCCEEDED
        @test gn.solution ≈ [1.0, 0.0, 0.5] atol = 1.0e-6
        # The rescue, run on that solve as after a failed restoration phase: the regular phase
        # resumes from a point with a lower violation than where restoration started.
        solve_wb() = (s = MadNLPSolver(wb; linear_solver = LDLSolver, print_level = MadNLPCore.ERROR, feasibility_fallback = false); MadNLPCore.solve!(s); s)
        solver = solve_wb()
        theta_ref = MadNLPCore.get_RR(solver).theta_ref
        @test MadNLPCore.feasibility_rescue!(solver, MadNLPCore.RESTORATION_FAILED; from_entry = true) == MadNLPCore.REGULAR
        @test MadNLPCore.get_theta(MadNLPCore.get_c(solver)) < 0.9 * theta_ref
        # A rejected rescue leaves the solver at the restoration phase's last iterate, and a
        # limit reached during the rescue is reported.
        for (field, value, expected) in (
                (:feasibility_fallback_max_iter, 0, MadNLPCore.RESTORATION_FAILED),
                (:max_wall_time, 0.0, MadNLPCore.MAXIMUM_WALLTIME_EXCEEDED),
            )
            solver = solve_wb()
            x_last = copy(MadNLPCore.full(MadNLPCore.get_x(solver)))
            c_last = copy(MadNLPCore.get_c(solver))
            setfield!(MadNLPCore.get_opt(solver), field, value)
            @test MadNLPCore.feasibility_rescue!(solver, MadNLPCore.RESTORATION_FAILED; from_entry = true) == expected
            @test MadNLPCore.full(MadNLPCore.get_x(solver)) == x_last
            @test MadNLPCore.get_c(solver) == c_last
        end
        # The rescue of a failed regular phase starts from the current point, and here resumes
        # the regular phase at a lower violation.
        solver = solve_wb()
        theta_before = MadNLPCore.get_theta(MadNLPCore.get_c(solver))
        @test MadNLPCore.feasibility_rescue!(solver, MadNLPCore.SEARCH_DIRECTION_BECOMES_TOO_SMALL; from_entry = false) == MadNLPCore.REGULAR
        @test MadNLPCore.get_theta(MadNLPCore.get_c(solver)) < 0.9 * theta_before
        # The Gauss-Newton workspaces follow a method chosen after the first restoration.
        solver = solve_wb()
        MadNLPCore.get_opt(solver).feasibility_fallback_method = :gauss_newton
        @test MadNLPCore.feasibility_rescue!(solver, MadNLPCore.SEARCH_DIRECTION_BECOMES_TOO_SMALL; from_entry = false) == MadNLPCore.REGULAR
        # Diverging iterates are retried a bounded number of times.
        solver = MadNLPSolver(nlp; linear_solver = LDLSolver, print_level = MadNLPCore.ERROR, diverging_iterates_tol = 1.0)
        stats = MadNLPCore.solve!(solver)
        @test stats.status == MadNLPCore.DIVERGING_ITERATES
        @test 1 <= MadNLPCore.get_RR(solver).n_rescues <= MadNLPCore.MAX_RESCUES
        # An unknown method is refused.
        @test_throws ErrorException madnlp(nlp; linear_solver = LDLSolver, print_level = MadNLPCore.ERROR, feasibility_fallback_method = :typo)
        # A solve that never enters restoration is unchanged.
        a = madnlp(nlp; linear_solver = LDLSolver, print_level = MadNLPCore.ERROR, feasibility_fallback = false)
        b = madnlp(nlp; linear_solver = LDLSolver, print_level = MadNLPCore.ERROR)
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
