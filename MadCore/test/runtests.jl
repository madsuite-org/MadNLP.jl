using Test, MadCore, SparseArrays, LinearAlgebra

@testset "MadCore" begin
    @testset "Linear solver markers without their backend" begin
        # MUMPS_seq_jll is not a test dependency, so its extension is not loaded.
        err = try
            MadCore.create_linear_solver(MumpsSolver, sparse([1.0;;]))
        catch e
            e
        end
        @test err isa MethodError
        @test occursin("Load MUMPS_seq_jll", sprint(showerror, err))
        @test_throws MethodError MadCore.default_options(MumpsSolver)
    end

    @testset "LinearSolverBarrier forwards is_factorized" begin
        struct FailedSolver <: MadCore.AbstractLinearSolver{Float64} end
        MadCore.is_factorized(::FailedSolver) = false
        @test !MadCore.is_factorized(MadCore.LinearSolverBarrier(FailedSolver()))
    end

    @testset "ScaledShiftedSolver forwards is_factorized and is_async" begin
        struct FailedAsyncSolver <: MadCore.AbstractLinearSolver{Float64} end
        MadCore.create_linear_solver(::Type{FailedAsyncSolver}, A; kwargs...) = FailedAsyncSolver()
        MadCore.factorize!(S::FailedAsyncSolver) = S
        MadCore.is_factorized(::FailedAsyncSolver) = false
        MadCore.is_async(::FailedAsyncSolver) = true
        # lower triangle of [2 1; 1 0], the zero diagonal entry stored
        A = sparse([1, 2, 2], [1, 1, 2], [2.0, 1.0, 0.0])
        S = MadCore.create_kkt_linear_solver(FailedAsyncSolver, A, 1; opt = nothing, dual_shift = 1.0e-8)
        @test S isa MadCore.ScaledShiftedSolver
        MadCore.factorize!(S)
        @test !MadCore.is_factorized(S)
        @test MadCore.is_async(S)
    end

    # a given permutation, as an ordering of the KKT system
    struct GivenOrdering <: MadCore.AbstractOrdering
        perm::Vector{Int}
    end
    MadCore.ordering_permutation(o::GivenOrdering, ::SparseMatrixCSC) = o.perm

    @testset "Orderings without their backend" begin
        # Metis and AMD are not test dependencies, so their extensions are not loaded.
        A = sparse([1, 2, 2], [1, 1, 2], [2.0, 1.0, 0.0])
        for (o, pkg) in ((MetisOrdering(), "Metis"), (AMDOrdering(), "AMD"))
            err = try
                MadCore.ordering_permutation(o, A)
            catch e
                e
            end
            @test err isa MethodError
            @test occursin("Load $pkg", sprint(showerror, err))
        end
        @test MadCore.ordering_permutation(DefaultOrdering(), A) == [1, 2]
    end

    @testset "An ordering needs a solver that accepts a permutation" begin
        A = sparse([1, 2, 2], [1, 1, 2], [2.0, 1.0, 0.0])
        @test_throws ArgumentError MadCore.create_kkt_linear_solver(FailedAsyncSolver, A, 1; opt = nothing, kkt_ordering = GivenOrdering([2, 1]))
        @test_throws ArgumentError MadCore.create_kkt_linear_solver(LapackCPUSolver, A, 1; kkt_ordering = GivenOrdering([2, 1]))
        # an ordering that does not return a permutation
        @test_throws "did not return a permutation" MadCore.create_kkt_linear_solver(FailedAsyncSolver, A, 1; opt = nothing, kkt_ordering = GivenOrdering([1, 1]))
    end

    @testset "LDLSolver through its extension" begin
        using LDLFactorizations
        @test MadCore.is_supported(LDLSolver, Float64)
        @test MadCore.input_type(LDLSolver) == :csc
        @test MadCore.default_options(LDLSolver) isa MadCore.AbstractOptions

        # Lower triangle of a symmetric indefinite matrix, as passed by the KKT systems.
        A = sparse([4.0 1.0 0.0; 1.0 -3.0 2.0; 0.0 2.0 5.0])
        tril_A = SparseMatrixCSC{Float64,Int32}(tril(A))
        solver = MadCore.create_linear_solver(LDLSolver, tril_A)
        @test solver isa MadCore.AbstractLinearSolver
        @test parentmodule(typeof(solver)) === Base.get_extension(MadCore, :MadCoreLDLExt)
        MadCore.factorize!(solver)
        @test MadCore.is_factorized(solver)
        b = [1.0, 2.0, 3.0]
        x = copy(b)
        MadCore.solve_linear_system!(solver, x)
        @test A * x ≈ b
        @test MadCore.inertia(solver) == (2, 0, 1)

        # the same matrix in a given pivot order, handed to LDLFactorizations as its P
        opt = MadCore.default_options(LDLSolver)
        S = MadCore.create_kkt_linear_solver(LDLSolver, tril_A, 2; opt, kkt_ordering = GivenOrdering([3, 1, 2]))
        @test S.inner.P == [3, 1, 2]
        @test opt.ldl_user_perm === nothing
        MadCore.factorize!(S)
        x = copy(b)
        MadCore.solve_linear_system!(S, x)
        @test A * x ≈ b
        @test MadCore.inertia(S) == (2, 0, 1)
    end
end
