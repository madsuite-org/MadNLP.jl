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
    end

    @testset "cuDSS orderings" begin
        # Lower triangle of a symmetric 3x3 matrix.
        A = sparse([1, 2, 3, 3], [1, 1, 2, 3], [2.0, 1.0, 1.0, 3.0], 3, 3)
        @test MadCore.ordering_permutation(Val(MadCore.DEFAULT_ORDERING), A, Int[]) === nothing
        @test MadCore.ordering_permutation(Val(MadCore.USER_ORDERING), A, [2, 1, 3]) == [2, 1, 3]
        @test_throws ArgumentError MadCore.ordering_permutation(Val(MadCore.USER_ORDERING), A, Int[])
        @test_throws ArgumentError MadCore.ordering_permutation(Val(MadCore.USER_ORDERING), A, [1, 1, 2])
        # Without Metis and AMD loaded, the error names the package to load.
        msg(o) = sprint(showerror, try MadCore.ordering_permutation(Val(o), A, Int[]) catch e; e end)
        if Base.get_extension(MadCore, :MadCoreMetisExt) === nothing
            @test occursin("Load Metis", msg(MadCore.METIS_ORDERING))
        end
        if Base.get_extension(MadCore, :MadCoreAMDExt) === nothing
            for o in (MadCore.AMD_ORDERING, MadCore.SYMAMD_ORDERING, MadCore.COLAMD_ORDERING)
                @test occursin("Load AMD", msg(o))
            end
        end
        @eval using AMD, Metis
        for o in (MadCore.METIS_ORDERING, MadCore.AMD_ORDERING, MadCore.SYMAMD_ORDERING, MadCore.COLAMD_ORDERING)
            p = Base.invokelatest(MadCore.ordering_permutation, Val(o), A, Int[])
            @test length(p) == size(A, 1) && isperm(p)
        end
    end
end
