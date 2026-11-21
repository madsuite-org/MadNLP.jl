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

    @testset "Orderings without their backend" begin
        # Metis and AMD are not test dependencies, so their extensions are not loaded.
        A = sparse([1, 2, 2], [1, 1, 2], [2.0, 1.0, 0.0])
        for (o, pkg) in ((MetisOrdering(), "Metis"), (AMDOrdering(), "AMD"), (ColAMDOrdering(), "AMD"))
            err = try
                MadCore.ordering_permutation(o, A)
            catch e
                e
            end
            @test err isa MethodError
            @test occursin("Load $pkg", sprint(showerror, err))
        end
        @test MadCore.ordering_permutation(UserOrdering([2, 1]), A) == [2, 1]
        @test_throws ArgumentError MadCore.ordering_permutation(UserOrdering([1, 2, 3]), A)
    end

    @testset "PermutedSolver forwards is_factorized and is_async" begin
        S = MadCore.create_kkt_linear_solver(FailedAsyncSolver, sparse([1, 2, 2], [1, 1, 2], [2.0, 1.0, 0.0]), 1; opt = nothing, kkt_ordering = UserOrdering([2, 1]))
        @test S isa MadCore.PermutedSolver
        MadCore.factorize!(S)
        @test !MadCore.is_factorized(S)
        @test MadCore.is_async(S)
    end

    @testset "PermutedSolver analyzes the current values" begin
        struct AnalyzingSolver <: MadCore.AbstractLinearSolver{Float64}
            A::SparseMatrixCSC{Float64, Int}
            seen::Vector{Float64}
        end
        MadCore.create_linear_solver(::Type{AnalyzingSolver}, A; kwargs...) = AnalyzingSolver(A, Float64[])
        MadCore.analyze!(S::AnalyzingSolver) = (append!(empty!(S.seen), nonzeros(S.A)); true)
        A = sparse([1, 2, 2], [1, 1, 2], [0.0, 0.0, 0.0])
        S = MadCore.create_kkt_linear_solver(AnalyzingSolver, A, 1; opt = nothing, kkt_ordering = UserOrdering([2, 1]))
        nonzeros(A) .= [1.0, 2.0, 3.0]
        @test MadCore.analyze!(S)
        # P A Pᵀ = [3 2; 2 1]: the lower triangle by columns
        @test S.inner.seen == [3.0, 2.0, 1.0]
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

        # the same matrix, permuted
        S = MadCore.create_kkt_linear_solver(LDLSolver, tril_A, 2; kkt_ordering = UserOrdering([3, 1, 2]))
        @test S isa MadCore.PermutedSolver
        @test nonzeros(S.Ap) isa Vector{Float64} && eltype(rowvals(S.Ap)) == Int32
        MadCore.factorize!(S)
        x = copy(b)
        MadCore.solve_linear_system!(S, x)
        @test A * x ≈ b
        @test MadCore.inertia(S) == (2, 0, 1)
        @test occursin("with user ordering", MadCore.introduce(S))
    end
end
