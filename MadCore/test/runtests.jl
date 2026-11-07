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

    @testset "Public API" begin
        names = MadCore.public_names()
        @test allunique(names)
        @test all(name -> isdefined(MadCore, name), names)
        @test !any(startswith("_"), string.(names))
        @test !(:set_kkt! in names)
        @test !(:defined_names in names)
        if VERSION >= v"1.11"
            @test all(name -> Base.ispublic(MadCore, name), names)
            @test !Base.ispublic(MadCore, :defined_names)
        end
    end
end
