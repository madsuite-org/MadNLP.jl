using Test, MadNLP, MadNLPTests
using NLPModels
using Quadmath
import SparseArrays: sparse

@testset "MadNLP test" begin
    @testset "User-facing names are public in MadNLP" begin
        # option values and types a user writes as `MadNLP.name`; none may warn in the REPL
        user_names = (
            :TRACE, :DEBUG, :INFO, :NOTICE, :WARN, :ERROR, :SOLVE_SUCCEEDED,
            :SparseKKTSystem, :DenseKKTSystem, :DenseCondensedKKTSystem, :SparseCondensedKKTSystem,
            :SparseCallback, :DenseCallback, :BFGS, :DampedBFGS, :CompactLBFGS, :ExactHessian,
            :LU, :QR, :CHOLESKY, :BUNCHKAUFMAN, :RichardsonIterator, :MonotoneUpdate,
            :QualityFunctionUpdate, :InertiaBased, :InertiaFree, :MadNLPOptions, :get_kkt,
        )
        @test all(n -> isdefined(MadNLP, n), user_names)
        if VERSION >= v"1.11"
            @test all(n -> Base.ispublic(MadNLP, n), user_names)
            @test all(n -> Base.ispublic(MadNLP, n), MadNLP.MadCore.public_names())
            @test all(n -> Base.ispublic(MadNLP, n), MadNLP.MadNLPCore.PUBLIC_NAMES)
            @test all(n -> Base.ispublic(MadNLP.MadNLPCore, n), MadNLP.MadNLPCore.PUBLIC_NAMES)
        end
    end
    @testset "Matrix tools" begin
        include("matrix_test.jl")
    end

    @testset "KKTSystem" begin
        include("kkt_test.jl")
    end

    @testset "MadNLP test" begin
        include("madnlp_test.jl")
        include("presets_test.jl")
        include("barrier_test.jl")
        include("restoration_test.jl")
        include("condensed_preset_test.jl")
        include("madnlp_dense.jl")
        include("madnlp_quasi_newton.jl")
        include("regression_test.jl")
    end

    @testset "Schur complement" begin
        include("schur_test.jl")
    end

    @testset "Extensions" begin
        include("krylov_test.jl")
        include("ordering_test.jl")
        # MKL Pardiso was not tested in CI before; enable once its tests pass.
        # include("pardiso_test.jl")
    end

    include("zero_compile_test.jl")

    # last: loading AppleAccelerate forwards BLAS and LAPACK to Accelerate
    Sys.isapple() && @testset "Apple Accelerate" begin
        include("accelerate_test.jl")
    end
end # @testset
