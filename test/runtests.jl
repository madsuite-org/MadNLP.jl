using Test, MadNLP, MadNLPTests, MINLPTests
using NLPModels
using Quadmath
import MathOptInterface
import SparseArrays: sparse

@testset "MadNLP test" begin
    @testset "Matrix tools" begin
        include("matrix_test.jl")
    end

    @testset "KKTSystem" begin
        include("kkt_test.jl")
    end

    @testset "MOI interface" begin
        include("MOI_interface_test.jl")
    end

    @testset "MadNLP test" begin
        include("madnlp_test.jl")
        include("sensitivity_test.jl")
        include("madnlp_dense.jl")
        include("madnlp_quasi_newton.jl")
    end

    @testset "MINLP test" begin
        include("minlp_test.jl")
    end

    @testset "Schur complement" begin
        include("schur_test.jl")
    end
end # @testset
