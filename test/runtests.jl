using Test, MadNLP, MadNLPTests
using NLPModels
using Quadmath
import SparseArrays: sparse

@testset "MadNLP test" begin
    @testset "Matrix tools" begin
        include("matrix_test.jl")
    end

    @testset "KKTSystem" begin
        include("kkt_test.jl")
    end

    @testset "MadNLP test" begin
        include("madnlp_test.jl")
        include("condensed_border_test.jl")
        include("madnlp_dense.jl")
        include("madnlp_quasi_newton.jl")
        include("regression_test.jl")
    end

    @testset "Schur complement" begin
        include("schur_test.jl")
    end
end # @testset
