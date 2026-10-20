using Test, MadNLP, MINLPTests
import MathOptInterface

@testset "MOI interface" begin
    include("MOI_interface_test.jl")
end

@testset "MINLP test" begin
    include("minlp_test.jl")
end

