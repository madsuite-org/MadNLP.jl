using Test, MadNLP, MadNLPTests, HSL, Quadmath

testset = [
    [
        "HSL-Ma27",
        ()->MadNLP.Optimizer(
            linear_solver=Ma27Solver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "HSL-Ma57",
        ()->MadNLP.Optimizer(
            linear_solver=Ma57Solver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "HSL-Ma77",
        ()->MadNLP.Optimizer(
            linear_solver=Ma77Solver,
            print_level=MadNLP.ERROR),
        ["unbounded"]
    ],
    [
        "HSL-Ma86",
        ()->MadNLP.Optimizer(
            linear_solver=Ma86Solver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "HSL-Ma97",
        ()->MadNLP.Optimizer(
            linear_solver=Ma97Solver,
            print_level=MadNLP.ERROR),
        []
    ],
]

@testset "HSL solvers" begin
    if LIBHSL_isfunctional()
        for (hsl_solver, package) in [(Ma27Solver, "ma27"),
                                      (Ma57Solver, "ma57"),
                                      (Ma77Solver, "ma77"),
                                      (Ma86Solver, "ma86"),
                                      (Ma97Solver, "ma97")]
            @testset "$package" begin
                # MadNLPTests.test_linear_solver(hsl_solver,Float32)
                @testset "Float64" begin
                    MadNLPTests.test_linear_solver(hsl_solver,Float64)
                end
                if MadNLP.is_supported(hsl_solver, Float128)
                    @testset "Float128" begin
                        MadNLPTests.test_linear_solver(hsl_solver,Float128)
                    end
                end
            end
        end
        for (name,optimizer_constructor,exclude) in testset
            test_madnlp(name,optimizer_constructor,exclude)
        end
    end
end
