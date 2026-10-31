using Pardiso

# MKL Pardiso is not available on every platform; Panua Pardiso needs a license.
if Pardiso.mkl_is_available()
    @testset "PardisoMKLSolver" begin
        MadNLPTests.test_linear_solver(PardisoMKLSolver, Float32)
        MadNLPTests.test_linear_solver(PardisoMKLSolver, Float64)
        test_madnlp(
            "PardisoMKL",
            () -> MadNLP.Optimizer(linear_solver = PardisoMKLSolver, print_level = MadNLP.ERROR),
            ["eigmina"],
        )
    end
end
