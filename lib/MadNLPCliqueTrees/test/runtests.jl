using Test, MadNLP, MadNLPCliqueTrees, MadNLPTests, CliqueTrees
using CliqueTrees.Multifrontal: DynamicRegularization, GMW81, SE99
using SparseArrays

@testset "MadNLPCliqueTrees test" begin
    # Default LDL algorithm
    MadNLPTests.test_linear_solver(CliqueTreesSolver, Float32)
    MadNLPTests.test_linear_solver(CliqueTreesSolver, Float64)

    # Cholesky algorithm on positive definite matrix
    @testset "Cholesky algorithm" begin
        for T in (Float32, Float64)
            row = Int32[1,2,2]; col = Int32[1,1,2]; val = T[1., .1, 2.]
            b = T[1.0, 3.0]
            sol = T[0.8542713567839195, 1.4572864321608041]
            csc = sparse(row, col, val, 2, 2)
            opt = CliqueTreesOptions(cliquetrees_algorithm=MadNLP.CHOLESKY)
            M = CliqueTreesSolver(csc; opt=opt)
            MadNLP.factorize!(M)
            @test MadNLP.is_inertia(M)
            @test MadNLP.inertia(M) == (2, 0, 0)
            x = MadNLP.solve_linear_system!(M, copy(b))
            @test MadNLPTests.solcmp(x, sol)
        end
    end

    test_madnlp(
        "CliqueTrees",
        () -> MadNLP.Optimizer(linear_solver = CliqueTreesSolver, print_level = MadNLP.ERROR),
        ["eigmina"],
    )

    @testset "Elimination orderings" begin
        for alg in (AMF(), MMD(), MF())
            test_madnlp(
                "CliqueTrees/$(nameof(typeof(alg)))",
                () -> MadNLP.Optimizer(
                    linear_solver = CliqueTreesSolver,
                    print_level = MadNLP.ERROR,
                    cliquetrees_ordering = alg,
                ),
                ["eigmina"],
            )
        end
    end

    @testset "Dynamic Regularization" begin
        reg = DynamicRegularization()

        test_madnlp(
            "CliqueTrees/$(nameof(typeof(reg)))",
            () -> MadNLP.Optimizer(
                linear_solver = CliqueTreesSolver,
                print_level = MadNLP.ERROR,
                cliquetrees_regularization = reg,
            ),
            [],
        )
    end

    @testset "GMW81" begin
        reg = GMW81()

        test_madnlp(
            "CliqueTrees/$(nameof(typeof(reg)))",
            () -> MadNLP.Optimizer(
                linear_solver = CliqueTreesSolver,
                print_level = MadNLP.ERROR,
                cliquetrees_regularization = reg,
            ),
            ["unbounded"],
        )
    end

    @testset "SE99" begin
        reg = SE99()

        test_madnlp(
            "CliqueTrees/$(nameof(typeof(reg)))",
            () -> MadNLP.Optimizer(
                linear_solver = CliqueTreesSolver,
                print_level = MadNLP.ERROR,
                cliquetrees_regularization = reg,
            ),
            ["eigmina"],
        )
    end
end
