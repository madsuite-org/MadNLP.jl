const OPTIMIZER = ()->MadNLP.Optimizer(
    kkt_system=MadNLP.DenseKKTSystem,
    linear_solver=MadNLP.LapackCPUSolver,
    print_level=MadNLP.ERROR
)

@testset "MINLPTests" begin
    ###
    ### src/nlp tests.
    ###

    MINLPTests.test_nlp(
        OPTIMIZER,
        exclude = [
            "005_011",  # Uses the function `\`
            "006_010",  # User-defined function without Hessian (autodiff only provides 1st order)
            "009_010",  # Objective is non-smooth
        ],
        objective_tol = 1e-5,
        primal_tol = 1e-5,
        dual_tol = NaN,
    )

    ###
    ### src/nlp-cvx tests.
    ###

    MINLPTests.test_nlp_cvx(
        OPTIMIZER,
    )
end

