using Krylov

@testset "KrylovIterator" begin
    test_madnlp(
        "KrylovIterator",
        () -> MadNLP.Optimizer(iterator = KrylovIterator, print_level = MadNLP.ERROR),
        [],
    )
end
