using Pardiso  # GivenOrdering is defined in ordering_test.jl, included before

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
        @testset "kkt_ordering is the pivot order" begin
            # an arrowhead matrix whose dense row and column come first: eliminated last, it
            # fills nothing, first the whole factor
            n = 40
            d = [i == 1 ? 4.0n : (iseven(i) ? -4.0 : 4.0) for i in 1:n]
            A = SparseMatrixCSC{Float64, Int32}(sparse(vcat(1:n, 2:n), vcat(1:n, fill(1, n - 1)), vcat(d, ones(n - 1)), n, n))
            sizes = map((vcat(2:n, 1), collect(1:n))) do p
                S = MadNLP.create_kkt_linear_solver(PardisoMKLSolver, copy(A), n; kkt_ordering = GivenOrdering(p))
                @test S.opt.pardiso_user_perm == p
                MadNLP.factorize!(S)
                Pardiso.get_iparm(S.ps, 18)  # the entries of the factor
            end
            @test sizes[1] < 3n
            @test sizes[2] > n^2 / 4
        end
    end
end
