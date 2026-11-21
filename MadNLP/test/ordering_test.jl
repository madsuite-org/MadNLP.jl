using LinearAlgebra, SparseArrays, Random
import AMD, Metis  # load MadCoreAMDExt and MadCoreMetisExt

const ORDERINGS = (
    MadNLP.MetisOrdering(), MadNLP.AMDOrdering(), MadNLP.SymAMDOrdering(), MadNLP.ColAMDOrdering(),
)

# The lower triangle of a random symmetric indefinite matrix [H Jᵀ; J -D], every diagonal
# entry stored, as a sparse KKT system assembles it.
function random_kkt(rng, n, m)
    H = sprandn(rng, n, n, 0.2)
    H = H + H' + 4n * I
    J = sprandn(rng, m, n, 0.3) + sparse(1:m, 1:m, ones(m), m, n)
    K = [H J'; J -Diagonal(rand(rng, m) .+ 1)]
    return SparseMatrixCSC{Float64, Int32}(sparse(LowerTriangular(K)))
end

# the options of a solver that relies on the block structure, as SDSSolver's do
@kwdef mutable struct BlockOptions <: MadNLP.AbstractOptions
    n_primal::Int = -1
end

@testset "Fill-reducing orderings" begin
    rng = MersenneTwister(1)
    n, m = 30, 12
    A = random_kkt(rng, n, m)
    K = Symmetric(A, :L)
    b = randn(rng, n + m)

    @testset "permutations" begin
        for o in (ORDERINGS..., MadNLP.DefaultOrdering(), MadNLP.UserOrdering(randperm(rng, n + m)))
            p = MadNLP.ordering_permutation(o, A)
            @test p isa Vector{Int}
            @test length(p) == n + m && isperm(p)
        end
        @test_throws ArgumentError MadNLP.ordering_permutation(MadNLP.UserOrdering([1, 1]), sparse([1.0 0; 1 1]))
        @test_throws ArgumentError MadNLP.ordering_permutation(MadNLP.UserOrdering([1]), sparse([1.0 0; 1 1]))
    end

    @testset "PermutedSolver with $(nameof(ls)) and $(nameof(typeof(o)))" for ls in (MadNLP.MumpsSolver, MadNLP.LDLSolver),
            o in (ORDERINGS..., MadNLP.UserOrdering(randperm(rng, n + m)))
        S = MadNLP.create_kkt_linear_solver(ls, copy(A), n; kkt_ordering = o)
        @test S isa MadNLP.PermutedSolver
        @test occursin("ordering", MadNLP.introduce(S))
        MadNLP.factorize!(S)
        @test MadNLP.is_factorized(S)
        x = MadNLP.solve_linear_system!(S, copy(b))
        @test K * x ≈ b
        @test MadNLP.is_inertia(S) == MadNLP.is_inertia(MadNLP.create_linear_solver(ls, A))
        MadNLP.is_inertia(S) && @test MadNLP.inertia(S) == (n, 0, m)
        # new values on the same pattern reach the permuted matrix
        nonzeros(S.A) .*= 2
        MadNLP.factorize!(S)
        @test 2 * (K * MadNLP.solve_linear_system!(S, copy(b))) ≈ b
        # the permutation allocates nothing beyond the solve of the inner solver
        c = copy(b)
        @test (@allocated MadNLP.solve_linear_system!(S, c)) <= (@allocated MadNLP.solve_linear_system!(S.inner, c))
    end

    @testset "ordering inside the scaled and shifted factorization" begin
        S = MadNLP.create_kkt_linear_solver(
            MadNLP.MumpsSolver, A, n; kkt_ordering = MadNLP.AMDOrdering(), kkt_scaler = MadNLP.RuizScaler(), dual_shift = 1.0e-10,
        )
        @test S isa MadNLP.ScaledShiftedSolver
        @test S.inner isa MadNLP.PermutedSolver
        MadNLP.factorize!(S)
        @test MadNLP.inertia(S) == (n, 0, m)
        @test K * MadNLP.solve_linear_system!(S, copy(b)) ≈ b rtol = 1.0e-6
    end

    @testset "solvers that rely on the block structure" begin
        @test_throws ArgumentError MadNLP.create_kkt_linear_solver(
            MadNLP.MumpsSolver, A, n; opt = BlockOptions(), kkt_ordering = MadNLP.MetisOrdering(),
        )
    end

    @testset "madnlp with $(nameof(ls))" for ls in (MadNLP.MumpsSolver, MadNLP.LDLSolver)
        models = (
            () -> MadNLP.SparseWrapperModel(Array, MadNLPTests.HS15Model()),
            () -> MadNLP.SparseWrapperModel(Array, MadNLPTests.DenseDummyQP(zeros(50); m = 20, equality_cons = [1, 4, 7])),
        )
        for model in models, kkt in (MadNLP.SparseKKTSystem, MadNLP.SparseCondensedKKTSystem)
            ref = madnlp(model(); linear_solver = ls, kkt_system = kkt, print_level = MadNLP.ERROR)
            @test ref.status == MadNLP.SOLVE_SUCCEEDED
            N = size(MadNLP.MadNLPSolver(model(); linear_solver = ls, kkt_system = kkt, print_level = MadNLP.ERROR).kkt.aug_com, 1)
            for o in (ORDERINGS..., MadNLP.UserOrdering(randperm(rng, N)))
                res = madnlp(model(); linear_solver = ls, kkt_system = kkt, kkt_ordering = o, print_level = MadNLP.ERROR)
                @test res.status == ref.status
                @test res.solution ≈ ref.solution atol = 1.0e-6
                @test res.objective ≈ ref.objective atol = 1.0e-8
                @test abs(res.iter - ref.iter) <= 2
            end
        end
    end

    @testset "kkt_ordering reaches the linear solver" begin
        solver = MadNLP.MadNLPSolver(MadNLP.SparseWrapperModel(Array, MadNLPTests.HS15Model()); kkt_ordering = MadNLP.MetisOrdering(), print_level = MadNLP.ERROR)
        @test solver.kkt.linear_solver.wrapped isa MadNLP.PermutedSolver
        @test MadNLP.solve!(solver).status == MadNLP.SOLVE_SUCCEEDED
    end

    test_madnlp(
        "METIS ordering with MUMPS",
        () -> MadNLP.Optimizer(kkt_ordering = MadNLP.MetisOrdering(), print_level = MadNLP.ERROR),
        [],
    )
    test_madnlp(
        "AMD ordering under ShiftedPreset",
        () -> MadNLP.Optimizer(preset = MadNLP.ShiftedPreset, kkt_ordering = MadNLP.AMDOrdering(), print_level = MadNLP.ERROR),
        [],
    )
end
