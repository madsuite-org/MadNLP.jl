using LinearAlgebra, SparseArrays, Random
import AMD, Metis  # load MadCoreAMDExt and MadCoreMetisExt

const ORDERINGS = (MadNLP.MetisOrdering(), MadNLP.AMDOrdering())

# A given permutation, to check that the linear solver factorizes in the order it is handed.
struct GivenOrdering <: MadNLP.AbstractOrdering
    perm::Vector{Int}
end
MadNLP.ordering_permutation(o::GivenOrdering, ::SparseMatrixCSC) = o.perm

# The lower triangle of a random symmetric indefinite matrix [H Jᵀ; J -D], every diagonal
# entry stored, as a sparse KKT system assembles it.
function random_kkt(rng, n, m)
    H = sprandn(rng, n, n, 0.2)
    H = H + H' + 4n * I
    J = sprandn(rng, m, n, 0.3) + sparse(1:m, 1:m, ones(m), m, n)
    K = [H J'; J -Diagonal(rand(rng, m) .+ 1)]
    return SparseMatrixCSC{Float64, Int32}(sparse(LowerTriangular(K)))
end

# The lower triangle of an arrowhead matrix whose dense row and column come first: eliminated
# first, it fills the whole factor; eliminated last, it fills nothing.
function arrowhead(n; definite = false)
    d = [i == 1 ? 4.0n : (!definite && iseven(i) ? -4.0 : 4.0) for i in 1:n]
    return SparseMatrixCSC{Float64, Int32}(sparse(vcat(1:n, 2:n), vcat(1:n, fill(1, n - 1)), vcat(d, ones(n - 1)), n, n))
end

const MUMPSExt = Base.get_extension(MadNLP.MadCore, :MadCoreMUMPSExt)
const LDLExt = Base.get_extension(MadNLP.MadCore, :MadCoreLDLExt)
const SuiteSparseExt = Base.get_extension(MadNLP.MadCore, :MadCoreSuiteSparseExt)

# The entries of the factor, as each solver reports them (MUMPS: its real space).
factor_size(S::MUMPSExt.MumpsState) = S.mumps_struc.infog[9]
factor_size(S::LDLExt.LDLState) = S.inner.Lp[end] - 1
factor_size(S::SuiteSparseExt.UmfpackState) = nnz(S.inner.L)
factor_size(S::SuiteSparseExt.CHOLMODState) = nnz(S.inner)
# the option that holds the permutation
user_perm(opt::MUMPSExt.MumpsOptions) = opt.mumps_user_perm
user_perm(opt::LDLExt.LDLFactorizationsOptions) = opt.ldl_user_perm
user_perm(opt::SuiteSparseExt.UmfpackOptions) = opt.umfpack_user_perm
user_perm(opt::SuiteSparseExt.CHOLMODOptions) = opt.cholmod_user_perm
user_perm(S::MadNLP.MadCore.AbstractLinearSolver) = user_perm(S.opt)

const SOLVERS = (MadNLP.MumpsSolver, MadNLP.LDLSolver, MadNLP.UmfpackSolver, MadNLP.CHOLMODSolver)
# the matrices each solver factorizes: CHOLMOD's Cholesky needs a definite one
definite(ls) = ls == MadNLP.CHOLMODSolver

@testset "Fill-reducing orderings" begin
    rng = MersenneTwister(1)
    n, m = 30, 12
    A = random_kkt(rng, n, m)
    K = Symmetric(A, :L)
    b = randn(rng, n + m)

    @testset "permutations" begin
        for o in (ORDERINGS..., MadNLP.DefaultOrdering())
            p = MadNLP.ordering_permutation(o, A)
            @test p isa Vector{Int}
            @test length(p) == n + m && isperm(p)
        end
    end

    @testset "$(nameof(ls)) factorizes in the given order" for ls in SOLVERS
        N = 40
        B = arrowhead(N; definite = definite(ls))
        x0 = MadNLP.solve_linear_system!(MadNLP.factorize!(MadNLP.create_linear_solver(ls, B)), ones(N))
        sizes = map((vcat(2:N, 1), collect(1:N))) do p
            S = MadNLP.create_kkt_linear_solver(ls, copy(B), N; kkt_ordering = GivenOrdering(p))
            @test user_perm(S) == p
            MadNLP.factorize!(S)
            @test MadNLP.solve_linear_system!(S, ones(N)) ≈ x0
            factor_size(S)
        end
        # the dense column last fills nothing, first the whole factor (or, with UMFPACK's row
        # pivoting, much of it): the solver keeps the order it is handed
        @test 2 * sizes[1] < sizes[2]
    end

    @testset "$(nameof(ls)) with $(nameof(typeof(o)))" for ls in SOLVERS, o in ORDERINGS
        B = definite(ls) ? SparseMatrixCSC{Float64, Int32}(sparse(LowerTriangular(Matrix(K)^2 + I))) : A
        opt = MadNLP.default_options(ls)
        S = MadNLP.create_kkt_linear_solver(ls, copy(B), n; opt, kkt_ordering = o)
        @test user_perm(S) == MadNLP.ordering_permutation(o, B)
        @test user_perm(opt) === nothing  # the options of the KKT system are not changed
        MadNLP.factorize!(S)
        @test MadNLP.is_factorized(S)
        @test Symmetric(B, :L) * MadNLP.solve_linear_system!(S, copy(b)) ≈ b
        MadNLP.is_inertia(S) && !definite(ls) && @test MadNLP.inertia(S) == (n, 0, m)
    end

    @testset "ordering inside the scaled and shifted factorization" begin
        S = MadNLP.create_kkt_linear_solver(
            MadNLP.MumpsSolver, A, n; kkt_ordering = MadNLP.AMDOrdering(), kkt_scaler = MadNLP.RuizScaler(), dual_shift = 1.0e-10,
        )
        @test S isa MadNLP.ScaledShiftedSolver
        @test user_perm(S.inner) == MadNLP.ordering_permutation(MadNLP.AMDOrdering(), A)
        MadNLP.factorize!(S)
        @test MadNLP.inertia(S) == (n, 0, m)
        @test K * MadNLP.solve_linear_system!(S, copy(b)) ≈ b rtol = 1.0e-6
    end

    @testset "a solver without a user permutation" begin
        @test_throws ArgumentError MadNLP.create_kkt_linear_solver(
            MadNLP.LapackCPUSolver, A, n; kkt_ordering = MadNLP.MetisOrdering(),
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
            for o in ORDERINGS
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
        @test user_perm(solver.kkt.linear_solver.wrapped) isa Vector{Int}
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
