cuda_testset = [
    [
        "CUDSS, Ruiz scaling, dual shift",
        () -> MadNLP.Optimizer(
            kkt_system = MadNLP.SparseKKTSystem,
            linear_solver = MadCore.CUDSSSolver,
            cudss_pivot_epsilon = 1.0e-10,
            kkt_scaler = MadNLP.RuizScaler(),
            dual_shift = 1.0e-10,
            print_level = MadNLP.ERROR,
        ),
        [],
    ],
    [
        "CUDSS, Jacobi scaling",
        () -> MadNLP.Optimizer(
            kkt_system = MadNLP.SparseKKTSystem,
            linear_solver = MadCore.CUDSSSolver,
            kkt_scaler = MadNLP.JacobiScaler(),
            print_level = MadNLP.ERROR,
        ),
        [],
    ],
    # Temporarily commented out since LapackCUDASolver does not currently support sparse callbacks
    [
        "CUDSS",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.CUDSSSolver,
            print_level = MadNLP.ERROR
        ),
        [],
    ],
    [
        "CUDSS-AMD",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.CUDSSSolver,
            print_level = MadNLP.ERROR,
            kkt_ordering = MadCore.AMDOrdering(),
        ),
        [],
    ],
    [
        "CUDSS-METIS",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.CUDSSSolver,
            print_level = MadNLP.ERROR,
            kkt_ordering = MadCore.MetisOrdering(),
        ),
        [],
    ],
    [
        "SDS-METIS",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.SDSSolver,
            print_level = MadNLP.ERROR,
            kkt_ordering = MadCore.MetisOrdering(),
        ),
        [],
    ],
    [
        "CUDSS-IR",
        # cuDSS's refinement fails (CUDSS_STATUS_IR_FAILED) on eigmina when matching and the
        # pivot epsilon of the default preset are both on
        () -> MadNLP.Optimizer(
            kkt_system = MadNLP.SparseCondensedKKTSystem,
            linear_solver = MadCore.CUDSSSolver,
            print_level = MadNLP.ERROR,
            cudss_ir = 1,
            cudss_matching = false,
        ),
        [],
    ],
    [
        "CUDSS-HYBRID",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.CUDSSSolver,
            print_level = MadNLP.ERROR,
            cudss_hybrid_memory = true,
        ),
        [],
    ],
    [
        "CUDSS-NOPIVOTING",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.CUDSSSolver,
            print_level = MadNLP.ERROR,
            cudss_pivoting = false,
        ),
        [],
    ],
    [
        # Bunch-Kaufman pivoting with matching: the analysis reads the values, so it is
        # deferred to the first factorization and redone when a solve is rejected
        "CUDSS-BUNCHKAUFMAN",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.CUDSSSolver,
            print_level = MadNLP.ERROR,
            cudss_pivot_type = 'B',
            cudss_matching = true,
            cudss_deferred_analysis = true,
            cudss_reanalyze_on_improve = true,
        ),
        [],
    ],
    [
        "CUDSS-ANALYSIS-EACH",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.CUDSSSolver,
            print_level = MadNLP.ERROR,
            cudss_matching = true,
            cudss_analysis_each_factorization = true,
        ),
        [],
    ],
    [
        "CUDSS-NOREFACTORIZATION",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.CUDSSSolver,
            print_level = MadNLP.ERROR,
            cudss_pivot_type = 'B',
            cudss_refactorization = false,
        ),
        [],
    ],
    [
        "Formulation K2",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.CUDSSSolver,
            print_level = MadNLP.ERROR,
            kkt_system = MadNLP.SparseKKTSystem,
        ),
        [],
    ],
    [
        "Formulation K2.5",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.CUDSSSolver,
            print_level = MadNLP.ERROR,
            kkt_system = MadNLP.ScaledSparseKKTSystem,
        ),
        [],
    ],
    [
        "LapackGPU-BUNCHKAUFMAN",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.LapackCUDASolver,
            lapack_algorithm = MadNLP.BUNCHKAUFMAN,
            print_level = MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackGPU-LU",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.LapackCUDASolver,
            lapack_algorithm = MadNLP.LU,
            print_level = MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackGPU-QR",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.LapackCUDASolver,
            lapack_algorithm = MadNLP.QR,
            print_level = MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackGPU-EVD",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.LapackCUDASolver,
            lapack_algorithm = MadNLP.EVD,
            print_level = MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackGPU-CHOLESKY",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.LapackCUDASolver,
            lapack_algorithm = MadNLP.CHOLESKY,
            print_level = MadNLP.ERROR,
        ),
        ["infeasible", "lootsma", "eigmina", "lp_examodels_issue75"], # KKT system not PD
    ],
    [
        "SDS",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.SDSSolver,
            print_level = MadNLP.ERROR,
        ),
        [],
    ],
    [
        "SDS-BUNCHKAUFMAN",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.SDSSolver,
            print_level = MadNLP.ERROR,
            preset = MadNLP.AugmentedPreset,
        ),
        [],
    ],
    [
        "SDS-ANALYSIS-EACH",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.SDSSolver,
            print_level = MadNLP.ERROR,
            sds_matching = "algo6",
            sds_analysis_each_factorization = true,
        ),
        [],
    ],
    [
        "SDS-CONDENSED",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.SDSSolver,
            print_level = MadNLP.ERROR,
            preset = MadNLP.CondensedPreset,
        ),
        [],
    ],
]

rocm_testset = [
    [
        "LapackROCmSolver-LU",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.LapackROCmSolver,
            lapack_algorithm = MadNLP.LU,
            print_level = MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackROCmSolver-QR",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.LapackROCmSolver,
            lapack_algorithm = MadNLP.QR,
            print_level = MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackROCmSolver-EVD",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.LapackROCmSolver,
            lapack_algorithm = MadNLP.EVD,
            print_level = MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackROCmSolver-CHOLESKY",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.LapackROCmSolver,
            lapack_algorithm = MadNLP.CHOLESKY,
            print_level = MadNLP.ERROR,
        ),
        ["infeasible", "lootsma", "eigmina", "lp_examodels_issue75"], # KKT system not PD
    ],
    [
        "SDS",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.SDSSolver,
            print_level = MadNLP.ERROR,
        ),
        [],
    ],
    [
        "SDS-BUNCHKAUFMAN",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.SDSSolver,
            print_level = MadNLP.ERROR,
            preset = MadNLP.AugmentedPreset,
        ),
        [],
    ],
    [
        "SDS-ANALYSIS-EACH",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.SDSSolver,
            print_level = MadNLP.ERROR,
            sds_matching = "algo6",
            sds_analysis_each_factorization = true,
        ),
        [],
    ],
    [
        "SDS-CONDENSED",
        () -> MadNLP.Optimizer(
            linear_solver = MadCore.SDSSolver,
            print_level = MadNLP.ERROR,
            preset = MadNLP.CondensedPreset,
        ),
        [],
    ],
]

@testset "GPU solvers" begin
    if HAS_CUDA && CUDA.functional()
        MadNLPTests.test_linear_solver(MadCore.LapackCUDASolver, Float32)
        MadNLPTests.test_linear_solver(MadCore.LapackCUDASolver, Float64)
        # Test LapackGPU wrapper
        for (name, optimizer_constructor, exclude) in cuda_testset
            test_madnlp(name, optimizer_constructor, exclude; Arr = CuArray)
        end
    end
    if HAS_AMDGPU && AMDGPU.functional()
        MadNLPTests.test_linear_solver(MadCore.LapackROCmSolver, Float32)
        MadNLPTests.test_linear_solver(MadCore.LapackROCmSolver, Float64)
        for (name, optimizer_constructor, exclude) in rocm_testset
            test_madnlp(name, optimizer_constructor, exclude; Arr = ROCArray)
        end
    end
end

# One scaling sweep on the device against the host, over a pattern with both triangles that has
# a column longer than MadCore.SCALING_LONG_COLUMN (its own work group) and many short ones.
function test_scaling_sweep(Arr)
    MC = MadNLP.MadCore
    n = 3000
    nlong = MC.SCALING_LONG_COLUMN + 1000
    A = SparseArrays.spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
    A[2:nlong, 1] .= 1.0
    A[1, 2:nlong] .= 1.0
    long_cols = findall(>(MC.SCALING_LONG_COLUMN), diff(A.colptr))
    @test long_cols == [1]
    absnz = rand(SparseArrays.nnz(A)) .+ 0.1
    d = rand(n) .+ 0.5
    dnew = similar(d)
    MC.scaling_sweep!(dnew, d, A.colptr, A.rowval, absnz, long_cols)
    dnew_device = Arr(zeros(n))
    MC.scaling_sweep!(dnew_device, Arr(d), Arr(Int32.(A.colptr)), Arr(Int32.(A.rowval)), Arr(absnz), Arr(Int32.(long_cols)))
    return @test Array(dnew_device) ≈ dnew
end

@testset "Scaling sweep with a long column" begin
    HAS_CUDA && CUDA.functional() && test_scaling_sweep(CuArray)
    HAS_AMDGPU && AMDGPU.functional() && test_scaling_sweep(ROCArray)
end

# A given permutation, to check that the linear solver factorizes in the order it is handed.
struct GivenOrdering <: MadCore.AbstractOrdering
    perm::Vector{Int}
end
MadCore.ordering_permutation(o::GivenOrdering, ::SparseArrays.SparseMatrixCSC) = o.perm

# The lower triangle of an arrowhead matrix whose dense row and column come first: eliminated
# first, it fills the whole factor; eliminated last, it fills nothing.
function arrowhead(n)
    d = [i == 1 ? 4.0n : (iseven(i) ? -4.0 : 4.0) for i in 1:n]
    return SparseArrays.sparse(vcat(1:n, 2:n), vcat(1:n, fill(1, n - 1)), vcat(d, ones(n - 1)), n, n)
end

function test_given_order(ls, factor_size)
    n = 40
    A = arrowhead(n)
    K = A + copy(A') - SparseArrays.spdiagm(0 => [A[i, i] for i in 1:n])
    b = randn(n)
    sizes = map((vcat(2:n, 1), collect(1:n))) do p
        S = MadCore.create_kkt_linear_solver(ls, CUDA.CUSPARSE.CuSparseMatrixCSC{Float64, Cint}(A), n; kkt_ordering = GivenOrdering(p))
        MadCore.factorize!(S)
        x = Array(MadCore.solve_linear_system!(S, CuVector(b)))
        @test K * x ≈ b
        @test MadCore.inertia(S) == (n ÷ 2, 0, n ÷ 2)
        factor_size(S)
    end
    # the dense column last fills nothing, first the whole factor: the solver keeps the order
    # it is handed, where its own ordering would have avoided the fill of the second
    @test sizes[1] < 3n
    return @test sizes[2] > n^2 / 4
end

@testset "kkt_ordering is the pivot order of the GPU solver" begin
    if HAS_CUDA && CUDA.functional()
        @testset "CUDSSSolver" test_given_order(MadCore.CUDSSSolver, S -> CUDSS.cudss_get(S.inner, "lu_nnz"))
        @testset "SDSSolver" test_given_order(MadCore.SDSSolver, S -> SparseDirectSolver.getparam(S.inner, "lu_nnz"))
    end
end
