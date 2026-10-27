cuda_testset = [
    # Temporarily commented out since LapackCUDASolver does not currently support sparse callbacks
    [
        "CUDSS",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR
        ),
        [],
    ],
    [
        "CUDSS-AMD",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_ordering=MadNLPGPU.AMD_ORDERING,
        ),
        [],
    ],
    [
        "CUDSS-SYMAMD",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_ordering=MadNLPGPU.SYMAMD_ORDERING,
        ),
        [],
    ],
    [
        "CUDSS-COLAMD",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_ordering=MadNLPGPU.COLAMD_ORDERING,
        ),
        [],
    ],
    [
        "CUDSS-METIS",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_ordering=MadNLPGPU.METIS_ORDERING,
        ),
        [],
    ],
    [
        "CUDSS-IR",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_ir=1,
        ),
        [],
    ],
    [
        "CUDSS-HYBRID",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_hybrid_memory=true,
        ),
        [],
    ],
    [
        "CUDSS-NOPIVOTING",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
            cudss_pivoting=false,
        ),
        [],
    ],
    [
        "Formulation K2",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
            kkt_system=MadNLP.SparseKKTSystem,
        ),
        [],
    ],
    [
        "Formulation K2.5",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
            kkt_system=MadNLP.ScaledSparseKKTSystem,
        ),
        [],
    ],
    [
        "LapackGPU-BUNCHKAUFMAN",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.LapackCUDASolver,
            lapack_algorithm=MadNLP.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackGPU-LU",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.LapackCUDASolver,
            lapack_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackGPU-QR",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.LapackCUDASolver,
            lapack_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackGPU-EVD",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.LapackCUDASolver,
            lapack_algorithm=MadNLP.EVD,
            print_level=MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackGPU-CHOLESKY",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.LapackCUDASolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR,
        ),
        ["infeasible", "lootsma", "eigmina", "lp_examodels_issue75"], # KKT system not PD
    ],
]

rocm_testset = [
    [
        "LapackROCmSolver-LU",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.LapackROCmSolver,
            lapack_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackROCmSolver-QR",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.LapackROCmSolver,
            lapack_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackROCmSolver-EVD",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.LapackROCmSolver,
            lapack_algorithm=MadNLP.EVD,
            print_level=MadNLP.ERROR,
        ),
        [],
    ],
    [
        "LapackROCmSolver-CHOLESKY",
        ()->MadNLP.Optimizer(
            linear_solver = MadNLPGPU.LapackROCmSolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR,
        ),
        ["infeasible", "lootsma", "eigmina", "lp_examodels_issue75"], # KKT system not PD
    ],
]

@testset "MadNLPGPU test" begin
    if HAS_CUDA && CUDA.functional()
        MadNLPTests.test_linear_solver(MadNLPGPU.LapackCUDASolver, Float32)
        MadNLPTests.test_linear_solver(MadNLPGPU.LapackCUDASolver, Float64)
        # Test LapackGPU wrapper
        for (name,optimizer_constructor,exclude) in cuda_testset
            test_madnlp(name,optimizer_constructor,exclude; Arr=CuArray)
        end
    end
    if HAS_AMDGPU && AMDGPU.functional()
        MadNLPTests.test_linear_solver(MadNLPGPU.LapackROCmSolver, Float32)
        MadNLPTests.test_linear_solver(MadNLPGPU.LapackROCmSolver, Float64)
        for (name,optimizer_constructor,exclude) in rocm_testset
            test_madnlp(name,optimizer_constructor,exclude; Arr=ROCArray)
        end
    end
end
