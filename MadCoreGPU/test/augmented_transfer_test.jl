# The augmented KKT matrix assembled on the device equals the host's. HS15 has a bounded
# variable, so the COO form of the matrix carries two entries in a diagonal slot (the Hessian
# diagonal and the primal-dual diagonal), which the device transfer! must sum.
_host_csc(A::SparseArrays.SparseMatrixCSC) = A
_host_csc(A) = SparseArrays.SparseMatrixCSC(size(A)..., Array(A.colPtr), Array(A.rowVal), Array(A.nzVal))

function test_augmented_transfer(AT, linear_solver)
    return @testset "Augmented KKT matrix on $(nameof(AT))" begin
        MNC = MadNLP.MadNLPCore
        nlp = MNC.HS15Model()
        kw = (; kkt_system = MadNLP.SparseKKTSystem, preset = MadNLP.AugmentedPreset, print_level = MadNLP.ERROR)
        host = MadNLP.MadNLPSolver(nlp; linear_solver = MadNLP.LapackCPUSolver, kw...)
        device = MadNLP.MadNLPSolver(MadNLP.SparseWrapperModel(AT, nlp); linear_solver, kw...)
        Ks = map((host, device)) do solver
            MNC.initialize!(solver)
            MNC.set_aug_diagonal!(solver.kkt, solver)
            MadNLP.MadCore.build_kkt!(solver.kkt)
            return Matrix(_host_csc(solver.kkt.aug_com))
        end
        @test size(Ks[1]) == size(Ks[2])
        @test any(!iszero, Ks[1])
        @test Ks[2] ≈ Ks[1]
    end
end
