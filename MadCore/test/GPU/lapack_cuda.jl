# The LAPACK solver on CUDA: the device follows the input matrix.
using CUDA, LinearAlgebra, SparseArrays

@testset "LapackSolver on CUDA" begin
    n = 20
    for T in (Float32, Float64), alg in (MadCore.BUNCHKAUFMAN, MadCore.LU, MadCore.QR, MadCore.EVD, MadCore.CHOLESKY), legacy in (true, false)
        B = rand(T, n, n)
        A = Matrix(Symmetric(B * B' + n * I))
        alg == MadCore.CHOLESKY || (A[1, 1] = -A[1, 1])
        opt = MadCore.LapackOptions(lapack_algorithm = alg)
        M = MadCore.LapackSolver(CuMatrix(LowerTriangular(A)); opt, legacy)
        @test M.fact isa CuMatrix{T} && !(M isa MadCore.LapackCPUSolver)
        MadCore.factorize!(M)
        x = CUDA.ones(T, n)
        MadCore.solve_linear_system!(M, x)
        @test A * Array(x) ≈ ones(T, n) rtol = (T == Float32 ? 1.0e-3 : 1.0e-9)
    end
    A = Matrix(Symmetric(rand(8, 8) + 8I))
    L = Matrix(LowerTriangular(A))
    # A sparse device matrix is copied into a dense device factor.
    M = MadCore.create_linear_solver(MadCore.LapackCUDASolver, CUDA.CUSPARSE.CuSparseMatrixCSC(sparse(L)))
    @test M.fact isa CuMatrix{Float64}
    MadCore.factorize!(M)
    x = CUDA.ones(8)
    MadCore.solve_linear_system!(M, x)
    @test A * Array(x) ≈ ones(8)
    # The marker does not choose the device: a host matrix gets the host solver, and
    # LapackCPUSolver factorizes a device matrix on the host.
    @test MadCore.create_linear_solver(MadCore.LapackCUDASolver, L) isa MadCore.LapackCPUSolver
    M = MadCore.create_linear_solver(MadCore.LapackCPUSolver, CuMatrix(L))
    @test M.fact isa Matrix{Float64}
    MadCore.factorize!(M)
    x = CUDA.ones(8)
    MadCore.solve_linear_system!(M, x)
    @test A * Array(x) ≈ ones(8)
end
