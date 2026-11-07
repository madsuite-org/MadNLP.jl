atol = rtol = 1e-4

m = 2
n = 2
row = Int32[1,2,2]
col = Int32[1,1,2]
val = Float64[1.,.1,2.]

coo = MadNLP.SparseMatrixCOO(m,n,row,col,val)
csc = sparse(row,col,val,m,n)
dense = Array(csc)

b = [1.0,3.0]
x = similar(b)

@testset "SparseMatrixCOO" begin
    @test MadNLP.size(coo) == (2,2)
    @test coo[1,1] == 1.
end

@testset "LAPACK" begin
    sol= [0.8542713567839195, 1.4572864321608041]
    M = MadNLP.LapackCPUSolver(dense)
    MadNLP.introduce(M)
    MadNLP.improve!(M)
    MadNLP.factorize!(M)
    MadNLP.is_inertia(M) && (MadNLP.inertia(M) = (2,0,0))
    x = MadNLP.solve_linear_system!(M,copy(b))
    @test solcmp(x,sol)
end


MadNLPTests.test_linear_solver(LDLSolver,Float32)
MadNLPTests.test_linear_solver(LDLSolver,Float64)
MadNLPTests.test_linear_solver(LDLSolver,Float128)
MadNLPTests.test_linear_solver(UmfpackSolver,Float64)
MadNLPTests.test_linear_solver(CHOLMODSolver,Float64)

import SparseArrays
import SparseArrays: CHOLMOD as SuiteSparseCHOLMOD
using LinearAlgebra: I
using Random: MersenneTwister

@testset "LDLSolver with a failed factorization" begin
    K = sparse(Int32[2, 2], Int32[1, 2], [1.0, 0.0], 2, 2)
    M = MadNLP.create_linear_solver(LDLSolver, K; opt = MadNLP.default_options(LDLSolver))
    MadNLP.factorize!(M)
    @test !MadNLP.is_factorized(M)
    @test MadNLP.inertia(M) == (0, 2, 0)
    x = [1.0, 2.0]
    MadNLP.solve_linear_system!(M, x)
    @test x == [1.0, 2.0]
end

@testset "CHOLMOD index width" begin
    K = sparse(Int32[1, 3, 2, 3, 3], Int32[1, 1, 2, 2, 3], [4.0, 1.0, 3.0, 1.0, -2.0], 3, 3)
    rhs = [1.0, 2.0, 3.0]
    sol = [4.0 0.0 1.0; 0.0 3.0 1.0; 1.0 1.0 -2.0] \ rhs
    for (max_lnz, Ti) in ((Inf, Int32), (0, Int64)), T in (Float32, Float64)
        opt = MadNLP.default_options(CHOLMODSolver)
        opt.cholmod_algorithm = MadNLP.LDL
        A = SparseArrays.SparseMatrixCSC{T, Int32}(K)
        M = MadNLP.create_linear_solver(CHOLMODSolver, A; opt, max_lnz_int32 = max_lnz)
        @test typeof(M.inner).parameters[2] == Ti
        MadNLP.factorize!(M)
        @test MadNLP.inertia(M) == (2, 0, 1)
        x = T.(rhs)
        MadNLP.solve_linear_system!(M, x)
        @test x ≈ sol rtol = sqrt(eps(T))
        # New values reach CHOLMOD and the solve buffers are reused
        A.nzval .*= 2
        MadNLP.factorize!(M)
        x = T.(rhs)
        MadNLP.solve_linear_system!(M, x)
        @test x ≈ sol / 2 rtol = sqrt(eps(T))
    end
    # A fill-reducing ordering that is not the identity, on both factorizations. The dense
    # block makes CHOLMOD choose the supernodal path for Cholesky.
    n = 300
    W = SparseArrays.sprand(MersenneTwister(1), n, n, 0.05)
    W[1:80, 1:80] .= 1.0
    S = W'W + n * SparseArrays.sparse(1.0I, n, n)
    rhs = collect(1.0:n)
    for algorithm in (MadNLP.LDL, MadNLP.CHOLESKY)
        opt = MadNLP.default_options(CHOLMODSolver)
        opt.cholmod_algorithm = algorithm
        A = SparseArrays.SparseMatrixCSC{Float64, Int32}(SparseArrays.tril(S))
        M = MadNLP.create_linear_solver(CHOLMODSolver, A; opt)
        @test M.perm != 1:n
        @test (unsafe_load(SuiteSparseCHOLMOD.typedpointer(M.inner)).is_super != 0) == (algorithm == MadNLP.CHOLESKY)
        MadNLP.factorize!(M)
        @test MadNLP.inertia(M) == (n, 0, 0)
        @test MadNLP.solve_linear_system!(M, copy(rhs)) ≈ Matrix(S) \ rhs
        # CHOLMOD repacks the factor once, at the second factorization
        MadNLP.factorize!(M)
        @test (@allocated MadNLP.factorize!(M)) < 1000
    end
    # Solves reuse their buffers, so a solve does not allocate vectors of length n
    n = 10_000
    D = SparseArrays.spdiagm(0 => ones(n))
    M = MadNLP.create_linear_solver(CHOLMODSolver, SparseArrays.SparseMatrixCSC{Float64, Int32}(D))
    MadNLP.factorize!(M)
    x = ones(n)
    MadNLP.solve_linear_system!(M, x)
    # On Julia 1.10 this solve still allocates about four vectors of length n
    VERSION >= v"1.11" && @test (@allocated MadNLP.solve_linear_system!(M, x)) < 1000
    # A positive definite matrix goes through the supernodal Cholesky path
    P = sparse(Int32[1, 2, 2], Int32[1, 1, 2], [2.0, 1.0, 3.0], 2, 2)
    M = MadNLP.create_linear_solver(CHOLMODSolver, P)
    MadNLP.factorize!(M)
    @test MadNLP.inertia(M) == (2, 0, 0)
    @test MadNLP.solve_linear_system!(M, [1.0, 1.0]) ≈ [2.0 1.0; 1.0 3.0] \ [1.0, 1.0]
    # A failed Cholesky factorization leaves the right-hand side unchanged
    M = MadNLP.create_linear_solver(CHOLMODSolver, K)
    MadNLP.factorize!(M)
    @test MadNLP.inertia(M) == (0, 3, 0)
    @test MadNLP.solve_linear_system!(M, copy(rhs)) == rhs
end

MadNLPTests.test_linear_solver(LapackCPUSolver,Float32)
MadNLPTests.test_linear_solver(LapackCPUSolver,Float64)
