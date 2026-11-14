# AccelerateSolver: the sparse LDLᵀ of Apple Accelerate (macOS only). Loading
# AppleAccelerate also forwards BLAS and LAPACK to Accelerate, so this file runs last.
using AppleAccelerate, LinearAlgebra, Random, SparseArrays
const AA = AppleAccelerate

@testset "AccelerateSolver" begin
    for T in (Float32, Float64)
        MadNLPTests.test_linear_solver(MadNLP.AccelerateSolver, T)
    end

    @testset "Inertia" begin
        kind = AA.SparseFactorizationLDLTTPP
        rng = Random.Xoshiro(1)
        for trial in 1:10
            n = 40
            K = sprandn(rng, n, n, 0.15)
            K = K + K' + spdiagm(randn(rng, n))
            A = SparseArrays.SparseMatrixCSC{Float64, Int32}(sparse(tril(K)))
            opt = MadNLP.default_options(MadNLP.AccelerateSolver)
            opt.accelerate_factorization = kind
            M = MadNLP.create_linear_solver(MadNLP.AccelerateSolver, A; opt)
            @test MadNLP.is_inertia(M)
            ev = eigvals(Symmetric(Matrix(K)))
            for _ in 1:2 # the first factorization, then a refactorization
                MadNLP.factorize!(M)
                @test MadNLP.inertia(M) == (count(>(0), ev), 0, count(<(0), ev))
                b = rand(rng, n)
                x = copy(b)
                MadNLP.solve_linear_system!(M, x)
                @test Symmetric(Matrix(A), :L) * x ≈ b rtol = 1.0e-8
                nonzeros(A) .*= 2
                ev .*= 2
            end
        end
    end

    @testset "No inertia, $kind" for kind in (AA.SparseFactorizationLDLTSBK, AA.SparseFactorizationLDLTUnpivoted)
        opt = MadNLP.default_options(MadNLP.AccelerateSolver)
        opt.accelerate_factorization = kind
        M = MadNLP.create_linear_solver(MadNLP.AccelerateSolver, sparse(tril(sparse(1.0I, 3, 3))); opt)
        @test !MadNLP.is_inertia(M)
    end

    @testset "Edge cases, $T" for T in (Float32, Float64)
        build(A; kw...) = begin
            opt = MadNLP.default_options(MadNLP.AccelerateSolver)
            for (k, v) in kw
                setproperty!(opt, k, v)
            end
            MadNLP.create_linear_solver(MadNLP.AccelerateSolver, A; opt)
        end
        # indefinite: inertia in single and double precision
        A = SparseArrays.SparseMatrixCSC{T, Int32}(sparse(tril(T[2 1 0; 1 -3 1; 0 1 4])))
        M = build(A)
        MadNLP.factorize!(M)
        @test MadNLP.inertia(M) == (2, 0, 1)
        # Cholesky fails on it, then recovers on a positive definite matrix
        M = build(A; accelerate_factorization = AA.SparseFactorizationCholesky)
        MadNLP.factorize!(M)
        @test !MadNLP.is_factorized(M)
        @test MadNLP.inertia(M) == (0, 3, 0)
        nonzeros(A) .= T[2, 1, 3, 1, 4]
        MadNLP.factorize!(M)
        @test MadNLP.is_factorized(M)
        @test MadNLP.inertia(M) == (3, 0, 0)
        x = T[1, 2, 3]
        MadNLP.solve_linear_system!(M, x)
        @test Symmetric(Matrix(A), :L) * x ≈ T[1, 2, 3] rtol = sqrt(eps(T))
        # a singular matrix reports a zero pivot rather than failing
        S = SparseArrays.SparseMatrixCSC{T, Int32}(sparse(tril(T[1 0 0; 0 2 0; 0 0 0])))
        M = build(S)
        MadNLP.factorize!(M)
        @test MadNLP.inertia(M) == (2, 1, 0)
    end

    @testset "improve! raises the pivot threshold, $T" for T in (Float32, Float64)
        n = 30
        K = spdiagm(-1 => ones(n - 1), 0 => [(-1)^i * (2 + i / n) for i in 1:n], 1 => ones(n - 1))
        A = SparseArrays.SparseMatrixCSC{T, Int32}(sparse(tril(K)))
        # a zero tolerance above every pivot makes them all zero, so the inertia shows
        # which options the factorization used
        opt = MadNLP.default_options(MadNLP.AccelerateSolver)
        M = MadNLP.create_linear_solver(MadNLP.AccelerateSolver, A; opt)
        MadNLP.factorize!(M)
        @test MadNLP.inertia(M) == (n ÷ 2, 0, n ÷ 2)
        M.opt.accelerate_zero_tolerance = 1.0e3   # read by improve!
        @test MadNLP.improve!(M)
        @test MadNLP.inertia(M) == (n ÷ 2, 0, n ÷ 2)  # the current factors are kept
        MadNLP.factorize!(M)
        @test MadNLP.inertia(M) == (0, n, 0)          # the next factorization uses the new options
        MadNLP.factorize!(M)
        @test MadNLP.inertia(M) == (0, n, 0)          # and so do the refactorizations
        x = T.(1:n)
        @test MadNLP.solve_linear_system!(M, copy(x)) isa Vector{T}
        # up to accelerate_pivtolmax
        while MadNLP.improve!(M) end
        @test M.nfoptions.pivotTolerance == M.opt.accelerate_pivtolmax
    end

    @testset "Options" begin
        nlp = MadNLPTests.HS15Model()
        r0 = madnlp(nlp; print_level = MadNLP.ERROR)
        for opts in (
                (;),
                (; accelerate_order = AA.SparseOrderMetis, accelerate_scaling = AA.SparseScalingUser),
                (; accelerate_order = AA.SparseOrderAMD, accelerate_pivtol = 1.0e-2),
                (; accelerate_factorization = AA.SparseFactorizationLDLTSBK),
            )
            r = madnlp(
                MadNLP.SparseWrapperModel(Vector{Float64}, nlp);
                linear_solver = MadNLP.AccelerateSolver, print_level = MadNLP.ERROR, opts...,
            )
            @test r.status == MadNLP.SOLVE_SUCCEEDED
            @test r.objective ≈ r0.objective
        end
    end

    MadNLPTests.test_madnlp(
        "AccelerateSolver",
        () -> MadNLP.Optimizer(linear_solver = MadNLP.AccelerateSolver, print_level = MadNLP.ERROR),
        [],
    )
end
