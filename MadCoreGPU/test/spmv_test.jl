# MadCore._spmv! on device matrices, through KrylovPreconditioners' operators, against the host.
using SparseArrays, LinearAlgebra
function test_spmv(Arr, SpMat)
    MadCore = MadNLP.MadCore
    m, n = 300, 200
    A = sprand(m, n, 0.05)
    dA = SpMat(A)
    x, y = rand(n), rand(m)
    xt, yt = rand(m), rand(n)
    α, β = 0.7, -1.3
    @test Array(MadCore._spmv!(Arr(copy(y)), dA, Arr(x), α, β)) ≈ α * A * x + β * y
    @test Array(MadCore._spmv!(Arr(copy(yt)), dA', Arr(xt), α, β)) ≈ α * A' * xt + β * yt
    # contiguous views of longer vectors, as the KKT vectors are passed
    ybuf = Arr([zeros(5); y; zeros(3)])
    xbuf = Arr([zeros(2); x])
    MadCore._spmv!(view(ybuf, 6:(5 + m)), dA, view(xbuf, 3:(2 + n)), α, β)
    @test Array(ybuf)[6:(5 + m)] ≈ α * A * x + β * y
    @test all(iszero, Array(ybuf)[1:5]) && all(iszero, Array(ybuf)[(6 + m):end])
    # the products ran through kept operators, not the fallback
    ext = Base.get_extension(MadCoreGPU, :MadCoreGPUKrylovPreconditionersExt)
    @test ext !== nothing
    entry = ext._OPERATORS[objectid(dA)]
    return @test entry.ops['N'] !== nothing && entry.ops['C'] !== nothing
end

if HAS_CUDA && CUDA.functional()
    @testset "SpMV through KrylovOperator (CUDA)" test_spmv(CuArray, CUDA.CUSPARSE.CuSparseMatrixCSC)
end
if HAS_AMDGPU && AMDGPU.functional()
    @testset "SpMV through KrylovOperator (ROCm)" test_spmv(ROCArray, AMDGPU.rocSPARSE.ROCSparseMatrixCSC)
end
