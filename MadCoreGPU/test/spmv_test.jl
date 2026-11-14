# hess_op and jac_op of a GPU condensed KKT system, through KrylovPreconditioners' operators,
# against the host.
using SparseArrays, LinearAlgebra
function test_kept_operators(Arr, SpMat)
    m, n = 300, 200
    A = sprand(m, n, 0.05)
    dA = SpMat(A)
    α, β = 0.7, -1.3
    ext = Base.get_extension(MadCoreGPU, :MadCoreGPUKrylovPreconditionersExt)
    @test ext !== nothing
    for (transa, opA) in (('N', A), ('T', transpose(A)))
        op = MadCoreGPU.kept_op(dA, transa)
        @test op isa ext.KeptOperator            # the products run through a kept operator
        p, q = size(opA)
        x, y = rand(q), rand(p)
        @test Array(mul!(Arr(copy(y)), op, Arr(x), α, β)) ≈ α * opA * x + β * y
        # contiguous views of longer vectors, as the KKT vectors are passed
        ybuf = Arr([zeros(5); y; zeros(3)])
        xbuf = Arr([zeros(2); x])
        mul!(view(ybuf, 6:(5 + p)), op, view(xbuf, 3:(2 + q)), α, β)
        @test Array(ybuf)[6:(5 + p)] ≈ α * opA * x + β * y
        @test all(iszero, Array(ybuf)[1:5]) && all(iszero, Array(ybuf)[(6 + p):end])
    end
    return
end

if HAS_CUDA && CUDA.functional()
    @testset "Kept sparse operators (CUDA)" test_kept_operators(CuArray, CUDA.CUSPARSE.CuSparseMatrixCSC)
end
if HAS_AMDGPU && AMDGPU.functional()
    @testset "Kept sparse operators (ROCm)" test_kept_operators(ROCArray, AMDGPU.rocSPARSE.ROCSparseMatrixCSC)
end
