using Test, MadNLP, MadNLPGPU, MadNLPTests

const HAS_CUDA = try
    using CUDA
    using CUDSS
    true
catch
    false
end

const HAS_AMDGPU = try
    using AMDGPU
    true
catch
    false
end

@testset "MadNLPGPU test" begin
    include("madnlpgpu_test.jl")
    if HAS_CUDA && CUDA.functional()
        include("densekkt_cuda.jl")
        include("condensed_mul_cuda.jl")
        include("condensed_border_cuda.jl")
        # Need to add support for CompactLBFGS in SparseCondensedKKTSystem (Issue #563)
        # include("sparsekkt_cuda.jl")
        include("schur_cuda_test.jl")
    end
    if HAS_AMDGPU && AMDGPU.functional()
        include("densekkt_rocm.jl")
        # Need to add support for CompactLBFGS in SparseCondensedKKTSystem (Issue #563)
        # include("sparsekkt_rocm.jl")
    end
end
