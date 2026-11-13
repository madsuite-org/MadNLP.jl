using Test, MadNLP, MadCoreGPU, MadNLPTests
import SparseArrays

# The GPU entry points set the option presets for GPU models; CI installs the one
# matching the runner's GPU.
const HAS_CUDA = try
    using CUDA, CUDSS, cuMadNLP
    true
catch
    false
end

const HAS_AMDGPU = try
    using AMDGPU, rocMadNLP
    true
catch
    false
end

# defines SDS_TESTS
include("sds_test.jl")

@testset "MadCoreGPU test" begin
    include("cpu_test.jl")
    include("gpu_test.jl")
    include("timer_test.jl")
    include("augmented_transfer_test.jl")
    if HAS_CUDA && CUDA.functional()
        test_augmented_transfer(CuArray, MadCoreGPU.LapackCUDASolver)
        test_gpu_presets(CuArray, Float64)
        if SDS_TESTS
            test_sds_presets(CuArray, Float64)
            for T in (Float32, Float64)
                test_sds_linear_solver(CuArray, T)
            end
            test_sds_against(CuArray, MadCoreGPU.CUDSSSolver)
        end
        include("densekkt_cuda.jl")
        include("condensed_mul_cuda.jl")
        include("condensed_border_cuda.jl")
        # Need to add support for CompactLBFGS in SparseCondensedKKTSystem (Issue #563)
        # include("sparsekkt_cuda.jl")
        include("schur_cuda_test.jl")
        include("zero_compile_cuda.jl")
    end
    if HAS_AMDGPU && AMDGPU.functional()
        test_augmented_transfer(ROCArray, MadCoreGPU.LapackROCmSolver)
        test_gpu_presets(ROCArray, Float64)
        if SDS_TESTS
            test_sds_presets(ROCArray, Float64)
            for T in (Float32, Float64)
                test_sds_linear_solver(ROCArray, T)
            end
        end
        include("densekkt_rocm.jl")
        # Need to add support for CompactLBFGS in SparseCondensedKKTSystem (Issue #563)
        # include("sparsekkt_rocm.jl")
    end
end
