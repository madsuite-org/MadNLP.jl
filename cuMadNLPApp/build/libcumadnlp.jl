# The image of libcumadnlp: cuMadNLPApp, and the device path cuMadNLPAppWarmup's workload
# compiled, plus the method instances traced runs of a built library still compiled
# (precompile_statements.jl, collected with CUMADNLPAPP_TRACE=1).
using CUDACore, cuMadNLP, cuMadNLPApp, cuMadNLPAppWarmup
const GPUCompiler = first(m for m in Base.loaded_modules_array() if nameof(m) == :GPUCompiler)
const GPUToolbox = first(m for m in Base.loaded_modules_array() if nameof(m) == :GPUToolbox)
# CUDA.jl keys launched kernels by (objectid, hash, function) and compares keys with
# isequal specialized on the kernel function type: precompile it for every GPU kernel
# function (KernelAbstractions names them gpu_*) of the loaded modules.
for m in Base.loaded_modules_array(), n in names(m; all = true)
    startswith(string(n), "gpu_") && isdefined(m, n) || continue
    f = getfield(m, n)
    f isa Function || continue
    K = Tuple{UInt64, UInt64, typeof(f)}
    precompile(isequal, (K, K))
end
# the kernel hook of a traced device solve
precompile(cuMadNLPApp.kernel_hook, (GPUCompiler.CompilerJob{GPUCompiler.PTXCompilerTarget, CUDACore.CUDACompilerParams},))
# reached only in some runs: a long GPU synchronize waits through the CUDA worker
precompile(Tuple{typeof(GPUToolbox.perform!), GPUToolbox.WaitRequest{typeof(CUDACore.worker_synchronize), Tuple{CUDACore.CuStream, CUDACore.CuContext}, CUDACore.cudaError_enum}})

cuMadNLPApp.MadNLPAppCore.precompile_image(joinpath(@__DIR__, "precompile_statements.jl"))
