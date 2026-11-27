# The image of libmadnlp: MadNLPApp, whose precompile workload (src/warmup.jl) compiled the
# CPU solve path, and the device path MadNLPAppCUDAWarmup's workload compiled, plus the method
# instances that traced runs of a built library still compiled (precompile_statements.jl,
# collected with MADNLPAPP_TRACE=1; see check.sh).
using CUDACore, cuMadNLP, MadNLPApp, MadNLPAppCUDAWarmup
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
precompile(MadNLPApp.kernel_hook, (GPUCompiler.CompilerJob{GPUCompiler.PTXCompilerTarget, CUDACore.CUDACompilerParams},))
# reached only in some runs: a long GPU synchronize waits through the CUDA worker
precompile(Tuple{typeof(GPUToolbox.perform!), GPUToolbox.WaitRequest{typeof(CUDACore.worker_synchronize), Tuple{CUDACore.CuStream, CUDACore.CuContext}, CUDACore.cudaError_enum}})


module PrecompileStatements end
let file = joinpath(@__DIR__, "precompile_statements.jl")
    isfile(file) && MadNLPApp.CNLPLib.eval_statements(PrecompileStatements, file)
end
# Base.merge_names builds a Symbol[...] when it merges keyword-argument names not known
# at compile time (the solver's options)
for n in 4:16
    precompile(Tuple{typeof(Base.getindex), Type{Symbol}, ntuple(_ -> Symbol, n)...})
end
precompile(Tuple{typeof(Base.getindex), Type{Symbol}, Symbol, Symbol, Symbol, Vararg{Symbol}})
# stdout and stderr may be a file, a pipe or a terminal
for io in (Base.PipeEndpoint, Base.TTY, IOStream), f in (println, print)
    precompile(f, (io, String))
    precompile(flush, (io,))
end
