# MadNLP on a CUDA GPU (cuMadNLP), for the cnlp solver interface's device entry point:
# `cnlp_solve_device` solves a problem evaluated through its device callbacks, with the
# solver's vectors on the GPU. Loaded when the build environment has cuMadNLP and CUDACore,
# which is what makes the CUDA build of libmadnlp.
module MadNLPAppCUDAExt

import CNLPLib, CUDACore, cuMadNLP
using CNLPLib: CNLPDeviceProblem, CNLPProblem, CNLPResult, ProblemModel
using MadNLPApp: MadNLPApp, c_solve

function __init__()
    # option values name cuMadNLP's types too (linear_solver=CUDSSSolver, ...)
    cuMadNLP in MadNLPApp.OPTION_MODULES || push!(MadNLPApp.OPTION_MODULES, cuMadNLP)
    return nothing
end

function c_solve_device(
        p::Ptr{CNLPProblem}, d::Ptr{CNLPDeviceProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}},
        vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPResult}
    )
    solve() = c_solve(() -> ProblemModel(p, d), nopt, keys, vals, result)
    MadNLPApp.trace_on() || return solve()
    # MADNLPAPP_TRACE=1: also report the GPU kernels compiled at run time
    empty!(COMPILED)
    rc = CUDACore.GPUCompiler.with_compile_hook(solve, kernel_hook)
    println(
        stderr, string(
            "madnlp: GPU kernels compiled at run time: ", length(COMPILED),
            isempty(COMPILED) ? "" : " (" * join(COMPILED, ", ") * ")"
        )
    )
    flush(stderr)
    return rc
end

# GPUCompiler reports every kernel lookup to the hook; a lookup with no compiled image
# cached is a kernel about to be compiled at run time.
const COMPILED = String[]
const COMPILED_LOCK = ReentrantLock()
function kernel_hook(job)
    res = CUDACore.GPUCompiler.cached_results(CUDACore.CUDACompilerResults, job)
    if res === nothing || res.image === nothing
        name = string(job.source.def.name)
        lock(COMPILED_LOCK) do
            name in COMPILED || push!(COMPILED, name)
        end
    end
    return nothing
end

"The cnlp solver interface's device entry point (`cnlp_solver.h` 0.2): CUDA."
Base.@ccallable function cnlp_solve_device(
        p::Ptr{CNLPProblem}, d::Ptr{CNLPDeviceProblem}, nopt::Int64,
        keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}},
        result::Ptr{CNLPResult}
    )::Int64
    dev = unsafe_load(d).device
    if dev != CNLPLib.CNLP_DEVICE_CUDA
        CNLPLib.set_failure!(result, "madnlp: device $dev is not supported (CUDA is $(CNLPLib.CNLP_DEVICE_CUDA))")
        return Int64(1)
    end
    if !CUDACore.functional()
        CNLPLib.set_failure!(result, "madnlp: no usable CUDA GPU")
        return Int64(2)
    end
    return c_solve_device(p, d, nopt, keys, vals, result)
end

# A library model solved through cnlp_solve_device: what the build runs to compile the
# device path (build/libmadnlp.jl), and a test helper.
function solve_lib_device(path::String, model::String, args::Vector{String})
    lib = CNLPLib.load(path)
    vals = Any[something(tryparse(Int, a), tryparse(Float64, a), a) for a in args]
    m = CNLPLib.CNLPModel(lib, Symbol(model), vals...)
    md = CNLPLib.CNLPModel(lib, Symbol(model), vals...; device = :cuda)
    h = CNLPLib.problem(m)
    d = CNLPLib.device_problem(h, md)
    n, k = m.meta.nvar, m.meta.ncon
    x, y, zl, zu = zeros(n), zeros(k), zeros(n), zeros(n)
    r = Ref(CNLPResult(pointer(x), pointer(y), pointer(zl), pointer(zu), 0.0, -1, -1, ntuple(_ -> 0x00, 128)))
    ks = ["print_level"]; vs = ["0"]
    kp = [pointer(s) for s in ks]; vp = [pointer(s) for s in vs]
    GC.@preserve h d x y zl zu r ks vs kp vp begin
        cnlp_solve_device(
            Base.unsafe_convert(Ptr{CNLPProblem}, h), Base.unsafe_convert(Ptr{CNLPDeviceProblem}, d),
            Int64(1), pointer(kp), pointer(vp), Base.unsafe_convert(Ptr{CNLPResult}, r)
        )
    end
    return r[].status
end

# The build compiles the device path with build/cuda/MadNLPAppCUDAWarmup, a package whose
# precompile workload calls `solve_lib_device`: CNLPLib's own CUDA extension, which it
# needs, is not loaded while this extension precompiles.

end # module
