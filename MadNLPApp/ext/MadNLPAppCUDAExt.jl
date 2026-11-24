# MadNLP on a CUDA GPU (cuMadNLP), for the cnlp solver interface's device entry point:
# `cnlp_solve_device` solves a problem evaluated through its device callbacks, with the
# solver's vectors on the GPU. Loaded when the build environment has cuMadNLP and CUDACore,
# which is what makes the CUDA build of libmadnlp.
module MadNLPAppCUDAExt

import CNLPLib, CUDACore, cuMadNLP
using CNLPLib: CNLPDeviceProblem, CNLPProblem, CNLPResult, ProblemModel
using MadNLPApp: MadNLPApp, c_solve
using PrecompileTools: @setup_workload, @compile_workload

function __init__()
    # option values name cuMadNLP's types too (linear_solver=CUDSSSolver, ...)
    cuMadNLP in MadNLPApp.OPTION_MODULES || push!(MadNLPApp.OPTION_MODULES, cuMadNLP)
    return nothing
end

c_solve_device(
    p::Ptr{CNLPProblem}, d::Ptr{CNLPDeviceProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}},
    vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPResult}
) =
    c_solve(() -> ProblemModel(p, d), nopt, keys, vals, result)

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

# The device solve path, compiled at build time: each `LIB MODEL ARGS...` of
# MADNLPAPP_WARMUP_RUNS whose library has a CUDA tier, solved through cnlp_solve_device.
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

@setup_workload begin
    runs = get(ENV, "MADNLPAPP_WARMUP_RUNS", "")
    @compile_workload begin
        if !isempty(runs) && CUDACore.functional()
            for line in eachline(runs)
                w = split(line)
                (isempty(w) || startswith(w[1], "#")) && continue
                l = CNLPLib.load(String(w[1]))
                CNLPLib.Libdl.dlsym(l.handle, Symbol(w[2], "_device"); throw_error = false) === nothing && continue
                solve_lib_device(String(w[1]), String(w[2]), String.(w[3:end]))
            end
        end
    end
end

end # module
