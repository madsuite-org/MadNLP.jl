# What every process of a benchmark run loads: the packages, the benchmark module, the sets
# and the configurations. `DEVICE` (`:cpu` or `:cuda`) and `SET_NAMES` are defined by run.jl
# before this file is included, on this process and on every worker.

using MadNLP, MadNLPHSL, ExaModels, ExaModelsPower, COPSBenchmark, ExaMittelmann, NLPModels, NLPModelsIpopt
# CUTEst: the compiled SIF decoders of CUTEst.jl on the host, ExaCUTEst models on a device
"cutest" in SET_NAMES && (DEVICE == :cuda ? (@eval using ExaCUTEst) : (@eval using CUTEst))
DEVICE == :cuda && @eval using CUDA, CUDSS, MadNLPGPU

include(joinpath(@__DIR__, "src", "MadNLPBenchmark.jl"))
include(joinpath(@__DIR__, "sets.jl"))
include(joinpath(@__DIR__, "configs.jl"))
using .MadNLPBenchmark

# set => (case => build), filled on first use
const CASES = Dict{String, Dict{String, Function}}()

# Solve one case of a set with the named configurations on this process, telling `events`
# (a channel on the coordinating process) how far it has got. Workers that share a GPU also
# share `token`: only its holder runs a timed solve, while the others build and compile.
function solve_here(set, case, config_names, options, warmup, events, token)
    guard = function (solve)
        put!(events, :waiting)
        token === nothing || take!(token)
        put!(events, :solve)
        try
            return solve()
        finally
            token === nothing || put!(token, 1)
            put!(events, :released)
        end
    end
    builds = get!(() -> Dict(SETS[set]()), CASES, set)
    chosen = [c for c in configs(DEVICE) if first(c) in config_names]
    solve_case(
        case, builds[case], chosen; options, warmup, report = event -> put!(events, event), prepare = compile_model, guard,
        backend = DEVICE == :cuda ? CUDABackend() : nothing,
        release = set == "cutest" && DEVICE == :cpu ? finalize : (m -> nothing),
        reclaim = DEVICE == :cuda ? CUDA.reclaim : (() -> nothing),
    )
    put!(events, :done)
    return nothing
end

# Compile the callbacks of an ExaModel before the timed solve. A device kernel compiles when it
# first launches, so this also evaluates them once at the starting point.
compile_model(m) = m isa ExaModels.AbstractExaModel ? ExaModels.precompile_callbacks(m) : nothing

worker_threads() = Dict{String, Any}(
    "julia_threads" => Threads.nthreads(), "blas_threads" => MadNLPBenchmark.LinearAlgebra.BLAS.get_num_threads(),
)

# Compile the solve path of this process on tiny problems, so that the first real instance a
# new worker gets is not solved twice for it.
function warm_up(config_names, options)
    chosen = [c for c in configs(DEVICE) if first(c) in config_names]
    backend = DEVICE == :cuda ? CUDABackend() : nothing
    for form in ACOPF_FORMS
        solve_case("warm-up", last(acopf_case("case14_ieee", form)), chosen; backend, options, warmup = false, prepare = compile_model)
    end
    "cutest" in SET_NAMES && DEVICE == :cpu &&
        solve_case("warm-up", b -> cutest_host_model("HS15"), chosen; options, warmup = false, release = finalize)
    return nothing
end
