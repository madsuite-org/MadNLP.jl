# What every process of a benchmark run loads: the packages, the benchmark module, the sets
# and the configurations. `DEVICE` (`:cpu` or `:cuda`) and `SET_NAMES` are defined by run.jl
# before this file is included, on this process and on every worker.

using MadNLP, MadNLPHSL, ExaModels, ExaModelsPower, COPSBenchmark, ExaMittelmann
# CUTEst: the compiled SIF decoders of CUTEst.jl on the host, ExaCUTEst models on a device
"cutest" in SET_NAMES && (DEVICE == :cuda ? (@eval using ExaCUTEst) : (@eval using CUTEst))
DEVICE == :cuda && @eval using CUDA, CUDSS, MadNLPGPU

include(joinpath(@__DIR__, "src", "MadNLPBenchmark.jl"))
include(joinpath(@__DIR__, "sets.jl"))
include(joinpath(@__DIR__, "configs.jl"))
using .MadNLPBenchmark

# set => (case => build), filled on first use
const CASES = Dict{String, Dict{String, Function}}()

# Solve one case of a set with the named configurations on this process.
function solve_here(set, case, config_names, options, warmup)
    builds = get!(() -> Dict(SETS[set]()), CASES, set)
    chosen = [c for c in configs(DEVICE) if first(c) in config_names]
    return solve_case(
        case, builds[case], chosen; options, warmup,
        backend = DEVICE == :cuda ? CUDABackend() : nothing,
        release = set == "cutest" && DEVICE == :cpu ? finalize : (m -> nothing),
        reclaim = DEVICE == :cuda ? CUDA.reclaim : (() -> nothing),
    )
end
