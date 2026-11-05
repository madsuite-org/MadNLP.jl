# Run a benchmark: every instance of the requested sets with the requested configurations.
#
#     julia --project=benchmark benchmark/run.jl <set>[,<set>...] [options]
#
# See benchmark/README.md for the sets, the configurations, the options and the protocol.

function parse_args(args)
    isempty(args) && error(
        "usage: run.jl <set>[,<set>...] [--config=a,b] [--device=cpu|cuda] [--out=DIR] " *
            "[--only=case,...] [--tol=1e-8] [--wall=7200] [--no-warmup] [--overwrite] [--verbose]"
    )
    opt = Dict{String, String}("device" => "cpu", "config" => "default", "tol" => "1e-8", "wall" => "7200")
    flags = Set{String}()
    for a in args[(begin + 1):end]
        m = match(r"^--([a-z-]+)(?:=(.*))?$", a)
        m === nothing && error("unknown argument $a")
        m[2] === nothing ? push!(flags, m[1]) : (opt[m[1]] = m[2])
    end
    return String.(split(args[begin], ",")), opt, flags
end
const SET_NAMES, OPT, FLAGS = parse_args(ARGS)
const DEVICE = Symbol(OPT["device"])
DEVICE in (:cpu, :cuda) || error("--device is cpu or cuda")

using MadNLP, MadNLPHSL, ExaModels, ExaModelsPower, COPSBenchmark, ExaMittelmann
"cutest" in SET_NAMES && @eval using CUTEst
DEVICE == :cuda && @eval using CUDA, CUDSS, MadNLPGPU

include(joinpath(@__DIR__, "src", "MadNLPBenchmark.jl"))
include(joinpath(@__DIR__, "sets.jl"))
include(joinpath(@__DIR__, "configs.jl"))
using .MadNLPBenchmark

function main()
    available = configs(DEVICE)
    wanted = String.(split(OPT["config"], ","))
    unknown = setdiff(wanted, first.(available))
    isempty(unknown) || error("no configuration $(join(unknown, ", ")) on $DEVICE; available: $(join(first.(available), ", "))")
    chosen = [c for c in available if first(c) in wanted]
    backend = DEVICE == :cuda ? CUDABackend() : nothing
    device_name = DEVICE == :cuda ? CUDA.name(CUDA.device()) : nothing
    outdir = get(OPT, "out", joinpath(@__DIR__, "results", MadNLPBenchmark.hardware_tag(device_name)))
    options = (;
        tol = parse(Float64, OPT["tol"]), max_wall_time = parse(Float64, OPT["wall"]),
        print_level = "verbose" in FLAGS ? MadNLP.INFO : MadNLP.ERROR, rethrow_error = false,
    )
    meta = Dict{String, Any}("device" => String(DEVICE), "madnlp_commit" => madnlp_commit())
    DEVICE == :cuda && merge!(meta, Dict("gpu_name" => device_name, "cuda_runtime" => string(CUDA.runtime_version())))
    for set in SET_NAMES
        haskey(SETS, set) || error("no set $set; available: $(join(sort!(collect(keys(SETS))), ", "))")
        set == "cutest" && DEVICE != :cpu && error("the cutest set runs on the host only")
        cases = SETS[set]()
        if haskey(OPT, "only")
            only = String.(split(OPT["only"], ","))
            cases = [c for c in cases if first(c) in only]
        end
        @info "set $set: $(length(cases)) instances, configurations $(join(first.(chosen), ", ")), results in $outdir"
        run_benchmark(
            set, cases, chosen; outdir, backend, options, meta,
            warmup = !("no-warmup" in FLAGS), overwrite = "overwrite" in FLAGS,
            release = set == "cutest" ? finalize : (m -> nothing),
            reclaim = DEVICE == :cuda ? CUDA.reclaim : (() -> nothing),
        )
    end
    return
end

# The commit of the MadNLP checkout the benchmark runs from, with "-dirty" if it has local changes.
function madnlp_commit()
    root = dirname(@__DIR__)
    try
        sha = readchomp(`git -C $root rev-parse HEAD`)
        return isempty(readchomp(`git -C $root status --porcelain -- src lib ext`)) ? sha : sha * "-dirty"
    catch
        return "unknown"
    end
end

main()
