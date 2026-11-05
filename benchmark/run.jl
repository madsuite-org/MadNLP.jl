# Run a benchmark: every instance of the requested sets with the requested configurations.
#
#     julia --project=benchmark benchmark/run.jl <set>[,<set>...] [options]
#
# See benchmark/README.md for the sets, the configurations, the options and the protocol.

using Distributed

function parse_args(args)
    isempty(args) && error(
        "usage: run.jl <set>[,<set>...] [--config=a,b] [--device=cpu|cuda] [--workers=auto|N] [--out=DIR] " *
            "[--only=case,...] [--tol=1e-8] [--wall=7200] [--no-warmup] [--overwrite] [--verbose]"
    )
    opt = Dict{String, String}("device" => "cpu", "config" => "default", "tol" => "1e-8", "wall" => "7200", "workers" => "auto")
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

include(joinpath(@__DIR__, "session.jl"))

# The number of processes that solve. On a CUDA device, one per GPU. On the host, `auto` is a
# quarter of the physical cores, at most 8: the solves are sequential, and processes that
# share memory channels slow each other down.
function worker_count()
    requested = OPT["workers"] == "auto" ? nothing : parse(Int, OPT["workers"])
    if DEVICE == :cuda
        gpus = length(CUDA.devices())
        return requested === nothing ? gpus : min(requested, gpus)
    end
    cores = something(MadNLPBenchmark.physical_cores(), Sys.CPU_THREADS)
    return requested === nothing ? clamp(cores ÷ 4, 1, 8) : requested
end

# The processes that solve: this one alone, or `n` workers that load the same session.
function start_workers(n)
    n > 1 || return [myid()]
    pids = addprocs(n; exeflags = ["--project=$(Base.active_project())", "--startup-file=no"])
    session = joinpath(@__DIR__, "session.jl")
    # the GPUs this process sees, by the names a worker can be restricted to
    gpus = DEVICE == :cuda ? split(get(ENV, "CUDA_VISIBLE_DEVICES", join(0:(n - 1), ",")), ",") : String[]
    @sync for (i, pid) in enumerate(pids)
        @async begin
            remotecall_wait(Core.eval, pid, Main, :(const DEVICE = $(QuoteNode(DEVICE)); const SET_NAMES = $SET_NAMES))
            # a GPU each; on the host a directory each, where CUTEst.jl decodes its problems
            remotecall_wait(Core.eval, pid, Main, DEVICE == :cuda ? :(ENV["CUDA_VISIBLE_DEVICES"] = $(String(gpus[i]))) : :(cd(mktempdir())))
            remotecall_wait(Core.eval, pid, Main, :(include($session)))
        end
    end
    return pids
end

function main()
    available = first.(configs(DEVICE))
    chosen = String.(split(OPT["config"], ","))
    unknown = setdiff(chosen, available)
    isempty(unknown) || error("no configuration $(join(unknown, ", ")) on $DEVICE; available: $(join(available, ", "))")
    device_name = DEVICE == :cuda ? CUDA.name(CUDA.device()) : nothing
    outdir = get(OPT, "out", joinpath(@__DIR__, "results", MadNLPBenchmark.hardware_tag(device_name)))
    options = (;
        tol = parse(Float64, OPT["tol"]), max_wall_time = parse(Float64, OPT["wall"]),
        print_level = "verbose" in FLAGS ? MadNLP.INFO : MadNLP.ERROR, rethrow_error = false,
    )
    warmup = !("no-warmup" in FLAGS)
    meta = Dict{String, Any}("device" => String(DEVICE), "madnlp_commit" => madnlp_commit(), "options" => string(options))
    if DEVICE == :cuda
        merge!(
            meta, Dict(
                "gpu_name" => device_name, "gpu_count" => length(CUDA.devices()),
                "gpu_memory_gb" => round(CUDA.totalmem(CUDA.device()) / 2^30; digits = 1),
                "gpu_driver" => string(CUDA.driver_version()), "cuda_runtime" => string(CUDA.runtime_version()),
            )
        )
    end
    config_options = Dict(configs(DEVICE))
    pids = start_workers(worker_count())
    for set in SET_NAMES
        haskey(SETS, set) || error("no set $set; available: $(join(sort!(collect(keys(SETS))), ", "))")
        cases = first.(SETS[set]())
        haskey(OPT, "only") && (cases = intersect(cases, String.(split(OPT["only"], ","))))
        @info "set $set: $(length(cases)) instances, configurations $(join(chosen, ", ")), $(length(pids)) process(es), results in $outdir"
        run_benchmark(
            set, cases, chosen; outdir, pids, meta, overwrite = "overwrite" in FLAGS,
            meta_of = config -> Dict{String, Any}("config_options" => string(config_options[config])),
        ) do pid, case, todo
            pid == myid() ? solve_here(set, case, todo, options, warmup) :
                remotecall_fetch(Main.solve_here, pid, set, case, todo, options, warmup)
        end
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
