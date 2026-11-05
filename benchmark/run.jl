# Run a benchmark: every instance of the requested sets with the requested configurations.
#
#     julia --project=benchmark benchmark/run.jl <set>[,<set>...] [options]
#
# See benchmark/README.md for the sets, the configurations, the options and the protocol.

using Distributed

function parse_args(args)
    isempty(args) && error(
        "usage: run.jl <set>[,<set>...] [--config=a,b] [--device=cpu|cuda] [--workers=auto|N] [--out=DIR] " *
            "[--only=case,...] [--tol=1e-8] [--wall=7200] [--margin=S] [--no-warmup] [--overwrite] [--verbose]"
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

# The heap-size hint is what bounds a worker in practice: without it the GC lets the heap
# track the free memory of the machine, and one large problem can reach tens of GB. A worker
# that still grows past RSS_LIMIT_GB is replaced between cases.
const HEAP_HINT = "6G"
const RSS_LIMIT_GB = 8.0

# The number of processes that solve. On a CUDA device, one per GPU. On the host, `auto` follows
# the size of the machine: 80% of the physical cores, as each solve is sequential, and no more
# workers at their memory limit than fit in half of the memory.
function worker_count()
    requested = OPT["workers"] == "auto" ? nothing : parse(Int, OPT["workers"])
    if DEVICE == :cuda
        gpus = length(CUDA.devices())
        return requested === nothing ? gpus : min(requested, gpus)
    end
    cores = something(MadNLPBenchmark.physical_cores(), Sys.CPU_THREADS)
    by_memory = floor(Int, 0.5 * Sys.total_memory() / 2^30 / RSS_LIMIT_GB)
    return requested === nothing ? max(1, min(floor(Int, 0.8 * cores), by_memory)) : requested
end

# A worker process and the GPU it is restricted to, if any. A killed worker is replaced in its slot.
mutable struct Slot
    pid::Int
    os_pid::Int
    gpu::String
end

function start_worker!(slot::Slot, chosen, options)
    slot.pid = only(addprocs(1; exeflags = ["--project=$(Base.active_project())", "--startup-file=no", "--heap-size-hint=$HEAP_HINT"]))
    remotecall_wait(Core.eval, slot.pid, Main, :(const DEVICE = $(QuoteNode(DEVICE)); const SET_NAMES = $SET_NAMES))
    isempty(slot.gpu) || remotecall_wait(Core.eval, slot.pid, Main, :(ENV["CUDA_VISIBLE_DEVICES"] = $(slot.gpu)))
    remotecall_wait(Core.eval, slot.pid, Main, :(include($(joinpath(@__DIR__, "session.jl")))))
    slot.os_pid = remotecall_fetch(getpid, slot.pid)
    remotecall_wait(Main.warm_up, slot.pid, chosen, options)
    return slot
end

function resident_gb(os_pid)
    for line in eachline("/proc/$os_pid/status")
        startswith(line, "VmRSS:") && return parse(Float64, split(line)[2]) / 2^20
    end
    return 0.0
end

function start_workers(n, chosen, options)
    # the GPUs this process sees, by the names a worker can be restricted to
    gpus = DEVICE == :cuda ? split(get(ENV, "CUDA_VISIBLE_DEVICES", join(0:(n - 1), ",")), ",") : fill("", n)
    slots = [Slot(0, 0, String(gpus[i])) for i in 1:n]
    @sync for slot in slots
        @async start_worker!(slot, chosen, options)
    end
    return slots
end

# Solve one case on a worker, and kill the worker if a build or a solve outlasts `limit`
# seconds. The events arrive on a channel of this process, so that a worker stuck in a
# library call, which cannot answer a query, is still detected. The instance is recorded with
# what is known of it, and the configurations that finished before the kill keep their rows.
function solve_on(slot, set, case, todo, chosen, options, warmup, limit)
    events = RemoteChannel(() -> Channel{Any}(Inf))
    remote_do(Main.solve_here, slot.pid, set, case, todo, options, warmup, events)
    rows = Pair{String, Vector{Any}}[]
    size = (0, 0, 0, 0)
    last = time()
    while true
        if isready(events)
            event = take!(events)
            if event === :done
                if isfile("/proc/$(slot.os_pid)/status") && resident_gb(slot.os_pid) > RSS_LIMIT_GB
                    rmprocs(slot.pid; waitfor = 30)
                    start_worker!(slot, chosen, options)
                end
                return rows
            end
            event isa Tuple && (size = event)
            event isa Pair && push!(rows, event)
            last = time()
        elseif time() - last > limit
            @warn "$case: no progress for $(round(Int, limit)) s, killing worker $(slot.pid)"
            run(ignorestatus(`kill -9 $(slot.os_pid)`))
            try
                rmprocs(slot.pid; waitfor = 0)
            catch
            end
            start_worker!(slot, chosen, options)
            finished = first.(rows)
            return [rows; [c => MadNLPBenchmark.failure_row(case, MadNLPBenchmark.KILLED_STATUS, size) for c in todo if c ∉ finished]]
        elseif !(slot.pid in workers())   # the worker died on its own: a crash inside a library
            @warn "$case: worker $(slot.pid) died"
            start_worker!(slot, chosen, options)
            finished = first.(rows)
            return [rows; [c => MadNLPBenchmark.failure_row(case, MadNLPBenchmark.ERROR_STATUS, size) for c in todo if c ∉ finished]]
        else
            sleep(0.2)
        end
    end
    return
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
    validate(configs(DEVICE))
    sets = map(SET_NAMES) do set
        haskey(SETS, set) || error("no set $set; available: $(join(sort!(collect(keys(SETS))), ", "))")
        cases = first.(SETS[set]())
        haskey(OPT, "only") && (cases = intersect(cases, String.(split(OPT["only"], ","))))
        @info "set $set: $(length(cases)) instances, configurations $(join(chosen, ", "))"
        set => cases
    end
    if "cutest" in SET_NAMES && DEVICE == :cpu
        decode_cutest(["HS15"; last(only(s for s in sets if first(s) == "cutest"))])
    end
    slots = start_workers(worker_count(), chosen, options)
    @info "$(length(slots)) worker(s), results in $outdir"
    # the threads a solve actually gets are those of a worker, not of this process
    merge!(meta, remotecall_fetch(Main.worker_threads, first(slots).pid))
    # a build or a solve gets the wall limit and a margin for compilation and evaluation
    wall = options.max_wall_time
    limit = wall + parse(Float64, get(OPT, "margin", string(max(120.0, 0.2 * wall))))
    meta["watchdog_s"] = limit
    run_benchmark(
        sets, chosen; outdir, slots, meta, overwrite = "overwrite" in FLAGS,
        meta_of = config -> Dict{String, Any}("config_options" => string(config_options[config])),
    ) do slot, set, case, todo
        solve_on(slot, set, case, todo, chosen, options, warmup, limit)
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
