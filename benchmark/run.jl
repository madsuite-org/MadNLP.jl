# Run a benchmark: every instance of the requested sets with the requested configurations.
#
#     julia --project=benchmark benchmark/run.jl <set>[,<set>...] [options]
#
# See benchmark/README.md for the sets, the configurations, the options and the protocol.

using Distributed

function parse_args(args)
    isempty(args) && error(
        "usage: run.jl <set>[,<set>...] [--config=a,b] [--device=cpu|cuda|cpu,cuda] [--workers=auto|N] [--workers-per-gpu=2] [--root=DIR] [--out=DIR] " *
            "[--only=case,...] [--tol=1e-8] [--wall=7200] [--margin=S] [--status-every=600] [--no-warmup] [--overwrite] [--verbose]"
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

# Several devices: one run of this script per device, side by side, each sized on its own. The
# run fails if one of them does.
if occursin(',', OPT["device"])
    haskey(OPT, "out") && error("--out names one directory; with several devices use --root")
    args = filter(!startswith("--device="), ARGS)
    legs = map(split(OPT["device"], ",")) do device
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(@__FILE__) $args --device=$device`
        run(pipeline(cmd; stdin = devnull, stdout, stderr); wait = false)
    end
    atexit(() -> foreach(kill, legs))
    foreach(wait, legs)
    exit(all(success, legs) ? 0 : 1)
end

const DEVICE = Symbol(OPT["device"])
DEVICE in (:cpu, :cuda) || error("--device is cpu or cuda")

include(joinpath(@__DIR__, "session.jl"))

# The heap-size hint is what bounds a worker in practice: without it the GC lets the heap
# track the free memory of the machine, and one large problem can reach tens of GB. A worker
# that still grows past RSS_LIMIT_GB is replaced between cases.
const HEAP_HINT = "6G"
const RSS_LIMIT_GB = 8.0

# A worker above this share of the machine's memory is killed during a solve, as is one past
# RSS_LIMIT_GB when the machine is nearly out of memory, before the system's out-of-memory
# killer takes the job or the runner with it.
const KILL_MEMORY_FRACTION = 0.4

# The number of processes that solve. On a CUDA device, `--workers-per-gpu` per GPU. On the host, `auto` follows
# what this process may use, the allocation of a batch job included: 75% of its physical cores,
# as each solve is sequential and a device leg may run beside this one, and 3 GB per worker out
# of 75% of the memory, as workers rarely reach their limit at the same time.
function worker_count()
    requested = OPT["workers"] == "auto" ? nothing : parse(Int, OPT["workers"])
    requested === nothing || return requested
    DEVICE == :cuda && return parse(Int, get(OPT, "workers-per-gpu", "2")) * length(CUDA.devices())
    allowed = ccall(:uv_available_parallelism, Cuint, ()) / Sys.CPU_THREADS   # the share of the machine's threads
    cores = allowed * something(MadNLPBenchmark.physical_cores(), Sys.CPU_THREADS)
    by_memory = floor(Int, 0.75 * Sys.total_memory() / 2^30 / 3)
    return requested === nothing ? max(1, min(floor(Int, 0.75 * cores), by_memory)) : requested
end

# A worker process and the GPU it is restricted to, if any. A killed worker is replaced in its slot.
mutable struct Slot
    pid::Int
    os_pid::Int
    gpu::String
    token::Union{Nothing, RemoteChannel}   # one per GPU, see `solve_here`
    case::String     # what the worker is on, for the status report
    state::Symbol
    since::Float64
end

# Seconds workers spent waiting for a GPU and seconds the GPUs spent in timed solves: if the
# first is large there are more workers per GPU than the solves can use, and if the second is
# far below the length of the run there are too few.
const TOKEN_TIME = Dict("waiting" => 0.0, "solving" => 0.0)

# Where the solver logs go, one file per instance and configuration.
const LOG_DIR = Ref("")

# A worker does not reliably notice that this process has gone, so it is stopped on the way out.
const WORKER_PIDS = Int[]
# With a direct system call: `run` can hang here when the exit comes from a signal.
atexit(() -> foreach(pid -> ccall(:kill, Cint, (Cint, Cint), pid, 9), WORKER_PIDS))

function start_worker!(slot::Slot, chosen, options)
    slot.pid = only(addprocs(1; exeflags = ["--project=$(Base.active_project())", "--startup-file=no", "--heap-size-hint=$HEAP_HINT"]))
    slot.os_pid = remotecall_fetch(getpid, slot.pid)
    push!(WORKER_PIDS, slot.os_pid)
    remotecall_wait(Core.eval, slot.pid, Main, :(const DEVICE = $(QuoteNode(DEVICE)); const SET_NAMES = $SET_NAMES; const LOG_DIR = $(LOG_DIR[])))
    isempty(slot.gpu) || remotecall_wait(Core.eval, slot.pid, Main, :(ENV["CUDA_VISIBLE_DEVICES"] = $(slot.gpu)))
    remotecall_wait(Core.eval, slot.pid, Main, :(include($(joinpath(@__DIR__, "session.jl")))))
    remotecall_wait(Main.warm_up, slot.pid, chosen, options)
    return slot
end

# Decode the problems in one more process, one after the other, while the workers solve the
# sets ahead of CUTEst. `decoded` collects the names as they become safe to load.
function decode_in_background(names, decoded)
    try
        pid = only(addprocs(1; exeflags = ["--project=$(Base.active_project())", "--startup-file=no"]))
        push!(WORKER_PIDS, remotecall_fetch(getpid, pid))
        remotecall_wait(Core.eval, pid, Main, :(const DEVICE = $(QuoteNode(DEVICE)); const SET_NAMES = $SET_NAMES))
        remotecall_wait(Core.eval, pid, Main, :(include($(joinpath(@__DIR__, "session.jl")))))
        progress = RemoteChannel(() -> Channel{String}(Inf))
        remote_do(Main.decode_cutest, pid, names, progress)
        for name in progress
            push!(decoded, name)
        end
    catch err
        @warn "decoding stopped" exception = err
    end
    return
end

# Wait for a problem to be decoded, or for the decoding to end: a problem it could not decode
# then fails to load and is recorded as an error.
function wait_decoded(slot, name, decoded, decoder)
    name in decoded && return
    slot.case, slot.state, slot.since = name, :decoding, time()
    while name ∉ decoded && !istaskdone(decoder)
        sleep(1)
    end
    return
end

function resident_gb(os_pid)
    isfile("/proc/$os_pid/status") || return 0.0
    try
        for line in eachline("/proc/$os_pid/status")
            startswith(line, "VmRSS:") && return parse(Float64, split(line)[2]) / 2^20
        end
    catch   # the process ended between the check and the read
    end
    return 0.0
end

function start_workers(n, chosen, options)
    # the GPUs this process sees, by the names a worker can be restricted to
    gpus = DEVICE == :cuda ? split(get(ENV, "CUDA_VISIBLE_DEVICES", join(0:(length(CUDA.devices()) - 1), ",")), ",") : [""]
    tokens = Dict(gpu => (ch = RemoteChannel(() -> Channel{Int}(1)); put!(ch, 1); ch) for gpu in gpus if !isempty(gpu))
    slots = [Slot(0, 0, String(gpus[mod1(i, length(gpus))]), get(tokens, gpus[mod1(i, length(gpus))], nothing), "", :starting, time()) for i in 1:n]
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
    slot.pid in workers() || start_worker!(slot, chosen, options)   # a failed restart is retried here
    slot.case, slot.state, slot.since = "$set/$case", :building, time()
    events = RemoteChannel(() -> Channel{Any}(Inf))
    remote_do(Main.solve_here, slot.pid, set, case, todo, options, warmup, events, slot.token)
    rows = Pair{String, Vector{Any}}[]
    size = (0, 0, 0, 0)
    last = time()
    state = :working   # :waiting for the GPU, :solving with it, or :working without it
    # the rows of the configurations a dead worker did not finish, and a new worker
    function replace_worker(status)
        while isready(events)   # what the worker sent before it ended
            event = take!(events)
            event === :solve && (state = :solving)
            event === :released && (state = :working)
            event isa Pair && push!(rows, event)
        end
        # a killed worker cannot hand back its token
        state === :solving && slot.token !== nothing && !isready(slot.token) && put!(slot.token, 1)
        restart(slot, chosen, options)
        finished = first.(rows)
        COMPLETED[] += 1
        slot.case, slot.state, slot.since = "", :idle, time()
        return [rows; [c => MadNLPBenchmark.failure_row(case, status, size) for c in todo if c ∉ finished]]
    end
    while true
        if isready(events)
            event = take!(events)
            if event === :done
                COMPLETED[] += 1
                slot.case, slot.state, slot.since = "", :idle, time()
                failed = any(r -> last_status(r) == MadNLPBenchmark.ERROR_STATUS, rows)
                # a device context can be unusable after an error, such as running out of memory
                if resident_gb(slot.os_pid) > RSS_LIMIT_GB || (failed && DEVICE == :cuda)
                    rmprocs(slot.pid; waitfor = 30)
                    restart(slot, chosen, options)
                end
                return rows
            end
            event isa Tuple && (size = event)
            event isa Pair && push!(rows, event)
            now = time()
            event === :solve && (TOKEN_TIME["waiting"] += now - last)
            event === :released && (TOKEN_TIME["solving"] += now - last)
            event === :waiting && (state = :waiting)
            event === :solve && (state = :solving)
            event === :released && (state = :working)
            event in (:waiting, :solve, :released) && ((slot.state, slot.since) = (state, now))
            last = now
        elseif state !== :waiting && time() - last > limit   # waiting for a GPU is not a hang
            @warn "$case: no progress for $(round(Int, limit)) s, killing worker $(slot.pid)"
            kill_worker(slot)
            return replace_worker(MadNLPBenchmark.KILLED_STATUS)
        elseif resident_gb(slot.os_pid) > KILL_MEMORY_FRACTION * Sys.total_memory() / 2^30 ||
                (resident_gb(slot.os_pid) > RSS_LIMIT_GB && Sys.free_memory() < 0.05 * Sys.total_memory())
            @warn "$case: worker $(slot.pid) uses $(round(Int, resident_gb(slot.os_pid))) GB, killing it"
            kill_worker(slot)
            return replace_worker(MadNLPBenchmark.KILLED_STATUS)
        elseif !(slot.pid in workers())   # the worker died on its own: a crash inside a library
            @warn "$case: worker $(slot.pid) died"
            return replace_worker(MadNLPBenchmark.ERROR_STATUS)
        else
            sleep(0.2)
        end
    end
    return
end

last_status(row) = last(row)[2]

const COMPLETED = Ref(0)

# What every worker is on, longest first, so that a stall shows while it happens, and the
# results so far: in `summary.md` beside the result files and, with `log`, in the log.
function report_status(slots, total, started, outdir, wall; log = true)
    lines = ["status after $(round(Int, (time() - started) / 60)) min: $(COMPLETED[]) of $total instances done"]
    for slot in sort(slots; by = s -> s.since)
        slot.state === :idle && continue
        push!(lines, "  worker $(slot.pid): $(slot.state) $(slot.case) for $(round(Int, time() - slot.since)) s, $(round(resident_gb(slot.os_pid); digits = 1)) GB")
    end
    idle = count(s -> s.state === :idle, slots)
    idle > 0 && push!(lines, "  $idle idle")
    push!(lines, "  machine: $(round(Sys.free_memory() / 2^30; digits = 1)) GB of memory free, load $(round(Sys.loadavg()[1]; digits = 1))")
    status = join(lines, "\n")
    tables = sprint(io -> summarize(io, outdir; wall))
    log && @info status * "\n\n" * tables
    path = joinpath(outdir, "summary.md")
    write(path * ".tmp", "```\n", status, "\n```\n\n", tables)
    mv(path * ".tmp", path; force = true)
    return
end

function kill_worker(slot)
    run(ignorestatus(`kill -9 $(slot.os_pid)`))
    try
        rmprocs(slot.pid; waitfor = 0)
    catch
    end
    return
end

# A new worker for the slot. A failure here must not end the run: the next case tries again.
function restart(slot, chosen, options)
    try
        start_worker!(slot, chosen, options)
    catch err
        @warn "could not start a worker" exception = err
    end
    return
end

function main()
    available = first.(configs(DEVICE))
    chosen = String.(split(OPT["config"], ","))
    unknown = setdiff(chosen, available)
    isempty(unknown) || error("no configuration $(join(unknown, ", ")) on $DEVICE; available: $(join(available, ", "))")
    device_name = DEVICE == :cuda ? CUDA.name(CUDA.device()) : nothing
    root = get(OPT, "root", joinpath(@__DIR__, "results"))
    outdir = get(OPT, "out", joinpath(root, MadNLPBenchmark.hardware_tag(device_name)))
    options = (;
        tol = parse(Float64, OPT["tol"]), acceptable_tol = parse(Float64, OPT["tol"]), max_wall_time = parse(Float64, OPT["wall"]),
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
    decoded = Set{String}()
    decoder = nothing
    if "cutest" in SET_NAMES && DEVICE == :cpu
        decode_cutest(["HS15"])   # the problem a worker warms up on
        decoder = @async decode_in_background(last(only(s for s in sets if first(s) == "cutest")), decoded)
    end
    LOG_DIR[] = joinpath(outdir, "logs")
    slots = start_workers(worker_count(), chosen, options)
    @info "$(length(slots)) worker(s), results in $outdir"
    # the threads a solve actually gets are those of a worker, not of this process
    merge!(meta, remotecall_fetch(Main.worker_threads, first(slots).pid))
    meta["hsl_version"] in ("none", MadNLPBenchmark.HSL_VERSION) ||
        @warn "this benchmark version uses HSL $(MadNLPBenchmark.HSL_VERSION), this run $(meta["hsl_version"]): its results are not comparable"
    # a build or a solve gets the wall limit and a margin for compilation and evaluation
    wall = options.max_wall_time
    limit = wall + parse(Float64, get(OPT, "margin", string(max(120.0, 0.2 * wall))))
    meta["watchdog_s"] = limit
    started = time()
    every = parse(Float64, get(OPT, "status-every", "600"))
    total = sum(length ∘ last, sets)
    logged = time()
    reporter = @async while true
        sleep(min(every, 5))
        try
            log = time() - logged >= every
            log && (logged = time())
            report_status(slots, total, started, outdir, wall; log)
        catch err
            @warn "could not report the status" exception = err
        end
    end
    run_benchmark(
        sets, chosen; outdir, slots, meta, overwrite = "overwrite" in FLAGS, group = instance_group,
        meta_of = config -> Dict{String, Any}("config_options" => string(config_options[config])),
    ) do slot, set, case, todo
        set == "cutest" && decoder !== nothing && wait_decoded(slot, case, decoded, decoder)
        solve_on(slot, set, case, todo, chosen, options, warmup, limit)
    end
    report_status(slots, total, started, outdir, wall)
    DEVICE == :cuda && @info "GPU time: $(round(Int, TOKEN_TIME["solving"])) s in timed solves, $(round(Int, TOKEN_TIME["waiting"])) s of workers waiting"
    return
end

# The commit of the MadNLP checkout the benchmark runs from, with "-dirty" if it has local changes.
function madnlp_commit()
    root = dirname(@__DIR__)
    try
        sha = readchomp(`git -C $root rev-parse HEAD`)
        return isempty(readchomp(`git -C $root status --porcelain -- MadCore MadCoreGPU MadNLP MadNLPCore cuMadNLP`)) ? sha : sha * "-dirty"
    catch
        return "unknown"
    end
end

main()
