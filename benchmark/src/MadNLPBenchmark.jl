"""
    MadNLPBenchmark

The benchmarking loop behind `benchmark/run.jl`: it solves every instance of a problem set
with every requested configuration and writes one CSV file per set and configuration, with a
metadata file beside it. `benchmark/README.md` states the protocol.
"""
module MadNLPBenchmark

using MadNLP, NLPModels
using Dates, DelimitedFiles, LinearAlgebra, Printf, TOML

export run_benchmark, solve_case, read_results, read_meta, result_files, summarize, hardware_line, solved, score_time, sgm, size_class

"The columns of a result file, in order."
const HEADER = [
    "case", "status", "iters", "nvar", "ncon", "nnzj", "nnzh",
    "obj", "primal_feas", "dual_feas",
    "n_fact", "n_backsolve", "n_ir", "n_obj", "n_grad", "n_con", "n_jac", "n_hess",
    "t_total", "t_linsolve", "t_eval", "t_init",
    # seconds Julia spent compiling during the recorded solve, and how many times the instance
    # was solved to obtain a solve free of compilation
    "t_compile", "n_solves",
    # violation of the model's own constraints and bounds at the returned point, evaluated
    # independently of the solver: equality rows, all rows beyond [lcon, ucon], variables beyond
    # [lvar, uvar]. The solver's primal_feas is the residual of its scaled, relaxed problem,
    # where an equality can sit at the edge of its relaxed bounds under a clean status.
    "viol_eq", "viol_con", "viol_var",
]

# Status recorded for an instance whose worker was killed for overrunning the wall limit, and
# for one that could not be built or whose solve threw. MadNLP checks its wall limit between
# iterations only, so a single long factorization or evaluation can run far past it.
const KILLED_STATUS = -100
const ERROR_STATUS = -101

const T_TOTAL = findfirst(==("t_total"), HEADER)

sanitize(s) = replace(strip(String(s)), r"[^A-Za-z0-9.+-]+" => "_")

"A directory name for the machine: the CPU model, or the name of the device if one is given."
hardware_tag(device_name = nothing) =
    sanitize(device_name === nothing ? Sys.cpu_info()[1].model : device_name)

dims(m) = (NLPModels.get_nvar(m), NLPModels.get_ncon(m), NLPModels.get_nnzj(m), NLPModels.get_nnzh(m))

function violation(m, stats)
    x = Array(stats.solution)
    c = Array(NLPModels.cons(m, convert(typeof(NLPModels.get_x0(m)), x)))
    lcon, ucon = Array(NLPModels.get_lcon(m)), Array(NLPModels.get_ucon(m))
    lvar, uvar = Array(NLPModels.get_lvar(m)), Array(NLPModels.get_uvar(m))
    viol_eq = maximum(abs.(ifelse.(lcon .== ucon, c .- lcon, 0.0)); init = 0.0)
    viol_con = maximum(max.(lcon .- c, c .- ucon, 0); init = 0.0)
    viol_var = maximum(max.(lvar .- x, x .- uvar, 0); init = 0.0)
    return (viol_eq, viol_con, viol_var)
end

function result_row(case, m, stats, t_compile, n_solves)
    c = stats.counters
    viol = try
        violation(m, stats)
    catch err
        @warn "$case: could not evaluate the violation of the returned point" exception = err
        (NaN, NaN, NaN)
    end
    return Any[
        case, Int(stats.status), stats.iter, dims(m)...,
        stats.objective, stats.primal_feas, stats.dual_feas,
        c.factorization_cnt, c.backsolve_cnt, c.ir,
        c.obj_cnt, c.obj_grad_cnt, c.con_cnt, c.con_jac_cnt, c.lag_hess_cnt,
        c.total_time, c.linear_solver_time, c.eval_function_time, c.init_time,
        t_compile, n_solves,
        viol...,
    ]
end

error_row(case, m) = failure_row(case, ERROR_STATUS, m === nothing ? (0, 0, 0, 0) : dims(m))

"The row of an instance that has no solve to report; `d` is `(nvar, ncon, nnzj, nnzh)`."
function failure_row(case, status, d)
    row = Any[NaN for _ in HEADER]
    row[1:3] = [case, status, 0]
    row[4:7] .= d
    for name in ("n_fact", "n_backsolve", "n_ir", "n_obj", "n_grad", "n_con", "n_jac", "n_hess", "n_solves")
        row[findfirst(==(name), HEADER)] = 0
    end
    return row
end

function write_meta(path, fields)
    meta = Dict{String, Any}(
        "timestamp" => string(Dates.now()),
        "host" => gethostname(),
        "julia_version" => string(VERSION),
        "cpu_model" => String(strip(Sys.cpu_info()[1].model)),
        "cpu_threads" => Sys.CPU_THREADS,
        "memory_gb" => round(Sys.total_memory() / 2^30; digits = 1),
        "os" => string(Sys.KERNEL, " ", Sys.ARCH),
    )
    cores = physical_cores()
    cores === nothing || (meta["cpu_cores"] = cores)
    # the versions of the loaded packages that the active project depends on directly
    direct = keys(get(TOML.parsefile(Base.active_project()), "deps", Dict{String, Any}()))
    for mod in Base.loaded_modules_array()
        name = string(nameof(mod))
        v = pkgversion(mod)
        name in direct && v !== nothing && (meta["pkg_" * name] = string(v))
    end
    merge!(meta, fields)
    open(io -> TOML.print(io, meta; sorted = true), path, "w")
    return
end

# The number of physical cores, where the system reports the topology (Linux).
function physical_cores()
    isfile("/proc/cpuinfo") || return nothing
    cores = Set{Tuple{String, String}}()
    socket = ""
    for line in eachline("/proc/cpuinfo")
        m = match(r"^(physical id|core id)\s*:\s*(\S+)", line)
        m === nothing && continue
        m[1] == "physical id" ? (socket = m[2]) : push!(cores, (socket, m[2]))
    end
    return isempty(cores) ? nothing : length(cores)
end

# Model types this process has already solved with a configuration.
const WARMED = Set{Tuple{String, DataType}}()

# Seconds of Julia compilation tolerated inside a recorded solve, and the most solves of one
# instance by one configuration.
const COMPILE_TOLERANCE = 0.01
const MAX_SOLVES = 3

# A first solve of a model type below this many seconds is repeated in any case: loading device
# code costs a fraction of a second that Julia's compilation timer does not see, which matters
# for a short solve and not for a long one.
const REPEAT_BELOW = 10.0

# One solve, with the time Julia spent compiling during it, in seconds.
function timed_solve(m, kwargs)
    Base.cumulative_compile_timing(true)
    before = Base.cumulative_compile_time_ns()[1]
    run_solver = get(kwargs, :solver, madnlp)
    stats = try
        run_solver(m; (k => v for (k, v) in pairs(kwargs) if k != :solver)...)
    finally
        Base.cumulative_compile_timing(false)
    end
    return stats, (Base.cumulative_compile_time_ns()[1] - before) / 1.0e9
end

"""
    solve(m, config, kwargs; warmup = true, guard) -> (stats, t_compile, n_solves)

Solve `m` and return the statistics of a solve that carries no compilation. The solve is
repeated while Julia compiled during it, and once more if it was short and the first time
`config` met the type of `m` in this process. A solve that reaches the wall limit is kept as
it is. `t_compile` is the
compilation time of the solve that is returned.
"""
function solve(m, config, kwargs; warmup::Bool = true, guard = f -> f())
    first_of_type = (config, typeof(m)) ∉ WARMED
    stats, t_compile = guard(() -> timed_solve(m, kwargs))
    push!(WARMED, (config, typeof(m)))
    n = 1
    while warmup && n < MAX_SOLVES && stats.status != MadNLP.MAXIMUM_WALLTIME_EXCEEDED &&
            (t_compile > COMPILE_TOLERANCE || (first_of_type && n == 1 && stats.counters.total_time < REPEAT_BELOW))
        stats, t_compile = guard(() -> timed_solve(m, kwargs))
        n += 1
    end
    return stats, t_compile, n
end

"""
    solve_case(case, build, configs; backend, options, warmup, prepare, release, reclaim, report)

Build the model of `case` and solve it with every configuration of `configs`
(`name => options`). Returns `name => row` for each, where `row` is a row of a result file.
`prepare` is called with the model before its solves, to compile what can be compiled
outside the timed region; `release` with the model after its solves, for models that hold outside
resources; `reclaim` afterwards, to return device memory before the next model is built.
`guard` wraps every timed solve. `report` is called with `(nvar, ncon, nnzj, nnzh)` once the
model is built and with `name => row` after each configuration, so that a process watching
this one knows how far it got if it never returns.
"""
function solve_case(
        case, build, configs; backend = nothing, options = (;), warmup::Bool = true,
        prepare = m -> nothing, release = m -> nothing, reclaim = () -> nothing, report = event -> nothing,
        guard = f -> f(),
    )
    m = try
        model = build(backend)
        report(dims(model))
        prepare(model)
        model
    catch err
        @warn "$case: the model could not be built" exception = (err, catch_backtrace())
        nothing
    end
    rows = map(configs) do (config, opts)
        row = if m === nothing
            error_row(case, m)
        else
            try
                result_row(case, m, solve(m, config, (; options..., opts...); warmup, guard)...)
            catch err
                @warn "$case [$config]: the solve threw" exception = (err, catch_backtrace())
                error_row(case, m)
            end
        end
        report(config => row)
        config => row
    end
    m === nothing || release(m)
    m = nothing
    GC.gc()
    reclaim()
    return rows
end

"""
    run_benchmark(solve_on, sets, configs; outdir, slots = [1], meta, meta_of, overwrite)

Solve every case of `sets` (`set => case names`) with every configuration of `configs`
(names) and write `<outdir>/<set>_<config>.csv`, with a metadata file beside each.

`solve_on(slot, set, case, configs)` solves one case with the named configurations on the
worker `slot` and returns `name => row` for each. The cases of all the sets form one queue,
handed out to the slots as they become free, so that a slow instance holds up one worker
only. Cases with the same `group(set, case)` go to one worker in turn, so that instances of
one compiled model type are compiled once. This process collects the rows and writes the files after every case, so an
interrupted run keeps its rows.

- `meta`: extra fields for the metadata files; `meta_of(config)`: fields of one configuration.
- `overwrite`: start the result files afresh. By default a case already present in a result
  file is skipped, so that an interrupted run is resumed by the same command.
"""
function run_benchmark(
        solve_on, sets, configs; outdir::AbstractString, slots = [1], group = (set, case) -> case,
        meta = Dict{String, Any}(), meta_of = config -> Dict{String, Any}(), overwrite::Bool = false,
    )
    mkpath(outdir)
    file(set, config) = joinpath(outdir, "$(set)_$(config).csv")
    rows = Dict{Tuple{String, String}, Matrix{Any}}()
    queue = Channel{Vector{Tuple{String, String, Vector{String}}}}(Inf)
    for (set, cases) in sets
        done = Dict{String, Set{String}}()
        for config in configs
            path = file(set, config)
            rows[(set, config)] = reshape(Any[HEADER...], 1, :)
            done[config] = Set{String}()
            if !overwrite && isfile(path) && filesize(path) > 0
                old = readdlm(path, ',', Any)
                old[1, :] == HEADER || error("$path has different columns; move it away or pass overwrite = true")
                rows[(set, config)] = old
                union!(done[config], string.(old[2:end, 1]))
            end
            write_meta(
                path * ".meta.toml",
                merge(Dict{String, Any}("set" => set, "config" => config, "workers" => length(slots)), meta, meta_of(config)),
            )
        end
        groups = Dict{Any, Vector{Tuple{String, String, Vector{String}}}}()
        order = []
        for case in cases
            todo = [config for config in configs if case ∉ done[config]]
            isempty(todo) && continue
            key = group(set, case)
            haskey(groups, key) || push!(order, key)
            push!(get!(groups, key, []), (set, case, todo))
        end
        foreach(key -> put!(queue, groups[key]), order)
    end
    close(queue)
    written = ReentrantLock()
    @sync for slot in slots
        @async for tasks in queue, (set, case, todo) in tasks
            results = try
                solve_on(slot, set, case, todo)
            catch err   # the process failed, not the solver
                @warn "$case: worker failed" exception = (err, catch_backtrace())
                [config => error_row(case, nothing) for config in todo]
            end
            lock(written) do
                for (config, row) in results
                    @info @sprintf("%-11s %-32s %-22s status %4d  iter %5d  time %10.3f s", set, case, config, row[2], row[3], row[T_TOTAL])
                    rows[(set, config)] = vcat(rows[(set, config)], permutedims(row))
                    writedlm(file(set, config), rows[(set, config)], ',')
                end
            end
        end
    end
    return [file(set, config) for (set, _) in sets for config in configs]
end

# ------------------------------------------------------------------------------------------
# Reading and scoring result files

"The rows of a result file, given by its path or its contents, as a vector of `NamedTuple`s."
read_results(path::AbstractString) = open(read_results, path)
function read_results(io::IO)
    data = readdlm(io, ',', Any)
    names = Tuple(Symbol.(data[1, :]))
    return [NamedTuple{names}(Tuple(data[i, :])) for i in axes(data, 1)[(begin + 1):end]]
end

"The metadata written beside the result file `path`, as a dictionary."
read_meta(path::AbstractString) = TOML.parsefile(path * ".meta.toml")

"The result files of a directory: `(set, config) => path`."
function result_files(dir::AbstractString)
    files = Dict{Tuple{String, String}, String}()
    for f in sort!(readdir(dir))
        m = match(r"^([a-z]+)_(.+)\.csv$", f)
        m === nothing || (files[(m[1], m[2])] = joinpath(dir, f))
    end
    return files
end

"""
    summarize(io, dir; wall = 7200.0)

Print, as Markdown, the machines of the runs in `dir` and, per set and configuration, the
instances solved and the shifted geometric mean of the solve time by size class.
"""
function summarize(io::IO, dir::AbstractString; wall = 7200.0)
    files = result_files(dir)
    println(io, "### ", basename(rstrip(dir, '/')), "\n")
    machines = unique!([hardware_line(read_meta(f)) for f in values(files) if isfile(f * ".meta.toml")])
    foreach(m -> println(io, "- ", m), machines)
    isempty(machines) || println(io)
    for set in sort!(unique(first.(keys(files))))
        println(io, "**", set, "** (solved, SGM in seconds; a failure counts as ", round(Int, wall), " s)\n")
        println(io, "| configuration | small | medium | large | total |")
        println(io, "|---|---|---|---|---|")
        for (s, config) in sort!(collect(keys(files)))
            s == set || continue
            rows = read_results(files[(s, config)])
            cells = map((:small, :medium, :large, :total)) do class
                sel = class == :total ? rows : [r for r in rows if size_class(r) == class]
                isempty(sel) ? "–" : @sprintf("%d / %d, %.2f", count(solved, sel), length(sel), sgm(score_time.(sel, wall)))
            end
            println(io, "| ", config, " | ", join(cells, " | "), " |")
        end
        println(io)
    end
    return
end

"One line that names the machine of a run, from its metadata."
function hardware_line(meta)
    cores = haskey(meta, "cpu_cores") ? "$(meta["cpu_cores"]) cores, " : ""
    cpu = "$(meta["cpu_model"]) ($(cores)$(meta["cpu_threads"]) threads), $(meta["memory_gb"]) GB of memory"
    gpu = haskey(meta, "gpu_name") ? "; $(meta["gpu_name"]), $(meta["gpu_memory_gb"]) GB, driver $(meta["gpu_driver"]), CUDA runtime $(meta["cuda_runtime"])" : ""
    return "$(meta["host"]): $(cpu)$(gpu)"
end

"`SOLVE_SUCCEEDED` or `SOLVED_TO_ACCEPTABLE_LEVEL`."
solved(r) = r.status == Int(MadNLP.SOLVE_SUCCEEDED) || r.status == Int(MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)

"The time an instance is scored with: its solve time if solved, the wall limit otherwise."
score_time(r, wall) = solved(r) && r.t_total isa Real && isfinite(r.t_total) ? min(Float64(r.t_total), wall) : wall

"Shifted geometric mean of `times` with shift `shift`."
sgm(times; shift = 10.0) = isempty(times) ? NaN : exp(sum(t -> log(t + shift), times) / length(times)) - shift

"Size class of an instance by the nonzeros of its Jacobian and Hessian together."
size_class(r) = (n = r.nnzj + r.nnzh; n < 2^18 ? :small : n < 2^20 ? :medium : :large)

end # module
