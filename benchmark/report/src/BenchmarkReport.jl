"""
    BenchmarkReport

Reading, scoring and reporting the results of `benchmark/run.jl`. It has no solver among its
dependencies, so that a report can be built on a machine without them. Result files are treated
as untrusted input, since the report is public.
"""
module BenchmarkReport

using Dates, DelimitedFiles, Printf, TOML

export write_index, ResultSet, locate, HSL_VERSION, ipopt_reference, BENCHMARK_VERSION, short, write_report, read_results, read_meta,
    result_files, summarize, solver_name, hardware_line, solved, score_time, sgm, size_class

const BENCHMARK_VERSION, HSL_VERSION = let project = TOML.parsefile(joinpath(dirname(dirname(@__DIR__)), "Project.toml"))
    project["version"], project["benchmark"]["hsl"]
end

# Status recorded for an instance whose worker was killed for overrunning the wall limit, and
# for one that could not be built or whose solve threw.
const KILLED_STATUS = -100
const ERROR_STATUS = -101

# The status codes of MadNLP that the report tells apart.
const SOLVE_SUCCEEDED = 1
const SOLVED_TO_ACCEPTABLE_LEVEL = 2
const MAXIMUM_ITERATIONS_EXCEEDED = 6
const MAXIMUM_WALLTIME_EXCEEDED = 7
const STATUS_NAMES = Dict(
    3 => "search direction too small", 4 => "diverging iterates", 5 => "infeasible problem detected",
    6 => "iteration limit", 7 => "time limit", -1 => "restoration failed", -2 => "invalid number detected",
    -3 => "error in step computation", -4 => "not enough degrees of freedom", -5 => "user stop", -6 => "internal error",
    -7 => "invalid objective", -8 => "invalid gradient", -9 => "invalid constraints", -10 => "invalid Jacobian",
    -11 => "invalid Hessian",
)

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
    metas = [read_meta(f) for f in values(files) if isfile(f * ".meta.toml")]
    solvers = unique!(sort!([solver_name(m["device"]) for m in metas]))
    println(io, "### ", join([solvers; basename(rstrip(dir, '/'))], ", ", " on "), "\n")
    machines = unique!([hardware_line(m) for m in metas])
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

"The package that solves on `device`: MadNLP on the host, cuMadNLP on a CUDA device."
solver_name(device) = device == "cuda" ? "cuMadNLP" : "MadNLP"

"One line that names the machine of a run, from its metadata."
function hardware_line(meta)
    cores = haskey(meta, "cpu_cores") ? "$(meta["cpu_cores"]) cores, " : ""
    cpu = "$(meta["cpu_model"]) ($(cores)$(meta["cpu_threads"]) threads), $(meta["memory_gb"]) GB of memory"
    gpu = haskey(meta, "gpu_name") ? "; $(meta["gpu_name"]), $(meta["gpu_memory_gb"]) GB, driver $(meta["gpu_driver"]), CUDA runtime $(meta["cuda_runtime"])" : ""
    return "$(meta["host"]): $(cpu)$(gpu)"
end

"`SOLVE_SUCCEEDED` or `SOLVED_TO_ACCEPTABLE_LEVEL`."
solved(r) = r.status == SOLVE_SUCCEEDED || r.status == SOLVED_TO_ACCEPTABLE_LEVEL

"The time an instance is scored with: its solve time if solved, the wall limit otherwise."
score_time(r, wall) = solved(r) && r.t_total isa Real && isfinite(r.t_total) ? min(Float64(r.t_total), wall) : wall

"Shifted geometric mean of `times` with shift `shift`."
sgm(times; shift = 10.0) = isempty(times) ? NaN : exp(sum(t -> log(t + shift), times) / length(times)) - shift

"Size class of an instance by the nonzeros of its Jacobian and Hessian together."
size_class(r) = (n = r.nnzj + r.nnzh; n < 2^18 ? :small : n < 2^20 ? :medium : :large)

# ------------------------------------------------------------------------------------------
# Result sets

const ARCHIVE_BRANCH = "benchmark-results"
# The clone of the repository that holds the archive branch.
const REPOSITORY = Ref(dirname(dirname(dirname(@__DIR__))))
# The clone may belong to another user, and a server may keep it up to date by itself, with
# `FETCH[]` false.
const FETCH = Ref(true)
git_command(args...) = (root = REPOSITORY[]; `git -c safe.directory=$root -C $root $args`)
git(args...) = readchomp(git_command(args...))

# The revision of the archive branch, up to date if the clone may fetch; `nothing` without one.
function archive_revision()
    FETCH[] || return "origin/$ARCHIVE_BRANCH"
    remote = get(ENV, "MADNLP_BENCHMARK_REMOTE", "origin")
    return success(git_command("fetch", "--quiet", remote, ARCHIVE_BRANCH)) ? "FETCH_HEAD" : nothing
end
# a full commit hash as it is shown; any other revision as it is
short(rev) = occursin(r"^[0-9a-f]{40}$", rev) ? rev[1:8] : rev

# The results of one side: per (device, set, config) the rows by case, the machines per
# device, and what to call it.
struct ResultSet
    label::String
    rows::Dict{Tuple{String, String, String}, Dict{String, Any}}
    machines::Dict{String, Vector{String}}
    revs::Dict{Tuple{String, String}, String}      # (device, config) => what was run
    harness::Dict{Tuple{String, String}, String}   # (device, config) => version of this benchmark
    logs::Dict{Tuple{String, String, String, String}, String}   # (device, set, case, config) => address of the solver log
end

ResultSet(label) = ResultSet(label, Dict(), Dict(), Dict(), Dict(), Dict())

# A row as the report may use it: every column but the name of the instance is a number.
number(x) = x isa Real ? x : something(tryparse(Float64, string(x)), NaN)
clean_row(r) = NamedTuple{keys(r)}(Tuple(name === :case ? string(r[name]) : number(r[name]) for name in keys(r)))

# Files of the names below become pages and addresses of the report.
const DEVICES = ("cpu", "cuda", "amdgpu")
safe_name(s) = occursin(r"^[A-Za-z0-9_.+-]+$", s) && !startswith(s, ".")

function add!(rs::ResultSet, set, config, meta, rows)
    device = string(get(meta, "device", ""))
    (device in DEVICES && safe_name(set) && safe_name(config)) || return rs
    rs.rows[(device, set, config)] = Dict(string(r.case) => clean_row(r) for r in rows)
    abbreviate(commit) = replace(commit, r"^([0-9a-f]{8})[0-9a-f]{32}" => s"\1")
    commit = abbreviate(get(meta, "madnlp_commit", "unknown"))
    haskey(meta, "first_commit") && (commit = abbreviate(meta["first_commit"]) * " to " * commit)
    rs.harness[(device, config)] = get(meta, "benchmark_version", BENCHMARK_VERSION)
    rs.revs[(device, config)] = config == "ipopt" ? get(meta, "ipopt_version", "?") : commit
    line = hardware_line(meta)
    line in get!(rs.machines, device, String[]) || push!(rs.machines[device], line)
    return rs
end

function from_directory(dir, label)
    rs = ResultSet(label)
    for (root, _, _) in walkdir(dir), ((set, config), path) in result_files(root)
        isfile(path * ".meta.toml") && add!(rs, set, config, read_meta(path), read_results(path))
    end
    return rs
end

# The archived results of `rev`: for every device, those of the nearest commit on the
# first-parent history of `rev` that has results for that device. `nothing` if the archive
# has none.
function from_archive(rev; depth = 200)
    archive = archive_revision()
    archive === nothing && return nothing
    paths = split(git("ls-tree", "-r", "--name-only", archive), '\n'; keepempty = false)
    show(path) = read(git_command("show", "$archive:$path"), String)
    by_commit = Dict{String, Vector{Tuple{String, String, String}}}()   # commit => (path, set, config)
    for path in paths
        m = match(r"^[^/]+/([0-9a-f]{40})/([a-z]+)_(.+)\.csv$", path)
        m === nothing || push!(get!(by_commit, m[1], []), (String(path), m[2], m[3]))
    end
    rs = ResultSet("")
    used = Dict{String, String}()   # device => commit
    for sha in split(git("rev-list", "--first-parent", "-n", string(depth), rev), '\n')
        for (path, set, config) in get(by_commit, sha, [])
            meta = TOML.parse(show(path * ".meta.toml"))
            get!(used, meta["device"], sha) == sha || continue   # a nearer commit covers this device
            add!(rs, set, config, meta, read_results(IOBuffer(show(path))))
        end
    end
    isempty(used) && return nothing
    target = git("rev-parse", rev)
    # one entry per commit used; the devices are named only when they come from different commits
    commits = unique!(sort!(collect(values(used))))
    label = join(
        map(commits) do sha
            name = sha == target ? "`$(sha[1:8])`" : "`$(sha[1:8])`, the nearest archived ancestor of `$(short(rev))`"
            length(commits) == 1 ? name : name * " (" * join(sort!([d for (d, c) in used if c == sha]), ", ") * ")"
        end, "; ",
    )
    return ResultSet(label, rs.rows, rs.machines, rs.revs, rs.harness, rs.logs)
end

# The Ipopt reference of a version of this benchmark, `ipopt/<version>/` on the archive branch:
# one run for all the machines and commits that are compared under that version. `nothing` if
# it has not been archived.
function ipopt_reference(version = BENCHMARK_VERSION)
    archive = archive_revision()
    archive === nothing && return nothing
    show(path) = read(git_command("show", "$archive:$path"), String)
    rs = ResultSet("Ipopt reference")
    for path in split(git("ls-tree", "-r", "--name-only", archive, "ipopt/$version/"), '\n'; keepempty = false)
        m = match(r"/([a-z]+)_ipopt\.csv$", path)
        m === nothing || add!(rs, m[1], "ipopt", TOML.parse(show(path * ".meta.toml")), read_results(IOBuffer(show(path))))
        log = match(r"/logs/([a-z]+)/(.*)__ipopt\.txt$", path)
        log === nothing || (rs.logs[("cpu", log[1], log[2], "ipopt")] = String(path))
    end
    return isempty(rs.rows) ? nothing : rs
end

function locate(arg, label)
    isdir(arg) && return from_directory(arg, something(label, "`$arg`"))
    rs = from_archive(arg)
    return rs === nothing || label === nothing ? rs : ResultSet(label, rs.rows, rs.machines, rs.revs, rs.harness, rs.logs)
end

include("site.jl")

end
