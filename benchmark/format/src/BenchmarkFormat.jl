"""
    BenchmarkFormat

The result files of `benchmark/run.jl`: reading and scoring them. It has no solver among its
dependencies, so that the archive job and the website (madsuite-org/benchmark-report) need none.
"""
module BenchmarkFormat

using Dates, DelimitedFiles, Printf, TOML

export HSL_VERSION, BENCHMARK_VERSION, ARCHIVE_BRANCH, read_results, read_meta, result_files, summarize, solver_name,
    hardware_line, solved, score_time, sgm, size_class

const BENCHMARK_VERSION, HSL_VERSION = let project = TOML.parsefile(joinpath(dirname(dirname(@__DIR__)), "Project.toml"))
    project["version"], project["benchmark"]["hsl"]
end

# Status recorded for an instance whose worker was killed for overrunning the wall limit, and
# for one that could not be built or whose solve threw.
const KILLED_STATUS = -100
const ERROR_STATUS = -101

# The status codes of MadNLP that count as solved.
const SOLVE_SUCCEEDED = 1
const SOLVED_TO_ACCEPTABLE_LEVEL = 2

const ARCHIVE_BRANCH = "benchmark-results"

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
instances solved and the shifted geometric mean of the solve time by size class. The wall limit
recorded in the metadata takes precedence over `wall`.
"""
function summarize(io::IO, dir::AbstractString; wall = 7200.0)
    files = result_files(dir)
    metas = [read_meta(f) for f in values(files) if isfile(f * ".meta.toml")]
    solvers = unique!(sort!([solver_name(m["device"]) for m in metas]))
    println(io, "### ", join([solvers; basename(rstrip(dir, '/'))], ", ", " on "), "\n")
    machines = unique!([hardware_line(m) for m in metas])
    limits = unique!([read_meta(f)["wall_limit_s"] for f in values(files) if isfile(f * ".meta.toml") && haskey(read_meta(f), "wall_limit_s")])
    length(limits) == 1 && (wall = only(limits))
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

end
