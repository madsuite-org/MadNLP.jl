# Summarize result files: instances solved and shifted geometric mean of the solve time, by
# size class, for every set and configuration found in the given directories.
#
#     julia --project=benchmark benchmark/summary.jl [DIR ...] [--wall=7200]
#
# Without a directory, every directory under benchmark/results is summarized.

using MadNLP
include(joinpath(@__DIR__, "src", "MadNLPBenchmark.jl"))
using .MadNLPBenchmark
using Printf

function result_files(dir)
    files = Dict{Tuple{String, String}, String}()   # (set, config) => path
    for f in sort!(readdir(dir))
        m = match(r"^([a-z]+)_(.+)\.csv$", f)
        m === nothing || (files[(m[1], m[2])] = joinpath(dir, f))
    end
    return files
end

function summarize(io, dir; wall)
    files = result_files(dir)
    println(io, "### ", basename(rstrip(dir, '/')), "\n")
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

function main(args)
    wall = 7200.0
    dirs = String[]
    for a in args
        m = match(r"^--wall=(.*)$", a)
        m === nothing ? push!(dirs, a) : (wall = parse(Float64, m[1]))
    end
    if isempty(dirs)
        root = joinpath(@__DIR__, "results")
        dirs = isdir(root) ? [joinpath(root, d) for d in sort!(readdir(root)) if isdir(joinpath(root, d))] : String[]
    end
    isempty(dirs) && error("no result directory")
    foreach(d -> summarize(stdout, d; wall), dirs)
    return
end

main(ARGS)
