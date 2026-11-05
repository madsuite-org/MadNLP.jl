# Summarize result files: the machine, and per set and configuration the instances solved and
# the shifted geometric mean of the solve time, by size class.
#
#     julia --project=benchmark benchmark/summary.jl [DIR ...] [--wall=7200]
#
# Without a directory, every directory under benchmark/results is summarized.

using MadNLP
include(joinpath(@__DIR__, "src", "MadNLPBenchmark.jl"))
using .MadNLPBenchmark

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
