# Write the benchmark report, a static site, from result directories.
#
#     julia --project=benchmark benchmark/report.jl ROOT SITE [--base=REV|DIR] [--wall=7200] [--title=TEXT] [--link=URL]
#
# ROOT holds one directory of result files per machine, as run.jl writes them. SITE is the
# directory of the report. `--base` adds the results of a directory or an archived revision.

using MadNLP
using MadNLPBenchmark

function main(args)
    opt = Dict{String, String}("wall" => "7200", "title" => "MadNLP benchmark")
    pos = String[]
    for a in args
        m = match(r"^--(base|wall|title|link)=(.*)$", a)
        m === nothing ? push!(pos, a) : (opt[m[1]] = m[2])
    end
    length(pos) == 2 || error("usage: report.jl ROOT SITE [--base=REV|DIR] [--wall=7200] [--title=TEXT] [--link=URL]")
    base = haskey(opt, "base") ? locate(opt["base"], nothing) : nothing
    write_report(pos[2], pos[1]; base, wall = parse(Float64, opt["wall"]), title = opt["title"], link = get(opt, "link", nothing), reference = ipopt_reference(), finished = true)
    return
end

main(ARGS)
