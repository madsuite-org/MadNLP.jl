# Draw performance profiles of the solve time from result files: one figure per set, one curve
# per directory and configuration that has that set.
#
#     julia --project=benchmark/plots benchmark/plots/profiles.jl DIR [DIR ...] [--out=DIR]
#
# Each DIR holds result files, for instance one directory per machine under benchmark/results,
# or the results of two commits. The figures are written as PNG and PDF to `--out` (default
# the working directory). An instance counts as solved with MadNLP status 1 or 2; times are
# floored at 1 ms so that a ratio stays finite.

using BenchmarkProfiles, DelimitedFiles, Plots

function read_times(path)
    data = readdlm(path, ',', Any)
    col(name) = findfirst(==(name), vec(data[1, :]))
    case, status, time = col("case"), col("status"), col("t_total")
    return Dict(
        string(data[i, case]) => (data[i, status] in (1, 2) && data[i, time] isa Real ? max(Float64(data[i, time]), 1.0e-3) : -1.0)
            for i in axes(data, 1)[(begin + 1):end]
    )
end

function main(args)
    dirs = filter(!startswith("--"), args)
    out = something(findlast(startswith("--out="), args) |> i -> i === nothing ? nothing : args[i][7:end], pwd())
    isempty(dirs) && error("usage: profiles.jl DIR [DIR ...] [--out=DIR]")
    mkpath(out)
    series = Dict{String, Vector{Pair{String, Dict{String, Float64}}}}()   # set => label => times
    for dir in dirs, f in sort!(readdir(dir))
        m = match(r"^([a-z]+)_(.+)\.csv$", f)
        m === nothing && continue
        label = length(dirs) == 1 ? String(m[2]) : "$(basename(rstrip(dir, '/'))) $(m[2])"
        push!(get!(series, String(m[1]), []), label => read_times(joinpath(dir, f)))
    end
    for (set, curves) in sort!(collect(series); by = first)
        if length(curves) < 2
            @info "$set: one curve only, nothing to compare"
            continue
        end
        cases = sort!(collect(intersect((keys(last(c)) for c in curves)...)))
        T = [last(c)[case] for case in cases, c in curves]
        performance_profile(PlotsBackend(), T, first.(curves); title = "$set: solve time, $(length(cases)) instances", legend = :bottomright)
        for ext in ("png", "pdf")
            savefig(joinpath(out, "profile_$set.$ext"))
        end
        @info "$set: $(length(curves)) curves over $(length(cases)) instances"
    end
    return
end

main(ARGS)
