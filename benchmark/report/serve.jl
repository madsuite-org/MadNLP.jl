# Build the reports of running benchmarks, on a server outside CI.
#
#     julia --project=benchmark/report benchmark/report/serve.jl LIVE SITE --repo=CLONE
#
# LIVE holds a folder `pr-<N>` per pull request, kept up to date by the benchmark job, which
# replaces `run.json` in it after every copy. Each new `run.json` rebuilds `SITE/pr-<N>`, and
# `SITE/index.html` lists the benchmarks and `SITE/status.json` gives the state of every pull
# request. CLONE is a clone of the repository
# with `origin/benchmark-results`, for the archived results; this script never fetches.

using BenchmarkReport, Dates, FileWatching, TOML

function json_field(text, name)
    m = match(Regex("\"$name\"\\s*:\\s*\"([^\"]*)\""), text)
    return m === nothing ? "" : String(m[1])
end

const RUNS = Dict{String, NamedTuple}()

function build(live, site, name)
    text = read(joinpath(live, name, "run.json"), String)
    repository, pr, base = json_field(text, "repository"), json_field(text, "pr"), json_field(text, "base")
    (occursin(r"^[\w.-]+/[\w.-]+$", repository) && occursin(r"^\d+$", pr)) || error("$name: bad run.json")
    finished = json_field(text, "state") != "running"
    write_report(
        joinpath(site, name), joinpath(live, name);
        base = occursin(r"^[0-9a-f]{40}$", base) ? locate(base, "master") : nothing, reference = ipopt_reference(),
        wall = 7200.0, title = "MadNLP benchmark report (pr #$pr)", link = "https://github.com/$repository/pull/$pr", finished,
    )
    updated = Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ")
    RUNS[name] = (; name, pr, commit = json_field(text, "commit"), state = json_field(text, "state"), updated)
    write_index(site, collect(values(RUNS)); repository)
    entries = ["\"$(r.name)\":{\"state\":\"$(r.state)\",\"updated\":\"$(r.updated)\"}" for r in sort!(collect(values(RUNS)); by = r -> r.name)]
    write(joinpath(site, "status.json.tmp"), "{", join(entries, ","), "}\n")
    chmod(joinpath(site, "status.json.tmp"), 0o644)
    mv(joinpath(site, "status.json.tmp"), joinpath(site, "status.json"); force = true)
    return
end

function watch(live, site, name)
    while true
        try
            isfile(joinpath(live, name, "run.json")) && build(live, site, name)
        catch err
            @warn "$name: the report was not updated" exception = err
        end
        file, event = watch_folder(joinpath(live, name))
        # the copy is complete when run.json has been replaced
        while file != "run.json"
            file, event = watch_folder(joinpath(live, name))
        end
    end
    return
end

function main(args)
    options = filter(startswith("--"), args)
    live, site = filter(!startswith("--"), args)
    for option in options
        m = match(r"^--repo=(.*)$", option)
        m === nothing ? error("unknown option $option") : (BenchmarkReport.REPOSITORY[] = m[1])
    end
    BenchmarkReport.FETCH[] = false
    watching = Set{String}()
    while true
        for name in readdir(live)
            if occursin(r"^pr-\d+$", name) && isdir(joinpath(live, name)) && name ∉ watching
                push!(watching, name)
                errormonitor(@async watch(live, site, name))
            end
        end
        watch_folder(live)
    end
    return
end

main(ARGS)
