# Compare two sets of results, a base and a candidate, and print the comparison as Markdown:
# the machines of both, and per device, set and configuration the instances solved, the
# shifted geometric mean times and their ratio.
#
#     julia --project=benchmark benchmark/compare.jl BASE CANDIDATE [--wall=7200] [--threshold=1.1]
#                                                    [--base-label=TEXT] [--candidate-label=TEXT]
#
# BASE and CANDIDATE are each a directory of result files (searched with its subdirectories,
# so `benchmark/results` covers every machine) or a git revision of MadNLP. A revision is
# looked up in the result archive, the `benchmark-results` branch, which stores the results
# of a commit under `<hardware>/<commit>/`. For a device the commit has no archived results
# for, the nearest ancestor that has some is used and named in the output. When the archive
# has nothing for BASE, the results of CANDIDATE are printed on their own.

using MadNLP
using MadNLPBenchmark
using Printf, TOML

const ARCHIVE_BRANCH = "benchmark-results"
const ROOT = dirname(@__DIR__)
git(args...) = readchomp(`git -C $ROOT $args`)
# a full commit hash as it is shown; any other revision as it is
short(rev) = occursin(r"^[0-9a-f]{40}$", rev) ? rev[1:8] : rev

# The results of one side: per (device, set, config) the rows by case, the machines per
# device, and what to call it.
struct ResultSet
    label::String
    rows::Dict{Tuple{String, String, String}, Dict{String, Any}}
    machines::Dict{String, Vector{String}}
end

function add!(rs::ResultSet, set, config, meta, rows)
    device = meta["device"]
    rs.rows[(device, set, config)] = Dict(string(r.case) => r for r in rows)
    line = hardware_line(meta)
    line in get!(rs.machines, device, String[]) || push!(rs.machines[device], line)
    return rs
end

function from_directory(dir, label)
    rs = ResultSet(label, Dict(), Dict())
    for (root, _, _) in walkdir(dir), ((set, config), path) in result_files(root)
        isfile(path * ".meta.toml") && add!(rs, set, config, read_meta(path), read_results(path))
    end
    return rs
end

# The archived results of `rev`: for every device, those of the nearest commit on the
# first-parent history of `rev` that has results for that device. `nothing` if the archive
# has none.
function from_archive(rev; depth = 200)
    remote = get(ENV, "MADNLP_BENCHMARK_REMOTE", "origin")
    success(`git -C $ROOT fetch --quiet $remote $ARCHIVE_BRANCH`) || return nothing
    paths = split(git("ls-tree", "-r", "--name-only", "FETCH_HEAD"), '\n'; keepempty = false)
    show(path) = read(`git -C $ROOT show FETCH_HEAD:$path`, String)
    by_commit = Dict{String, Vector{Tuple{String, String, String}}}()   # commit => (path, set, config)
    for path in paths
        m = match(r"^[^/]+/([0-9a-f]{40})/([a-z]+)_(.+)\.csv$", path)
        m === nothing || push!(get!(by_commit, m[1], []), (String(path), m[2], m[3]))
    end
    rs = ResultSet("", Dict(), Dict())
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
    return ResultSet(label, rs.rows, rs.machines)
end

function locate(arg, label)
    isdir(arg) && return from_directory(arg, something(label, "`$arg`"))
    rs = from_archive(arg)
    return rs === nothing || label === nothing ? rs : ResultSet(label, rs.rows, rs.machines)
end

# The results of one side on their own: used when there is nothing to compare them with.
function report(io, rs::ResultSet; wall)
    for device in sort!(collect(keys(rs.machines)))
        println(io, "- ", device, " runs on ", join(rs.machines[device], "; "))
    end
    println(io, "\nSGM is the shifted geometric mean of the solve time in seconds (shift 10 s, a failure counts as ", round(Int, wall), " s).\n")
    println(io, "| device | set | configuration | instances | solved | SGM |\n|---|---|---|---|---|---|")
    for key in sort!(collect(keys(rs.rows)))
        rows = collect(values(rs.rows[key]))
        @printf(io, "| %s | %s | %s | %d | %d | %.2f |\n", key..., length(rows), count(solved, rows), sgm(score_time.(rows, wall)))
    end
    return
end

function compare(io, base::ResultSet, cand::ResultSet; wall, threshold)
    println(io, "## MadNLP benchmark\n")
    println(io, "| | base | candidate |\n|---|---|---|")
    println(io, "| results of | ", base.label, " | ", cand.label, " |")
    devices = sort!(collect(union(keys(base.machines), keys(cand.machines))))
    for device in devices
        println(io, "| ", device, " runs on | ", join(get(base.machines, device, ["–"]), "<br>"), " | ", join(get(cand.machines, device, ["–"]), "<br>"), " |")
    end
    different = [d for d in devices if haskey(base.machines, d) && haskey(cand.machines, d) && base.machines[d] != cand.machines[d]]
    isempty(different) || println(io, "\n**The ", join(different, " and "), " runs of the two sides were made on different machines: their times are not comparable.** Solved counts and iteration counts are.")
    common = sort!(collect(intersect(keys(base.rows), keys(cand.rows))))
    if isempty(common)
        println(io, "\nNo device, set and configuration in common.")
        return
    end
    println(io, "\nOver the instances both sides have run. SGM is the shifted geometric mean of the solve time in seconds (shift 10 s, a failure counts as ", round(Int, wall), " s); the time ratio is the candidate's shifted time over the base's, so below 1 the candidate is faster.\n")
    println(io, "| device | set | configuration | instances | solved: base | solved: candidate | SGM: base | SGM: candidate | time ratio | iterations ratio |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    changed = String[]
    regressed = String[]
    for key in common
        a, b = base.rows[key], cand.rows[key]
        cases = sort!(collect(intersect(keys(a), keys(b))))
        isempty(cases) && continue
        ga = sgm([score_time(a[c], wall) for c in cases]); gb = sgm([score_time(b[c], wall) for c in cases])
        na, nb = count(c -> solved(a[c]), cases), count(c -> solved(b[c]), cases)
        ratio = (gb + 10) / (ga + 10)
        both = [c for c in cases if solved(a[c]) && solved(b[c]) && a[c].iters > 0]
        # geometric mean of the iteration ratios over the instances both solve
        iters = isempty(both) ? NaN : exp(sum(c -> log(b[c].iters / a[c].iters), both) / length(both))
        @printf(io, "| %s | %s | %s | %d | %d | %d | %.2f | %.2f | %.2f | %s |\n", key..., length(cases), na, nb, ga, gb, ratio, isnan(iters) ? "–" : @sprintf("%.2f", iters))
        (nb < na || ratio > threshold) && push!(regressed, "$(key[1]) / $(key[2]) / $(key[3])")
        for c in cases
            solved(a[c]) == solved(b[c]) && continue
            push!(changed, @sprintf("| %s | %s | %s | %s | %d | %d |", key..., c, a[c].status, b[c].status))
        end
    end
    isempty(regressed) || println(io, "\n**Possible regression** (fewer instances solved, or a time ratio above ", threshold, "): ", join(regressed, "; "), ".")
    if !isempty(changed)
        println(io, "\n<details><summary>Instances whose outcome changed (", length(changed), ")</summary>\n")
        println(io, "| device | set | configuration | instance | status: base | status: candidate |\n|---|---|---|---|---|---|")
        foreach(l -> println(io, l), changed)
        println(io, "\n</details>")
    end
    return
end

function main(args)
    opt = Dict{String, String}("wall" => "7200", "threshold" => "1.1")
    pos = String[]
    for a in args
        m = match(r"^--(wall|threshold|base-label|candidate-label)=(.*)$", a)
        m === nothing ? push!(pos, a) : (opt[m[1]] = m[2])
    end
    length(pos) == 2 || error("usage: compare.jl BASE CANDIDATE [--wall=7200] [--threshold=1.1] [--base-label=TEXT] [--candidate-label=TEXT]")
    wall = parse(Float64, opt["wall"])
    base = locate(pos[1], get(opt, "base-label", nothing))
    cand = locate(pos[2], get(opt, "candidate-label", nothing))
    cand === nothing && error("no results for $(pos[2])")
    if base === nothing
        println("## MadNLP benchmark\n")
        println("Results of ", cand.label, ". The archive has no results for `", short(pos[1]), "` or its ancestors to compare them with.\n")
        report(stdout, cand; wall)
    else
        compare(stdout, base, cand; wall, threshold = parse(Float64, opt["threshold"]))
    end
    return
end

main(ARGS)
