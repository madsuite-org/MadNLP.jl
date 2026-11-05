# Compare two sets of result files, a base and a candidate, set by set and configuration by
# configuration, and print the comparison as Markdown.
#
#     julia --project=benchmark benchmark/compare.jl BASE CANDIDATE [--wall=7200] [--hardware=TAG]
#                                                      [--threshold=1.1] [--candidate-label=TEXT]
#
# BASE and CANDIDATE are each a directory of result files or a git revision of MadNLP. A
# revision is looked up in the result archive, the `benchmark-results` branch, which
# stores the results of a commit under `<hardware>/<commit>/`; if that commit has no
# archived results, the nearest ancestor that has some is used and named in the output.

using MadNLP
include(joinpath(@__DIR__, "src", "MadNLPBenchmark.jl"))
using .MadNLPBenchmark
using Printf

const ARCHIVE_BRANCH = "benchmark-results"
const ROOT = dirname(@__DIR__)
git(args...) = readchomp(`git -C $ROOT $args`)

# Materialize the archived results of `rev` (or of its nearest archived ancestor) for
# `hardware` in a temporary directory. Returns the directory and the commit used.
function archived(rev, hardware; depth = 200)
    remote = get(ENV, "MADNLP_BENCHMARK_REMOTE", "origin")
    run(`git -C $ROOT fetch --quiet $remote $ARCHIVE_BRANCH`)
    hardware in split(git("ls-tree", "--name-only", "FETCH_HEAD"), '\n') ||
        error("the archive has no results for the hardware $hardware")
    have = Set(split(git("ls-tree", "--name-only", "FETCH_HEAD:$hardware"), '\n'; keepempty = false))
    for sha in split(git("rev-list", "--first-parent", "-n", string(depth), rev), '\n')
        sha in have || continue
        dir = mktempdir()
        for f in split(git("ls-tree", "--name-only", "FETCH_HEAD:$hardware/$sha"), '\n'; keepempty = false)
            write(joinpath(dir, f), read(`git -C $ROOT show FETCH_HEAD:$hardware/$sha/$f`))
        end
        return dir, String(sha)
    end
    error("no archived results on $hardware for $rev or its last $depth ancestors")
end

function locate(arg, hardware)
    isdir(arg) && return arg, arg
    hardware === nothing && error("$arg is not a directory; to compare with an archived revision pass --hardware=TAG")
    dir, sha = archived(arg, hardware)
    return dir, sha == git("rev-parse", arg) ? "`$(sha[1:8])`" : "`$(sha[1:8])` (nearest archived ancestor of `$arg`)"
end

function results(dir)
    out = Dict{Tuple{String, String}, Dict{String, Any}}()
    for f in readdir(dir)
        m = match(r"^([a-z]+)_(.+)\.csv$", f)
        m === nothing && continue
        out[(m[1], m[2])] = Dict(string(r.case) => r for r in read_results(joinpath(dir, f)))
    end
    return out
end

function compare(io, base, cand, base_label, cand_label; wall, threshold)
    A, B = results(base), results(cand)
    common = sort!(collect(intersect(keys(A), keys(B))))
    println(io, "## MadNLP benchmark\n")
    println(
        io, "Base: ", base_label, ". Candidate: ", cand_label, ". Over the instances both have run; ",
        "SGM is the shifted geometric mean of the solve time in seconds (shift 10 s, a failure counts as ", round(Int, wall), " s).\n"
    )
    isempty(common) && (println(io, "No set and configuration in common."); return)
    println(io, "| set | configuration | instances | solved (base) | solved | SGM (base) | SGM | ratio | iterations ratio |")
    println(io, "|---|---|---|---|---|---|---|---|---|")
    changed = String[]
    regressed = String[]
    for key in common
        a, b = A[key], B[key]
        cases = sort!(collect(intersect(keys(a), keys(b))))
        isempty(cases) && continue
        ga = sgm([score_time(a[c], wall) for c in cases]); gb = sgm([score_time(b[c], wall) for c in cases])
        both = [c for c in cases if solved(a[c]) && solved(b[c]) && a[c].iters > 0]
        # geometric mean of the iteration ratios over the instances both solve
        iters = isempty(both) ? NaN : exp(sum(c -> log(b[c].iters / a[c].iters), both) / length(both))
        na, nb = count(c -> solved(a[c]), cases), count(c -> solved(b[c]), cases)
        ratio = (gb + 10) / (ga + 10)
        @printf(io, "| %s | %s | %d | %d | %d | %.2f | %.2f | %.2f | %.2f |\n", key[1], key[2], length(cases), na, nb, ga, gb, ratio, iters)
        (nb < na || ratio > threshold) && push!(regressed, "`$(key[1])` / `$(key[2])`")
        for c in cases
            solved(a[c]) == solved(b[c]) && continue
            push!(changed, @sprintf("| %s | %s | %s | %d | %d |", key[1], key[2], c, a[c].status, b[c].status))
        end
    end
    println(io, "\nThe ratio is the candidate's shifted time over the base's: below 1 the candidate is faster.")
    isempty(regressed) || println(io, "\n**Possible regression** (fewer instances solved, or a ratio above ", threshold, "): ", join(regressed, ", "), ".")
    if !isempty(changed)
        println(io, "\nInstances whose outcome changed (MadNLP status):\n")
        println(io, "| set | configuration | instance | base | candidate |\n|---|---|---|---|---|")
        foreach(l -> println(io, l), changed)
    end
    return
end

function main(args)
    opt = Dict("wall" => "7200", "threshold" => "1.1")
    pos = String[]
    for a in args
        m = match(r"^--(wall|hardware|threshold|candidate-label)=(.*)$", a)
        m === nothing ? push!(pos, a) : (opt[m[1]] = m[2])
    end
    length(pos) == 2 || error("usage: compare.jl BASE CANDIDATE [--wall=7200] [--hardware=TAG] [--threshold=1.1] [--candidate-label=TEXT]")
    hardware = get(opt, "hardware", nothing)
    base, base_label = locate(pos[1], hardware)
    cand, cand_label = locate(pos[2], hardware)
    compare(
        stdout, base, cand, base_label, get(opt, "candidate-label", cand_label);
        wall = parse(Float64, opt["wall"]), threshold = parse(Float64, opt["threshold"])
    )
    return
end

main(ARGS)
