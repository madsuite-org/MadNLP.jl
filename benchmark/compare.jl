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
using Printf

# The results of one side on their own: used when there is nothing to compare them with.
function report(io, rs::ResultSet; wall)
    for device in sort!(collect(keys(rs.machines)))
        println(io, "- ", solver_name(device), " runs on ", join(rs.machines[device], "; "))
    end
    println(io, "\nSGM is the shifted geometric mean of the solve time in seconds (shift 10 s, a failure counts as ", round(Int, wall), " s).\n")
    println(io, "| solver | set | configuration | instances | solved | SGM |\n|---|---|---|---|---|---|")
    for key in sort!(collect(keys(rs.rows)))
        rows = collect(values(rs.rows[key]))
        @printf(io, "| %s | %s | %s | %d | %d | %.2f |\n", solver_name(key[1]), key[2], key[3], length(rows), count(solved, rows), sgm(score_time.(rows, wall)))
    end
    return
end

function compare(io, base::ResultSet, cand::ResultSet; wall, threshold)
    println(io, "## MadNLP benchmark\n")
    println(io, "| | base | candidate |\n|---|---|---|")
    println(io, "| results of | ", base.label, " | ", cand.label, " |")
    devices = sort!(collect(union(keys(base.machines), keys(cand.machines))))
    for device in devices
        println(io, "| ", solver_name(device), " runs on | ", join(get(base.machines, device, ["–"]), "<br>"), " | ", join(get(cand.machines, device, ["–"]), "<br>"), " |")
    end
    different = [d for d in devices if haskey(base.machines, d) && haskey(cand.machines, d) && base.machines[d] != cand.machines[d]]
    isempty(different) || println(io, "\n**The ", join(solver_name.(different), " and "), " runs of the two sides were made on different machines: their times are not comparable.** Solved counts and iteration counts are.")
    common = sort!(collect(intersect(keys(base.rows), keys(cand.rows))))
    if isempty(common)
        println(io, "\nNo solver, set and configuration in common.")
        return
    end
    println(io, "\nOver the instances both sides have run. SGM is the shifted geometric mean of the solve time in seconds (shift 10 s, a failure counts as ", round(Int, wall), " s); the time ratio is the candidate's shifted time over the base's, so below 1 the candidate is faster.\n")
    println(io, "| solver | set | configuration | instances | solved: base | solved: candidate | SGM: base | SGM: candidate | time ratio | iterations ratio |")
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
        @printf(io, "| %s | %s | %s | %d | %d | %d | %.2f | %.2f | %.2f | %s |\n", solver_name(key[1]), key[2], key[3], length(cases), na, nb, ga, gb, ratio, isnan(iters) ? "–" : @sprintf("%.2f", iters))
        (nb < na || ratio > threshold) && push!(regressed, "$(solver_name(key[1])) / $(key[2]) / $(key[3])")
        for c in cases
            solved(a[c]) == solved(b[c]) && continue
            push!(changed, @sprintf("| %s | %s | %s | %s | %d | %d |", solver_name(key[1]), key[2], key[3], c, a[c].status, b[c].status))
        end
    end
    isempty(regressed) || println(io, "\n**Possible regression** (fewer instances solved, or a time ratio above ", threshold, "): ", join(regressed, "; "), ".")
    if !isempty(changed)
        println(io, "\n<details><summary>Instances whose outcome changed (", length(changed), ")</summary>\n")
        println(io, "| solver | set | configuration | instance | status: base | status: candidate |\n|---|---|---|---|---|---|")
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
