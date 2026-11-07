# Archive result files on the `benchmark-results` branch, under `<hardware>/<commit>/`.
#
#     julia --project=benchmark benchmark/archive.jl [DIR] [--commit=REV] [--wall=7200] [--dry-run]
#     julia --project=benchmark benchmark/archive.jl DIR --ipopt-reference [--dry-run]
#
# DIR (default `benchmark/results`) holds one directory of result files per machine, as
# run.jl writes them. The commit is the one recorded in the metadata of the result files;
# `--commit` names it instead. A summary of each machine's results is archived with them
# (`summary.md`). `--note=TEXT` adds a paragraph to the commit message. `--dry-run` prepares the
# commit and prints it without pushing.
#
# `--ipopt-reference` archives the `ipopt` result files of DIR, which holds them directly, as the
# Ipopt reference of this version of the benchmark, under `ipopt/<version>/`.

using MadNLP
using MadNLPBenchmark

const ARCHIVE_BRANCH = "benchmark-results"
const ROOT = dirname(@__DIR__)
git(args...) = readchomp(`git -C $ROOT $args`)

function main(args)
    opt = Dict{String, String}("wall" => "7200", "note" => "")
    flags, pos = Set{String}(), String[]
    for a in args
        m = match(r"^--(commit|wall|note)=(.*)$", a)
        m !== nothing ? (opt[m[1]] = m[2]) : startswith(a, "--") ? push!(flags, a) : push!(pos, a)
    end
    dir = isempty(pos) ? joinpath(@__DIR__, "results") : only(pos)
    "--ipopt-reference" in flags && return archive_reference(dir, flags, opt["note"])
    machines = [d for d in sort!(readdir(dir)) if isdir(joinpath(dir, d)) && !isempty(result_files(joinpath(dir, d)))]
    isempty(machines) && error("no result files under $dir")

    # the commit the results belong to: one, and without local changes
    commits = unique!([read_meta(f)["madnlp_commit"] for d in machines for f in values(result_files(joinpath(dir, d)))])
    commit = if haskey(opt, "commit")
        git("rev-parse", opt["commit"])
    else
        length(commits) == 1 || error("the result files are from several commits ($(join(commits, ", "))); pass --commit")
        endswith(only(commits), "-dirty") && error("the results are from a checkout with local changes; pass --commit to archive them anyway")
        only(commits)
    end

    remote = get(ENV, "MADNLP_BENCHMARK_REMOTE", "origin")
    run(`git -C $ROOT fetch --quiet $remote $ARCHIVE_BRANCH`)
    tree = mktempdir()
    run(`git -C $ROOT worktree add --quiet --detach $tree FETCH_HEAD`)
    try
        for d in machines
            dest = mkpath(joinpath(tree, d, commit))
            for f in readdir(joinpath(dir, d); join = true)
                isfile(f) && cp(f, joinpath(dest, basename(f)); force = true)
            end
            open(io -> summarize(io, dest; wall = parse(Float64, opt["wall"])), joinpath(dest, "summary.md"), "w")
        end
        run(`git -C $tree add -A`)
        if isempty(readchomp(`git -C $tree status --porcelain`))
            println("nothing new to archive for $commit")
            return
        end
        run(`git -C $tree commit --quiet -m "Results of $commit ($(join(machines, ", ")))" -m $(opt["note"])`)
        println(readchomp(`git -C $tree show --stat --format=%s HEAD`))
        if "--dry-run" in flags
            println("dry run: not pushed")
        else
            run(`git -C $tree push --quiet $remote HEAD:$ARCHIVE_BRANCH`)
            println("archived on $ARCHIVE_BRANCH")
        end
    finally
        run(`git -C $ROOT worktree remove --force $tree`)
    end
    return
end

function archive_reference(dir, flags, note)
    remote = get(ENV, "MADNLP_BENCHMARK_REMOTE", "origin")
    run(`git -C $ROOT fetch --quiet $remote $ARCHIVE_BRANCH`)
    tree = mktempdir()
    run(`git -C $ROOT worktree add --quiet --detach $tree FETCH_HEAD`)
    try
        dest = mkpath(joinpath(tree, "ipopt", BENCHMARK_VERSION))
        files = filter(f -> occursin(r"_ipopt\.csv(\.meta\.toml)?$", f), readdir(dir; join = true))
        isempty(files) && error("no ipopt result files in $dir")
        foreach(f -> cp(f, joinpath(dest, basename(f)); force = true), files)
        run(`git -C $tree add -A`)
        run(`git -C $tree commit --quiet -m "Ipopt reference of benchmark $BENCHMARK_VERSION" -m $note`)
        println(readchomp(`git -C $tree show --stat --format=%s HEAD`))
        "--dry-run" in flags ? println("dry run: not pushed") : run(`git -C $tree push --quiet $remote HEAD:$ARCHIVE_BRANCH`)
    finally
        run(`git -C $ROOT worktree remove --force $tree`)
    end
    return
end

main(ARGS)
