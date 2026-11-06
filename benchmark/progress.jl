# Post the results so far as a comment on a pull request, and keep updating that comment. The
# publishing job of the workflow later replaces it with the final comparison, by the same marker.
#
#     julia --project=benchmark benchmark/progress.jl ROOT [--every=1800] [--wall=7200] [--once]
#
# ROOT holds one directory of result files per machine. The environment gives PR, COMMIT,
# GITHUB_REPOSITORY and GH_TOKEN. Runs until it is stopped; a failed update only skips a turn.

using MadNLP, Dates, Downloads
include(joinpath(@__DIR__, "src", "MadNLPBenchmark.jl"))
using .MadNLPBenchmark

const MARKER = "<!-- madnlp-benchmark -->"
const API = get(ENV, "GITHUB_API_URL", "https://api.github.com")

function request(method, url, body)
    response = IOBuffer()
    payload = "{\"body\":\"" * escape_json(body) * "\"}"
    Downloads.request(
        url; method, input = IOBuffer(payload), output = response, throw = true,
        headers = ["Authorization" => "Bearer $(ENV["GH_TOKEN"])", "Accept" => "application/vnd.github+json", "Content-Type" => "application/json"],
    )
    return String(take!(response))
end

function escape_json(s)
    out = IOBuffer()
    for c in s
        c == '"' ? print(out, "\\\"") : c == '\\' ? print(out, "\\\\") : c == '\n' ? print(out, "\\n") :
            c < ' ' ? print(out, "\\u", string(UInt16(c); base = 16, pad = 4)) : print(out, c)
    end
    return String(take!(out))
end

# The id of an earlier comment of this workflow on the pull request, if any.
function find_comment(repo, pr)
    response = IOBuffer()
    Downloads.request(
        "$API/repos/$repo/issues/$pr/comments?per_page=100"; output = response,
        headers = ["Authorization" => "Bearer $(ENV["GH_TOKEN"])", "Accept" => "application/vnd.github+json"],
    )
    text = String(take!(response))
    at = findfirst(MARKER, text)
    at === nothing && return nothing
    ids = collect(eachmatch(r"\"id\":\s*([0-9]+),\s*\"node_id\":\s*\"IC_", text[1:first(at)]))
    return isempty(ids) ? nothing : ids[end][1]
end

function body(root, commit, wall)
    out = IOBuffer()
    println(out, MARKER)
    println(out, "## MadNLP benchmark of `$commit`: in progress\n")
    println(out, "Results so far, updated ", Dates.format(now(UTC), "yyyy-mm-dd HH:MM"), " UTC. The counts cover the instances finished so far.\n")
    for dir in sort!(readdir(root; join = true))
        isdir(dir) && !isempty(result_files(dir)) && summarize(out, dir; wall)
    end
    return String(take!(out))
end

function main(args)
    root = first(args)
    opt(name, default) = (i = findfirst(startswith("--$name="), args); i === nothing ? default : parse(Float64, split(args[i], "=")[2]))
    every, wall = opt("every", 1800.0), opt("wall", 7200.0)
    repo, pr, commit = ENV["GITHUB_REPOSITORY"], ENV["PR"], ENV["COMMIT"]
    while true
        try
            id = find_comment(repo, pr)
            text = body(root, commit, wall)
            if id === nothing
                request("POST", "$API/repos/$repo/issues/$pr/comments", text)
            else
                request("PATCH", "$API/repos/$repo/issues/comments/$id", text)
            end
        catch err
            @warn "could not update the comment" exception = err
        end
        "--once" in args && return
        sleep(every)
    end
end

main(ARGS)
