using Documenter
using MadNLP
import MadCore, MadNLPCore

makedocs(
    sitename = "MadNLP.jl",
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX()
    ),
    # The docstrings of the internals are defined in MadCore and MadNLPCore; only
    # the exported names of MadNLP must appear in the manual.
    modules = [MadNLP, MadNLPCore, MadCore],
    checkdocs_ignored_modules = [MadNLPCore, MadCore],
    repo = "https://github.com/madsuite-org/MadNLP.jl/blob/{commit}{path}#{line}",
    checkdocs = :exports,
    clean=true,
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Quickstart" => "quickstart.md",
        "Algorithm" => "algorithm.md",
        "Options" => "options.md",
        "Tutorials" => [
            "GPU acceleration" => "tutorials/gpu.md",
            "Multi-precision" => "tutorials/multiprecision.md",
            "Warm-start" => "tutorials/warmstart.md",
            "Quasi-Newton" => "tutorials/quasi_newton.md",
            "Custom KKT system" => "tutorials/kktsystem.md",
        ],
        "Manual" => [
            "KKT systems" => "man/kkt.md",
            "Linear Solvers" => "man/linear_solvers.md",
        ],
        "API Reference" => [
            "IPM solver" => "lib/ipm.md",
            "Barrier strategies" => "lib/barrier.md",
            "Callback wrappers" => "lib/callbacks.md",
            "KKT systems" => "lib/kkt.md",
            "Linear Solvers" => "lib/linear_solvers.md",
        ]
    ]
)

# Every open pull request pushes its preview to gh-pages, so a push can lose the race with
# another and be rejected. Each attempt clones gh-pages again, so a retry pushes onto the
# current branch.
for attempt in 1:3
    try
        deploydocs(
            repo = "github.com/madsuite-org/MadNLP.jl.git",
            target = "build",
            devbranch = "master",
            devurl = "dev",
            push_preview = true,
        )
        break
    catch e
        attempt == 3 && rethrow()
        @warn "deploydocs failed (attempt $attempt of 3), retrying" exception = e
        sleep(30 * attempt)
    end
end
