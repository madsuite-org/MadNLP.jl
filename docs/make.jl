using Documenter
using MadNLP

makedocs(
    sitename = "MadNLP.jl",
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX()
    ),
    modules = [MadNLP],
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

deploydocs(
    repo = "github.com/madsuite-org/MadNLP.jl.git",
    target = "build",
    devbranch = "master",
    devurl = "dev",
    push_preview = true,
)
