# The report is public and its input comes from benchmark jobs: nothing in a result folder may
# reach a page as markup, and only result files may be published.
#
#     julia --project=benchmark/report benchmark/report/test/runtests.jl

using BenchmarkReport, Test

function hostile_results(root)
    dir = mkpath(joinpath(root, "machine"))
    columns = [
        "case", "status", "iters", "nvar", "ncon", "nnzj", "nnzh", "obj", "primal_feas", "dual_feas", "n_fact",
        "n_backsolve", "n_ir", "n_obj", "n_grad", "n_con", "n_jac", "n_hess", "t_total", "t_linsolve", "t_eval", "t_init",
        "t_compile", "n_solves", "viol_eq", "viol_con", "viol_var",
    ]
    row = fill("1", length(columns))
    row[1] = "<script>alert(1)</script>"
    row[3] = "<img src=x onerror=alert(2)>"
    write(joinpath(dir, "smoke_default.csv"), join(columns, ",") * "\n" * join(row, ",") * "\n")
    write(
        joinpath(dir, "smoke_default.csv.meta.toml"), """
        device = "cuda"
        host = "<script>alert(3)</script>"
        cpu_model = "x"
        cpu_threads = 1
        memory_gb = 1.0
        gpu_name = "x"
        gpu_memory_gb = 1.0
        gpu_driver = "x"
        cuda_runtime = "x"
        madnlp_commit = "<script>alert(4)</script>"
        """
    )
    write(joinpath(dir, "summary.md"), "```\nstatus <script>\n  worker 2: <b>x <img> for 3 s, 1.0 GB\n```\n")
    write(joinpath(dir, "evil.html"), "<script>alert(5)</script>")
    symlink("/etc/passwd", joinpath(dir, "secret.txt"))
    return
end

markup(page) = (text = read(page, String); occursin(r"<(img|b>x|iframe)|<script>alert", text[1:(findlast("<script>", text).start - 1)]))

mktempdir() do root
    results, site = mkpath(joinpath(root, "results")), joinpath(root, "site")
    hostile_results(results)
    write_report(site, results; wall = 60.0, title = "<i>title</i>", finished = false)
    pages = filter(endswith(".html"), readdir(site; join = true))
    @test length(pages) >= 3
    @test !any(markup, pages)
    published = joinpath(site, "data", "machine")
    @test isfile(joinpath(published, "smoke_default.csv"))
    @test !ispath(joinpath(published, "evil.html"))
    @test !ispath(joinpath(published, "secret.txt"))
    @test isfile(joinpath(site, "status.json"))
end
