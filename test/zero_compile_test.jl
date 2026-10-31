# The precompile workload covers solving a problem with the default options: in a fresh
# Julia process, that compiles nothing of MadNLP at run time.

# Run `code` in a fresh process with this test environment and Julia's default flags, and
# return the traced precompile statements that mention a module matching `pattern`
# (methods, closures, constructors, and Base methods on its types), and whether the trace
# holds the planted `zero_compile_probe` (a positive control: the trace was written and the
# filter still matches its format). With `warm`, the code first runs once untraced to build
# the package images: `Pkg.test` precompiles for its own flags (bounds checking), so the
# default-flag images of MadNLP are built here once more.
function compiled_at_run_time(code, pattern; warm = true)
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    code = code * "\nzero_compile_probe(x) = x + 1; zero_compile_probe(1)\n"
    cmd(args...) = `$julia --startup-file=no --project=$(Base.active_project()) $args -e $code`
    warm && run(cmd())
    trace = tempname()
    run(cmd("--trace-compile=$trace"))
    lines = readlines(trace)
    rm(trace)
    return filter(l -> occursin(pattern, l), lines),
           any(l -> occursin("typeof(Main.zero_compile_probe)", l), lines)
end

@testset "No compilation at run time" begin
    compiled, probe = compiled_at_run_time("""
        using MadNLP
        madnlp(MadNLP.HS15Model(); print_level = MadNLP.ERROR)
        madnlp(MadNLP.PrecompileChainModel(100, Float64[]); print_level = MadNLP.ERROR)
        madnlp(MadNLP.PrecompileChainModel(100, Float64[]; fixed = true); print_level = MadNLP.ERROR)
        """, r"\bMadNLP\.")
    @test probe
    isempty(compiled) || foreach(l -> println(first(l, 300)), compiled)
    @test isempty(compiled)
end
