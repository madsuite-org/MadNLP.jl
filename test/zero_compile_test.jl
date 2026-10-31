# The precompile workload covers solving a problem with the default options: in a fresh
# Julia process, that compiles no method of MadNLP at run time.

# Run `code` in a fresh process with this test environment and Julia's default flags
# (`Pkg.test` may run with bounds checking or coverage, for which there are no package
# images), and return the traced precompile statements of methods of `mod`. The code runs
# twice: the first run builds the package images, the second is the one traced.
function compiled_at_run_time(code, mod)
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    cmd(args...) = `$julia --startup-file=no --project=$(Base.active_project()) $args -e $code`
    run(cmd())
    trace = tempname()
    run(cmd("--trace-compile=$trace"))
    return filter(l -> occursin("typeof($mod.", l), readlines(trace))
end

@testset "No compilation at run time" begin
    compiled = compiled_at_run_time("""
        using MadNLP
        madnlp(MadNLP.HS15Model(); print_level = MadNLP.ERROR)
        madnlp(MadNLP.PrecompileChainModel(100, Float64[]); print_level = MadNLP.ERROR)
        madnlp(MadNLP.PrecompileChainModel(100, Float64[]; fixed = true); print_level = MadNLP.ERROR)
        """, "MadNLP")
    isempty(compiled) || foreach(l -> println(first(l, 300)), compiled)
    @test isempty(compiled)
end
