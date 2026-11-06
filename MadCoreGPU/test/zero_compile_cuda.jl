# The CUDA precompile workload covers a GPU solve with the default options: in a fresh
# Julia process, that compiles nothing of cuMadNLP and the packages it loads at run time, also when
# a package loaded before them adds a `getindex` method for its own index type (as
# ExaModels does), which invalidates code compiled for abstract arrays.

# As in MadNLP/test/zero_compile_test.jl: the traced statements of a fresh process that
# mention a module matching `pattern`, and whether the planted probe was traced.
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

@testset "No compilation at run time (CUDA)" begin
    for (k, prelude) in enumerate(("", "struct Index end; Base.getindex(::AbstractArray, ::Index) = nothing"))
        compiled, probe = compiled_at_run_time("""
            using CUDA, CUDSS
            $prelude
            using cuMadNLP, MadNLPCore
            madnlp(MadNLPCore.PrecompileChainModel(100, CuVector{Float64}(undef, 0)); print_level = MadNLPCore.ERROR)
            """, r"\b(cuMadNLP|Mad(NLPCore|Core|CoreGPU)(\w*Ext)?)\."; warm = k == 1)
        # The test problem's constructor, called from the script's top level, is not part
        # of what the workload is for.
        filter!(l -> !occursin("MadNLPCore.PrecompileChainModel{", l), compiled)
        @test probe
        isempty(compiled) || foreach(l -> println(first(l, 300)), compiled)
        @test isempty(compiled)
    end
end
