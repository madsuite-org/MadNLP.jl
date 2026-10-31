# The CUDA precompile workload covers a GPU solve with the default options: in a fresh
# Julia process, that compiles nothing of MadNLP or MadNLPGPU at run time, also when a
# package loaded before MadNLP adds a `getindex` method for its own index type (as
# ExaModels does), which invalidates code compiled for abstract arrays.

# As in MadNLP's test/zero_compile_test.jl: the traced statements of a fresh process that
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
            using MadNLP, MadNLPGPU
            madnlp(MadNLP.PrecompileChainModel(100, CuVector{Float64}(undef, 0)); print_level = MadNLP.ERROR)
            """, r"\bMadNLP(GPU(CUDAExt)?)?\."; warm = k == 1)
        @test probe
        isempty(compiled) || foreach(l -> println(first(l, 300)), compiled)
        @test isempty(compiled)
    end
end
