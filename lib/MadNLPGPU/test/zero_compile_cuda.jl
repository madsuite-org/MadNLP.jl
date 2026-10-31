# The CUDA precompile workload covers a GPU solve with the default options: in a fresh
# Julia process, that compiles no method of MadNLP or MadNLPGPU at run time, also when a
# package loaded before MadNLP adds a `getindex` method for its own index type (as
# ExaModels does), which invalidates code compiled for abstract arrays.

function compiled_at_run_time(code, mods)
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    cmd(args...) = `$julia --startup-file=no --project=$(Base.active_project()) $args -e $code`
    run(cmd())
    trace = tempname()
    run(cmd("--trace-compile=$trace"))
    return filter(l -> any(m -> occursin("typeof($m.", l), mods), readlines(trace))
end

@testset "No compilation at run time (CUDA)" begin
    for prelude in ("", "struct Index end; Base.getindex(::AbstractArray, ::Index) = nothing")
        compiled = compiled_at_run_time("""
            using CUDA, CUDSS
            $prelude
            using MadNLP, MadNLPGPU
            madnlp(MadNLP.PrecompileChainModel(100, CuVector{Float64}(undef, 0)); print_level = MadNLP.ERROR)
            """, ("MadNLP", "MadNLPGPU"))
        isempty(compiled) || foreach(l -> println(first(l, 300)), compiled)
        @test isempty(compiled)
    end
end
