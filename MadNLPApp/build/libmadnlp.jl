# The image of libmadnlp: MadNLPApp, whose precompile workload (src/warmup.jl) compiled the
# solve path, plus the method instances that traced runs of a built library still compiled
# (precompile_statements.jl, collected with MADNLPAPP_TRACE=1; see check.sh).
using MadNLPApp
# the CUDA build (build.jl ... cuda): loading these loads MadNLPAppCUDAExt, cnlp_solve_device
if get(ENV, "MADNLPAPP_DEVICE", "") == "cuda"
    using CUDACore, cuMadNLP
    # the device path, compiled here: each MADNLPAPP_WARMUP_RUNS model whose library has a
    # CUDA tier, solved through cnlp_solve_device (precompiling the extension cannot do it)
    let ext = Base.get_extension(MadNLPApp, :MadNLPAppCUDAExt), runs = get(ENV, "MADNLPAPP_WARMUP_RUNS", "")
        ext === nothing && error("MadNLPAppCUDAExt is not loaded")
        if !isempty(runs) && CUDACore.functional()
            for line in eachline(runs)
                w = split(line)
                (isempty(w) || startswith(w[1], "#")) && continue
                l = MadNLPApp.CNLPLib.load(String(w[1]))
                MadNLPApp.CNLPLib.Libdl.dlsym(l.handle, Symbol(w[2], "_device"); throw_error = false) === nothing && continue
                st = ext.solve_lib_device(String(w[1]), String(w[2]), String.(w[3:end]))
                println(stderr, "warm-up on the device: ", w[2], " status ", st)
            end
        end
    end
end

module PrecompileStatements
    for (id, m) in Base.loaded_modules
        name = Symbol(id.name)
        isdefined(@__MODULE__, name) || Core.eval(@__MODULE__, :(const $name = $m))
    end
    file = joinpath(@__DIR__, "precompile_statements.jl")
    for line in (isfile(file) ? eachline(file) : ())
        startswith(line, "precompile(") || continue
        try
            Core.eval(@__MODULE__, Meta.parse(line))
        catch
            println(stderr, "stale precompile statement: ", line[1:min(end, 160)])
        end
    end
end

# Base.merge_names builds a Symbol[...] when it merges keyword-argument names not known at
# compile time (the solver's options); package-mode traces do not show it (the sysimage has
# it), an image without it compiles it on every run.
for n in 4:16
    precompile(Tuple{typeof(Base.getindex), Type{Symbol}, ntuple(_ -> Symbol, n)...})
end
precompile(Tuple{typeof(Base.getindex), Type{Symbol}, Symbol, Symbol, Symbol, Vararg{Symbol}})
# stdout and stderr may be a file, a pipe or a terminal
for io in (Base.PipeEndpoint, Base.TTY, IOStream), f in (println, print)
    precompile(f, (io, String))
    precompile(flush, (io,))
end
