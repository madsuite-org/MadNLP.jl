# The image of libmadnlp: MadNLPApp, whose precompile workload (src/warmup.jl) compiled the
# solve path, plus the method instances that traced runs of a built library still compiled
# (precompile_statements.jl, collected with MADNLPAPP_TRACE=1; see check.sh).
using MadNLPApp
module PrecompileStatements end
let file = joinpath(@__DIR__, "precompile_statements.jl")
    isfile(file) && MadNLPApp.CNLPLib.eval_statements(PrecompileStatements, file)
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
