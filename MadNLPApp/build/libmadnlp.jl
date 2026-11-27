# The image of libmadnlp: MadNLPApp, whose precompile workload (src/warmup.jl) compiled the
# solve path, plus the method instances that traced runs of a built library still compiled
# (precompile_statements.jl, collected with MADNLPAPP_TRACE=1; see check.sh).
using MadNLPApp
MadNLPApp.MadNLPAppCore.precompile_image(joinpath(@__DIR__, "precompile_statements.jl"))
