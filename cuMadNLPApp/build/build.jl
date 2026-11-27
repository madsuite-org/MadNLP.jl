# Build libcumadnlp, the cnlp solver `cumadnlp` (MadNLP on a CUDA GPU):
#
#   julia --project=cuMadNLPApp/build cuMadNLPApp/build/build.jl OUT [trim]
#
#   OUT/cumadnlp/lib/libcumadnlp.so        cnlp_solve_device (+ a private Julia runtime)
#   OUT/cumadnlp/lib/libcnlp_cumadnlp.so   the same, under the name `cnlp` looks for
#
# trim is juliac's trim mode (default "no"). Needs a CUDA GPU: the warm-up solves on it.
include(joinpath(@__DIR__, "..", "..", "MadNLPAppCore", "build", "build_library.jl"))

trim = length(ARGS) >= 2 ? ARGS[2] : "no"
dest, t = build_library("cumadnlp", joinpath(@__DIR__, "libcumadnlp.jl"), @__DIR__, ARGS[1]; trim)
println("built $(dest) in $(round(t; digits = 1)) s (trim = $trim)")
