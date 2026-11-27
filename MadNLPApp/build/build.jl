# Build libmadnlp and the madnlp executable into one bundle:
#
#   julia --project=MadNLPApp/build MadNLPApp/build/build.jl OUT [trim]
#
#   OUT/madnlp/lib/libmadnlp.so       madnlp_solve, cnlp_solve, madnlp_main (+ a private Julia runtime)
#   OUT/madnlp/lib/libcnlp_madnlp.so  the same, under the name `cnlp` looks for
#   OUT/madnlp/bin/madnlp             the AMPL .nl solver, a launcher for madnlp_main
#   OUT/madnlp/include/madnlp.h
#
# trim is juliac's trim mode (default "no"). The environment variables of the workload
# (MADNLPAPP_WARMUP_RUNS, MADNLPAPP_WARMUP_NL; src/warmup.jl) choose what it solves. The GPU
# solver is built by cuMadNLPApp/build/build.jl.
include(joinpath(@__DIR__, "..", "..", "MadNLPAppCore", "build", "build_library.jl"))

app = dirname(@__DIR__)
trim = length(ARGS) >= 2 ? ARGS[2] : "no"
dest, t = build_library("madnlp", joinpath(@__DIR__, "libmadnlp.jl"), app, ARGS[1]; trim)

mkpath(joinpath(dest, "bin")); mkpath(joinpath(dest, "include"))
cp(joinpath(app, "include", "madnlp.h"), joinpath(dest, "include", "madnlp.h"); force = true)
run(`cc -std=c99 -O2 -o $(joinpath(dest, "bin", "madnlp")) $(joinpath(@__DIR__, "madnlp.c"))
     -L$(joinpath(dest, "lib")) -lmadnlp -Wl,-rpath,\$ORIGIN/../lib`)
println("built $(dest) in $(round(t; digits = 1)) s (trim = $trim)")
