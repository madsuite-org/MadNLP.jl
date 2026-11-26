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
using JuliaC, Random

out = abspath(ARGS[1])
trim = length(ARGS) >= 2 ? ARGS[2] : "no"
app = dirname(@__DIR__)
project = app
dest = joinpath(out, "madnlp")
rm(dest; recursive = true, force = true); mkpath(dest)

# the CPUs the binary runs on: any x86-64, with code cloned for the common extensions
cpu = get(ENV, "JULIA_CPU_TARGET", "generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)")
img = JuliaC.ImageRecipe(
    file = joinpath(@__DIR__, "libmadnlp.jl"), project = project, cpu_target = cpu, output_type = "--output-lib", add_ccallables = true,
    trim_mode = trim, julia_args = ["--experimental"], verbose = true
)
t = @elapsed JuliaC.compile_products(img)
link = JuliaC.LinkRecipe(image_recipe = img, outname = joinpath(dest, "libmadnlp"), rpath = JuliaC.RPATH_BUNDLE)
JuliaC.link_products(link)
# a salt of its own for the privatized runtime (see ExaModelsCompiler: identical salts collide)
fetch(
    Threads.@spawn begin
        Random.seed!(rand(Random.RandomDevice(), UInt128))
        JuliaC.bundle_products(JuliaC.BundleRecipe(link_recipe = link, output_dir = dest, privatize = true))
    end
)
rm(joinpath(dest, "share"); recursive = true, force = true)
symlink("libmadnlp.so", joinpath(dest, "lib", "libcnlp_madnlp.so"))

mkpath(joinpath(dest, "bin")); mkpath(joinpath(dest, "include"))
cp(joinpath(app, "include", "madnlp.h"), joinpath(dest, "include", "madnlp.h"); force = true)
run(`cc -std=c99 -O2 -o $(joinpath(dest, "bin", "madnlp")) $(joinpath(@__DIR__, "madnlp.c"))
     -L$(joinpath(dest, "lib")) -lmadnlp -Wl,-rpath,\$ORIGIN/../lib`)
println("built $(dest) in $(round(t; digits = 1)) s (trim = $trim)")
