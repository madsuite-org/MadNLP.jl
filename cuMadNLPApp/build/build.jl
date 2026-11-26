# Build libcumadnlp, the cnlp solver `cumadnlp` (MadNLP on a CUDA GPU):
#
#   julia --project=cuMadNLPApp/build cuMadNLPApp/build/build.jl OUT [trim]
#
#   OUT/cumadnlp/lib/libcumadnlp.so        cnlp_solve_device (+ a private Julia runtime)
#   OUT/cumadnlp/lib/libcnlp_cumadnlp.so   the same, under the name `cnlp` looks for
#
# trim is juliac's trim mode (default "no"). Needs a CUDA GPU: the warm-up solves on it.
using JuliaC, Random

out = abspath(ARGS[1])
trim = length(ARGS) >= 2 ? ARGS[2] : "no"
dest = joinpath(out, "cumadnlp")
rm(dest; recursive = true, force = true); mkpath(dest)

# the CPUs the binary runs on: any x86-64, with code cloned for the common extensions
cpu = get(ENV, "JULIA_CPU_TARGET", "generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)")
img = JuliaC.ImageRecipe(
    file = joinpath(@__DIR__, "libcumadnlp.jl"), project = @__DIR__, cpu_target = cpu, output_type = "--output-lib", add_ccallables = true,
    trim_mode = trim, julia_args = ["--experimental"], verbose = true
)
t = @elapsed JuliaC.compile_products(img)
link = JuliaC.LinkRecipe(image_recipe = img, outname = joinpath(dest, "libcumadnlp"), rpath = JuliaC.RPATH_BUNDLE)
JuliaC.link_products(link)
# a salt of its own for the privatized runtime (identical salts collide)
fetch(
    Threads.@spawn begin
        Random.seed!(rand(Random.RandomDevice(), UInt128))
        JuliaC.bundle_products(JuliaC.BundleRecipe(link_recipe = link, output_dir = dest, privatize = true))
    end
)
rm(joinpath(dest, "share"); recursive = true, force = true)
symlink("libcumadnlp.so", joinpath(dest, "lib", "libcnlp_cumadnlp.so"))
println("built $(dest) in $(round(t; digits = 1)) s (trim = $trim)")
