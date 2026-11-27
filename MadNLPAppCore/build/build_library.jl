# The build of a MadNLP cnlp solver library, shared by MadNLPApp/build/build.jl and
# cuMadNLPApp/build/build.jl (included from there, in their build environments):
#
#   OUT/NAME/lib/libNAME.so        the image, with a private Julia runtime (bundle)
#   OUT/NAME/lib/libcnlp_NAME.so   the same, under the name `cnlp` looks for
using JuliaC, Random

function build_library(name::AbstractString, image::AbstractString, project::AbstractString,
                       out::AbstractString; trim::AbstractString = "no")
    dest = joinpath(abspath(out), name)
    rm(dest; recursive = true, force = true); mkpath(dest)
    # the CPUs the binary runs on: any x86-64, with code cloned for the common extensions
    cpu = get(ENV, "JULIA_CPU_TARGET", "generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)")
    img = JuliaC.ImageRecipe(
        file = image, project = project, cpu_target = cpu, output_type = "--output-lib", add_ccallables = true,
        trim_mode = trim, julia_args = ["--experimental"], verbose = true
    )
    t = @elapsed JuliaC.compile_products(img)
    link = JuliaC.LinkRecipe(image_recipe = img, outname = joinpath(dest, "lib" * name), rpath = JuliaC.RPATH_BUNDLE)
    JuliaC.link_products(link)
    # a salt of its own for the privatized runtime (identical salts collide)
    fetch(
        Threads.@spawn begin
            Random.seed!(rand(Random.RandomDevice(), UInt128))
            JuliaC.bundle_products(JuliaC.BundleRecipe(link_recipe = link, output_dir = dest, privatize = true))
        end
    )
    rm(joinpath(dest, "share"); recursive = true, force = true)
    symlink("lib" * name * ".so", joinpath(dest, "lib", "libcnlp_" * name * ".so"))
    return dest, t
end
