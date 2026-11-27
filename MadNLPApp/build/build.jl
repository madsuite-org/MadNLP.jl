# Build libmadnlp and the madnlp executable into one bundle:
#
#   julia --project=MadNLPApp/build MadNLPApp/build/build.jl OUT [trim]
#
#   OUT/madnlp/lib/libmadnlp.so       madnlp_solve, cnlp_solve, cnlp_solve_device, madnlp_main
#                                     (+ a private Julia runtime)
#   OUT/madnlp/lib/libcnlp_madnlp.so  the same, under the name `cnlp` looks for
#   OUT/madnlp/bin/madnlp             the AMPL .nl solver, a launcher for madnlp_main
#   OUT/madnlp/include/madnlp.h
#
# trim is juliac's trim mode (default "no"). The environment variables of the workload
# (MADNLPAPP_WARMUP_RUNS, MADNLPAPP_WARMUP_NL; src/warmup.jl) choose what it solves. Needs a
# CUDA GPU: the device warm-up (MadNLPAppCUDAWarmup) solves on it.
using JuliaC, Random, SHA

function build_library(
        name::AbstractString, image::AbstractString, project::AbstractString,
        out::AbstractString; trim::AbstractString = "no"
    )
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
    # share/julia/artifacts stays: the libraries the image's packages load (MUMPS, OpenBLAS32,
    # METIS, the CUDA libraries, ...), so that the bundle needs nothing from a Julia installation
    symlink("lib" * name * ".so", joinpath(dest, "lib", "libcnlp_" * name * ".so"))
    return dest, t
end

app = dirname(@__DIR__)
trim = length(ARGS) >= 2 ? ARGS[2] : "no"
# the build environment: MadNLPApp and MadNLPAppCUDAWarmup, whose workload compiles the device path
dest, t = build_library("madnlp", joinpath(@__DIR__, "libmadnlp.jl"), @__DIR__, ARGS[1]; trim)

# No HSL in the bundle, ever (its license does not allow passing it on): HSL_jll's placeholder
# is all the build may carry. Its libraries define the HSL entry points as stubs of a byte or
# so; a real HSL's are kilobytes (ma57ad_ is 11 kB), so any library defining one of them with
# a body fails the build.
for (root, _, files) in walkdir(dest), f in files
    p = joinpath(root, f)
    (isfile(p) && !islink(p) && occursin(".so", f)) || continue
    syms = try
        read(`nm -D -S --defined-only $p`, String)
    catch
        ""
    end
    for l in eachline(IOBuffer(syms))
        w = split(l)
        length(w) == 4 && w[4] in ("ma57ad_", "ma27ad_", "ma97_analyse_d", "ma86_factor_d") &&
            parse(Int, w[2]; base = 16) > 256 || continue
        rm(dest; recursive = true)
        error("$p defines the HSL routine $(w[4]): the bundle must not carry HSL")
    end
end

mkpath(joinpath(dest, "bin")); mkpath(joinpath(dest, "include"))
cp(joinpath(app, "include", "madnlp.h"), joinpath(dest, "include", "madnlp.h"); force = true)
run(`cc -std=c99 -O2 -o $(joinpath(dest, "bin", "madnlp")) $(joinpath(@__DIR__, "madnlp.c"))
     -L$(joinpath(dest, "lib")) -lmadnlp -Wl,-rpath,\$ORIGIN/../lib`)

# The CUDA libraries a GPU solve loads (cuBLAS, cuSPARSE, cuSOLVER, cuDSS, ...: about 2 GB)
# are a separate download: moved out of the bundle into one artifact, OUT/artifacts/cuda-HASH
# (runtime/ and cudss/), listed in the bundle's ARTIFACTS file for the cuda device, so `cnlp
# install` fetches it only for a GPU and every library naming the same artifact shares one
# copy (see cnlplib's docs/src/models.md). It is privatized as one unit, so it holds all the
# libraries those load: its own copies of nvJitLink and nvRTC as well, which the bundle also
# keeps (CUDA_Compiler_jll loads them when the library starts). The CUDA runtime library
# itself stays (CUDACore's initialization needs it), and so do the artifact folders, empty:
# the JLLs then load nothing and a GPU solve is refused (deps/, "MadNLPApp:").
function content_hash(dir)
    ctx = SHA.SHA256_CTX()
    for (root, _, files) in walkdir(dir), f in sort(files)
        p = joinpath(root, f)
        SHA.update!(ctx, codeunits(relpath(p, dir)))
        SHA.update!(ctx, islink(p) ? codeunits(readlink(p)) : SHA.sha256(read(p)))
    end
    return bytes2hex(SHA.digest!(ctx))[1:40]
end
arts = joinpath(dest, "share", "julia", "artifacts")
with(f) = only(filter(d -> isfile(joinpath(arts, d, "lib", f)), readdir(arts)))
runtime, compiler, cudss = with("libcublas.so"), with("libnvJitLink.so"), with("libcudss.so")
outart = joinpath(abspath(ARGS[1]), "artifacts"); mkpath(outart)
stage = mktempdir(outart)
mkpath(joinpath(stage, "runtime")); mkpath(joinpath(stage, "cudss"))
for f in readdir(joinpath(arts, runtime, "lib"))
    startswith(f, "libcudart") || mv(joinpath(arts, runtime, "lib", f), joinpath(stage, "runtime", f))
end
for f in readdir(joinpath(arts, compiler, "lib"))
    startswith(f, "libnvJitLink") || startswith(f, "libnvrtc") || continue
    cp(joinpath(arts, compiler, "lib", f), joinpath(stage, "runtime", f); follow_symlinks = false)
end
for f in readdir(joinpath(arts, cudss))
    mv(joinpath(arts, cudss, f), joinpath(stage, "cudss", f))
end
# cuDSS loads cuBLAS and nvRTC by name, looked up beside it ($ORIGIN): links to runtime/
for f in readdir(joinpath(stage, "runtime"))
    l = joinpath(stage, "cudss", "lib", f)
    ispath(l) || islink(l) || symlink(joinpath("..", "..", "runtime", f), l)
end
a = string("cuda-", content_hash(stage))
rm(joinpath(outart, a); recursive = true, force = true); mv(stage, joinpath(outart, a))
lines = [
    string(a, "/runtime share/julia/artifacts/", runtime, "/lib cuda"),
    string(a, "/cudss share/julia/artifacts/", cudss, " cuda"),
]
open(io -> foreach(l -> println(io, l), lines), joinpath(dest, "ARTIFACTS"), "a")
println("the CUDA libraries, a separate download: ", join(lines, "; "))
println("built $(dest) in $(round(t; digits = 1)) s (trim = $trim)")
