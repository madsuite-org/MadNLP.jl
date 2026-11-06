using Pkg

ROOT = pwd()

# Develop the packages of this repository first: they are not registered.
Pkg.develop([
    PackageSpec(path=joinpath(ROOT, "MadCore")),
    PackageSpec(path=joinpath(ROOT, "MadNLPCore")),
    PackageSpec(path=joinpath(ROOT, "MadNLP")),
    PackageSpec(path=joinpath(ROOT, "MadCoreGPU")),
    PackageSpec(path=joinpath(ROOT, "cuMadNLP")),
    PackageSpec(path=joinpath(ROOT, "MadNLPTests")),
])
Pkg.update()
Base.compilecache(Base.identify_package("MadNLPTests"))
Base.compilecache(Base.identify_package("cuMadNLP"))
Base.compilecache(Base.identify_package("MadNLP"))
Pkg.instantiate()
