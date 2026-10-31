using Pkg

MADNLP_DIR = pwd()

Pkg.update()
Pkg.resolve()
Pkg.develop([
    PackageSpec(path=joinpath(MADNLP_DIR, "MadCore")),
    PackageSpec(path=joinpath(MADNLP_DIR, "MadNLPCore")),
    PackageSpec(path=MADNLP_DIR),
    PackageSpec(path=joinpath(MADNLP_DIR, "lib", "MadNLPGPU")),
    PackageSpec(path=joinpath(MADNLP_DIR, "lib", "MadNLPTests")),
])
Base.compilecache(Base.identify_package("MadNLPTests"))
Base.compilecache(Base.identify_package("MadNLPGPU"))
Base.compilecache(Base.identify_package("MadNLP"))
Pkg.instantiate()
