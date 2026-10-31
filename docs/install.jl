using Pkg

MADNLP_DIR = pwd()

Pkg.update()
Pkg.resolve()
Pkg.develop([
    PackageSpec(path=MADNLP_DIR),
    PackageSpec(path=joinpath(MADNLP_DIR, "lib", "MadNLPGPU")),
    PackageSpec(path=joinpath(MADNLP_DIR, "lib", "MadNLPTests")),
])
Pkg.add(PackageSpec(name = "MathOptInterface", rev = "bl/qp_block_data"))
Base.compilecache(Base.identify_package("MadNLPTests"))
Base.compilecache(Base.identify_package("MadNLPGPU"))
Base.compilecache(Base.identify_package("MadNLP"))
Pkg.instantiate()
