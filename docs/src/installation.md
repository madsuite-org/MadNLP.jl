# Installation
To install MadNLP, simply proceed to
```julia
pkg> add MadNLP
```

By default, MadNLP uses the sparse linear solver MUMPS.
In addition the user can install the following extensions to use a specialized linear solver.

## GPU support
MadNLPGPU provides GPU support for MadNLP:
```julia
pkg> add MadNLPGPU
```
MadNLPGPU automatically detects if you have a NVIDIA or an AMD GPU on your machine,
and adapts in consequence.

## HSL linear solver
Obtain a license and download [HSL_jll.jl](https://licences.stfc.ac.uk/products/Software/HSL/LibHSL).
Install the package into your current environment using:
```julia
import Pkg
Pkg.develop(path = "/full/path/to/HSL_jll.jl")
```

If the user has already compiled the HSL solver library, one can
simply override the path to the artifact by editing `~/.julia/artifacts/Overrides.toml`
```
# replace HSL_jll artifact /usr/local/lib/libhsl.so
ecece3e2c69a413a0e935cf52e03a3ad5492e137 = "/usr/local"
```

Once HSL_jll has been installed on your machine, you can install MadNLPHSL as
```julia
pkg> add MadNLPHSL
```

## Pardiso linear solver

To use Pardiso, the user needs to obtain the Pardiso shared libraries from
<https://panua.ch/>, provide the absolute path to the shared library:
```
julia> ENV["JULIA_PARDISO"] = "/path/to/pardiso/lib/"
```
and place the license file in the home directory.
After obtaining the library and the license file, run
```julia
pkg> add MadNLPPardiso
pkg> build MadNLPPardiso
```

The build process requires a C compiler.
