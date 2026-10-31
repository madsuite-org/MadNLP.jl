# Installation
To install MadNLP, run
```julia
pkg> add MadNLP
```

By default, MadNLP uses the sparse linear solver MUMPS.
The other linear solvers are package extensions: they are available as soon as the
package they wrap is loaded alongside MadNLP.

## GPU support
On NVIDIA GPUs, install cuMadNLP:
```julia
pkg> add cuMadNLP
```
On AMD GPUs, install rocMadNLP:
```julia
pkg> add rocMadNLP
```
Both load MadNLP and set the default options for models whose data live on the GPU.
MadNLPGPU is deprecated and loads cuMadNLP.

## HSL linear solvers
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

Once HSL_jll has been installed on your machine, `Ma27Solver`, `Ma57Solver`, `Ma77Solver`,
`Ma86Solver` and `Ma97Solver` are available in MadNLP, which loads HSL.jl.

## Pardiso linear solvers
The Pardiso solvers use [Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl):
```julia
pkg> add Pardiso
julia> using MadNLP, Pardiso
```
`PardisoMKLSolver` uses the Pardiso solver of Intel MKL.
`PanuaPardisoSolver` uses [Panua Pardiso](https://panua.ch/), which needs a license;
see the documentation of Pardiso.jl for its installation.

## Krylov iterative refinement
Loading [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) makes the option
`iterator = KrylovIterator` available, which refines the solution of the KKT system
with GMRES preconditioned by the factorized KKT system.
