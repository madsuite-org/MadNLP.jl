# cuMadNLPApp

MadNLP on a CUDA GPU (cuMadNLP) compiled ahead of time (`juliac`): `lib/libcumadnlp.so`
(also as `lib/libcnlp_cumadnlp.so`), the [cnlplib](https://cnlplib.madsuite.org) solver
`cumadnlp`. It solves a model through its device callbacks (`cnlp_solve_device`, a library
with a CUDA tier), with the solver's vectors on the GPU:

```sh
cnlp solve @emp:acp pglib_opf_case118_ieee --solver cumadnlp --device cuda
```

Options are MadNLP's, as `key=value` strings (`tol=1e-8`, `print_level=ERROR`, ...), and name
cuMadNLP's types too (`linear_solver=CUDSSSolver`). On the CPU, use [MadNLPApp](../MadNLPApp).

## Building

```sh
julia --project=cuMadNLPApp/build -e 'using Pkg; Pkg.instantiate()'
julia --project=cuMadNLPApp/build cuMadNLPApp/build/build.jl OUT
```

needs a CUDA GPU: the build solves small problems on it while it precompiles
(build/cuMadNLPAppWarmup), and compiles the solver's GPU kernels for compute capabilities
7.0 to 9.0 (`CUMADNLPAPP_CUDA_CAPS`), so that a solve compiles nothing at run time. When the
library loads it brings up the GPU and solves a tiny problem (`MADNLPAPP_WARMUP=0` skips it);
`CUMADNLPAPP_TRACE=1` reports what a solve still compiles.
