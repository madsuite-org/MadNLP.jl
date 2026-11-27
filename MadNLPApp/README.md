# MadNLPApp

MadNLP compiled ahead of time (`juliac`), for callers that are not Julia, on the CPU and on
a CUDA GPU. On the CPU it is MadNLPCore with MUMPS as the linear solver; the HSL solvers are
not included, since their license does not allow redistributing them. On the GPU it is
cuMadNLP (cuDSS). One bundle carries:

- `lib/libmadnlp.so` (also as `lib/libcnlp_madnlp.so`), a shared library with a C entry point, `madnlp_solve` (`include/madnlp.h`):
  a problem given as sizes, bounds, a starting point and evaluation callbacks, solved with
  MadNLP on the CPU. The same function is exported as `cnlp_solve`, so the library is also a
  solver for [cnlplib](https://cnlplib.madsuite.org) (`cnlp solve LIB:MODEL ... --solver madnlp`).
  `cnlp_solve_device` solves a model through its device (CUDA) callbacks, with the solver's
  vectors on the GPU; the GPU is brought up by the first device solve.
- `bin/madnlp`, an [AMPL](https://ampl.com) solver for `.nl` files (CPU):
  `madnlp model.nl [-AMPL] [key=value ...]`, options also read from the environment variable
  `madnlp_options`. With `-AMPL` the solution is written to `model.sol`.

Options are MadNLP's, as `key=value` strings: `tol=1e-8`, `max_iter=500`,
`print_level=INFO` (or `1`-`6`, `0` for quiet), `linear_solver=LapackCPUSolver`,
`hessian_approximation=CompactLBFGS`, ...

## HSL

The HSL linear solvers (`linear_solver=Ma27Solver`, `Ma57Solver`, `Ma77Solver`, `Ma86Solver`,
`Ma97Solver`) are supported, but the bundle does not carry HSL: its license does not allow
passing it on, and the build refuses a bundle with any HSL in it. A user brings their own
HSL_jll (from STFC), found in this order: the option `hsl_library=DIR`,
`JULIA_HSL_LIBRARY_PATH`, `artifacts.hsl` in `$CNLPLIB_HOME/config.toml`, and
`$CNLPLIB_HOME/solvers/madnlp/lib/julia`, where `cnlp hsl add` puts it. Without one, an HSL
solver is refused with what to do, and MUMPS stays the default. The compiled HSL code paths
come from `build/precompile_statements.jl` (method signatures traced once from HSL solves),
so the build needs no HSL either.

## Building

```sh
julia --project=MadNLPApp/build -e 'using Pkg; Pkg.instantiate()'
MADNLPAPP_WARMUP_RUNS=runs.txt MADNLPAPP_WARMUP_NL=model.nl \
    julia --project=MadNLPApp/build MadNLPApp/build/build.jl OUT
```

The bundle is `OUT/madnlp`. The build needs a CUDA GPU. It solves problems through the C
entry points while it precompiles, so that a solve compiles nothing at run time: on the CPU
(`src/warmup.jl`), and on the GPU (`build/MadNLPAppCUDAWarmup`, which also compiles the
solver's GPU kernels for compute capabilities 7.0 to 9.0, `MADNLPAPP_CUDA_CAPS`).
`MADNLPAPP_WARMUP_RUNS` is a file of lines `LIB MODEL ARGS...` (cnlp libraries and models),
and `MADNLPAPP_WARMUP_NL` lists `.nl` files. Methods that runs of a built library still
compiled are listed in `build/precompile_statements.jl` and compiled into the image.

## Checking that nothing compiles at run time

```sh
MadNLPApp/build/check.sh OUT/madnlp runs.txt model.nl CHECK
```

runs each case through `cnlp solve --solver OUT/madnlp/lib/libmadnlp.so` and `OUT/madnlp/bin/madnlp`
with `MADNLPAPP_TRACE=1`, with output to a file, a pipe and a terminal. It writes
`CHECK/compiled.tsv` (Julia compile time per run) and `CHECK/precompile_statements.jl`, and
fails when any run compiled something or did not report. `MADNLPAPP_TRACE=1` on any run
lists each method compiled at run time on stderr, with its compile time.
