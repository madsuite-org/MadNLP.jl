# MadNLPApp

MadNLP compiled ahead of time (`juliac`), for callers that are not Julia. One bundle carries:

- `lib/libmadnlp.so`, a shared library with a C entry point, `madnlp_solve` (`include/madnlp.h`):
  a problem given as sizes, bounds, a starting point and evaluation callbacks, solved with
  MadNLP on the CPU. The same function is exported as `cnlp_solve`, so the library is also a
  solver for [cnlplib](https://cnlplib.madsuite.org) (`cnlp solve LIB:MODEL ... --solver madnlp`).
- `bin/madnlp`, an [AMPL](https://ampl.com) solver for `.nl` files (CPU):
  `madnlp model.nl [-AMPL] [key=value ...]`, options also read from the environment variable
  `madnlp_options`. With `-AMPL` the solution is written to `model.sol`.

Options are MadNLP's, as `key=value` strings: `tol=1e-8`, `max_iter=500`,
`print_level=INFO` (or `1`-`6`, `0` for quiet), `linear_solver=LapackCPUSolver`,
`hessian_approximation=CompactLBFGS`, ...

## Building

```sh
julia --project=MadNLPApp/build -e 'using Pkg; Pkg.instantiate()'
MADNLPAPP_WARMUP_RUNS=runs.txt MADNLPAPP_WARMUP_NL=model.nl \
    julia --project=MadNLPApp/build MadNLPApp/build/build.jl OUT
```

The bundle is `OUT/madnlp`. The build solves problems through the C entry point while it
precompiles (`src/warmup.jl`), so that a solve compiles nothing at run time:
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
