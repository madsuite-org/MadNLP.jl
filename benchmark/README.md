# MadNLP benchmarks

This directory is the baseline benchmarking workflow of MadNLP: it solves standard problem
sets with a chosen solver configuration, records every solve in a CSV file, and compares the
results of two commits. Use it to measure a change to MadNLP, a new KKT system, a new linear
solver or a new option set against the current state of the solver.

The environment in `Project.toml` takes MadNLP, MadNLPGPU and MadNLPHSL from this checkout
through `[sources]`, so a benchmark always measures the code in the working tree. The problem
packages that are not registered are pinned to a commit there, so that the instances are the
same from one run to the next.

## Quick start

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'

julia --project=benchmark benchmark/run.jl smoke                       # a functional check
julia --project=benchmark benchmark/run.jl acopf,cops                  # two sets, default configuration
julia --project=benchmark benchmark/run.jl acopf --device=cuda --config=default,augmented_cudss

julia --project=benchmark benchmark/summary.jl                         # solved counts and mean times
julia --project=benchmark benchmark/compare.jl master HEAD --hardware=<tag>   # two commits
```

## Problem sets

| set | instances | source |
|:---|---:|:---|
| `acopf` | 66 | the typical-operating-point cases of pglib-opf, polar form, through ExaModelsPower |
| `cops` | 65 | the COPS problems of COPSBenchmark: nine scaling models at five sizes each, the other models at their canonical sizes |
| `mittelmann` | 47 | the instances scored in Hans Mittelmann's AMPL-NLP benchmark, through ExaMittelmann |
| `cutest` | 1338 | the CUTEst problems with at least as many free variables as equality constraints (`sets/cutest.txt`), through CUTEst.jl; host only |
| `smoke` | 10 | small ACOPF and COPS instances, for a functional check and for CI |

The sets are defined in `sets.jl`. `acopf`, `cops`, `mittelmann` and `smoke` are ExaModels
models and run on the host or on a CUDA device; `cutest` runs on the host.

## Configurations

A configuration is a named set of keyword arguments of `madnlp`: a KKT system, a linear
solver and their options. They are defined in `configs.jl`; add an entry there to benchmark
something new.

| configuration | device | KKT system | linear solver |
|:---|:---|:---|:---|
| `default` | cpu | MadNLP's default (`SparseKKTSystem`) | MadNLP's default (MUMPS) |
| `condensed_cholmod` | cpu | `SparseCondensedKKTSystem` | CHOLMOD |
| `augmented_ma27`, `augmented_ma57` | cpu | `SparseKKTSystem` | MA27, MA57; only with a licensed HSL_jll |
| `default` | cuda | MadNLPGPU's default (`SparseCondensedKKTSystem`) | cuDSS |
| `augmented_cudss` | cuda | `SparseKKTSystem` | cuDSS with Bunch-Kaufman pivoting |

## Options of `run.jl`

| option | default | meaning |
|:---|:---|:---|
| `--config=a,b` | `default` | the configurations to run |
| `--device=cpu\|cuda` | `cpu` | where the models and the solver live |
| `--out=DIR` | `results/<hardware>` | where the result files go; `<hardware>` is the CPU model or the GPU name |
| `--only=case,...` | all | restrict a set to the named instances |
| `--tol=1e-8` | `1e-8` | convergence tolerance |
| `--wall=7200` | `7200` | wall limit of a solve, in seconds |
| `--no-warmup` | | skip the discarded solves (see the protocol) |
| `--overwrite` | | start the result files afresh instead of resuming |
| `--verbose` | | print MadNLP's iteration log |

A run that is interrupted is resumed by the same command: an instance already in a result
file is skipped.

## Protocol

- **Options.** MadNLP's defaults, apart from what the configuration sets, the tolerance and
  the wall limit. A configuration that tunes other options says so in `configs.jl`.
- **Solved.** An instance is solved if MadNLP returns `SOLVE_SUCCEEDED` or
  `SOLVED_TO_ACCEPTABLE_LEVEL`. Every other exit, including the iteration limit with a good
  last iterate, is a failure.
- **Time.** The recorded time is MadNLP's total time of one solve, with the initialization,
  the linear solver and the function evaluations also recorded separately.
- **Compilation.** The first time a configuration meets a model type in a session, the
  instance is solved once and that solve is discarded, so that no recorded time includes
  compilation. All ACOPF cases share one model type and each COPS model family has its own, so
  only the first instance of each is solved twice. A discarded solve that reaches the wall
  limit is recorded instead of being repeated.
- **Aggregate.** The number of instances solved, and the shifted geometric mean of the solve
  time with a shift of 10 s, a failure counting as the wall limit. Instances are split into
  small, medium and large by the nonzeros of the Jacobian and the Hessian together: below
  2^18, below 2^20, and above.
- **Feasibility.** `primal_feas` is the residual of the solver's scaled, relaxed problem. The
  columns `viol_con` and `viol_var` are the violation of the model's own constraints and
  bounds at the returned point, evaluated independently of the solver.
- **Hardware.** Timings are comparable only on the same machine. Each result file has a
  metadata file beside it (`.meta.toml`) with the machine, the thread counts, the package
  versions, the MadNLP commit and the options.

## Result files

`run.jl` writes `<set>_<configuration>.csv` with one row per instance:

`case, status, iters, nvar, ncon, nnzj, nnzh, obj, primal_feas, dual_feas, n_fact,
n_backsolve, n_obj, n_grad, n_con, n_jac, n_hess, t_total, t_linsolve, t_eval, t_init,
viol_con, viol_var`

`status` is MadNLP's status code; `-100` marks an instance whose model could not be built or
whose solve threw.

## Comparing commits

`compare.jl BASE CANDIDATE` prints, per set and configuration, the instances solved, the
shifted geometric means and their ratio, the ratio of iteration counts, and every instance
whose outcome changed. BASE and CANDIDATE are each a directory of result files or a git
revision.

Results are archived on the `benchmark-results` branch of the repository, one
directory per machine and commit: `<hardware>/<commit>/<set>_<configuration>.csv`. A revision
given to `compare.jl` is looked up there (pass `--hardware=<tag>`); if that commit has no
archived results, its nearest archived ancestor is used and named in the output.

## CI

`.github/workflows/benchmark.yml` runs the benchmark on the self-hosted GPU runner:

- **On request, for a pull request:** add the `benchmark` label. Once the `build` workflow
  has succeeded on the head commit, the four sets run on the host and ACOPF, COPS and
  Mittelmann on the CUDA device, with a wall limit of 900 s per solve. A comment on the pull
  request then gives the table of instances solved and shifted geometric mean times, and the
  comparison with the archived results of the base commit. A push to a labeled pull request
  is benchmarked again.
- **For every commit on `master`:** the same run once `build` has succeeded, archived on
  `benchmark-results` with its summary table.
- **By hand:** the workflow can be dispatched with any sets, configurations and device, on
  any branch, and can archive its results.
