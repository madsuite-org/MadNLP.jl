# MadNLP benchmarks

This directory is the baseline benchmarking workflow of MadNLP: it solves standard problem
sets with a chosen solver configuration, records every solve in a CSV file, and compares the
results of two commits. Use it to measure a change to MadNLP, a new KKT system, a new linear
solver or a new option set against the current state of the solver.

The environment in `Project.toml` takes MadNLP, its core packages and cuMadNLP from this checkout
through `[sources]`, so a benchmark always measures the code in the working tree. The problem
packages that are not registered are pinned to a commit there, and the registered ones are
bounded in `[compat]`, so that the instances and the software around MadNLP are the same from
one run to the next. Widening a bound or moving a pin is a change to the benchmark itself.

## Quick start

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'

julia --project=benchmark benchmark/run.jl smoke                       # a functional check
julia --project=benchmark benchmark/run.jl acopf,cops,mittelmann,cutest           # the four sets on the host
julia --project=benchmark benchmark/run.jl acopf,cops,mittelmann,cutest --device=cuda

julia --project=benchmark benchmark/summary.jl                         # the machine, solved counts and mean times
julia --project=benchmark benchmark/compare.jl master benchmark/results           # against the archived results of master
```

## Problem sets

| set | instances | source |
|:---|---:|:---|
| `acopf` | 132 | the 66 typical-operating-point cases of pglib-opf, each in the polar and in the rectangular form, through ExaModelsPower |
| `cops` | 65 | the COPS problems of COPSBenchmark: nine scaling models at five sizes each, the other models at their canonical sizes |
| `mittelmann` | 47 | the instances scored in Hans Mittelmann's AMPL-NLP benchmark, through ExaMittelmann |
| `cutest` | 1338 | the CUTEst problems with at least as many free variables as equality constraints (`sets/cutest.txt`): through CUTEst.jl on the host, as ExaCUTEst models on a device |
| `smoke` | 12 | small ACOPF and COPS instances, for a functional check |

The sets are defined in `sets.jl`. Every set runs on the host and on a CUDA device.

## Configurations

A configuration is a named set of keyword arguments of `madnlp`: a KKT system, a linear
solver and their options. They are defined in `configs.jl`; add an entry there to benchmark
something new.

| configuration | device | KKT system | linear solver |
|:---|:---|:---|:---|
| `default` | cpu | MadNLP's default (`SparseKKTSystem`) | MadNLP's default (MUMPS) |
| `ipopt` | cpu | Ipopt's own | Ipopt's default (MUMPS), as a baseline; no time breakdown or factorization counts |
| `condensed_cholmod` | cpu | `SparseCondensedKKTSystem` | CHOLMOD |
| `augmented_ma27`, `augmented_ma57` | cpu | `SparseKKTSystem` | MA27, MA57; only with a licensed HSL_jll |
| `default` | cuda | cuMadNLP's default (`SparseCondensedKKTSystem`) | cuDSS |
| `augmented_cudss` | cuda | `SparseKKTSystem` | cuDSS with Bunch-Kaufman pivoting |

## Options of `run.jl`

| option | default | meaning |
|:---|:---|:---|
| `--config=a,b` | `default` | the configurations to run |
| `--device=cpu\|cuda\|cpu,cuda` | `cpu` | where the models and the solver live; `cpu,cuda` runs both side by side, each with its own result directory |
| `--workers=auto\|N` | `auto` | the number of processes that solve; see below |
| `--root=DIR` | `benchmark/results` | where the per-machine result directories go |
| `--out=DIR` | `<root>/<hardware>` | where the result files go; `<hardware>` is the CPU model or the GPU name |
| `--only=case,...` | all | restrict a set to the named instances |
| `--tol=1e-8` | `1e-8` | convergence tolerance |
| `--wall=7200` | `7200` | wall limit of a solve, in seconds |
| `--margin=S` | larger of 120 s and 20% of the limit | how far past the limit a worker is killed |
| `--status-every=600` | `600` | seconds between status reports |
| `--no-warmup` | | skip the discarded solves (see the protocol) |
| `--overwrite` | | start the result files afresh instead of resuming |
| `--verbose` | | print MadNLP's iteration log |

A run that is interrupted is resumed by the same command: an instance already in a result
file is skipped.

**Workers.** The instances of all the requested sets form one queue, handed out to worker
processes. On a CUDA device there are two workers per GPU (`--workers-per-gpu`), of which one solves at a time while the other builds its next model. On the host,
`auto` follows what the process may use, a batch job's allocation included: 75% of the physical
cores, limited to 3 GB per worker out of 75% of the memory. Workers that run side by side on the host share its
memory channels and slow each other down somewhat, so use `--workers=1` for the cleanest host
timings. Each worker has a heap-size hint of 6 GB and is replaced between cases if it grows
past 8 GB. The number of workers is recorded in the metadata.

**Wall limit.** MadNLP checks `max_wall_time` between iterations, so one long factorization
or evaluation can overrun it. The coordinating process therefore kills a worker when a build
or a solve has run for the wall limit plus a margin (`--margin`, by default the larger of
120 s and 20% of the limit), records the instance with status `-100`, and starts a new worker. A worker that passes 40% of
the machine's memory during a solve is treated the same way.

**Status.** Every ten minutes the log gives the instances done and, per worker, the instance,
its state (`building`, `waiting` for a GPU, `solving`, `working`) and for how long, longest
first, with its memory, followed by the solved counts and mean times so far. The same report
is written to `summary.md` beside the result files. In CI it appears live in the step's log on
GitHub.

**CUTEst.** On the host the problems are decoded once, before the workers start, and the
decoded files stay in CUTEst.jl's directory for later runs.

## Protocol

- **Options.** MadNLP's defaults, apart from what the configuration sets, the tolerance and
  the wall limit. A configuration that tunes other options says so in `configs.jl`.
- **Solved.** An instance is solved if MadNLP returns `SOLVE_SUCCEEDED` or
  `SOLVED_TO_ACCEPTABLE_LEVEL`. Every other exit, including the iteration limit with a good
  last iterate, is a failure.
- **Time.** The recorded time is MadNLP's total time of one solve, with the initialization,
  the linear solver and the function evaluations also recorded separately.
- **Compilation.** No recorded solve includes compilation. The callbacks of an ExaModel are
  compiled after the model is built, outside the timed region. Instances of one COPS model go
  to one worker, so that they are compiled once. Julia's compilation time is then measured
  around every solve, and a solve during which Julia compiled is repeated; an
  instance is also solved a second time when its model type is new to the configuration in
  that process, because the kernels of a device are compiled outside Julia's timer. The
  column `t_compile` is the compilation time of the recorded solve and `n_solves` the number
  of solves it took. A solve that reaches the wall limit is recorded as it is.
- **Aggregate.** The number of instances solved, and the shifted geometric mean of the solve
  time with a shift of 10 s, a failure counting as the wall limit. Instances are split into
  small, medium and large by the nonzeros of the Jacobian and the Hessian together: below
  2^18, below 2^20, and above.
- **Feasibility.** `primal_feas` is the residual of the solver's scaled, relaxed problem. The
  columns `viol_eq`, `viol_con` and `viol_var` are the violation of the model's own constraints and
  bounds at the returned point, evaluated independently of the solver.
- **Hardware.** Timings are comparable only on the same machine. Each result file has a
  metadata file beside it (`.meta.toml`) with the host, the CPU model, its cores and threads,
  the memory, the GPU with its memory, driver and CUDA runtime, the thread counts, the package
  versions, the MadNLP commit and the options. `summary.jl` names the machine above its
  tables.

## Result files

`run.jl` writes `<set>_<configuration>.csv` with one row per instance:

`case, status, iters, nvar, ncon, nnzj, nnzh, obj, primal_feas, dual_feas, n_fact,
n_backsolve, n_ir, n_obj, n_grad, n_con, n_jac, n_hess, t_total, t_linsolve, t_eval, t_init,
t_compile, n_solves, viol_eq, viol_con, viol_var`

`status` is MadNLP's status code; `-100` marks an instance whose worker was killed at the wall
limit, and `-101` one whose model could not be built or whose solve threw.

## Comparing and archiving

`compare.jl BASE CANDIDATE` prints, as Markdown, the machines of both sides and, per device,
set and configuration, the instances solved, the shifted geometric means and their ratio,
the ratio of iteration counts, and every instance whose outcome changed. It flags a set and
configuration with fewer instances solved or a time ratio above `--threshold` (1.1). BASE and
CANDIDATE are each a directory of result files or a git revision.

Results are archived on the `benchmark-results` branch of the repository, one directory per
machine and commit: `<hardware>/<commit>/<set>_<configuration>.csv`, with the metadata files
and a summary. A revision given to `compare.jl` is looked up there; for a device the commit
has no archived results for, its nearest archived ancestor is used and named in the output.
Results from different machines can be compared: the output names both machines and says
that their times are not comparable.

`archive.jl [DIR]` archives the result files of `DIR` (default `benchmark/results`) under
the commit recorded in their metadata.

## Plots

`benchmark/plots` is a separate environment, so that Plots stays out of a benchmark run. It
draws one performance profile of the solve time per set, with a curve per directory and
configuration:

```sh
julia --project=benchmark/plots -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/plots benchmark/plots/profiles.jl benchmark/results/*/ --out=profiles
```

## By hand

Everything CI does can be done from a checkout:

```sh
julia --project=benchmark benchmark/run.jl acopf,cops,mittelmann,cutest --device=cpu,cuda   # host and device, side by side

julia --project=benchmark benchmark/summary.jl                               # the tables
julia --project=benchmark benchmark/compare.jl master benchmark/results > comparison.md
gh pr comment <number> --body-file comparison.md                             # post it on a pull request

julia --project=benchmark benchmark/archive.jl                               # archive (needs push access)
```

The command fills the machine as CI does: workers follow the cores, the memory and the GPUs the
process can use, with the same limits, kills and resuming. In a batch job it follows the
allocation, so that one script is enough, and resubmitting it resumes the run:

```sh
#!/bin/bash
#SBATCH --nodes=1 --cpus-per-task=40 --mem=128G --gres=gpu:2 --time=24:00:00
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/run.jl acopf,cops,mittelmann,cutest --device=cpu,cuda
```

## CI

`.github/workflows/benchmark.yml` runs the benchmark on the self-hosted runner labeled
`benchmark`:

- **On request, for a pull request:** add the `benchmark` label. The four sets run with the default configuration on the
  host and on the CUDA device, with a wall limit of 900 s per solve. A comment on the pull
  request then compares the pull request with `master`: the machines, the instances solved
  and the shifted geometric mean times of both. A push to a labeled pull request
  stops the running benchmark and starts one on the new commit.
  While it runs, the comment shows the results so far, updated every ten minutes.
- **For every commit on `master`:** the same run once `build` has succeeded, archived on
  `benchmark-results` with its summary table.
- **By hand:** the workflow can be dispatched with any sets, configurations and device, on
  any branch, and can archive its results.
